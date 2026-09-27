//! First-party OpenAI Responses Code Interpreter container configuration.

use crate::codecs::CodecContext;
use crate::openai_containers::{OpenAiContainerRef, OpenAiContainerScope};
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile};
use serde_json::{json, Map, Value};
use std::collections::HashSet;

pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let Some(config) = req.hosted_code_interpreter() else {
        return Ok(());
    };
    let profile = context.profile();
    if profile.protocol != ProtocolFamily::OpenAiResponses
        || profile.provider_id.as_str() != "openai"
        || profile
            .extra
            .get("code_interpreter")
            .and_then(Value::as_str)
            != Some("openai_responses")
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare first-party Responses Code Interpreter support",
                profile.profile_name
            ),
        });
    }
    if (config.container.is_some() || !config.files.is_empty())
        && !crate::codecs::openai::responses::encode::is_official_openai_responses_profile(profile)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "scoped Code Interpreter containers and file mounts require the official OpenAI Responses endpoint".into(),
        });
    }
    if let Some(container) = &config.container {
        if config.memory_limit.is_some() || !config.files.is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "an explicit Code Interpreter container cannot be combined with automatic-container memory_limit or files settings".into(),
            });
        }
        validate_explicit_container(container, profile, context)?;
    } else {
        let mut file_ids = HashSet::with_capacity(config.files.len());
        for file in &config.files {
            let file = crate::codecs::validate_provider_file(file, profile, context)?;
            if file.protocol != ProtocolFamily::OpenAiResponses {
                return Err(crate::codecs::provider_file_protocol_error());
            }
            let id = file.file_id.as_str();
            if id.trim().is_empty()
                || id.trim() != id
                || id.len() > 512
                || id.chars().any(char::is_control)
                || id.contains('/')
                || id.contains('\\')
            {
                return Err(LlmError::InvalidRequest {
                    message: "Code Interpreter file IDs must be non-empty path-safe identifiers"
                        .into(),
                });
            }
            if !file_ids.insert(id) {
                return Err(LlmError::InvalidRequest {
                    message:
                        "Code Interpreter automatic file mounts cannot contain duplicate file IDs"
                            .into(),
                });
            }
        }
    }
    Ok(())
}

fn validate_explicit_container(
    container: &OpenAiContainerRef,
    profile: &ProviderProfile,
    context: &CodecContext,
) -> Result<(), LlmError> {
    let account_scope = context
        .account_scope()
        .filter(|scope| !scope.trim().is_empty())
        .ok_or_else(|| LlmError::UnsupportedCapability {
            message: "explicit Code Interpreter containers require RequestOptions.account_scope"
                .into(),
        })?;
    let expected_scope = OpenAiContainerScope::new(profile.profile_name.clone(), account_scope)
        .map_err(|error| LlmError::InvalidRequest {
            message: error.to_string(),
        })?;
    let expected = OpenAiContainerRef::from_id(&expected_scope, container.container_id()).map_err(
        |error| LlmError::InvalidRequest {
            message: error.to_string(),
        },
    )?;
    if &expected != container {
        return Err(LlmError::UnsupportedCapability {
            message: "Code Interpreter container belongs to a different profile, endpoint, or account scope".into(),
        });
    }
    Ok(())
}

pub(crate) fn apply(
    req: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    let Some(config) = req.hosted_code_interpreter() else {
        return Ok(());
    };
    validate(req, context)?;
    let container = if let Some(existing) = &config.container {
        json!(existing.container_id())
    } else {
        let mut container = json!({"type":"auto"});
        if let Some(limit) = config.memory_limit {
            container["memory_limit"] =
                serde_json::to_value(limit).expect("memory limit enum always serializes");
        }
        if !config.files.is_empty() {
            container["file_ids"] = Value::Array(
                config
                    .files
                    .iter()
                    .map(|file| json!(file.file_id))
                    .collect(),
            );
        }
        container
    };
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Code Interpreter requires tools to be an array".into(),
        })?
        .push(json!({"type":"code_interpreter","container":container}));

    let include = body.entry("include").or_insert_with(|| json!([]));
    let include = include
        .as_array_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Code Interpreter requires include to be an array".into(),
        })?;
    let outputs = Value::String("code_interpreter_call.outputs".into());
    if !include.contains(&outputs) {
        include.push(outputs);
    }
    Ok(())
}
