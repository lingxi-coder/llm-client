//! Qwen-specific hosted tools on the OpenAI-compatible Responses route.
//!
//! The wire is compatible with Responses, but Code Interpreter is a Qwen
//! adapter: it has no OpenAI container configuration, requires thinking, and
//! reports its own tool-call counter.

use crate::protocol::{
    ChatRequest, LlmError, ProtocolFamily, ProviderProfile, ReasoningEffort, ThinkingMode,
    ToolChoice,
};
use serde_json::{json, Map, Value};

const QWEN_CODE_INTERPRETER_MODELS: &[&str] = &["qwen3.8-max", "qwen3.8-flash"];

/// Whether this is one of the Qwen Responses profiles/models for which the
/// current provider catalog and first-party documentation confirm the tool.
pub(crate) fn supports_code_interpreter(profile: &ProviderProfile, model: &str) -> bool {
    profile.provider_id.as_str() == "qwen"
        && profile.protocol == ProtocolFamily::OpenAiResponses
        && profile.extra.get("web_search").and_then(Value::as_str) == Some("qwen")
        && QWEN_CODE_INTERPRETER_MODELS.contains(&model)
}

/// Validate the Qwen-specific contract when the caller requests this tool.
/// Returns `true` when this adapter owns the request, `false` when another
/// Responses provider should handle Code Interpreter.
pub(crate) fn validate(
    req: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<bool, LlmError> {
    let Some(config) = req.hosted_code_interpreter() else {
        return Ok(false);
    };
    if profile.provider_id.as_str() != "qwen" {
        return Ok(false);
    }

    let unsupported = |message: &str| LlmError::UnsupportedCapability {
        message: format!(
            "Qwen Code Interpreter on profile {:?}: {message}",
            profile.profile_name
        ),
    };
    if profile.protocol != ProtocolFamily::OpenAiResponses
        || profile.extra.get("web_search").and_then(Value::as_str) != Some("qwen")
    {
        return Err(unsupported(
            "use a Qwen Responses profile that declares the Qwen web-search adapter",
        ));
    }
    if !QWEN_CODE_INTERPRETER_MODELS.contains(&model) {
        return Err(unsupported(
            "the selected model is outside the verified Qwen 3.8 Max/Flash catalog",
        ));
    }
    if config.memory_limit.is_some() {
        return Err(unsupported(
            "Qwen accepts {type: code_interpreter} without OpenAI container memory settings",
        ));
    }
    if config.container.is_some() || !config.files.is_empty() {
        return Err(unsupported(
            "OpenAI container references and automatic file mounts are not part of the Qwen contract",
        ));
    }
    if !req.tools.is_empty() {
        return Err(LlmError::InvalidRequest {
            message: "Qwen Code Interpreter and custom function tools are mutually exclusive"
                .into(),
        });
    }
    if !matches!(req.tool_choice, ToolChoice::Auto) {
        return Err(unsupported(
            "only automatic tool choice is documented for this hosted tool",
        ));
    }
    if req.thinking.as_ref().is_some_and(|thinking| {
        thinking.mode == Some(ThinkingMode::Disabled)
            || thinking.effort == Some(ReasoningEffort::None)
    }) {
        return Err(LlmError::InvalidRequest {
            message: "Qwen Code Interpreter requires thinking to remain enabled".into(),
        });
    }
    if let Some(enabled) = profile.extra["body"].get("enable_thinking") {
        if enabled.as_bool() != Some(true) {
            return Err(LlmError::InvalidRequest {
                message: "Qwen Code Interpreter conflicts with extra.body.enable_thinking".into(),
            });
        }
    }
    if profile.extra["body"]["reasoning"]["effort"].as_str() == Some("none") {
        return Err(LlmError::InvalidRequest {
            message: "Qwen Code Interpreter conflicts with extra.body.reasoning.effort=none".into(),
        });
    }
    Ok(true)
}

/// Add the documented Qwen Responses tool and required thinking flag.
/// Returns `true` when the Qwen adapter handled the tool.
pub(crate) fn apply(
    req: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
    body: &mut Map<String, Value>,
) -> Result<bool, LlmError> {
    if !validate(req, profile, model)? {
        return Ok(false);
    }
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Qwen hosted tools require tools to be an array".into(),
        })?
        .push(json!({"type":"code_interpreter"}));
    body.insert("enable_thinking".into(), Value::Bool(true));
    Ok(true)
}
