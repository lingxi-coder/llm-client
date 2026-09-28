//! Typed first-party Anthropic Messages Web Fetch declaration.

use crate::codecs::CodecContext;
use crate::protocol::{
    CacheTtl, ChatRequest, ContentBlock, LlmError, ProtocolFamily, ProviderProfile,
};
use crate::providers::anthropic::types::{
    AnthropicFetchCaller, AnthropicFetchResponseInclusion, AnthropicWebFetchConfig,
};
use serde_json::{json, Value};

// Anthropic documents dynamic filtering for Claude 4.6 and later, plus Mythos
// Preview. Keep the wire IDs explicit; do not infer compatibility by prefix.
const DYNAMIC_FILTER_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
];

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

fn is_official_profile(profile: &ProviderProfile) -> bool {
    crate::providers::anthropic::code_execution::is_official_profile(profile)
        && profile.protocol == ProtocolFamily::AnthropicMessages
}

fn supports_typed_web_fetch(profile: &ProviderProfile) -> bool {
    is_official_profile(profile) || profile.protocol == ProtocolFamily::FoundryClaude
}

/// Native Web Fetch history can resume a pending server-side action. Pin such
/// requests to one connection after an uncertain execution outcome.
pub(crate) fn has_fetch(request: &ChatRequest) -> bool {
    request.hosted_anthropic_web_fetch().is_some()
        || request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| {
                matches!(
                    block,
                    ContentBlock::ProviderContent {
                        protocol: ProtocolFamily::AnthropicMessages,
                        value,
                    } if is_web_fetch_native_block(value)
                )
            })
}

fn is_web_fetch_native_block(value: &Value) -> bool {
    let Some(kind) = value.get("type").and_then(Value::as_str) else {
        return false;
    };
    if matches!(
        kind,
        "web_fetch_tool_result" | "web_fetch_tool_result_error"
    ) {
        return true;
    }
    kind == "server_tool_use" && value.get("name").and_then(Value::as_str) == Some("web_fetch")
}

pub(crate) fn validate_profile_extra(profile: &ProviderProfile) -> Result<(), LlmError> {
    if !supports_typed_web_fetch(profile) {
        return Ok(());
    }
    let raw_web_fetch = profile
        .extra
        .get("body")
        .and_then(|body| body.get("tools"))
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.starts_with("web_fetch_"))
            })
        });
    if raw_web_fetch {
        return Err(invalid(
            "Anthropic Web Fetch must use the typed hosted tool configuration",
        ));
    }
    Ok(())
}

pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let profile = context.profile();
    validate_profile_extra(profile)?;
    let Some(config) = request.hosted_anthropic_web_fetch() else {
        return Ok(());
    };
    if config.inline_definition && !is_official_profile(profile) {
        return Err(unsupported(
            "inline Anthropic Web Fetch definitions require the first-party Messages API",
        ));
    }
    request.validate_hosted_tools()?;

    let foundry_deployment = match profile.protocol {
        ProtocolFamily::AnthropicMessages if is_official_profile(profile) => None,
        ProtocolFamily::FoundryClaude => {
            let deployment = crate::hosting::foundry::require_deployment(context)?;
            if deployment.hosting == crate::protocol::FoundryHosting::Azure {
                if config.version
                    != crate::providers::anthropic::types::AnthropicWebFetchVersion::V20250910
                {
                    return Err(unsupported(
                        "Azure-hosted Foundry supports only Anthropic Web Fetch web_fetch_20250910",
                    ));
                }
                if config
                    .allowed_callers
                    .iter()
                    .any(|caller| *caller != AnthropicFetchCaller::Direct)
                {
                    return Err(unsupported(
                        "Azure-hosted Foundry Web Fetch supports direct callers only; Code Execution callers require Anthropic hosting",
                    ));
                }
            }
            Some(deployment)
        }
        _ => {
            return Err(unsupported(
                "Anthropic Web Fetch requires the first-party Anthropic Messages endpoint or an explicitly identified Foundry deployment",
            ));
        }
    };

    config.validate()?;
    let has_matching_inline_definition =
        config.inline_definition && has_inline_web_fetch_definition(request, &tool_value(config)?);
    if config.inline_definition && !has_matching_inline_definition {
        return Err(invalid(
            "inline Anthropic Web Fetch requires a matching typed tool_addition definition",
        ));
    }
    let dynamic_filtering_requested = config.version.supports_dynamic_filtering()
        && (config.allowed_callers.is_empty()
            || config
                .allowed_callers
                .iter()
                .any(|caller| *caller != AnthropicFetchCaller::Direct));
    let dynamic_filter_model = foundry_deployment
        .map(|deployment| deployment.model_id.as_str())
        .unwrap_or_else(|| context.request_model());
    if dynamic_filtering_requested && !DYNAMIC_FILTER_MODELS.contains(&dynamic_filter_model) {
        return Err(unsupported(format!(
            "Anthropic Web Fetch dynamic filtering is not documented for model {:?}",
            dynamic_filter_model
        )));
    }
    if request.tools.iter().any(|tool| tool.name == "web_fetch") {
        return Err(invalid(
            "client tool name web_fetch conflicts with Anthropic's hosted Web Fetch",
        ));
    }
    let deferred_fetch_is_surfaced = has_matching_inline_definition
        || (!config.inline_definition && has_web_fetch_reference_addition(request));
    if config.defer_loading
        && request.hosted_anthropic_tool_search().is_none()
        && !deferred_fetch_is_surfaced
    {
        return Err(invalid(
            "deferred Anthropic Web Fetch requires Anthropic hosted tool search on the same request",
        ));
    }
    if let Some(sources) = &config.url_sources {
        sources.validate(request)?;
    }
    Ok(())
}

fn has_web_fetch_reference_addition(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            let ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } = block
            else {
                return false;
            };
            if value.get("type").and_then(Value::as_str) != Some("tool_addition") {
                return false;
            }
            value.pointer("/tool/type").and_then(Value::as_str) == Some("tool_reference")
                && value.pointer("/tool/name").and_then(Value::as_str) == Some("web_fetch")
        })
    })
}

fn has_inline_web_fetch_definition(request: &ChatRequest, expected: &Value) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            let ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } = block
            else {
                return false;
            };
            value.get("type").and_then(Value::as_str) == Some("tool_addition")
                && value.pointer("/tool/type").and_then(Value::as_str) == Some("tool_definition")
                && value.pointer("/tool/definition") == Some(expected)
        })
    })
}

pub(crate) fn tool_value(config: &AnthropicWebFetchConfig) -> Result<Value, LlmError> {
    let mut tool = json!({
        "type": format!("web_fetch_{}", config.version.as_str()),
        "name": "web_fetch",
    });
    if !config.allowed_callers.is_empty() {
        tool["allowed_callers"] = Value::Array(
            config
                .allowed_callers
                .iter()
                .map(|caller| {
                    Value::String(
                        match caller {
                            AnthropicFetchCaller::Direct => "direct",
                            AnthropicFetchCaller::CodeExecution20250825 => {
                                "code_execution_20250825"
                            }
                            AnthropicFetchCaller::CodeExecution20260120 => {
                                "code_execution_20260120"
                            }
                            AnthropicFetchCaller::CodeExecution20260521 => {
                                "code_execution_20260521"
                            }
                        }
                        .to_owned(),
                    )
                })
                .collect(),
        );
    }
    if config.defer_loading {
        tool["defer_loading"] = json!(true);
    }
    if config.strict {
        tool["strict"] = json!(true);
    }
    if let Some(ttl) = config.cache_control {
        let mut control = json!({"type":"ephemeral"});
        match ttl {
            CacheTtl::FiveMinutes => {}
            CacheTtl::OneHour => control["ttl"] = json!("1h"),
            CacheTtl::ThirtyMinutes => {
                return Err(unsupported(
                    "Anthropic Web Fetch cache_control supports only 5m or 1h",
                ));
            }
        }
        tool["cache_control"] = control;
    }
    if let Some(max_uses) = config.max_uses {
        tool["max_uses"] = json!(max_uses);
    }
    if !config.allowed_domains.is_empty() {
        tool["allowed_domains"] = json!(config.allowed_domains);
    }
    if !config.blocked_domains.is_empty() {
        tool["blocked_domains"] = json!(config.blocked_domains);
    }
    if let Some(enabled) = config.citations {
        tool["citations"] = json!({"enabled": enabled});
    }
    if let Some(max_content_tokens) = config.max_content_tokens {
        tool["max_content_tokens"] = json!(max_content_tokens);
    }
    if let Some(use_cache) = config.use_cache {
        tool["use_cache"] = json!(use_cache);
    }
    if let Some(inclusion) = config.response_inclusion {
        let value = match inclusion {
            AnthropicFetchResponseInclusion::Full => "full",
            AnthropicFetchResponseInclusion::Excluded => "excluded",
        };
        tool["response_inclusion"] = json!(value);
    }
    if let Some(sources) = &config.url_sources {
        tool["url_sources"] = sources.to_value()?;
    }
    Ok(tool)
}

pub(crate) fn apply(
    request: &ChatRequest,
    body: &mut serde_json::Map<String, Value>,
) -> Result<(), LlmError> {
    let Some(config) = request.hosted_anthropic_web_fetch() else {
        return Ok(());
    };
    let tools = body
        .entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| invalid("Anthropic Web Fetch requires tools to be an array"))?;
    if !config.inline_definition {
        tools.push(tool_value(config)?);
    }
    Ok(())
}
