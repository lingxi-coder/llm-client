//! Alibaba Model Studio's hosted web extractor on the Qwen Responses route.
//!
//! This is an independent tool from web search. The documented Qwen contract
//! requires callers to enable both tools and thinking in one Responses call.

use crate::protocol::{
    ChatRequest, LlmError, ProtocolFamily, ProviderProfile, ReasoningEffort, ThinkingMode,
    ToolChoice,
};
use serde_json::{json, Map, Value};

const QWEN_WEB_EXTRACTOR_MODELS: &[&str] = &["qwen3.8-max", "qwen3.8-flash"];

/// Validate the Qwen-specific contract when the caller requests extraction.
/// Returns `true` when the Qwen adapter owns the request.
pub(crate) fn validate(
    req: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<bool, LlmError> {
    if !req.has_hosted_web_extractor() {
        return Ok(false);
    }

    let unsupported = |message: &str| LlmError::UnsupportedCapability {
        message: format!(
            "Qwen web extractor on profile {:?}: {message}",
            profile.profile_name
        ),
    };
    if profile.provider_id.as_str() != "qwen"
        || profile.protocol != ProtocolFamily::OpenAiResponses
        || profile.extra.get("web_search").and_then(Value::as_str) != Some("qwen")
    {
        return Err(unsupported(
            "use a Qwen Responses profile that declares the Qwen web-search adapter",
        ));
    }
    if !QWEN_WEB_EXTRACTOR_MODELS.contains(&model) {
        return Err(unsupported(
            "the selected model is outside the verified Qwen 3.8 Max/Flash catalog",
        ));
    }
    if req.hosted_web_search().is_none() {
        return Err(LlmError::InvalidRequest {
            message: "Qwen web extractor requires HostedTool::WebSearch in the same request".into(),
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
            message: "Qwen web extractor requires thinking to remain enabled".into(),
        });
    }
    if let Some(enabled) = profile.extra["body"].get("enable_thinking") {
        if enabled.as_bool() != Some(true) {
            return Err(LlmError::InvalidRequest {
                message: "Qwen web extractor conflicts with extra.body.enable_thinking".into(),
            });
        }
    }
    if profile.extra["body"]["reasoning"]["effort"].as_str() == Some("none") {
        return Err(LlmError::InvalidRequest {
            message: "Qwen web extractor conflicts with extra.body.reasoning.effort=none".into(),
        });
    }
    Ok(true)
}

/// Add the documented Qwen Responses tool and required thinking flag.
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
            message: "Qwen web extractor requires tools to be an array".into(),
        })?
        .push(json!({"type":"web_extractor"}));
    body.insert("enable_thinking".into(), Value::Bool(true));
    Ok(true)
}
