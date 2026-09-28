//! xAI Responses hosted MCP contract.
use crate::protocol::{LlmError, ProviderProfile};
use serde_json::Value;

pub(crate) fn is_official_xai_responses_profile(profile: &ProviderProfile) -> bool {
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    profile.provider_id.as_str() == "xai"
        && profile.protocol == crate::protocol::ProtocolFamily::OpenAiResponses
        && profile.extra.get("xai_remote_mcp").and_then(Value::as_str) == Some("xai_responses")
        && url.scheme() == "https"
        && url.host_str() == Some("api.x.ai")
        && url.path().trim_end_matches('/') == "/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

/// Request-level capability gate shared by the codec and preflight path.
pub(crate) fn validate_xai_remote_mcp(
    request: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    request.validate_hosted_tools()?;
    let configs: Vec<_> = request.xai_remote_mcp_servers().collect();
    if configs.is_empty() {
        return Ok(());
    }
    if request.remote_mcp_servers().next().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI and xAI remote MCP configurations cannot be mixed".into(),
        });
    }
    if !is_official_xai_responses_profile(profile) {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "xAI remote MCP requires an enabled official xAI Responses profile; {:?} is not enabled",
                profile.profile_name
            ),
        });
    }
    for config in configs {
        config.validate()?;
    }
    Ok(())
}

pub(crate) fn uses_openai_approval_semantics(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() != "xai"
}
