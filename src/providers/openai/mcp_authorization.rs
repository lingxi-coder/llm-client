//! OpenAI and xAI Responses MCP authentication fields.
use crate::{
    protocol::{LlmError, ProtocolFamily, ProviderProfile, Secret},
    transport::HttpRequest,
};
use std::collections::BTreeMap;

pub(crate) fn validate_mcp_profile(
    profile: &ProviderProfile,
    provider: &str,
    marker_key: &str,
    marker_value: &str,
    host: &str,
) -> Result<(), LlmError> {
    let endpoint = url::Url::parse(&profile.base_url).map_err(|_| LlmError::InvalidRequest {
        message: "remote MCP credential route is invalid".into(),
    })?;
    if profile.provider_id.as_str() != provider
        || profile.protocol != ProtocolFamily::OpenAiResponses
        || profile
            .extra
            .get(marker_key)
            .and_then(serde_json::Value::as_str)
            != Some(marker_value)
        || endpoint.scheme() != "https"
        || endpoint.host_str() != Some(host)
        || endpoint.path().trim_end_matches('/') != "/v1"
        || endpoint.username() != ""
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(LlmError::UnsupportedCapability {
            message: "remote MCP requires its enabled official Responses profile".into(),
        });
    }
    Ok(())
}

pub(crate) fn apply_mcp_authorizations(
    credentials: &BTreeMap<String, Secret<String>>,
    profile: &ProviderProfile,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    if credentials.is_empty() {
        return Ok(());
    }
    let endpoint = url::Url::parse(&request.url).map_err(|_| LlmError::InvalidRequest {
        message: "remote MCP request endpoint is invalid".into(),
    })?;
    let (provider, marker_key, marker_value, host) = if profile.provider_id.as_str() == "xai" {
        ("xai", "xai_remote_mcp", "xai_responses", "api.x.ai")
    } else {
        ("openai", "remote_mcp", "openai_responses", "api.openai.com")
    };
    validate_mcp_profile(profile, provider, marker_key, marker_value, host)?;
    if endpoint.scheme() != "https"
        || endpoint.host_str() != Some(host)
        || endpoint.path() != "/v1/responses"
        || endpoint.username() != ""
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(LlmError::UnsupportedCapability {
            message: "remote MCP credentials require the official matching Responses endpoint"
                .into(),
        });
    }
    let mut body: serde_json::Value =
        serde_json::from_slice(&request.body).map_err(|_| LlmError::InvalidRequest {
            message: "remote MCP request body is not JSON".into(),
        })?;
    let tools = body
        .get_mut("tools")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "remote MCP request has no tools array".into(),
        })?;
    let mut injected = std::collections::BTreeSet::new();
    for tool in tools {
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("mcp") {
            continue;
        }
        let Some(label) = tool
            .get("server_label")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        if let Some(secret) = credentials.get(&label) {
            tool["authorization"] = serde_json::Value::String(secret.expose_secret().clone());
            injected.insert(label);
        }
    }
    if injected.len() != credentials.len() {
        return Err(LlmError::InvalidRequest {
            message: "remote MCP tool encoding omitted an authorized server".into(),
        });
    }
    request.body = serde_json::to_vec(&body)
        .map_err(|_| LlmError::InvalidRequest {
            message: "remote MCP authorized request could not be encoded".into(),
        })?
        .into();
    Ok(())
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    use crate::protocol::ChatRequest;
    use crate::providers::dispatch::validate_mcp_authorizations;
    use bytes::Bytes;
    use serde_json::json;
    fn profile(provider: &str, base_url: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id":provider, "profile_name":"route", "base_url":base_url,
            "protocol":"open_ai_chat", "auth":"none"
        }))
        .unwrap()
    }

    fn request() -> HttpRequest {
        HttpRequest {
            method: "POST".into(),
            url: "https://openrouter.ai/api/v1/chat/completions".into(),
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        }
    }

    #[test]
    fn remote_mcp_token_is_added_only_to_matching_encoded_tool() {
        let chat: ChatRequest = serde_json::from_value(json!({
            "model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "hosted_tools":[{"type":"native","config":{"format":"openai.hosted_tool.v1","data":{"type":"remote_mcp","config":{
                "server_label":"docs", "server_url":"https://mcp.example.test"
            }}}}]
        }))
        .unwrap();
        let mut profile = profile("openai", "https://api.openai.com/v1");
        profile.protocol = ProtocolFamily::OpenAiResponses;
        profile.extra = serde_json::json!({"remote_mcp":"openai_responses"});
        let credentials = BTreeMap::from([("docs".into(), Secret::new("Bearer fresh".into()))]);
        validate_mcp_authorizations(&credentials, &chat, &profile).unwrap();
        let mut outgoing = request();
        outgoing.url = "https://api.openai.com/v1/responses".into();
        outgoing.body = serde_json::to_vec(&json!({
            "tools":[{"type":"mcp","server_label":"docs"},{"type":"web_search"}]
        }))
        .unwrap()
        .into();
        apply_mcp_authorizations(&credentials, &profile, &mut outgoing).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&outgoing.body).unwrap();
        assert_eq!(body["tools"][0]["authorization"], "Bearer fresh");
        assert!(body["tools"][1].get("authorization").is_none());
        assert!(!serde_json::to_string(&chat)
            .unwrap()
            .contains("Bearer fresh"));
    }
}
