//! Anthropic and hosted Claude MCP authentication fields.
use crate::{
    protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile, Secret},
    transport::HttpRequest,
};
use std::collections::BTreeMap;

pub(crate) fn foundry_mcp_endpoint(profile: &ProviderProfile) -> Result<url::Url, LlmError> {
    let endpoint = url::Url::parse(&format!(
        "{}/v1/messages",
        profile.base_url.trim_end_matches('/')
    ))
    .map_err(|_| LlmError::InvalidRequest {
        message: "Foundry MCP endpoint is invalid".into(),
    })?;
    if endpoint.scheme() != "https"
        || endpoint.host_str().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Foundry MCP credentials require the configured HTTPS Messages endpoint"
                .into(),
        });
    }
    Ok(endpoint)
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
    if (profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages)
        || profile.protocol == ProtocolFamily::FoundryClaude
    {
        let valid_foundry = profile.protocol == ProtocolFamily::FoundryClaude
            && endpoint == foundry_mcp_endpoint(profile)?;
        if !valid_foundry
            && (!crate::providers::anthropic::code_execution::is_official_profile(profile)
                || endpoint.scheme() != "https"
                || endpoint.host_str() != Some("api.anthropic.com")
                || endpoint.path() != "/v1/messages"
                || endpoint.username() != ""
                || endpoint.password().is_some()
                || endpoint.query().is_some()
                || endpoint.fragment().is_some())
        {
            return Err(LlmError::UnsupportedCapability {
                message:
                    "Anthropic MCP credentials require the matching Anthropic or configured Foundry Messages endpoint"
                        .into(),
            });
        }
        let mut body: serde_json::Value =
            serde_json::from_slice(&request.body).map_err(|_| LlmError::InvalidRequest {
                message: "Anthropic MCP request body is not JSON".into(),
            })?;
        let servers = body
            .get_mut("mcp_servers")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Anthropic MCP request has no typed server array".into(),
            })?;
        let mut injected = std::collections::BTreeSet::new();
        for server in servers {
            let Some(name) = server
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            if let Some(secret) = credentials.get(&name) {
                server["authorization_token"] =
                    serde_json::Value::String(secret.expose_secret().clone());
                injected.insert(name);
            }
        }
        if injected.len() != credentials.len() {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP tool encoding omitted an authorized server".into(),
            });
        }
        request.body = serde_json::to_vec(&body)
            .map_err(|_| LlmError::InvalidRequest {
                message: "Anthropic MCP authorized request could not be encoded".into(),
            })?
            .into();
        return Ok(());
    }
    Err(LlmError::UnsupportedCapability {
        message: "MCP credentials do not match an Anthropic Messages route".into(),
    })
}
/// Exact compact-JSON body growth caused by Anthropic's per-server
/// `authorization_token` fields. The serialized token is measured only; it is
/// never included in an error or diagnostic string.
pub(crate) fn anthropic_mcp_authorization_body_overhead(
    request: &ChatRequest,
    credentials: &BTreeMap<String, Secret<String>>,
) -> Result<usize, LlmError> {
    if credentials.is_empty() {
        return Ok(0);
    }
    let field_name_bytes =
        serde_json::to_vec("authorization_token").map_err(|_| LlmError::InvalidRequest {
            message: "Anthropic MCP authorization field size could not be measured".into(),
        })?;
    let mut overhead = 0usize;
    let mut injected = 0usize;
    for server in request.anthropic_mcp_servers() {
        let Some(secret) = credentials.get(server.name()) else {
            continue;
        };
        let token_bytes =
            serde_json::to_vec(secret.expose_secret()).map_err(|_| LlmError::InvalidRequest {
                message: "Anthropic MCP authorization size could not be measured".into(),
            })?;
        // Each typed server object already has `type`, `name`, and `url`, so
        // adding the field contributes one comma, its JSON key, a colon, and
        // the serialized JSON string value.
        let field_bytes = 1usize
            .checked_add(field_name_bytes.len())
            .and_then(|size| size.checked_add(1))
            .and_then(|size| size.checked_add(token_bytes.len()))
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "Anthropic authorization fields exceed the request size limit".into(),
            })?;
        overhead = overhead
            .checked_add(field_bytes)
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "Anthropic authorization fields exceed the request size limit".into(),
            })?;
        injected += 1;
    }
    if injected != credentials.len() {
        return Err(LlmError::InvalidRequest {
            message: "Anthropic MCP authorization does not match configured server names".into(),
        });
    }
    Ok(overhead)
}

#[cfg(test)]
mod policy_tests {
    use super::*;
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
            http1_header_layout: None,
            method: "POST".into(),
            url: "https://openrouter.ai/api/v1/chat/completions".into(),
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        }
    }

    #[test]
    fn anthropic_authorization_body_overhead_matches_compact_json_injection() {
        let chat: ChatRequest = serde_json::from_value(json!({
            "model":"claude-opus-5-5",
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "hosted_tools":[{"type":"native","config":{"format":"anthropic.hosted_tool.v1","data":{"type":"mcp","config":{
                "name":"docs", "url":"https://mcp.example.test/sse"
            }}}}]
        }))
        .unwrap();
        let credentials =
            BTreeMap::from([("docs".into(), Secret::new("token/quote\"slash\\雪".into()))]);
        let overhead = anthropic_mcp_authorization_body_overhead(&chat, &credentials).unwrap();
        let mut outgoing = request();
        outgoing.url = "https://api.anthropic.com/v1/messages".into();
        outgoing.body = serde_json::to_vec(&json!({
            "mcp_servers":[{"type":"url","name":"docs","url":"https://mcp.example.test/sse"}]
        }))
        .unwrap()
        .into();
        let base_len = outgoing.body.len();
        let mut profile = profile("anthropic", "https://api.anthropic.com");
        profile.protocol = ProtocolFamily::AnthropicMessages;
        apply_mcp_authorizations(&credentials, &profile, &mut outgoing).unwrap();
        assert_eq!(overhead, outgoing.body.len() - base_len);
    }
    #[test]
    fn foundry_mcp_credentials_are_bound_to_configured_https_endpoint() {
        let mut profile = profile(
            "custom-foundry",
            "https://account.services.ai.azure.com/anthropic/",
        );
        profile.protocol = ProtocolFamily::FoundryClaude;
        let credentials = BTreeMap::from([("docs".into(), Secret::new("fresh-token".into()))]);
        let mut outgoing = request();
        outgoing.url = "https://account.services.ai.azure.com/anthropic/v1/messages".into();
        outgoing.body = serde_json::to_vec(
            &json!({"mcp_servers":[{"type":"url","name":"docs","url":"https://mcp.example.test"}]}),
        )
        .unwrap()
        .into();
        apply_mcp_authorizations(&credentials, &profile, &mut outgoing).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&outgoing.body).unwrap();
        assert_eq!(body["mcp_servers"][0]["authorization_token"], "fresh-token");
        for url in [
            "https://other.services.ai.azure.com/anthropic/v1/messages",
            "http://account.services.ai.azure.com/anthropic/v1/messages",
            "https://account.services.ai.azure.com/anthropic/v1/messages?redirect=true",
            "https://account.services.ai.azure.com/v1/messages",
        ] {
            outgoing.url = url.into();
            assert!(apply_mcp_authorizations(&credentials, &profile, &mut outgoing).is_err());
        }
        profile.base_url = "http://account.services.ai.azure.com/anthropic".into();
        assert!(foundry_mcp_endpoint(&profile).is_err());
    }
}
