//! What a caller says about one request.

use crate::{
    protocol::{ChatRequest, HostedTool, LlmError, ProtocolFamily, ProviderProfile, Secret},
    transport::HttpRequest,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// Default wall-clock deadline for a completion request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// OpenRouter gateway response caching, independent of provider prompt caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRouterResponseCache {
    /// Send `X-OpenRouter-Cache: false`, including when a remote preset enables it.
    Disabled,
    /// Cache this exact request body. Refresh replaces only its matching entry.
    Enabled {
        ttl_seconds: Option<u32>,
        refresh: bool,
    },
}

pub(crate) fn validate_openrouter_response_cache(
    policy: Option<OpenRouterResponseCache>,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    let Some(policy) = policy else {
        return Ok(());
    };
    let base = url::Url::parse(&profile.base_url).map_err(|_| LlmError::InvalidRequest {
        message: "OpenRouter response-cache profile has an invalid endpoint".into(),
    })?;
    if profile.provider_id.as_str() != "openrouter"
        || !super::response_cache::official_profile_base(base.as_str())
    {
        return Err(LlmError::UnsupportedCapability {
            message: "gateway response caching requires an official OpenRouter HTTPS route".into(),
        });
    }
    if matches!(policy, OpenRouterResponseCache::Enabled { ttl_seconds: Some(ttl), .. } if !(1..=86_400).contains(&ttl))
    {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter response-cache TTL must be 1–86400 seconds".into(),
        });
    }
    Ok(())
}

pub(crate) fn apply_openrouter_response_cache(
    policy: Option<OpenRouterResponseCache>,
    profile: &ProviderProfile,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    validate_openrouter_response_cache(policy, profile)?;
    let Some(policy) = policy else {
        return Ok(());
    };
    let url = url::Url::parse(&request.url).map_err(|_| LlmError::InvalidRequest {
        message: "OpenRouter response-cache request has an invalid endpoint".into(),
    })?;
    if !super::response_cache::official_request_url(url.as_str()) {
        return Err(LlmError::UnsupportedCapability {
            message: "gateway response caching requires a documented OpenRouter API endpoint"
                .into(),
        });
    }
    match policy {
        OpenRouterResponseCache::Disabled => request
            .headers
            .push(("X-OpenRouter-Cache".into(), "false".into())),
        OpenRouterResponseCache::Enabled {
            ttl_seconds,
            refresh,
        } => {
            request
                .headers
                .push(("X-OpenRouter-Cache".into(), "true".into()));
            if let Some(ttl) = ttl_seconds {
                request
                    .headers
                    .push(("X-OpenRouter-Cache-TTL".into(), ttl.to_string()));
            }
            if refresh {
                request
                    .headers
                    .push(("X-OpenRouter-Cache-Clear".into(), "true".into()));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_mcp_authorizations(
    credentials: &BTreeMap<String, Secret<String>>,
    request: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    crate::codecs::anthropic_mcp::validate_profile_extra(profile)?;
    let openai_configs = request.remote_mcp_servers().collect::<Vec<_>>();
    let xai_configs = request.xai_remote_mcp_servers().collect::<Vec<_>>();
    let anthropic_configs = request.anthropic_mcp_servers().collect::<Vec<_>>();
    if !openai_configs.is_empty() || !xai_configs.is_empty() || !anthropic_configs.is_empty() {
        request.validate_hosted_tools()?;
    }
    let configured_families = usize::from(!openai_configs.is_empty())
        + usize::from(!xai_configs.is_empty())
        + usize::from(!anthropic_configs.is_empty());
    if configured_families > 1 {
        return Err(LlmError::UnsupportedCapability {
            message: "Anthropic, OpenAI, and xAI remote MCP configurations cannot be mixed".into(),
        });
    }
    if !openai_configs.is_empty() {
        validate_mcp_profile(
            profile,
            "openai",
            "remote_mcp",
            "openai_responses",
            "api.openai.com",
        )?;
    }
    if !xai_configs.is_empty() {
        validate_mcp_profile(
            profile,
            "xai",
            "xai_remote_mcp",
            "xai_responses",
            "api.x.ai",
        )?;
    }
    if !anthropic_configs.is_empty()
        && !crate::codecs::anthropic_code_execution::is_official_profile(profile)
        && profile.protocol != ProtocolFamily::FoundryClaude
    {
        return Err(LlmError::UnsupportedCapability {
            message: "typed Anthropic MCP requires Anthropic Messages or Foundry Claude".into(),
        });
    }
    if credentials.is_empty() {
        return Ok(());
    }
    if !anthropic_configs.is_empty() {
        if profile.protocol == ProtocolFamily::FoundryClaude {
            foundry_mcp_endpoint(profile)?;
        }
        let names = anthropic_configs
            .iter()
            .map(|config| config.name())
            .collect::<std::collections::BTreeSet<_>>();
        for (name, credential) in credentials {
            if !names.contains(name.as_str())
                || credential.expose_secret().trim().is_empty()
                || credential.expose_secret().len() > 16 * 1024
                || credential.expose_secret().chars().any(char::is_control)
            {
                return Err(LlmError::InvalidRequest {
                    message:
                        "Anthropic MCP authorization has an unknown server name or invalid secret"
                            .into(),
                });
            }
        }
        return Ok(());
    }
    let expected_provider = if !xai_configs.is_empty() {
        "xai"
    } else {
        "openai"
    };
    let expected_marker = if expected_provider == "xai" {
        ("xai_remote_mcp", "xai_responses", "api.x.ai")
    } else {
        ("remote_mcp", "openai_responses", "api.openai.com")
    };
    if openai_configs.is_empty() && xai_configs.is_empty()
        || validate_mcp_profile(
            profile,
            expected_provider,
            expected_marker.0,
            expected_marker.1,
            expected_marker.2,
        )
        .is_err()
    {
        return Err(LlmError::UnsupportedCapability {
            message: "remote MCP credentials do not match the active provider profile".into(),
        });
    }
    let labels = request
        .hosted_tools
        .iter()
        .filter_map(|tool| match tool {
            HostedTool::RemoteMcp(config) => Some(config.server_label()),
            HostedTool::XaiRemoteMcp(config) => Some(config.server_label()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    for (label, credential) in credentials {
        if !labels.contains(label.as_str())
            || credential.expose_secret().trim().is_empty()
            || credential.expose_secret().len() > 16 * 1024
        {
            return Err(LlmError::InvalidRequest {
                message: "remote MCP authorization has an unknown label or invalid secret".into(),
            });
        }
    }
    Ok(())
}

fn validate_mcp_profile(
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

fn foundry_mcp_endpoint(profile: &ProviderProfile) -> Result<url::Url, LlmError> {
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
            && (!crate::codecs::anthropic_code_execution::is_official_profile(profile)
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

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Optional host policy applied before signing the final wire bytes.
    pub finalizer: Option<std::sync::Arc<dyn RequestFinalizer>>,
    /// The secret this request authenticates with, if the profile needs one.
    ///
    /// Passed in rather than looked up: this crate does not hold, fetch or
    /// store credentials. The caller owns them — including expiry, refresh and
    /// whatever secure storage the platform provides — and hands over one
    /// already-valid credential per request. A profile whose `auth` is `None`
    /// wants `None` here.
    pub credential: Option<Secret<String>>,
    /// Credentials for named failover profiles. The primary credential is
    /// never sent to another connection unless supplied here for that profile.
    pub fallback_credentials: BTreeMap<String, Secret<String>>,
    /// Stable, non-secret account identity for stateful response continuation.
    /// A response ID is returned as a reusable `ContinuationRef` only when this
    /// is set; the same scope is required on the next request.
    pub account_scope: Option<String>,
    /// Stable, non-secret identity of the account used for provider file
    /// caching. Supply the same value only for the same provider account; when
    /// absent, uploaded files are scoped to the current request and are not
    /// reused across calls. Qwen automatic uploads are request-scoped even
    /// when this value is stable, because Qwen does not expire stored files.
    pub file_account_scope: Option<String>,
    /// Total request deadline, including response body reads. `complete()`
    /// defaults to 120 seconds when omitted, or two hours for a request with
    /// video content; `stream()` has no default total
    /// deadline and relies on the transport's idle-read timeout. Automatic
    /// Qwen cleanup uses only the remaining budget, then retries in the background.
    pub total_timeout: Option<Duration>,
    /// Per-request OpenRouter gateway response-cache policy. This is not a
    /// provider prompt-cache breakpoint or a client-side response cache.
    pub openrouter_response_cache: Option<OpenRouterResponseCache>,
    /// Fresh authorization values for hosted remote MCP servers, keyed by
    /// OpenAI/xAI `server_label` or Anthropic server `name`. Never serialized
    /// into ChatRequest/history.
    pub mcp_authorizations: BTreeMap<String, Secret<String>>,
}

/// Per-request transformation of the encoded request, before authentication.
/// Implementations must not dispatch requests or fetch credentials. The final
/// bytes are authenticated once and immutable after preparation.
pub trait RequestFinalizer: std::fmt::Debug + Send + Sync {
    fn finalize(
        &self,
        request: &mut crate::HttpRequest,
        profile: &crate::protocol::ProviderProfile,
    ) -> Result<(), crate::protocol::LlmError>;
}

#[cfg(test)]
mod tests {
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
            method: "POST".into(),
            url: "https://openrouter.ai/api/v1/chat/completions".into(),
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        }
    }

    #[test]
    fn response_cache_headers_are_separate_from_prompt_cache() {
        let profile = profile("openrouter", "https://openrouter.ai/api/v1");
        let mut outgoing = request();
        apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Enabled {
                ttl_seconds: Some(600),
                refresh: true,
            }),
            &profile,
            &mut outgoing,
        )
        .unwrap();
        assert_eq!(
            outgoing.headers,
            [
                ("X-OpenRouter-Cache".into(), "true".into()),
                ("X-OpenRouter-Cache-TTL".into(), "600".into()),
                ("X-OpenRouter-Cache-Clear".into(), "true".into()),
            ]
        );
        let mut disabled = request();
        apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Disabled),
            &profile,
            &mut disabled,
        )
        .unwrap();
        assert_eq!(
            disabled.headers,
            [("X-OpenRouter-Cache".into(), "false".into())]
        );
    }

    #[test]
    fn response_cache_rejects_non_openrouter_and_bad_ttl_before_http() {
        let mut request = request();
        assert!(apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Disabled),
            &profile("openai", "https://api.openai.com/v1"),
            &mut request,
        )
        .is_err());
        assert!(request.headers.is_empty());
        assert!(apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Enabled {
                ttl_seconds: Some(86_401),
                refresh: false,
            }),
            &profile("openrouter", "https://openrouter.ai/api/v1"),
            &mut request,
        )
        .is_err());
        assert!(request.headers.is_empty());
    }

    #[test]
    fn response_cache_accepts_only_documented_openrouter_endpoint_urls() {
        let canonical_profile = profile("openrouter", "https://openrouter.ai/api/v1");
        for path in [
            "/api/v1/chat/completions",
            "/api/v1/responses",
            "/api/v1/messages",
            "/api/v1/embeddings",
        ] {
            let mut outgoing = request();
            outgoing.url = format!("https://openrouter.ai{path}");
            apply_openrouter_response_cache(
                Some(OpenRouterResponseCache::Disabled),
                &canonical_profile,
                &mut outgoing,
            )
            .unwrap_or_else(|error| panic!("documented route {path} rejected: {error}"));
        }

        for endpoint in [
            "https://openrouter.ai/api/v1/custom/responses",
            "https://openrouter.ai/api/v1/responses?redirect=https://evil.test",
            "https://openrouter.ai:444/api/v1/responses",
            "https://user@openrouter.ai/api/v1/responses",
        ] {
            let mut outgoing = request();
            outgoing.url = endpoint.into();
            assert!(
                apply_openrouter_response_cache(
                    Some(OpenRouterResponseCache::Disabled),
                    &canonical_profile,
                    &mut outgoing,
                )
                .is_err(),
                "unexpectedly accepted {endpoint}"
            );
            assert!(outgoing.headers.is_empty());
        }

        for base in [
            "https://openrouter.ai/api/v1/custom",
            "https://openrouter.ai/api/v1?proxy=1",
            "https://openrouter.ai:444/api/v1",
        ] {
            let mut outgoing = request();
            assert!(
                apply_openrouter_response_cache(
                    Some(OpenRouterResponseCache::Disabled),
                    &profile("openrouter", base),
                    &mut outgoing,
                )
                .is_err(),
                "unexpectedly accepted base URL {base}"
            );
        }
    }

    #[test]
    fn remote_mcp_token_is_added_only_to_matching_encoded_tool() {
        let chat: ChatRequest = serde_json::from_value(json!({
            "model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "hosted_tools":[{"type":"remote_mcp","config":{
                "server_label":"docs", "server_url":"https://mcp.example.test"
            }}]
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

    #[test]
    fn anthropic_authorization_body_overhead_matches_compact_json_injection() {
        let chat: ChatRequest = serde_json::from_value(json!({
            "model":"claude-opus-5-5",
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "hosted_tools":[{"type":"anthropic_mcp","config":{
                "name":"docs", "url":"https://mcp.example.test/sse"
            }}]
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
