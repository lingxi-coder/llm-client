//! Provider-owned Claude Code Fast organization-status transport.
use crate::protocol::{JsonPayload, LlmError};
use crate::{HttpExecutor, HttpRequest, Transport};
use std::{fmt, time::Duration};

#[derive(Clone, Copy)]
pub enum FastModeCredential<'a> {
    ApiKey(&'a str),
    OAuth(&'a str),
}
impl fmt::Debug for FastModeCredential<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FastModeCredential(<redacted>)")
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastModeStatus {
    pub enabled: bool,
    pub disabled_reason: Option<String>,
}
#[derive(Debug, thiserror::Error)]
pub enum FastModeStatusError {
    #[error("invalid Fast status endpoint: {0}")]
    Endpoint(String),
    #[error("Fast status transport failed: {0}")]
    Transport(#[from] LlmError),
    #[error("Fast status endpoint returned HTTP {status}")]
    Http { status: u16, body: JsonPayload },
    #[error("Fast status response is null")]
    NullResponse,
}
impl FastModeStatusError {
    /// Native Rbe retry trigger; the host separately checks OAuth profile scope.
    pub fn requests_oauth_refresh(&self) -> bool {
        matches!(self, Self::Http { status: 401, .. })
            || matches!(self, Self::Http { status: 403, body } if body.string_at("").is_some_and(|message| message.contains("OAuth token has been revoked")))
    }
}
fn truthy(value: Option<JsonPayload>) -> bool {
    let Some(value) = value else {
        return false;
    };
    match value.kind_at("") {
        Some(crate::response_json::FlatJsonKind::Null) | None => false,
        Some(crate::response_json::FlatJsonKind::Boolean) => value.bool_at("").unwrap_or(false),
        Some(crate::response_json::FlatJsonKind::Number) => {
            value.number_at("").is_some_and(|number| number != 0.0)
        }
        Some(crate::response_json::FlatJsonKind::String) => value
            .root_string_code_units()
            .is_some_and(|units| !units.is_empty()),
        Some(
            crate::response_json::FlatJsonKind::Array | crate::response_json::FlatJsonKind::Object,
        ) => true,
    }
}
/// Native ox(): one GET, ten-second timeout, API-key or OAuth headers. The
/// embedding host owns refresh, caching, policy and storage; this is transport.
pub async fn fetch_status(
    transport: &dyn Transport,
    base_url: &str,
    credential: FastModeCredential<'_>,
    user_agent: &str,
) -> Result<FastModeStatus, FastModeStatusError> {
    let mut url =
        url::Url::parse(base_url).map_err(|e| FastModeStatusError::Endpoint(e.to_string()))?;
    if !matches!(url.scheme(), "https" | "http")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(FastModeStatusError::Endpoint(
            "expected an HTTP base without credentials, query or fragment".into(),
        ));
    }
    url.set_path(&format!(
        "{}/api/claude_code_penguin_mode",
        url.path().trim_end_matches('/')
    ));
    let mut headers = vec![("User-Agent".into(), user_agent.into())];
    match credential {
        FastModeCredential::ApiKey(key) => headers.push(("x-api-key".into(), key.into())),
        FastModeCredential::OAuth(token) => {
            headers.push(("Authorization".into(), format!("Bearer {token}")));
            headers.push(("anthropic-beta".into(), "oauth-2025-04-20".into()));
        }
    }
    let response = HttpExecutor::new(transport)
        .execute_bounded(
            HttpRequest {
                http1_header_layout: None,
                method: "GET".into(),
                url: url.into(),
                headers,
                body: Default::default(),
                timeout: Some(Duration::from_secs(10)),
            },
            64 * 1024,
        )
        .await?;
    let response_text = String::from_utf8_lossy(&response.body);
    let body =
        JsonPayload::parse(&response_text).unwrap_or_else(|_| JsonPayload::string(&response_text));
    if !(200..300).contains(&response.status) {
        return Err(FastModeStatusError::Http {
            status: response.status,
            body,
        });
    }
    if body.kind_at("") == Some(crate::response_json::FlatJsonKind::Null) {
        return Err(FastModeStatusError::NullResponse);
    }
    let enabled = truthy(body.payload_at("/enabled"));
    let disabled_reason = if enabled {
        None
    } else {
        let reason = body.string_at("/disabled_reason");
        Some(
            match reason.as_deref() {
                Some("free") => "free",
                Some("preference") => "preference",
                Some("extra_usage_disabled") => "extra_usage_disabled",
                Some("network_error") => "network_error",
                None if body.kind_at("/disabled_reason").is_none()
                    || body.kind_at("/disabled_reason")
                        == Some(crate::response_json::FlatJsonKind::Null) =>
                {
                    "preference"
                }
                _ => "unknown",
            }
            .into(),
        )
    };
    Ok(FastModeStatus {
        enabled,
        disabled_reason,
    })
}
