//! Read-only connection probes and paginated live model directories.
use super::*;
use crate::protocol::Secret;
use crate::{
    auth::{ApiKeyAuthenticator, BearerAuthenticator},
    Authenticator, HttpExecutor, Transport,
};

/// Selects a directory/auth protocol, independent of a model generation.
#[derive(Clone, Copy, Debug)]
pub enum ProbeProtocol {
    Anthropic,
    Google,
    OpenAi,
    ChatGpt,
}
/// Caller-owned secrets; Debug never exposes credentials.
pub struct ProbeCredential {
    pub token: Secret<String>,
    pub bearer: bool,
    pub account_id: Option<String>,
    pub fedramp: bool,
}
#[derive(Debug)]
pub struct ProbeResult {
    pub status: u16,
    /// None means a reachable endpoint did not provide a valid model directory.
    pub model_ids: Option<Vec<String>>,
}
/// Query a provider without generating text. One deadline covers all pages.
pub async fn probe(
    transport: &dyn Transport,
    base: &str,
    protocol: ProbeProtocol,
    credential: &ProbeCredential,
    timeout: Duration,
) -> Result<ProbeResult, LlmError> {
    let base = base.trim().trim_end_matches('/');
    let url = url::Url::parse(base).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid());
    }
    let family = match protocol {
        ProbeProtocol::Anthropic => ProtocolFamily::AnthropicMessages,
        ProbeProtocol::Google => ProtocolFamily::GeminiGenerateContent,
        _ => ProtocolFamily::OpenAiChat,
    };
    let root = base
        .strip_suffix("/models")
        .or_else(|| base.strip_suffix("/chat/completions"))
        .unwrap_or(base);
    let root = if matches!(protocol, ProbeProtocol::Anthropic) {
        root.strip_suffix("/v1").unwrap_or(root)
    } else {
        root
    };
    let profile: ProviderProfile = serde_json::from_value(serde_json::json!({"provider_id":"probe","profile_name":"probe","base_url":root,"protocol":family,"auth":"api_key","models":[],"chat_enabled":false})).map_err(|_| invalid())?;
    let directory = super::builtin()
        .into_iter()
        .find(|d| d.shape() == family)
        .ok_or_else(invalid)?;
    let deadline = crate::runtime::Deadline::after(Some(timeout));
    let executor = HttpExecutor::new(transport).with_deadline(deadline);
    let mut cursor = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut ids = Vec::new();
    loop {
        let mut request = directory.list_request(&profile, cursor.as_deref());
        if matches!(protocol, ProbeProtocol::ChatGpt) {
            request.url = format!("{root}/models");
        }
        if credential.bearer {
            BearerAuthenticator
                .apply(&mut request, &profile, Some(&credential.token))
                .await?;
            if matches!(protocol, ProbeProtocol::Anthropic) {
                request
                    .headers
                    .push(("anthropic-beta".into(), "oauth-2025-04-20".into()));
            }
        } else {
            ApiKeyAuthenticator
                .apply(&mut request, &profile, Some(&credential.token))
                .await?;
        }
        if matches!(protocol, ProbeProtocol::ChatGpt) {
            if let Some(id) = &credential.account_id {
                request
                    .headers
                    .push(("ChatGPT-Account-ID".into(), id.clone()));
            }
            if credential.fedramp {
                request
                    .headers
                    .push(("X-OpenAI-Fedramp".into(), "true".into()));
            }
        }
        let response = executor.execute_bounded(request, 16 * 1024 * 1024).await?;
        if !(200..300).contains(&response.status) {
            return Ok(ProbeResult {
                status: response.status,
                model_ids: None,
            });
        }
        let page = if matches!(protocol, ProbeProtocol::ChatGpt) {
            let body: Value = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
            let Some(rows) = body
                .get("models")
                .or_else(|| body.get("data"))
                .and_then(Value::as_array)
            else {
                return Ok(ProbeResult {
                    status: response.status,
                    model_ids: None,
                });
            };
            let ids = rows
                .iter()
                .map(|row| {
                    row.get("slug")
                        .or_else(|| row.get("id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<Option<Vec<_>>>();
            return Ok(ProbeResult {
                status: response.status,
                model_ids: ids,
            });
        } else {
            directory.decode_page(&response)
        };
        let Ok(page) = page else {
            return Ok(ProbeResult {
                status: response.status,
                model_ids: None,
            });
        };
        ids.extend(page.models.into_iter().map(|model| model.request_model));
        match page.next_cursor {
            Some(next) if seen.insert(next.clone()) && seen.len() <= 100 => cursor = Some(next),
            Some(_) => {
                return Err(LlmError::ProviderInternal {
                    message: "model directory pagination did not terminate".into(),
                })
            }
            None => {
                return Ok(ProbeResult {
                    status: response.status,
                    model_ids: Some(ids),
                })
            }
        }
    }
}
fn invalid() -> LlmError {
    LlmError::InvalidRequest {
        message: "invalid provider directory configuration".into(),
    }
}
