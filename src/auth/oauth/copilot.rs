//! GitHub Copilot device authorization and OAuth-token exchange.
//!
//! The caller owns polling sleeps, credential storage, and client-id selection.
//! Each operation makes exactly one bounded request through the injected SDK transport.

use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use serde_json::{json, Value};

use crate::{
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, Transport},
};

const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
pub const DEFAULT_GITHUB_DOMAIN: &str = "github.com";
pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub const COPILOT_TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";
pub const COPILOT_TOKEN_REFRESH_SKEW_SECS: u64 = 300;

/// Secret token returned by GitHub. The explicit accessor is for host storage
/// and request authentication; formatting the value never reveals it.
#[derive(Clone)]
pub struct CopilotSecret(String);

impl CopilotSecret {
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }
    #[doc(hidden)]
    pub fn token_for_storage(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for CopilotSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CopilotSecret(<redacted>)")
    }
}

#[derive(Clone)]
pub struct DeviceCodeResponse {
    pub user_code: String,
    pub verification_uri: String,
    pub device_code: String,
    pub interval_secs: u64,
    pub domain: String,
}

impl std::fmt::Debug for DeviceCodeResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceCodeResponse")
            .field("user_code", &self.user_code)
            .field("verification_uri", &self.verification_uri)
            .field("device_code", &"<redacted>")
            .field("interval_secs", &self.interval_secs)
            .field("domain", &self.domain)
            .finish()
    }
}

#[derive(Debug)]
pub enum PollOutcome {
    Success(CopilotSecret),
    Pending { interval_secs: u64 },
    SlowDown { interval_secs: u64 },
    Failed { error: String },
}

/// One-step device-flow client. Hosts handle scheduling and user interaction.
pub struct CopilotLogin {
    transport: Arc<dyn Transport>,
    client_id: String,
    user_agent: String,
}

impl CopilotLogin {
    pub fn new(
        transport: Arc<dyn Transport>,
        client_id: impl Into<String>,
        user_agent: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            client_id: client_id.into(),
            user_agent: user_agent.into(),
        }
    }

    pub async fn begin(&self, domain: &str) -> Result<DeviceCodeResponse, LlmError> {
        let url = github_url(domain, "/login/device/code")?;
        let response = post_json(
            self.transport.as_ref(),
            url,
            &self.user_agent,
            json!({"client_id": self.client_id, "scope": "read:user"}),
        )
        .await?;
        Ok(DeviceCodeResponse {
            user_code: field(&response, "user_code")?,
            verification_uri: field(&response, "verification_uri")?,
            device_code: field(&response, "device_code")?,
            interval_secs: response
                .get("interval")
                .and_then(Value::as_u64)
                .unwrap_or(5),
            domain: domain.to_string(),
        })
    }

    pub async fn poll_once(&self, dc: &DeviceCodeResponse) -> Result<PollOutcome, LlmError> {
        let url = github_url(&dc.domain, "/login/oauth/access_token")?;
        let response = post_json(
            self.transport.as_ref(),
            url,
            &self.user_agent,
            json!({"client_id": self.client_id, "device_code": dc.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}),
        )
        .await?;
        if let Some(token) = response.get("access_token").and_then(Value::as_str) {
            return Ok(PollOutcome::Success(CopilotSecret::new(token)));
        }
        match response.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => Ok(PollOutcome::Pending {
                interval_secs: dc.interval_secs,
            }),
            Some("slow_down") => Ok(PollOutcome::SlowDown {
                interval_secs: response
                    .get("interval")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| dc.interval_secs.saturating_add(5)),
            }),
            Some(code) => Ok(PollOutcome::Failed {
                error: safe_error_code(code),
            }),
            None => Ok(PollOutcome::Failed {
                error: "no access_token and no error in response".into(),
            }),
        }
    }
}

fn github_url(domain: &str, path: &str) -> Result<String, LlmError> {
    // The host supplies an authority, including an optional Enterprise port.
    // Reject URL components that could move the request to a different route.
    if domain.is_empty()
        || domain
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '/' | '\\' | '?' | '#' | '@'))
    {
        return Err(invalid("invalid GitHub domain"));
    }
    let url = url::Url::parse(&format!("https://{domain}{path}"))
        .map_err(|_| invalid("invalid GitHub domain"))?;
    if url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != path
    {
        return Err(invalid("invalid GitHub domain"));
    }
    Ok(url.into())
}

fn safe_error_code(code: &str) -> String {
    match code {
        "access_denied"
        | "expired_token"
        | "incorrect_device_code"
        | "incorrect_client_credentials"
        | "unsupported_grant_type"
        | "invalid_scope" => code.to_string(),
        _ => "unexpected device authorization error".into(),
    }
}

async fn post_json(
    transport: &dyn Transport,
    url: String,
    user_agent: &str,
    body: Value,
) -> Result<Value, LlmError> {
    let body = serde_json::to_vec(&body).map_err(|_| invalid("invalid Copilot request"))?;
    request_json(
        transport,
        HttpRequest {
            method: "POST".into(),
            url,
            headers: vec![
                ("Accept".into(), "application/json".into()),
                ("Content-Type".into(), "application/json".into()),
                ("User-Agent".into(), user_agent.into()),
            ],
            body: Bytes::from(body),
            timeout: Some(REQUEST_TIMEOUT),
        },
    )
    .await
}

async fn request_json(transport: &dyn Transport, request: HttpRequest) -> Result<Value, LlmError> {
    let response = HttpExecutor::new(transport)
        .execute_bounded(request, MAX_RESPONSE_BYTES)
        .await
        .map_err(|error| match error {
            LlmError::TransportTimeout { .. } => LlmError::TransportTimeout {
                message: "Copilot authorization request timed out".into(),
            },
            _ => LlmError::Transport {
                message: "Copilot authorization request failed".into(),
            },
        })?;
    if !(200..300).contains(&response.status) {
        if response.status == 404 {
            return Err(invalid("GitHub Copilot is not available for this account: no active Copilot subscription, or your organization has not authorized this app for Copilot. Check github.com/settings/copilot."));
        }
        return Err(LlmError::Transport {
            message: format!("Copilot authorization HTTP {}", response.status),
        });
    }
    serde_json::from_slice(&response.body).map_err(|_| invalid("invalid Copilot JSON response"))
}

fn field(body: &Value, key: &str) -> Result<String, LlmError> {
    body.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| invalid(&format!("copilot device-flow response missing '{key}'")))
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

/// Editor identity required by GitHub's Copilot token endpoint.
#[derive(Debug, Clone, Copy)]
pub struct ExchangeIdentity<'a> {
    pub user_agent: &'a str,
    pub editor_version: &'a str,
    pub plugin_version: &'a str,
    pub integration_id: &'a str,
}

#[derive(Clone)]
pub struct ExchangedToken {
    secret: CopilotSecret,
    pub expires_at: u64,
}

impl ExchangedToken {
    pub fn secret(&self) -> &CopilotSecret {
        &self.secret
    }
    pub fn into_secret(self) -> CopilotSecret {
        self.secret
    }
    pub fn bearer(&self) -> &str {
        self.secret.token_for_storage()
    }
    pub fn is_fresh(&self, now_unix_secs: u64) -> bool {
        self.expires_at > now_unix_secs.saturating_add(COPILOT_TOKEN_REFRESH_SKEW_SECS)
    }
}

impl std::fmt::Debug for ExchangedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExchangedToken")
            .field("secret", &self.secret)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Exchange once. The caller owns caching and refresh; failures never replay
/// the OAuth token through another transport or endpoint.
pub async fn exchange_copilot_token(
    transport: &dyn Transport,
    oauth_token: &str,
    identity: ExchangeIdentity<'_>,
) -> Result<ExchangedToken, LlmError> {
    let response = request_json(
        transport,
        HttpRequest {
            method: "GET".into(),
            url: COPILOT_TOKEN_EXCHANGE_URL.into(),
            headers: vec![
                ("Authorization".into(), format!("token {oauth_token}")),
                ("Accept".into(), "application/json".into()),
                ("Editor-Version".into(), identity.editor_version.into()),
                (
                    "Editor-Plugin-Version".into(),
                    identity.plugin_version.into(),
                ),
                (
                    "Copilot-Integration-Id".into(),
                    identity.integration_id.into(),
                ),
                ("User-Agent".into(), identity.user_agent.into()),
            ],
            body: Bytes::new(),
            timeout: Some(REQUEST_TIMEOUT),
        },
    )
    .await?;
    parse_exchange_response(&response)
}

fn parse_exchange_response(body: &Value) -> Result<ExchangedToken, LlmError> {
    let token = body
        .get("token")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty())
        .ok_or_else(|| invalid("copilot token-exchange response missing non-empty 'token'"))?;
    let expires_at = body
        .get("expires_at")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("copilot token-exchange response missing numeric 'expires_at'"))?;
    Ok(ExchangedToken {
        secret: CopilotSecret::new(token),
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{HttpResponse, StreamResponse};
    use async_trait::async_trait;
    use std::sync::Mutex;

    struct Mock {
        requests: Mutex<Vec<HttpRequest>>,
        responses: Mutex<Vec<(u16, Value)>>,
    }

    impl Mock {
        fn new(responses: Vec<(u16, Value)>) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::new(Vec::new()),
                responses: Mutex::new(responses),
            })
        }
    }

    #[async_trait]
    impl Transport for Mock {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.requests.lock().unwrap().push(request);
            let (status, body) = self.responses.lock().unwrap().remove(0);
            Ok(HttpResponse {
                status,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            }
            .into())
        }
    }

    #[test]
    fn enterprise_authority_preserves_port_and_rejects_url_components() {
        assert_eq!(
            github_url("github.example:8443", "/login/device/code").unwrap(),
            "https://github.example:8443/login/device/code"
        );
        for invalid_host in [
            "user@github.example",
            "github.example/path",
            "github.example?query",
            "https://github.example",
        ] {
            assert!(github_url(invalid_host, "/login/device/code").is_err());
        }
    }

    #[tokio::test]
    async fn begin_poll_and_exchange_use_one_bounded_sdk_request_each() {
        let mock = Mock::new(vec![
            (
                200,
                json!({"user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device",
                "device_code":"sensitive-device","interval":7}),
            ),
            (200, json!({"error":"slow_down"})),
            (200, json!({"access_token":"gho-secret"})),
            (
                200,
                json!({"token":"bearer-secret","expires_at":1_900_000_000_u64}),
            ),
        ]);
        let login = CopilotLogin::new(mock.clone(), COPILOT_CLIENT_ID, "LingXi-Code");
        let dc = login.begin("github.com").await.unwrap();
        assert!(!format!("{dc:?}").contains("sensitive-device"));
        assert!(matches!(
            login.poll_once(&dc).await.unwrap(),
            PollOutcome::SlowDown { interval_secs: 12 }
        ));
        let PollOutcome::Success(oauth) = login.poll_once(&dc).await.unwrap() else {
            panic!("expected token")
        };
        let identity = ExchangeIdentity {
            user_agent: "GitHubCopilotChat/0.26.7",
            editor_version: "vscode/1.99.3",
            plugin_version: "copilot-chat/0.26.7",
            integration_id: "vscode-chat",
        };
        let exchanged = exchange_copilot_token(mock.as_ref(), oauth.token_for_storage(), identity)
            .await
            .unwrap();
        assert_eq!(exchanged.bearer(), "bearer-secret");
        assert!(!format!("{exchanged:?}").contains("bearer-secret"));
        let requests = mock.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0].url, "https://github.com/login/device/code");
        assert_eq!(
            requests[1].url,
            "https://github.com/login/oauth/access_token"
        );
        assert_eq!(requests[3].url, COPILOT_TOKEN_EXCHANGE_URL);
        assert!(requests[3]
            .timeout
            .is_some_and(|timeout| timeout <= REQUEST_TIMEOUT && !timeout.is_zero()));
        assert!(requests[3]
            .headers
            .contains(&("Authorization".into(), "token gho-secret".into())));
        assert!(!format!("{:?}", requests[3]).contains("gho-secret"));
    }

    #[tokio::test]
    async fn status_and_transport_errors_do_not_echo_secrets_or_retry() {
        let mock = Mock::new(vec![(500, json!({"message":"gho-secret"}))]);
        let result = exchange_copilot_token(
            mock.as_ref(),
            "gho-secret",
            ExchangeIdentity {
                user_agent: "ua",
                editor_version: "editor",
                plugin_version: "plugin",
                integration_id: "id",
            },
        )
        .await;
        assert!(!format!("{result:?}").contains("gho-secret"));
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn injected_transport_error_is_redacted_and_not_retried() {
        struct LeakingTransport(Mutex<usize>);
        #[async_trait]
        impl Transport for LeakingTransport {
            async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
                *self.0.lock().unwrap() += 1;
                Err(LlmError::Transport {
                    message: "failed request containing gho-secret".into(),
                })
            }
        }
        let transport = LeakingTransport(Mutex::new(0));
        let error = exchange_copilot_token(
            &transport,
            "gho-secret",
            ExchangeIdentity {
                user_agent: "ua",
                editor_version: "editor",
                plugin_version: "plugin",
                integration_id: "id",
            },
        )
        .await
        .unwrap_err();
        assert!(!error.to_string().contains("gho-secret"));
        assert_eq!(*transport.0.lock().unwrap(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_transport_is_cancelled_at_request_deadline() {
        struct Stalled(Mutex<usize>);
        #[async_trait]
        impl Transport for Stalled {
            async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
                *self.0.lock().unwrap() += 1;
                std::future::pending().await
            }
        }
        let transport = Arc::new(Stalled(Mutex::new(0)));
        let login = CopilotLogin::new(transport.clone(), COPILOT_CLIENT_ID, "LingXi-Code");
        let handle = tokio::spawn(async move { login.begin(DEFAULT_GITHUB_DOMAIN).await });
        tokio::task::yield_now().await;
        tokio::time::advance(REQUEST_TIMEOUT).await;
        let result = handle.await.unwrap();
        assert!(matches!(result, Err(LlmError::TransportTimeout { .. })));
        assert_eq!(*transport.0.lock().unwrap(), 1);
    }

    #[test]
    fn exchange_parse_rejects_missing_fields_and_freshness_uses_skew() {
        assert!(parse_exchange_response(&json!({"expires_at":1})).is_err());
        assert!(parse_exchange_response(&json!({"token":"x"})).is_err());
        let token = parse_exchange_response(&json!({"token":"x","expires_at":1000})).unwrap();
        assert!(token.is_fresh(699));
        assert!(!token.is_fresh(700));
    }
}
