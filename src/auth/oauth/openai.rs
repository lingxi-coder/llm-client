//! OpenAI account OAuth wire protocol. Storage, callbacks, and refresh policy belong to the host.

use crate::auth::oauth::pkce::{generate_pkce, generate_state_token};
use crate::transport::{HttpExecutor, HttpRequest, Transport};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use bytes::Bytes;
use serde::{de, Deserialize, Deserializer, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use url::form_urlencoded::Serializer;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct OpenAiOAuthConfig {
    pub issuer: String,
    pub client_id: String,
    pub authorize_url: String,
    pub token_url: String,
    pub device_usercode_url: String,
    pub device_token_url: String,
    pub device_verify_url: String,
    pub scopes: String,
    pub codex_backend: String,
    pub loopback_ports: [u16; 2],
    pub authapi_base_url: String,
}

impl Default for OpenAiOAuthConfig {
    fn default() -> Self {
        let issuer = "https://auth.openai.com".to_string();
        Self {
            authorize_url: format!("{issuer}/oauth/authorize"),
            token_url: format!("{issuer}/oauth/token"),
            device_usercode_url: format!("{issuer}/api/accounts/deviceauth/usercode"),
            device_token_url: format!("{issuer}/api/accounts/deviceauth/token"),
            device_verify_url: format!("{issuer}/codex/device"),
            client_id: "app_EMoamEEZ73f0CkXaXp7hrann".into(),
            scopes: "openid profile email offline_access api.connectors.read api.connectors.invoke"
                .into(),
            codex_backend: "https://chatgpt.com/backend-api/codex".into(),
            loopback_ports: [1455, 1457],
            authapi_base_url: format!("{issuer}/api/accounts"),
            issuer,
        }
    }
}

impl OpenAiOAuthConfig {
    pub fn redirect_uri(&self, port: u16) -> String {
        format!("http://localhost:{port}/auth/callback")
    }
    pub fn device_redirect_uri(&self) -> String {
        format!("{}/deviceauth/callback", self.issuer.trim_end_matches('/'))
    }
    pub fn whoami_url(&self) -> String {
        format!(
            "{}/v1/user-auth-credential/whoami",
            self.authapi_base_url.trim_end_matches('/')
        )
    }
    pub fn build_authorize_url(&self, port: u16) -> (String, String, String) {
        self.build_authorize_url_with_redirect(&self.redirect_uri(port))
    }
    pub fn build_authorize_url_with_redirect(
        &self,
        redirect_uri: &str,
    ) -> (String, String, String) {
        let (verifier, challenge) = generate_pkce();
        let state = generate_state_token();
        let query = Serializer::new(String::new())
            .append_pair("response_type", "code")
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("scope", &self.scopes)
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("id_token_add_organizations", "true")
            .append_pair("codex_cli_simplified_flow", "true")
            .append_pair("state", &state)
            .append_pair("originator", "codex_cli_rs")
            .finish();
        (format!("{}?{query}", self.authorize_url), verifier, state)
    }
}

#[derive(Debug, Error)]
pub enum OAuthProtocolError {
    #[error("transport failure")]
    Transport,
    #[error("HTTP status {0}")]
    Status(u16),
    #[error("invalid response")]
    Decode,
    #[error("refresh credential rejected")]
    InvalidRefreshCredential,
}

async fn execute(
    transport: &dyn Transport,
    req: HttpRequest,
) -> Result<crate::transport::HttpResponse, OAuthProtocolError> {
    HttpExecutor::new(transport)
        .execute_bounded(req, MAX_RESPONSE_BYTES)
        .await
        .map_err(|_| OAuthProtocolError::Transport)
}

fn request(method: &str, url: String, content_type: Option<&str>, body: String) -> HttpRequest {
    let mut headers = vec![("accept".into(), "application/json".into())];
    if let Some(value) = content_type {
        headers.push(("content-type".into(), value.into()));
    }
    HttpRequest {
        http1_header_layout: None,
        method: method.into(),
        url,
        headers,
        body: Bytes::from(body),
        timeout: Some(REQUEST_TIMEOUT),
    }
}

fn decode<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, OAuthProtocolError> {
    serde_json::from_slice(body).map_err(|_| OAuthProtocolError::Decode)
}

fn success(response: crate::transport::HttpResponse) -> Result<Bytes, OAuthProtocolError> {
    if response.status == 200 {
        Ok(response.body)
    } else {
        Err(OAuthProtocolError::Status(response.status))
    }
}

#[derive(Deserialize)]
pub struct ExchangedTokens {
    pub id_token: Option<String>,
    pub access_token: String,
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: u64,
}

impl ExchangedTokens {
    /// The token endpoint may omit `expires_in` or return zero; both use the
    /// OpenAI OAuth default of one hour.
    pub fn effective_lifetime(&self) -> Duration {
        Duration::from_secs(if self.expires_in == 0 {
            3600
        } else {
            self.expires_in
        })
    }
}

impl fmt::Debug for ExchangedTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExchangedTokens")
            .field("id_token", &self.id_token.as_ref().map(|_| "[REDACTED]"))
            .field("access_token", &"[REDACTED]")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("expires_in", &self.expires_in)
            .finish()
    }
}

pub async fn exchange_code(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<ExchangedTokens, OAuthProtocolError> {
    let body = Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("code", code)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("client_id", &cfg.client_id)
        .append_pair("code_verifier", verifier)
        .finish();
    decode(&success(
        execute(
            transport,
            request(
                "POST",
                cfg.token_url.clone(),
                Some("application/x-www-form-urlencoded"),
                body,
            ),
        )
        .await?,
    )?)
}

pub async fn obtain_api_key(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
    id_token: &str,
) -> Result<String, OAuthProtocolError> {
    #[derive(Deserialize)]
    struct Response {
        access_token: String,
    }
    let body = Serializer::new(String::new())
        .append_pair(
            "grant_type",
            "urn:ietf:params:oauth:grant-type:token-exchange",
        )
        .append_pair("client_id", &cfg.client_id)
        .append_pair("requested_token", "openai-api-key")
        .append_pair("subject_token", id_token)
        .append_pair(
            "subject_token_type",
            "urn:ietf:params:oauth:token-type:id_token",
        )
        .finish();
    let response: Response = decode(&success(
        execute(
            transport,
            request(
                "POST",
                cfg.token_url.clone(),
                Some("application/x-www-form-urlencoded"),
                body,
            ),
        )
        .await?,
    )?)?;
    Ok(response.access_token)
}

pub async fn refresh_token(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
    refresh: &str,
) -> Result<ExchangedTokens, OAuthProtocolError> {
    #[derive(Serialize)]
    struct RefreshRequest<'a> {
        grant_type: &'static str,
        refresh_token: &'a str,
        client_id: &'a str,
    }
    let body = serde_json::to_string(&RefreshRequest {
        grant_type: "refresh_token",
        refresh_token: refresh,
        client_id: &cfg.client_id,
    })
    .map_err(|_| OAuthProtocolError::Decode)?;
    let response = execute(
        transport,
        request(
            "POST",
            cfg.token_url.clone(),
            Some("application/json"),
            body,
        ),
    )
    .await?;
    // Only an explicit OAuth invalid_grant or invalid_token rejection ends a session.
    if response.status == 400 || response.status == 401 || response.status == 403 {
        let kind = serde_json::from_slice::<serde_json::Value>(&response.body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| {
                        e.as_str()
                            .or_else(|| e.get("code").and_then(|c| c.as_str()))
                            .or_else(|| e.get("type").and_then(|c| c.as_str()))
                    })
                    .map(str::to_owned)
            });
        if matches!(
            kind.as_deref(),
            Some("invalid_grant" | "invalid_token" | "invalid_refresh_token")
        ) {
            return Err(OAuthProtocolError::InvalidRefreshCredential);
        }
    }
    decode(&success(response)?)
}

#[derive(Debug, Clone)]
pub struct DeviceUserCode {
    pub device_auth_id: String,
    pub user_code: String,
    pub interval: u64,
}
#[derive(Debug)]
pub enum PollOutcome {
    Ready {
        authorization_code: String,
        code_verifier: String,
    },
    Pending,
}
const DEFAULT_INTERVAL_SECS: u64 = 5;
fn default_interval() -> u64 {
    DEFAULT_INTERVAL_SECS
}
fn deserialize_interval<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Value {
        Num(u64),
        Str(String),
    }
    match Value::deserialize(deserializer)? {
        Value::Num(n) => Ok(n),
        Value::Str(s) => s.trim().parse().map_err(de::Error::custom),
    }
}
#[derive(Deserialize)]
struct UserCodeResponse {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(
        default = "default_interval",
        deserialize_with = "deserialize_interval"
    )]
    interval: u64,
}

pub async fn request_device_code(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
) -> Result<DeviceUserCode, OAuthProtocolError> {
    let body = serde_json::json!({"client_id": cfg.client_id}).to_string();
    let response: UserCodeResponse = decode(&success(
        execute(
            transport,
            request(
                "POST",
                cfg.device_usercode_url.clone(),
                Some("application/json"),
                body,
            ),
        )
        .await?,
    )?)?;
    Ok(DeviceUserCode {
        device_auth_id: response.device_auth_id,
        user_code: response.user_code,
        interval: response.interval,
    })
}

pub async fn poll_for_token(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
    device_auth_id: &str,
    user_code: &str,
) -> Result<PollOutcome, OAuthProtocolError> {
    #[derive(Deserialize)]
    struct ReadyResponse {
        authorization_code: String,
        code_verifier: String,
    }
    let body =
        serde_json::json!({"device_auth_id":device_auth_id,"user_code":user_code}).to_string();
    let response = execute(
        transport,
        request(
            "POST",
            cfg.device_token_url.clone(),
            Some("application/json"),
            body,
        ),
    )
    .await?;
    match response.status {
        200 => {
            let value: ReadyResponse = decode(&response.body)?;
            Ok(PollOutcome::Ready {
                authorization_code: value.authorization_code,
                code_verifier: value.code_verifier,
            })
        }
        403 | 404 => Ok(PollOutcome::Pending),
        other => Err(OAuthProtocolError::Status(other)),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct PatMetadata {
    #[serde(rename = "chatgpt_account_id")]
    pub account_id: Option<String>,
    #[serde(default, rename = "chatgpt_account_is_fedramp")]
    pub fedramp: bool,
    pub email: Option<String>,
    #[serde(rename = "chatgpt_plan_type")]
    pub plan: Option<String>,
}

pub async fn whoami(
    transport: &dyn Transport,
    cfg: &OpenAiOAuthConfig,
    pat: &str,
) -> Result<PatMetadata, OAuthProtocolError> {
    let mut req = request("GET", cfg.whoami_url(), None, String::new());
    req.headers
        .push(("authorization".into(), format!("Bearer {pat}")));
    decode(&success(execute(transport, req).await?)?)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdTokenClaims {
    pub account_id: Option<String>,
    pub fedramp: bool,
    pub email: Option<String>,
}

pub fn parse_id_token(jwt: &str) -> Option<IdTokenClaims> {
    let mut parts = jwt.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    let signature = parts.next()?;
    if parts.next().is_some() || signature.is_empty() {
        return None;
    }
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let auth = value.get("https://api.openai.com/auth");
    Some(IdTokenClaims {
        account_id: auth
            .and_then(|a| a.get("chatgpt_account_id"))
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        fedramp: auth
            .and_then(|a| a.get("chatgpt_account_is_fedramp"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        email: value
            .get("email")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
    })
}

pub struct OpenAiOAuthClient {
    config: OpenAiOAuthConfig,
    transport: Arc<dyn Transport>,
}
impl OpenAiOAuthClient {
    pub fn new(config: OpenAiOAuthConfig, transport: Arc<dyn Transport>) -> Self {
        Self { config, transport }
    }
    pub fn config(&self) -> &OpenAiOAuthConfig {
        &self.config
    }
    pub fn transport(&self) -> Arc<dyn Transport> {
        self.transport.clone()
    }
    pub fn build_authorize_url(&self, port: u16) -> (String, String, String) {
        self.config.build_authorize_url(port)
    }
    pub fn build_authorize_url_with_redirect(
        &self,
        redirect_uri: &str,
    ) -> (String, String, String) {
        self.config.build_authorize_url_with_redirect(redirect_uri)
    }
    pub async fn exchange_code_with_redirect(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
    ) -> Result<ExchangedTokens, OAuthProtocolError> {
        exchange_code(
            self.transport.as_ref(),
            &self.config,
            code,
            verifier,
            redirect_uri,
        )
        .await
    }
    pub async fn exchange_code(
        &self,
        code: &str,
        verifier: &str,
        port: u16,
    ) -> Result<ExchangedTokens, OAuthProtocolError> {
        self.exchange_code_with_redirect(code, verifier, &self.config.redirect_uri(port))
            .await
    }
    pub async fn obtain_api_key(&self, id_token: &str) -> Result<String, OAuthProtocolError> {
        obtain_api_key(self.transport.as_ref(), &self.config, id_token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::LlmError;
    use crate::transport::StreamResponse;
    use async_trait::async_trait;
    use futures::stream;
    use std::sync::Mutex;

    struct Mock {
        status: u16,
        response: String,
        sent: Mutex<Vec<HttpRequest>>,
    }
    #[async_trait]
    impl Transport for Mock {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.sent.lock().unwrap().push(request);
            let bytes = Bytes::from(self.response.clone());
            Ok(StreamResponse {
                status: self.status,
                headers: vec![],
                body: Box::pin(stream::once(async move { Ok(bytes) })),
            })
        }
    }
    fn mock(status: u16, response: &str) -> Mock {
        Mock {
            status,
            response: response.into(),
            sent: Mutex::new(vec![]),
        }
    }

    #[test]
    fn device_redirect_uri_uses_issuer_without_duplicate_slash() {
        let mut cfg = OpenAiOAuthConfig::default();
        assert_eq!(
            cfg.device_redirect_uri(),
            "https://auth.openai.com/deviceauth/callback"
        );
        cfg.issuer = "https://login.example.test/custom/".into();
        assert_eq!(
            cfg.device_redirect_uri(),
            "https://login.example.test/custom/deviceauth/callback"
        );
    }

    #[test]
    fn token_effective_lifetime_uses_default_only_for_zero() {
        let mut tokens = ExchangedTokens {
            id_token: None,
            access_token: "access".into(),
            refresh_token: None,
            expires_in: 0,
        };
        assert_eq!(tokens.effective_lifetime(), Duration::from_secs(3600));
        tokens.expires_in = 42;
        assert_eq!(tokens.effective_lifetime(), Duration::from_secs(42));
    }
    #[tokio::test]
    async fn exchange_posts_form_with_deadline_and_redacts_token_debug() {
        let transport = mock(
            200,
            r#"{"access_token":"SENSITIVE_ACCESS","refresh_token":"SENSITIVE_REFRESH","id_token":"SENSITIVE_ID","expires_in":42}"#,
        );
        let token = exchange_code(
            &transport,
            &OpenAiOAuthConfig::default(),
            "CODE",
            "VERIFIER",
            "http://localhost/callback",
        )
        .await
        .unwrap();
        assert_eq!(token.access_token, "SENSITIVE_ACCESS");
        let debug = format!("{token:?}");
        assert!(!debug.contains("SENSITIVE"));
        let sent = transport.sent.lock().unwrap();
        assert!(sent[0]
            .timeout
            .is_some_and(|t| t > Duration::ZERO && t <= Duration::from_secs(15)));
        assert_eq!(sent[0].method, "POST");
        assert!(String::from_utf8_lossy(&sent[0].body).contains("code_verifier=VERIFIER"));
    }
    #[tokio::test]
    async fn error_does_not_echo_response_or_refresh_secret() {
        let transport = mock(500, "echo SENSITIVE_REFRESH");
        let err = refresh_token(
            &transport,
            &OpenAiOAuthConfig::default(),
            "SENSITIVE_REFRESH",
        )
        .await
        .unwrap_err();
        assert_eq!(err.to_string(), "HTTP status 500");
        assert!(!format!("{err:?}").contains("SENSITIVE"));
    }
    #[tokio::test(start_paused = true)]
    async fn pending_send_respects_token_deadline() {
        struct NeverSend;
        #[async_trait]
        impl Transport for NeverSend {
            async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
                std::future::pending().await
            }
        }
        let error = exchange_code(
            &NeverSend,
            &OpenAiOAuthConfig::default(),
            "code",
            "verifier",
            "redirect",
        )
        .await
        .unwrap_err();
        assert!(matches!(error, OAuthProtocolError::Transport));
    }

    #[tokio::test(start_paused = true)]
    async fn pending_body_times_out_and_is_cancelled() {
        use futures::Stream;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::task::{Context, Poll};
        struct PendingBody(Arc<AtomicBool>);
        impl Stream for PendingBody {
            type Item = Result<Bytes, LlmError>;
            fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                Poll::Pending
            }
        }
        impl Drop for PendingBody {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        struct PendingTransport(Arc<AtomicBool>);
        #[async_trait]
        impl Transport for PendingTransport {
            async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
                Ok(StreamResponse {
                    status: 200,
                    headers: vec![],
                    body: Box::pin(PendingBody(self.0.clone())),
                })
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let transport = PendingTransport(dropped.clone());
        let error = exchange_code(
            &transport,
            &OpenAiOAuthConfig::default(),
            "code",
            "verifier",
            "redirect",
        )
        .await
        .unwrap_err();
        assert!(matches!(error, OAuthProtocolError::Transport));
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn only_explicit_invalid_grant_ends_refresh_session() {
        let cfg = OpenAiOAuthConfig::default();
        let plain_401 = mock(401, "{}");
        assert!(matches!(
            refresh_token(&plain_401, &cfg, "secret").await,
            Err(OAuthProtocolError::Status(401))
        ));
        let invalid = mock(400, r#"{"error":"invalid_grant"}"#);
        assert!(matches!(
            refresh_token(&invalid, &cfg, "secret").await,
            Err(OAuthProtocolError::InvalidRefreshCredential)
        ));
    }
}
