//! Anthropic OAuth wire protocol. The caller owns tokens, browser callbacks, and persistence.

use crate::transport::{HttpExecutor, HttpRequest, Transport};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use url::form_urlencoded;

const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);
const PROFILE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RESPONSE: usize = 1024 * 1024;
pub const BASE_API_URL: &str = "https://api.anthropic.com";
pub const OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";
pub const ROLES_URL_PATH: &str = "/api/oauth/claude_cli/roles";
pub const CLAUDE_CODE_OAUTH_SCOPES: &[&str] = &[
    "org:create_api_key",
    "user:profile",
    "user:inference",
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
];
pub const CLAUDE_CODE_INFERENCE_SCOPE: &str = "user:inference";
pub const LONG_LIVED_OAUTH_TOKEN_TTL_SECONDS: u64 = 31_536_000;
pub const REFRESH_GRANT_TYPE: &str = "refresh_token";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeAiOAuthConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub revocation_endpoint: String,
    pub profile_endpoint: String,
    pub client_id: String,
    pub redirect_uri: String,
    #[serde(default = "default_manual_redirect_uri")]
    pub manual_redirect_uri: String,
    pub scopes: Vec<String>,
}

fn default_manual_redirect_uri() -> String {
    "https://platform.claude.com/oauth/code/callback".into()
}

impl ClaudeAiOAuthConfig {
    pub fn default_with_port(port: u16) -> Self {
        Self {
            authorization_endpoint: "https://claude.com/cai/oauth/authorize".into(),
            token_endpoint: "https://platform.claude.com/v1/oauth/token".into(),
            revocation_endpoint: "https://platform.claude.com/v1/oauth/token/revoke".into(),
            profile_endpoint: format!("{BASE_API_URL}/api/oauth/profile"),
            client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e".into(),
            redirect_uri: format!("http://localhost:{port}/callback"),
            manual_redirect_uri: default_manual_redirect_uri(),
            scopes: CLAUDE_CODE_OAUTH_SCOPES
                .iter()
                .map(|s| (*s).into())
                .collect(),
        }
    }
    pub fn console_with_port(port: u16) -> Self {
        Self {
            authorization_endpoint: "https://platform.claude.com/oauth/authorize".into(),
            ..Self::default_with_port(port)
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AuthorizeOptions {
    pub scopes: Option<Vec<String>>,
    pub org_uuid: Option<String>,
    pub login_hint: Option<String>,
    pub login_method: Option<String>,
}

fn encode(value: &str) -> String {
    form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

pub fn format_authorize_url(
    config: &ClaudeAiOAuthConfig,
    redirect_uri: &str,
    challenge: &str,
    state: &str,
    options: &AuthorizeOptions,
) -> String {
    let scopes = options.scopes.as_ref().unwrap_or(&config.scopes).join(" ");
    let mut url = format!("{}?code=true&client_id={}&response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
        config.authorization_endpoint, encode(&config.client_id), encode(redirect_uri), encode(&scopes), encode(challenge), encode(state));
    for (name, value) in [
        ("orgUUID", &options.org_uuid),
        ("login_hint", &options.login_hint),
        ("login_method", &options.login_method),
    ] {
        if let Some(value) = value {
            url.push('&');
            url.push_str(name);
            url.push('=');
            url.push_str(&encode(value));
        }
    }
    url
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeAccount {
    #[serde(default)]
    pub uuid: String,
    #[serde(default)]
    pub email_address: String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeOrganization {
    #[serde(default)]
    pub uuid: String,
}
#[derive(Clone, Deserialize)]
pub struct ExchangeResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: u64,
    pub scope: Option<String>,
    pub account: Option<ExchangeAccount>,
    pub organization: Option<ExchangeOrganization>,
}
#[derive(Clone, Deserialize)]
pub struct RefreshResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: u64,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    InvalidAuthorizationCode,
    InvalidRefreshToken,
    Temporary(String),
}
impl std::fmt::Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAuthorizationCode => {
                write!(f, "Authentication failed: Invalid authorization code")
            }
            Self::InvalidRefreshToken => write!(f, "refresh credential rejected"),
            Self::Temporary(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for TokenError {}

fn safe_error(status: u16, body: &[u8]) -> String {
    let parsed: Option<serde_json::Value> = serde_json::from_slice(body).ok();
    let kind = parsed
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(|e| e.get("type").or_else(|| e.get("code")))
        .and_then(|v| v.as_str());
    let safe_kind = kind.filter(|s| {
        matches!(
            *s,
            "invalid_request"
                | "invalid_grant"
                | "unauthorized"
                | "access_denied"
                | "rate_limit_error"
                | "overloaded_error"
                | "api_error"
        )
    });
    match safe_kind {
        Some(kind) => format!("status {status} [{kind}]"),
        None => format!("status {status}"),
    }
}

#[derive(Serialize)]
struct ExchangeRequest<'a> {
    grant_type: &'a str,
    code: &'a str,
    redirect_uri: &'a str,
    client_id: &'a str,
    code_verifier: &'a str,
    state: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_in: Option<u64>,
}
#[derive(Serialize)]
struct RefreshRequest<'a> {
    grant_type: &'a str,
    refresh_token: &'a str,
    client_id: &'a str,
    scope: &'a str,
}

async fn token_post<T: for<'de> Deserialize<'de>>(
    transport: &dyn Transport,
    url: &str,
    body: impl Serialize,
    timeout: Duration,
    refresh: bool,
) -> Result<T, TokenError> {
    let body = serde_json::to_vec(&body)
        .map_err(|_| TokenError::Temporary("could not encode token request".into()))?;
    let response = HttpExecutor::new(transport)
        .execute_bounded(
            HttpRequest {
                method: "POST".into(),
                url: url.into(),
                headers: vec![
                    ("content-type".into(), "application/json".into()),
                    ("accept".into(), "application/json".into()),
                ],
                body: body.into(),
                timeout: Some(timeout),
            },
            MAX_RESPONSE,
        )
        .await
        .map_err(|_| TokenError::Temporary("token endpoint unavailable".into()))?;
    match response.status {
        200 => serde_json::from_slice(&response.body)
            .map_err(|_| TokenError::Temporary("invalid token endpoint response".into())),
        401 if !refresh => Err(TokenError::InvalidAuthorizationCode),
        400 | 401 | 403 if refresh && invalid_grant(&response.body) => {
            Err(TokenError::InvalidRefreshToken)
        }
        status => Err(TokenError::Temporary(safe_error(status, &response.body))),
    }
}

fn invalid_grant(body: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    let error = &value["error"];
    let rejected = [
        error.as_str(),
        error.get("type").and_then(|v| v.as_str()),
        error.get("code").and_then(|v| v.as_str()),
    ]
    .into_iter()
    .flatten()
    .any(|code| matches!(code, "invalid_grant" | "invalid_refresh_token"));
    rejected
}

pub async fn exchange_code(
    transport: &dyn Transport,
    config: &ClaudeAiOAuthConfig,
    code: &str,
    verifier: &str,
    state: &str,
    redirect_uri: &str,
    expires_in: Option<u64>,
) -> Result<ExchangeResponse, TokenError> {
    token_post(
        transport,
        &config.token_endpoint,
        ExchangeRequest {
            grant_type: "authorization_code",
            code,
            redirect_uri,
            client_id: &config.client_id,
            code_verifier: verifier,
            state,
            expires_in,
        },
        EXCHANGE_TIMEOUT,
        false,
    )
    .await
}
pub async fn refresh_token(
    transport: &dyn Transport,
    config: &ClaudeAiOAuthConfig,
    refresh_token: &str,
) -> Result<RefreshResponse, TokenError> {
    let scope = config.scopes.join(" ");
    token_post(
        transport,
        &config.token_endpoint,
        RefreshRequest {
            grant_type: REFRESH_GRANT_TYPE,
            refresh_token,
            client_id: &config.client_id,
            scope: &scope,
        },
        REFRESH_TIMEOUT,
        true,
    )
    .await
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct OAuthOrganization {
    #[serde(default)]
    pub organization_type: Option<String>,
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(default)]
    pub rate_limit_tier: Option<String>,
    #[serde(default)]
    pub billing_type: Option<String>,
    #[serde(default)]
    pub has_extra_usage_enabled: Option<bool>,
    #[serde(default)]
    pub subscription_created_at: Option<String>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct OAuthAccount {
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct OAuthProfileResponse {
    #[serde(default)]
    pub organization: Option<OAuthOrganization>,
    #[serde(default)]
    pub account: Option<OAuthAccount>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct UserRolesResponse {
    #[serde(default)]
    pub organization_role: Option<String>,
    #[serde(default)]
    pub workspace_role: Option<String>,
    #[serde(default)]
    pub organization_name: Option<String>,
}

async fn get<T: for<'de> Deserialize<'de>>(
    transport: &dyn Transport,
    url: String,
    headers: Vec<(String, String)>,
) -> Option<T> {
    let response = HttpExecutor::new(transport)
        .execute_bounded(
            HttpRequest {
                method: "GET".into(),
                url,
                headers,
                body: Default::default(),
                timeout: Some(PROFILE_TIMEOUT),
            },
            MAX_RESPONSE,
        )
        .await
        .ok()?;
    if response.status != 200 {
        return None;
    }
    serde_json::from_slice(&response.body).ok()
}
pub async fn fetch_profile_from_oauth_token(
    token: &str,
    transport: &dyn Transport,
) -> Option<OAuthProfileResponse> {
    get(
        transport,
        format!("{BASE_API_URL}/api/oauth/profile"),
        vec![
            ("Authorization".into(), format!("Bearer {token}")),
            ("Content-Type".into(), "application/json".into()),
        ],
    )
    .await
}
pub async fn fetch_profile_from_api_key(
    account_uuid: &str,
    api_key: &str,
    transport: &dyn Transport,
) -> Option<OAuthProfileResponse> {
    if account_uuid.is_empty() || api_key.is_empty() {
        return None;
    }
    let encoded: String = form_urlencoded::byte_serialize(account_uuid.as_bytes()).collect();
    get(
        transport,
        format!("{BASE_API_URL}/api/claude_cli_profile?account_uuid={encoded}"),
        vec![
            ("x-api-key".into(), api_key.into()),
            ("anthropic-beta".into(), OAUTH_BETA_HEADER.into()),
        ],
    )
    .await
}
pub async fn fetch_user_roles(token: &str, transport: &dyn Transport) -> Option<UserRolesResponse> {
    get(
        transport,
        format!("{BASE_API_URL}{ROLES_URL_PATH}"),
        vec![("Authorization".into(), format!("Bearer {token}"))],
    )
    .await
}

#[derive(Deserialize)]
struct LoginProfile {
    account: Option<LoginAccount>,
    organization: Option<LoginOrganization>,
}
#[derive(Deserialize)]
struct LoginAccount {
    #[serde(default)]
    email: String,
}
#[derive(Deserialize)]
struct LoginOrganization {
    #[serde(default)]
    uuid: String,
}

pub async fn fetch_login_identity(
    transport: &dyn Transport,
    profile_endpoint: &str,
    token: &str,
) -> Result<(String, String), String> {
    let response = HttpExecutor::new(transport)
        .execute_bounded(
            HttpRequest {
                method: "GET".into(),
                url: profile_endpoint.into(),
                headers: vec![
                    ("authorization".into(), format!("Bearer {token}")),
                    ("accept".into(), "application/json".into()),
                ],
                body: Default::default(),
                timeout: Some(Duration::from_secs(15)),
            },
            MAX_RESPONSE,
        )
        .await
        .map_err(|_| "profile endpoint unavailable".to_string())?;
    if response.status != 200 {
        return Err(format!("profile fetch failed: status {}", response.status));
    }
    let profile: LoginProfile = serde_json::from_slice(&response.body)
        .map_err(|_| "invalid profile response".to_string())?;
    Ok((
        profile.account.map(|a| a.email).unwrap_or_default(),
        profile.organization.map(|o| o.uuid).unwrap_or_default(),
    ))
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScopeUpgradeRequired {
    pub required: Vec<String>,
    #[serde(default)]
    pub granted: Vec<String>,
}
pub fn parse_scope_upgrade(body: &str) -> Option<ScopeUpgradeRequired> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let required = value
        .get("required_scopes")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect::<Vec<_>>();
    if required.is_empty() {
        return None;
    }
    let granted = value
        .get("granted_scopes")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    Some(ScopeUpgradeRequired { required, granted })
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
        body: &'static str,
        requests: Mutex<Vec<HttpRequest>>,
    }
    #[async_trait]
    impl Transport for Mock {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.requests.lock().unwrap().push(request);
            let body = self.body;
            Ok(StreamResponse {
                status: self.status,
                headers: vec![],
                body: Box::pin(stream::once(
                    async move { Ok(body.as_bytes().to_vec().into()) },
                )),
            })
        }
    }
    fn mock(status: u16, body: &'static str) -> Mock {
        Mock {
            status,
            body,
            requests: Mutex::new(vec![]),
        }
    }

    #[tokio::test]
    async fn refresh_rejects_only_explicit_invalid_grant_and_never_echoes_body() {
        let config = ClaudeAiOAuthConfig::default_with_port(1234);
        let rejected = mock(400, r#"{"error":"invalid_grant"}"#);
        assert!(matches!(
            refresh_token(&rejected, &config, "secret").await,
            Err(TokenError::InvalidRefreshToken)
        ));
        let temporary = mock(
            403,
            r#"{"error":{"type":"overloaded_error","message":"secret"}}"#,
        );
        assert!(matches!(
            refresh_token(&temporary, &config, "secret").await,
            Err(TokenError::Temporary(_))
        ));
        let malformed = mock(200, "secret malformed");
        let error = refresh_token(&malformed, &config, "secret")
            .await
            .err()
            .unwrap();
        assert!(!error.to_string().contains("secret"));
        let request = rejected.requests.lock().unwrap().pop().unwrap();
        assert!(request.timeout.is_some_and(
            |timeout| timeout > Duration::from_secs(14) && timeout <= Duration::from_secs(15)
        ));
    }

    #[test]
    fn authorize_encoding_and_scope_signal() {
        let config = ClaudeAiOAuthConfig::default_with_port(1234);
        let url = format_authorize_url(
            &config,
            &config.redirect_uri,
            "challenge",
            "state",
            &AuthorizeOptions::default(),
        );
        assert!(url.contains("scope=org%3Acreate_api_key+user%3Aprofile"));
        assert!(parse_scope_upgrade(r#"{"required_scopes":["user:file_upload"]}"#).is_some());
    }
}
