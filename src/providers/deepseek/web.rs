//! Independent DeepSeek Search/Fetch JSON APIs, separate from model generation.
//!
//! Each operation pins one profile and performs one HTTP request. The host owns
//! retries and account-wide pacing (Search and Fetch each allow 5 QPS). Errors
//! retain HTTP status and Retry-After so callers can make that decision.
use crate::{
    client::RequestOptions,
    protocol::{LlmError, ProviderProfile},
    providers::binding::ProviderBinding,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Published price per successful, nonempty request, in USD.
pub const REQUEST_PRICE_USD: f64 = 0.00075;

/// Independent search is available on exact first-party DeepSeek profiles.
/// Custom gateways are not silently redirected to a different origin.
pub fn supports(profile: &ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "deepseek" {
        return false;
    }
    url::Url::parse(&profile.base_url).is_ok_and(|url| {
        url.scheme() == "https"
            && url.host_str() == Some("api.deepseek.com")
            && url.port_or_known_default() == Some(443)
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && matches!(
                url.path().trim_end_matches('/'),
                "" | "/v1" | "/anthropic" | "/anthropic/v1"
            )
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekSearchRequest {
    pub query: String,
    #[serde(default = "default_count")]
    pub count: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ori_question: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub localization: Option<std::net::IpAddr>,
    #[serde(default)]
    pub need_content: bool,
}
fn default_count() -> u8 {
    10
}
impl DeepSeekSearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            count: 10,
            ori_question: None,
            localization: None,
            need_content: false,
        }
    }
    /// Encode documented site operators without accepting query syntax in domains.
    pub fn with_domains(
        mut self,
        allowed: &[String],
        blocked: &[String],
    ) -> Result<Self, LlmError> {
        self.validate()?;
        for domain in allowed.iter().chain(blocked) {
            if domain.is_empty()
                || domain.split('.').any(|label| {
                    label.is_empty()
                        || label.starts_with('-')
                        || label.ends_with('-')
                        || !label
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
                })
            {
                return Err(invalid("DeepSeek search filters must be domain names"));
            }
        }
        if !allowed.is_empty() {
            self.query.push(' ');
            self.query.push_str(
                &allowed
                    .iter()
                    .map(|domain| format!("site:{domain}"))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            );
        }
        for domain in blocked {
            self.query.push_str(&format!(" -site:{domain}"));
        }
        self.validate()?;
        Ok(self)
    }
    pub fn validate(&self) -> Result<(), LlmError> {
        validate_query(&self.query)?;
        if let Some(question) = &self.ori_question {
            validate_query(question)?;
        }
        if !(1..=50).contains(&self.count) {
            return Err(invalid("search count must be between 1 and 50"));
        }
        Ok(())
    }
}
fn validate_query(query: &str) -> Result<(), LlmError> {
    if query.trim().is_empty() || query.chars().count() > 128 {
        return Err(invalid("DeepSeek search query and original question must contain 1 to 128 Unicode characters and cannot be all whitespace"));
    }
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepSeekSearchHit {
    pub rerank_score: f64,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub summary: String,
    pub content: String,
    pub site_name: Option<String>,
    pub site_icon: Option<String>,
    pub date: Option<String>,
    pub authority: i8,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepSeekSearchResponse {
    pub results: Vec<DeepSeekSearchHit>,
}
impl DeepSeekSearchResponse {
    pub fn billable_requests(&self) -> u64 {
        u64::from(!self.results.is_empty())
    }
    pub fn cost_usd(&self) -> f64 {
        self.billable_requests() as f64 * REQUEST_PRICE_USD
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeepSeekFetchRequest {
    pub url: String,
}
impl DeepSeekFetchRequest {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
    pub fn validate(&self) -> Result<(), LlmError> {
        let parsed =
            url::Url::parse(&self.url).map_err(|_| invalid("invalid DeepSeek fetch URL"))?;
        if self.url.chars().count() > 8192
            || !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err(invalid("DeepSeek fetch requires an HTTP(S) URL with a host, no credentials, and at most 8192 characters"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeepSeekFetchResponse {
    pub title: String,
    pub content: String,
}
impl DeepSeekFetchResponse {
    pub fn billable_requests(&self) -> u64 {
        u64::from(!self.content.is_empty())
    }
    pub fn cost_usd(&self) -> f64 {
        self.billable_requests() as f64 * REQUEST_PRICE_USD
    }
}

#[derive(Clone, Copy)]
pub struct DeepSeekWeb<'a> {
    pub(crate) binding: &'a ProviderBinding,
}
impl DeepSeekWeb<'_> {
    pub async fn search(
        &self,
        request: &DeepSeekSearchRequest,
        options: &RequestOptions,
    ) -> Result<DeepSeekSearchResponse, LlmError> {
        request.validate()?;
        self.post("search", request, options).await
    }
    pub async fn fetch(
        &self,
        request: &DeepSeekFetchRequest,
        options: &RequestOptions,
    ) -> Result<DeepSeekFetchResponse, LlmError> {
        request.validate()?;
        self.post("fetch", request, options).await
    }
    async fn post<T: Serialize, R: serde::de::DeserializeOwned>(
        &self,
        operation: &str,
        body: &T,
        options: &RequestOptions,
    ) -> Result<R, LlmError> {
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(60)),
        ));
        let snapshot = self.binding.pin()?;
        let profile = snapshot
            .profile(self.binding.profile_name())
            .ok_or_else(|| invalid("DeepSeek profile is unavailable"))?;
        if !supports(profile) {
            return Err(LlmError::UnsupportedCapability {
                message: "DeepSeek Search/Fetch requires an exact first-party DeepSeek profile"
                    .into(),
            });
        }
        let mut request = HttpRequest {
            method: "POST".into(),
            url: format!("https://api.deepseek.com/v1/tools/{operation}"),
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("accept".into(), "application/json".into()),
            ],
            http1_header_layout: None,
            body: serde_json::to_vec(body)
                .map_err(|_| invalid("cannot encode DeepSeek web request"))?
                .into(),
            timeout: deadline.remaining()?,
        };
        // Search/Fetch always uses bearer auth, even when this binding's chat
        // wire is Anthropic. Preserve the bound profile/account for host auth.
        let mut auth_profile = profile.clone();
        auth_profile.auth = crate::protocol::AuthStrategy::Bearer;
        auth_profile.protocol = crate::protocol::ProtocolFamily::OpenAiChat;
        if let Some(extra) = auth_profile.extra.as_object_mut() {
            extra.remove("credential_header");
        }
        let authenticator = options
            .authenticator
            .as_ref()
            .map(|auth| auth.0.as_ref())
            .or_else(|| {
                snapshot
                    .runtime
                    .authenticators
                    .get(&profile.auth)
                    .map(|auth| auth.as_ref())
            })
            .ok_or_else(|| LlmError::Authentication {
                message: "no authenticator for DeepSeek web API".into(),
            })?;
        deadline
            .run(authenticator.apply(&mut request, &auth_profile, options.credential.as_ref()))
            .await??;
        if !request.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("authorization")
                && value
                    .strip_prefix("Bearer ")
                    .is_some_and(|key| !key.trim().is_empty())
        }) {
            return Err(LlmError::Authentication {
                message: "DeepSeek Search/Fetch requires bearer authentication".into(),
            });
        }
        request.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(snapshot.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, 16 * 1024 * 1024)
            .await?;
        if response.status != 200 {
            let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                serde_json::Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            let retry_after = response
                .header("retry-after")
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs);
            use crate::protocol::LlmErrorKind;
            let classification = match response.status {
                400 | 404 => LlmErrorKind::InvalidRequest,
                401 => LlmErrorKind::Authentication,
                402 => LlmErrorKind::QuotaExceeded,
                403 => LlmErrorKind::PermissionDenied,
                408 | 504 => LlmErrorKind::ProviderTimeout,
                413 => LlmErrorKind::RequestTooLarge,
                429 => LlmErrorKind::RateLimited,
                _ => LlmErrorKind::ProviderInternal,
            };
            let provider_text =
                crate::protocol::ProviderErrorText::from(format!("{} {}", response.status, body));
            return Err(LlmError::ProviderResponse {
                status: response.status,
                request_id: response.header("x-request-id").map(str::to_owned),
                body: Box::new(body),
                classification,
                retry_after,
            }
            .with_provider_error_text(provider_text));
        }
        serde_json::from_slice(&response.body).map_err(|_| LlmError::InvalidRequest {
            message: "invalid DeepSeek web API response".into(),
        })
    }
}
fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
