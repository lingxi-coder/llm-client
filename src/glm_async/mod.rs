//! GLM asynchronous chat-completion submission and one-shot result retrieval.
//!
//! This is a provider service, not a polling scheduler. The caller chooses
//! when to query again and owns the serialized, account-bound job reference.
//! Submission is never retried because a transport failure can happen after
//! the provider has accepted the request.

use crate::{
    codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec},
    files::provider_file_endpoint_fingerprint,
    protocol::{
        AuthStrategy, ChatRequest, CredentialConfig, LlmError, ProtocolFamily, ProviderId,
        ProviderProfile, Region, Secret,
    },
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt, time::Duration};
use thiserror::Error;
use url::Url;

const CHINA_BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// The account region is part of every job reference and must be selected by
/// the caller. Mainland BigModel documents async chat directly. The Z.AI
/// international docs currently list synchronous Chat Completion only, so
/// selecting that region is rejected until async chat is documented there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmAsyncRegion {
    ChinaMainland,
    International,
}

impl GlmAsyncRegion {
    fn base_url(self) -> Option<&'static str> {
        match self {
            Self::ChinaMainland => Some(CHINA_BASE_URL),
            Self::International => None,
        }
    }

    fn usage_region(self) -> Region {
        match self {
            Self::ChinaMainland => Region::ChinaMainland,
            Self::International => Region::International,
        }
    }
}

/// Explicit provider, region, account and credential binding for GLM async
/// chat. The profile supplies the selected model metadata and codec options;
/// its base URL must match the official endpoint selected by `region`.
#[derive(Clone)]
pub struct GlmAsyncConfig {
    profile: ProviderProfile,
    region: GlmAsyncRegion,
    account_scope: String,
    credential_scope: String,
    request_timeout: Duration,
}

impl GlmAsyncConfig {
    /// Bind one Zhipu profile to an explicit region, account and credential
    /// identity. The API key itself is supplied separately for each operation.
    pub fn new(
        profile: ProviderProfile,
        region: GlmAsyncRegion,
        account_scope: impl Into<String>,
        credential_scope: impl Into<String>,
    ) -> Result<Self, GlmAsyncError> {
        let account_scope = account_scope.into();
        let credential_scope = credential_scope.into();
        if profile.provider_id.as_str() != "zhipu" {
            return Err(invalid_config("profile provider_id must be `zhipu`"));
        }
        if profile.protocol != ProtocolFamily::OpenAiChat {
            return Err(invalid_config(
                "GLM async chat requires an OpenAI Chat Completions profile",
            ));
        }
        if profile.auth != AuthStrategy::Bearer {
            return Err(invalid_config(
                "GLM async chat requires Bearer authentication",
            ));
        }
        if !profile.supports_region(region.usage_region()) {
            return Err(invalid_config(
                "the selected GLM profile is not enabled for this region",
            ));
        }
        let Some(expected_base_url) = region.base_url() else {
            return Err(GlmAsyncError::UnsupportedRegion {
                region,
                message: "the current official Z.AI docs do not document async chat completion"
                    .into(),
            });
        };
        if profile.base_url.trim_end_matches('/') != expected_base_url {
            return Err(invalid_config(
                "the profile base URL does not match the explicitly selected GLM region",
            ));
        }
        if profile.profile_name.trim().is_empty()
            || account_scope.trim().is_empty()
            || credential_scope.trim().is_empty()
        {
            return Err(invalid_config(
                "profile, account scope and credential scope must be non-empty",
            ));
        }

        // This API receives credentials per operation. Do not retain a static
        // credential embedded in the provider profile clone.
        let mut profile = profile;
        profile.credential = CredentialConfig::None;

        Ok(Self {
            profile,
            region,
            account_scope,
            credential_scope,
            request_timeout: DEFAULT_TIMEOUT,
        })
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn profile_name(&self) -> &str {
        &self.profile.profile_name
    }

    pub fn region(&self) -> GlmAsyncRegion {
        self.region
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn credential_scope(&self) -> &str {
        &self.credential_scope
    }

    fn scope(&self, model: &str) -> GlmAsyncScope {
        GlmAsyncScope {
            provider_id: self.profile.provider_id.clone(),
            profile_name: self.profile.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(
                self.region
                    .base_url()
                    .expect("config rejects unsupported regions"),
            ),
            region: self.region,
            account_scope: self.account_scope.clone(),
            credential_scope: self.credential_scope.clone(),
            model: model.to_owned(),
        }
    }
}

impl fmt::Debug for GlmAsyncConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmAsyncConfig")
            .field("profile_name", &self.profile.profile_name)
            .field("region", &self.region)
            .field("account_scope", &self.account_scope)
            .field("credential_scope", &self.credential_scope)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

/// Per-operation API key. `credential_scope` is a stable, non-secret identity
/// chosen by the host and must match the one bound in [`GlmAsyncConfig`].
pub struct GlmAsyncCredentials {
    credential_scope: String,
    api_key: Secret<String>,
}

impl GlmAsyncCredentials {
    pub fn new(credential_scope: impl Into<String>, api_key: Secret<String>) -> Self {
        Self {
            credential_scope: credential_scope.into(),
            api_key,
        }
    }

    pub fn credential_scope(&self) -> &str {
        &self.credential_scope
    }

    fn validate(&self, expected: &str) -> Result<(), GlmAsyncError> {
        if self.credential_scope != expected {
            return Err(GlmAsyncError::ScopeMismatch {
                message: "credential scope does not match the configured GLM account".into(),
            });
        }
        if self.credential_scope.trim().is_empty() || self.api_key.expose_secret().trim().is_empty()
        {
            return Err(GlmAsyncError::InvalidCredentials {
                message: "credential scope and API key must be non-empty".into(),
            });
        }
        Ok(())
    }
}

impl fmt::Debug for GlmAsyncCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmAsyncCredentials")
            .field("credential_scope", &self.credential_scope)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// A typed provider-neutral conversation submitted to GLM's async endpoint.
/// The existing chat codec performs message/control validation and produces
/// the OpenAI-compatible wire body; the endpoint then forces `stream=false`.
pub type GlmAsyncRequest = ChatRequest;

/// Safe-to-persist identity for one accepted async chat request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAsyncScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub region: GlmAsyncRegion,
    pub account_scope: String,
    pub credential_scope: String,
    pub model: String,
}

/// Provider-generated task identity. Querying requires the exact profile,
/// region, endpoint, account, credential scope and model that created it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAsyncJobRef {
    pub scope: GlmAsyncScope,
    pub task_id: String,
}

impl GlmAsyncJobRef {
    pub fn scope(&self) -> &GlmAsyncScope {
        &self.scope
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }
}

/// Open task-state enum. Unrecognized provider values remain available as
/// `Other`, and the containing job always retains the complete native body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum GlmAsyncTaskStatus {
    Unknown,
    Processing,
    Success,
    Fail,
    Other(String),
}

impl From<String> for GlmAsyncTaskStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "PROCESSING" => Self::Processing,
            "SUCCESS" => Self::Success,
            "FAIL" => Self::Fail,
            "" => Self::Unknown,
            _ => Self::Other(value),
        }
    }
}

impl From<GlmAsyncTaskStatus> for String {
    fn from(status: GlmAsyncTaskStatus) -> Self {
        match status {
            GlmAsyncTaskStatus::Unknown => "UNKNOWN".into(),
            GlmAsyncTaskStatus::Processing => "PROCESSING".into(),
            GlmAsyncTaskStatus::Success => "SUCCESS".into(),
            GlmAsyncTaskStatus::Fail => "FAIL".into(),
            GlmAsyncTaskStatus::Other(value) => value,
        }
    }
}

/// One submission or one-shot result query, including the full native payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmAsyncJob {
    pub reference: GlmAsyncJobRef,
    pub task_status: GlmAsyncTaskStatus,
    pub request_id: Option<String>,
    pub model: String,
    pub native_error: Option<Value>,
    pub native: Value,
}

/// Submission and one-shot retrieval for the documented GLM async chat API.
pub struct GlmAsyncService<'a> {
    transport: &'a dyn Transport,
    config: GlmAsyncConfig,
}

impl<'a> GlmAsyncService<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        config: GlmAsyncConfig,
    ) -> Result<Self, GlmAsyncError> {
        Ok(Self { transport, config })
    }

    pub fn config(&self) -> &GlmAsyncConfig {
        &self.config
    }

    /// Submit exactly once. A transport failure, server failure, or malformed
    /// accepted response returns an explicit unknown outcome; callers must not
    /// automatically resubmit it.
    pub async fn submit(
        &self,
        request: &GlmAsyncRequest,
        credentials: &GlmAsyncCredentials,
    ) -> Result<GlmAsyncJob, GlmAsyncError> {
        credentials.validate(&self.config.credential_scope)?;
        if request.model.trim().is_empty() {
            return Err(invalid_request("model must be non-empty"));
        }
        if request.messages.is_empty() {
            return Err(invalid_request(
                "at least one conversation message is required",
            ));
        }

        let context = CodecContext::new(
            &self.config.profile,
            request.model.clone(),
            RequestMode::Complete,
        );
        let codec = crate::codecs::openai::chat::OpenAiChatCodec;
        codec.validate_request(request, &context)?;
        let encoded = codec.encode_request(EncodeRequest::new(request), &context)?;
        let mut body: Value = serde_json::from_slice(&encoded.body)
            .map_err(|_| invalid_request("encoded chat request is not valid JSON"))?;
        if !body.is_object() {
            return Err(invalid_request(
                "encoded chat request must be a JSON object",
            ));
        }
        body["stream"] = Value::Bool(false);
        let bytes = serde_json::to_vec(&body)
            .map_err(|_| invalid_request("encoded chat request cannot be serialized"))?;
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(GlmAsyncError::InvalidRequest {
                message: format!("request body exceeds {MAX_REQUEST_BYTES} bytes"),
            });
        }

        let http_request = HttpRequest {
            method: "POST".into(),
            url: async_chat_endpoint(self.config.region)?,
            headers: json_headers(credentials),
            body: Bytes::from(bytes),
            timeout: Some(self.config.request_timeout),
        };
        let response = HttpExecutor::new(self.transport)
            .execute_bounded(http_request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| GlmAsyncError::SubmitOutcomeUnknown {
                status: None,
                request_id: None,
                message: Box::new(format!(
                    "submission transport did not return a complete response: {source}"
                )),
                body: None,
                native: None,
            })?;

        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let native = decode_native(&response.body);
        if !(200..300).contains(&response.status) {
            let body_text = String::from_utf8_lossy(&response.body).into_owned();
            if response.status >= 500 || response.status == 408 {
                return Err(GlmAsyncError::SubmitOutcomeUnknown {
                    status: Some(response.status),
                    request_id,
                    message: Box::new("provider returned a server error after submission".into()),
                    body: Some(Box::new(body_text)),
                    native: native.map(Box::new),
                });
            }
            return Err(GlmAsyncError::Http {
                status: response.status,
                request_id,
                body: Box::new(body_text),
                native: native.map(Box::new),
            });
        }
        let Some(native) = native else {
            return Err(GlmAsyncError::SubmitOutcomeUnknown {
                status: Some(response.status),
                request_id,
                message: Box::new("provider accepted the request but returned invalid JSON".into()),
                body: Some(Box::new(
                    String::from_utf8_lossy(&response.body).into_owned(),
                )),
                native: None,
            });
        };
        let task_id = native
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty());
        let Some(task_id) = task_id else {
            return Err(GlmAsyncError::SubmitOutcomeUnknown {
                status: Some(response.status),
                request_id,
                message: Box::new(
                    "provider accepted the request but returned no async task ID".into(),
                ),
                body: Some(Box::new(
                    String::from_utf8_lossy(&response.body).into_owned(),
                )),
                native: Some(Box::new(native)),
            });
        };
        let reference = GlmAsyncJobRef {
            scope: self.config.scope(&request.model),
            task_id: task_id.to_owned(),
        };
        Ok(parse_job(reference, native, Some(request.model.clone())))
    }

    /// Retrieve one task snapshot. This makes one GET request and never polls
    /// or schedules another request on the caller's behalf.
    pub async fn get(
        &self,
        reference: &GlmAsyncJobRef,
        credentials: &GlmAsyncCredentials,
    ) -> Result<GlmAsyncJob, GlmAsyncError> {
        credentials.validate(&self.config.credential_scope)?;
        self.validate_reference(reference)?;
        let request = HttpRequest {
            method: "GET".into(),
            url: async_result_endpoint(self.config.region, &reference.task_id)?,
            headers: auth_header(credentials),
            body: Bytes::new(),
            timeout: Some(self.config.request_timeout),
        };
        let response = HttpExecutor::new(self.transport)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let native = decode_native(&response.body);
        if !(200..300).contains(&response.status) {
            return Err(GlmAsyncError::Http {
                status: response.status,
                request_id,
                body: Box::new(String::from_utf8_lossy(&response.body).into_owned()),
                native: native.map(Box::new),
            });
        }
        let Some(native) = native else {
            return Err(GlmAsyncError::ProviderResponse {
                status: response.status,
                request_id,
                message: Box::new("async result response is not valid JSON".into()),
                native: None,
            });
        };
        if native
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|returned_id| returned_id != reference.task_id)
        {
            return Err(GlmAsyncError::ProviderResponse {
                status: response.status,
                request_id,
                message: Box::new("async result ID does not match the requested task".into()),
                native: Some(Box::new(native)),
            });
        }
        Ok(parse_job(
            reference.clone(),
            native,
            Some(reference.scope.model.clone()),
        ))
    }

    fn validate_reference(&self, reference: &GlmAsyncJobRef) -> Result<(), GlmAsyncError> {
        if reference.scope != self.config.scope(&reference.scope.model)
            || reference.task_id.trim().is_empty()
        {
            return Err(GlmAsyncError::ScopeMismatch {
                message: "async job reference belongs to another provider, profile, region, endpoint, account, credential scope or model".into(),
            });
        }
        Ok(())
    }
}

impl fmt::Debug for GlmAsyncService<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmAsyncService")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Async chat currently has no documented cancellation operation.
///
/// This service deliberately exposes no `cancel` method. Dropping a caller's
/// wait does not claim to cancel an accepted provider task.
#[derive(Debug, Error)]
pub enum GlmAsyncError {
    #[error("invalid GLM async configuration: {message}")]
    InvalidConfig { message: String },
    #[error("GLM async region {region:?} is not documented: {message}")]
    UnsupportedRegion {
        region: GlmAsyncRegion,
        message: String,
    },
    #[error("invalid GLM async chat request: {message}")]
    InvalidRequest { message: String },
    #[error("invalid GLM async credentials: {message}")]
    InvalidCredentials { message: String },
    #[error("GLM async scope mismatch: {message}")]
    ScopeMismatch { message: String },
    #[error("GLM async provider HTTP error {status}: {body}")]
    Http {
        status: u16,
        request_id: Option<String>,
        body: Box<String>,
        native: Option<Box<Value>>,
    },
    #[error("GLM async provider response error: {message}")]
    ProviderResponse {
        status: u16,
        request_id: Option<String>,
        message: Box<String>,
        native: Option<Box<Value>>,
    },
    #[error("GLM async submission outcome is unknown: {message}")]
    SubmitOutcomeUnknown {
        status: Option<u16>,
        request_id: Option<String>,
        message: Box<String>,
        body: Option<Box<String>>,
        native: Option<Box<Value>>,
    },
    #[error(transparent)]
    Llm(Box<LlmError>),
}

impl From<LlmError> for GlmAsyncError {
    fn from(error: LlmError) -> Self {
        Self::Llm(Box::new(error))
    }
}

fn json_headers(credentials: &GlmAsyncCredentials) -> Vec<(String, String)> {
    let mut headers = auth_header(credentials);
    headers.push(("content-type".into(), "application/json".into()));
    headers
}

fn auth_header(credentials: &GlmAsyncCredentials) -> Vec<(String, String)> {
    vec![(
        "authorization".into(),
        format!("Bearer {}", credentials.api_key.expose_secret()),
    )]
}

fn async_chat_endpoint(region: GlmAsyncRegion) -> Result<String, GlmAsyncError> {
    endpoint(region, &["async", "chat", "completions"])
}

fn async_result_endpoint(region: GlmAsyncRegion, task_id: &str) -> Result<String, GlmAsyncError> {
    endpoint(region, &["async-result", task_id])
}

fn endpoint(region: GlmAsyncRegion, segments: &[&str]) -> Result<String, GlmAsyncError> {
    let base_url = region
        .base_url()
        .ok_or_else(|| GlmAsyncError::UnsupportedRegion {
            region,
            message: "the selected async endpoint is not documented".into(),
        })?;
    let mut url = Url::parse(base_url).map_err(|_| invalid_config("invalid endpoint"))?;
    url.path_segments_mut()
        .map_err(|_| invalid_config("invalid endpoint path"))?
        .extend(segments.iter().copied());
    Ok(url.into())
}

fn decode_native(body: &[u8]) -> Option<Value> {
    serde_json::from_slice(body).ok()
}

fn parse_job(
    reference: GlmAsyncJobRef,
    native: Value,
    fallback_model: Option<String>,
) -> GlmAsyncJob {
    let status = native
        .get("task_status")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let model = native
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(fallback_model)
        .unwrap_or_else(|| reference.scope.model.clone());
    let request_id = native
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let native_error = native.get("error").cloned();
    GlmAsyncJob {
        reference,
        task_status: GlmAsyncTaskStatus::from(status),
        request_id,
        model,
        native_error,
        native,
    }
}

fn invalid_config(message: impl Into<String>) -> GlmAsyncError {
    GlmAsyncError::InvalidConfig {
        message: message.into(),
    }
}

fn invalid_request(message: impl Into<String>) -> GlmAsyncError {
    GlmAsyncError::InvalidRequest {
        message: message.into(),
    }
}
