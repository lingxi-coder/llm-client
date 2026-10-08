//! OpenRouter's inline Batch inference API.
//!
//! This service intentionally has its own request and lifecycle types. The
//! OpenRouter API accepts an inline `requests` array and returns results in a
//! batch object; it does not use the OpenAI files/JSONL workflow.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId},
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeSet;

const API_BASE_URL: &str = "https://openrouter.ai/api/v1";
const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_REQUESTS: usize = 50_000;

/// Identity used to keep a batch reference tied to one OpenRouter connection
/// and account. Credentials are deliberately excluded from the reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
}

impl OpenRouterBatchScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, OpenRouterBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if profile_name.trim().is_empty() || account_scope.trim().is_empty() {
            return Err(invalid("profile name and account scope must be non-empty"));
        }
        Ok(Self {
            provider_id: ProviderId::from("openrouter"),
            profile_name,
            account_scope,
            endpoint_fingerprint: provider_file_endpoint_fingerprint(API_BASE_URL),
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), OpenRouterBatchError> {
        if self.provider_id.as_str() != "openrouter"
            || self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(API_BASE_URL)
        {
            return Err(invalid("OpenRouter Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// One of the request shapes documented for OpenRouter Batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterBatchEndpoint {
    ChatCompletions,
    Responses,
    Messages,
    Embeddings,
}

impl OpenRouterBatchEndpoint {
    pub fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
            Self::Messages => "/v1/messages",
            Self::Embeddings => "/v1/embeddings",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterBatchChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchChatContent {
    Text(String),
    Parts(Vec<OpenRouterBatchChatContentPart>),
}

impl From<String> for OpenRouterBatchChatContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for OpenRouterBatchChatContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterBatchChatContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenRouterBatchImageUrl },
    File { file: OpenRouterBatchChatFileUrl },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchChatFileUrl {
    pub filename: String,
    /// Public URL serialized in OpenRouter's `file_data` field.
    #[serde(rename = "file_data")]
    pub file_url: String,
}

impl OpenRouterBatchChatContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: OpenRouterBatchImageUrl { url: url.into() },
        }
    }

    pub fn file_url(filename: impl Into<String>, url: impl Into<String>) -> Self {
        Self::File {
            file: OpenRouterBatchChatFileUrl {
                filename: filename.into(),
                file_url: url.into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchChatMessage {
    pub role: OpenRouterBatchChatRole,
    pub content: OpenRouterBatchChatContent,
}

/// Typed body for `/v1/chat/completions`, including URL-based image and file parts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenRouterBatchChatBody {
    pub messages: Vec<OpenRouterBatchChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterBatchResponsesRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchResponsesInput {
    Text(String),
    Messages(Vec<OpenRouterBatchResponsesMessage>),
}

impl From<String> for OpenRouterBatchResponsesInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for OpenRouterBatchResponsesInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchResponsesMessage {
    pub role: OpenRouterBatchResponsesRole,
    pub content: OpenRouterBatchResponsesContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchResponsesContent {
    Text(String),
    Parts(Vec<OpenRouterBatchResponsesContentPart>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterBatchResponsesContentPart {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    Image { image_url: String },
    #[serde(rename = "input_file")]
    File { file_url: String },
}

impl OpenRouterBatchResponsesContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            image_url: url.into(),
        }
    }

    pub fn file_url(url: impl Into<String>) -> Self {
        Self::File {
            file_url: url.into(),
        }
    }
}

/// Typed body for `/v1/responses`, including URL-based image and file parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchResponsesBody {
    pub input: OpenRouterBatchResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterBatchMessagesRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchMessagesContent {
    Text(String),
    Parts(Vec<OpenRouterBatchMessagesContentPart>),
}

impl From<String> for OpenRouterBatchMessagesContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for OpenRouterBatchMessagesContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterBatchMessagesSourceType {
    Url,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchMessagesUrlSource {
    #[serde(rename = "type")]
    pub source_type: OpenRouterBatchMessagesSourceType,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterBatchMessagesContentPart {
    Text {
        text: String,
    },
    Image {
        source: OpenRouterBatchMessagesUrlSource,
    },
    Document {
        source: OpenRouterBatchMessagesUrlSource,
    },
}

impl OpenRouterBatchMessagesContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: OpenRouterBatchMessagesUrlSource {
                source_type: OpenRouterBatchMessagesSourceType::Url,
                url: url.into(),
            },
        }
    }

    pub fn document_url(url: impl Into<String>) -> Self {
        Self::Document {
            source: OpenRouterBatchMessagesUrlSource {
                source_type: OpenRouterBatchMessagesSourceType::Url,
                url: url.into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchMessagesMessage {
    pub role: OpenRouterBatchMessagesRole,
    pub content: OpenRouterBatchMessagesContent,
}

/// Typed body for `/v1/messages`, including URL-based image and document parts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenRouterBatchMessagesBody {
    pub messages: Vec<OpenRouterBatchMessagesMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchEmbeddingInput {
    Text(String),
    Texts(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterBatchEncodingFormat {
    Float,
    Base64,
}

/// Typed text-only body for `/v1/embeddings`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenRouterBatchEmbeddingsBody {
    pub input: OpenRouterBatchEmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<OpenRouterBatchEncodingFormat>,
}

/// Endpoint-specific, typed request body. Unsupported or undocumented body
/// fields cannot be forwarded through this API.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum OpenRouterBatchRequestBody {
    ChatCompletions(OpenRouterBatchChatBody),
    Responses(OpenRouterBatchResponsesBody),
    Messages(OpenRouterBatchMessagesBody),
    Embeddings(OpenRouterBatchEmbeddingsBody),
}

impl OpenRouterBatchRequestBody {
    fn matches_endpoint(&self, endpoint: OpenRouterBatchEndpoint) -> bool {
        matches!(
            (endpoint, self),
            (
                OpenRouterBatchEndpoint::ChatCompletions,
                Self::ChatCompletions(_)
            ) | (OpenRouterBatchEndpoint::Responses, Self::Responses(_))
                | (OpenRouterBatchEndpoint::Messages, Self::Messages(_))
                | (OpenRouterBatchEndpoint::Embeddings, Self::Embeddings(_))
        )
    }
}

/// One item in OpenRouter's inline `requests` array.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenRouterBatchLine {
    pub custom_id: String,
    pub body: OpenRouterBatchRequestBody,
}

/// Validated requests sharing a single documented endpoint and model.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouterBatchInput {
    endpoint: OpenRouterBatchEndpoint,
    model: String,
    provider_only: Vec<String>,
    requests: Vec<OpenRouterBatchLine>,
}

impl OpenRouterBatchInput {
    pub fn new(
        endpoint: OpenRouterBatchEndpoint,
        model: impl Into<String>,
        requests: Vec<OpenRouterBatchLine>,
    ) -> Result<Self, OpenRouterBatchError> {
        let model = model.into();
        validate_model(&model)?;
        if requests.is_empty() || requests.len() > MAX_REQUESTS {
            return Err(invalid("Batch input requires 1 to 50,000 requests"));
        }
        let mut ids = BTreeSet::new();
        for request in &requests {
            if request.custom_id.trim().is_empty()
                || request.custom_id.chars().count() > 256
                || request.custom_id.chars().any(char::is_control)
                || !ids.insert(request.custom_id.as_str())
            {
                return Err(invalid(
                    "custom_id must be unique and contain 1 to 256 non-control characters",
                ));
            }
            if !request.body.matches_endpoint(endpoint) {
                return Err(invalid(
                    "each request body must match the selected Batch endpoint",
                ));
            }
            validate_body(&request.body)?;
        }
        Ok(Self {
            endpoint,
            model,
            provider_only: Vec::new(),
            requests,
        })
    }

    /// Restrict routing to one or more OpenRouter provider slugs. OpenRouter
    /// Batch supports `provider.only`; its other synchronous preferences are
    /// intentionally not accepted here.
    pub fn with_provider_only(
        mut self,
        providers: Vec<String>,
    ) -> Result<Self, OpenRouterBatchError> {
        if providers.is_empty() || providers.len() > 32 {
            return Err(invalid("provider.only must contain 1 to 32 provider slugs"));
        }
        let mut unique = BTreeSet::new();
        for provider in &providers {
            if provider.is_empty()
                || provider.len() > 128
                || !provider.split('/').all(is_provider_slug_segment)
                || !unique.insert(provider.as_str())
            {
                return Err(invalid(
                    "provider.only contains an invalid or duplicate slug",
                ));
            }
        }
        validate_provider_modalities(self.endpoint, &self.requests, &providers)?;
        self.provider_only = providers;
        Ok(self)
    }

    pub fn endpoint(&self) -> OpenRouterBatchEndpoint {
        self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Durable identity for a submitted or listed batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterBatchJobRef {
    scope: OpenRouterBatchScope,
    batch_id: String,
    endpoint: OpenRouterBatchEndpoint,
    model: String,
}

impl OpenRouterBatchJobRef {
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn scope(&self) -> &OpenRouterBatchScope {
        &self.scope
    }

    pub fn endpoint(&self) -> OpenRouterBatchEndpoint {
        self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenRouterBatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
    Other(String),
}

impl OpenRouterBatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }

    fn filter_value(&self) -> Option<&'static str> {
        match self {
            Self::Validating => Some("validating"),
            Self::InProgress => Some("in_progress"),
            Self::Completed => Some("completed"),
            Self::Failed => Some("failed"),
            Self::Expired => Some("expired"),
            Self::Cancelled => Some("cancelled"),
            Self::Finalizing | Self::Cancelling | Self::Other(_) => None,
        }
    }
}

impl Serialize for OpenRouterBatchStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::Validating => "validating",
            Self::InProgress => "in_progress",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Other(value) => value,
        })
    }
}

impl<'de> Deserialize<'de> for OpenRouterBatchStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "validating" => Self::Validating,
            "in_progress" => Self::InProgress,
            "finalizing" => Self::Finalizing,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelling" => Self::Cancelling,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenRouterBatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterBatchResponse {
    pub status_code: u16,
    pub request_id: Option<String>,
    pub body: Value,
    pub native: Value,
}

/// Per-request output. The provider response body and complete result object
/// are retained, while an item-level `error` remains independent of batch
/// status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterBatchResult {
    pub id: Option<String>,
    pub custom_id: String,
    pub response: Option<OpenRouterBatchResponse>,
    pub error: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterBatchJob {
    pub reference: OpenRouterBatchJobRef,
    pub status: OpenRouterBatchStatus,
    pub completion_window: Option<String>,
    pub request_counts: Option<OpenRouterBatchRequestCounts>,
    pub results: Option<Vec<OpenRouterBatchResult>>,
    pub error: Option<Value>,
    pub created_at: Option<i64>,
    pub finalized_at: Option<i64>,
    pub usage: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpenRouterBatchListOptions {
    pub limit: Option<u8>,
    pub after: Option<String>,
    pub statuses: Vec<OpenRouterBatchStatus>,
    /// Include batches created strictly after a Unix timestamp or ISO-8601 date/time.
    pub created_after: Option<String>,
    /// Include batches created strictly before a Unix timestamp or ISO-8601 date/time.
    pub created_before: Option<String>,
}

impl OpenRouterBatchListOptions {
    pub fn created_after(mut self, timestamp: impl Into<String>) -> Self {
        self.created_after = Some(timestamp.into());
        self
    }

    pub fn created_before(mut self, timestamp: impl Into<String>) -> Self {
        self.created_before = Some(timestamp.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterBatchPage {
    pub jobs: Vec<OpenRouterBatchJob>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterBatchDeletion {
    pub reference: OpenRouterBatchJobRef,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenRouterBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid OpenRouter Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid OpenRouter Batch response: {0}")]
    InvalidResponse(String),
    #[error("OpenRouter Batch returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of OpenRouter Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("outcome of OpenRouter Batch {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        reason: String,
    },
}

/// OpenRouter Batch API client. It makes one HTTP call per operation and
/// leaves polling, retry decisions, and deletion timing to the caller.
#[derive(Clone)]
pub struct OpenRouterBatchService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: OpenRouterBatchScope,
}

impl<'a> OpenRouterBatchService<'a> {
    fn pin(&self) -> Result<Self, crate::protocol::LlmError> {
        let mut pinned = self.clone();
        pinned.binding = self
            .binding
            .as_ref()
            .map(crate::providers::binding::ProviderBinding::pinned)
            .transpose()?;
        Ok(pinned)
    }

    pub(crate) fn with_binding(
        mut self,
        binding: &crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding.clone());
        self
    }

    fn request_credential<'o>(
        &self,
        request_options: &'o crate::RequestOptions,
    ) -> Result<&'o str, OpenRouterBatchError> {
        if request_options
            .account_scope
            .as_deref()
            .is_some_and(|account| account != self.scope.account_scope())
        {
            return Err(crate::protocol::LlmError::PermissionDenied {
                message: "request account scope does not match the native resource scope".into(),
            }
            .into());
        }

        let credential = request_options
            .credential
            .as_ref()
            .map(|value| value.expose_secret().as_str())
            .unwrap_or_default();
        if credential.trim().is_empty() {
            return Err(invalid("OpenRouter API key must be non-empty"));
        }
        Ok(credential)
    }

    pub fn new(
        http: &'a dyn Transport,
        scope: OpenRouterBatchScope,
    ) -> Result<Self, OpenRouterBatchError> {
        scope.validate()?;

        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &OpenRouterBatchScope {
        &self.scope
    }

    /// Submit one inline request array. A transport or body-read failure can
    /// happen after OpenRouter accepted it, so this method never retries.
    pub async fn submit(
        &self,
        input: &OpenRouterBatchInput,
        request_options: &crate::RequestOptions,
    ) -> Result<OpenRouterBatchJob, OpenRouterBatchError> {
        let pinned_service = self.pin()?;
        let wire = SubmitWire {
            endpoint: input.endpoint.path(),
            model: &input.model,
            provider: (!input.provider_only.is_empty()).then_some(ProviderWire {
                only: &input.provider_only,
            }),
            completion_window: "24h",
            requests: &input.requests,
        };
        let body = serde_json::to_vec(&wire)
            .map_err(|_| invalid("Batch input could not be encoded as JSON"))?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(invalid("inline Batch request exceeds 64 MiB"));
        }
        let request =
            pinned_service.request("POST", "/batches", Some(Bytes::from(body)), request_options)?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("submit", source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let job = pinned_service.decode_job(&response.body).map_err(|error| {
            OpenRouterBatchError::OutcomeUnknownResponse {
                operation: "submit",
                reason: error.to_string(),
            }
        })?;
        if job.reference.endpoint != input.endpoint || job.reference.model != input.model {
            return Err(OpenRouterBatchError::OutcomeUnknownResponse {
                operation: "submit",
                reason: "created Batch does not match the submitted endpoint and model".into(),
            });
        }
        Ok(job)
    }

    pub async fn get(
        &self,
        reference: &OpenRouterBatchJobRef,
        request_options: &crate::RequestOptions,
    ) -> Result<OpenRouterBatchJob, OpenRouterBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!("/batches/{}", reference.batch_id);
        let response = pinned_service
            .send("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let job = pinned_service.decode_job(&response.body)?;
        ensure_same_job(reference, &job.reference)?;
        Ok(job)
    }

    pub async fn list(
        &self,
        options: &OpenRouterBatchListOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<OpenRouterBatchPage, OpenRouterBatchError> {
        let pinned_service = self.pin()?;
        let limit = options.limit.unwrap_or(20);
        if !(1..=100).contains(&limit) {
            return Err(invalid("Batch list limit must be between 1 and 100"));
        }
        if options
            .after
            .as_deref()
            .is_some_and(|value| !valid_batch_id(value))
        {
            return Err(invalid("Batch list cursor is invalid"));
        }
        for status in &options.statuses {
            if status.filter_value().is_none() {
                return Err(invalid(
                    "list filters support validating, in_progress, completed, failed, expired, and cancelled",
                ));
            }
        }
        let created_after = options
            .created_after
            .as_deref()
            .map(parse_created_at_filter)
            .transpose()?;
        let created_before = options
            .created_before
            .as_deref()
            .map(parse_created_at_filter)
            .transpose()?;
        if created_after
            .zip(created_before)
            .is_some_and(|(after, before)| after >= before)
        {
            return Err(invalid("created_after must be earlier than created_before"));
        }
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query.append_pair("limit", &limit.to_string());
        if let Some(after) = &options.after {
            query.append_pair("after", after);
        }
        for status in &options.statuses {
            query.append_pair("status", status.filter_value().expect("validated status"));
        }
        if let Some(created_after) = &options.created_after {
            query.append_pair("created_after", created_after);
        }
        if let Some(created_before) = &options.created_before {
            query.append_pair("created_before", created_before);
        }
        let path = format!("/batches?{}", query.finish());
        let response = pinned_service
            .send("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let wire: BatchListWire = decode_json(&response.body)?;
        let mut jobs = Vec::with_capacity(wire.data.len());
        for batch in wire.data {
            let job = pinned_service.decode_job_value(batch)?;
            jobs.push(job);
        }
        Ok(OpenRouterBatchPage {
            jobs,
            has_more: wire.has_more,
            first_id: wire.first_id,
            last_id: wire.last_id,
        })
    }

    /// Delete a terminal batch and its retained request/result artifacts.
    /// OpenRouter rejects an in-flight batch with HTTP 409; this service does
    /// not silently cancel or retry the deletion.
    pub async fn delete(
        &self,
        reference: &OpenRouterBatchJobRef,
        request_options: &crate::RequestOptions,
    ) -> Result<OpenRouterBatchDeletion, OpenRouterBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!("/batches/{}", reference.batch_id);
        let request = pinned_service.request("DELETE", &path, None, request_options)?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("delete", source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let wire: DeleteWire = decode_json(&response.body)?;
        if wire.id != reference.batch_id {
            return Err(OpenRouterBatchError::InvalidResponse(
                "deleted Batch ID does not match the requested reference".into(),
            ));
        }
        let mut native = wire.extra;
        native.insert("id".into(), Value::String(wire.id));
        Ok(OpenRouterBatchDeletion {
            reference: reference.clone(),
            native: Value::Object(native),
        })
    }

    fn decode_job(&self, body: &[u8]) -> Result<OpenRouterBatchJob, OpenRouterBatchError> {
        self.decode_job_value(decode_json(body)?)
    }

    fn decode_job_value(&self, native: Value) -> Result<OpenRouterBatchJob, OpenRouterBatchError> {
        let wire: BatchWire = serde_json::from_value(native.clone())
            .map_err(|error| OpenRouterBatchError::InvalidResponse(error.to_string()))?;
        if !valid_batch_id(&wire.id) {
            return Err(OpenRouterBatchError::InvalidResponse(
                "provider returned an invalid Batch ID".into(),
            ));
        }
        validate_model(&wire.model).map_err(|_| {
            OpenRouterBatchError::InvalidResponse("provider returned an invalid Batch model".into())
        })?;
        let endpoint = endpoint_from_path(&wire.endpoint).ok_or_else(|| {
            OpenRouterBatchError::InvalidResponse(
                "provider returned an unsupported Batch endpoint".into(),
            )
        })?;
        let reference = OpenRouterBatchJobRef {
            scope: self.scope.clone(),
            batch_id: wire.id,
            endpoint,
            model: wire.model,
        };
        let results = wire
            .results
            .map(|results| {
                let mut custom_ids = BTreeSet::new();
                results
                    .into_iter()
                    .map(|result| {
                        let decoded = decode_result(result)?;
                        if !custom_ids.insert(decoded.custom_id.clone()) {
                            return Err(OpenRouterBatchError::InvalidResponse(
                                "provider returned duplicate result custom_id values".into(),
                            ));
                        }
                        Ok(decoded)
                    })
                    .collect()
            })
            .transpose()?;
        Ok(OpenRouterBatchJob {
            reference,
            status: wire.status,
            completion_window: wire.completion_window,
            request_counts: wire.request_counts,
            results,
            error: wire.error,
            created_at: wire.created_at,
            finalized_at: wire.finalized_at,
            usage: wire.usage,
            native,
        })
    }

    fn validate_reference(
        &self,
        reference: &OpenRouterBatchJobRef,
    ) -> Result<(), OpenRouterBatchError> {
        reference.scope.validate()?;
        if reference.scope != self.scope
            || !valid_batch_id(&reference.batch_id)
            || validate_model(&reference.model).is_err()
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    async fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<crate::transport::HttpResponse, OpenRouterBatchError> {
        let request = self.request(method, path, body, request_options)?;
        Ok(HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await?)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<HttpRequest, OpenRouterBatchError> {
        if !path.starts_with('/') || path.contains('#') {
            return Err(invalid("invalid internal Batch route"));
        }
        let mut headers = vec![(
            "Authorization".into(),
            format!("Bearer {}", self.request_credential(request_options)?),
        )];
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            http1_header_layout: None,
            method: method.into(),
            url: format!("{API_BASE_URL}{path}"),
            headers,
            body: body.unwrap_or_default(),
            timeout: request_options.total_timeout,
        })
    }
}

#[derive(Serialize)]
struct SubmitWire<'a> {
    endpoint: &'static str,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<ProviderWire<'a>>,
    completion_window: &'static str,
    requests: &'a [OpenRouterBatchLine],
}

#[derive(Serialize)]
struct ProviderWire<'a> {
    only: &'a [String],
}

#[derive(Deserialize)]
struct BatchWire {
    id: String,
    endpoint: String,
    model: String,
    status: OpenRouterBatchStatus,
    #[serde(default)]
    completion_window: Option<String>,
    #[serde(default)]
    request_counts: Option<OpenRouterBatchRequestCounts>,
    #[serde(default)]
    results: Option<Vec<ResultWire>>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    finalized_at: Option<i64>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct ResultWire {
    #[serde(default)]
    id: Option<String>,
    custom_id: String,
    #[serde(default)]
    response: Option<ResponseWire>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct ResponseWire {
    status_code: u16,
    #[serde(default)]
    request_id: Option<String>,
    body: Value,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

fn decode_result(wire: ResultWire) -> Result<OpenRouterBatchResult, OpenRouterBatchError> {
    if wire.custom_id.trim().is_empty() || wire.custom_id.chars().any(char::is_control) {
        return Err(OpenRouterBatchError::InvalidResponse(
            "provider returned a result with an invalid custom_id".into(),
        ));
    }
    if wire.response.is_some() == wire.error.is_some() {
        return Err(OpenRouterBatchError::InvalidResponse(
            "each Batch result must contain exactly one response or error".into(),
        ));
    }
    let response = wire.response.map(|response| {
        let mut native = response.extra;
        native.insert("status_code".into(), Value::from(response.status_code));
        native.insert(
            "request_id".into(),
            response
                .request_id
                .as_ref()
                .map(|request_id| Value::String(request_id.clone()))
                .unwrap_or(Value::Null),
        );
        native.insert("body".into(), response.body.clone());
        OpenRouterBatchResponse {
            status_code: response.status_code,
            request_id: response.request_id,
            body: response.body,
            native: Value::Object(native),
        }
    });
    let mut native = wire.extra;
    if let Some(id) = &wire.id {
        native.insert("id".into(), Value::String(id.clone()));
    }
    native.insert("custom_id".into(), Value::String(wire.custom_id.clone()));
    native.insert(
        "response".into(),
        response
            .as_ref()
            .map(|response| response.native.clone())
            .unwrap_or(Value::Null),
    );
    native.insert("error".into(), wire.error.clone().unwrap_or(Value::Null));
    Ok(OpenRouterBatchResult {
        id: wire.id,
        custom_id: wire.custom_id,
        response,
        error: wire.error,
        native: Value::Object(native),
    })
}

#[derive(Deserialize)]
struct BatchListWire {
    #[serde(default)]
    data: Vec<Value>,
    #[serde(default)]
    first_id: Option<String>,
    #[serde(default)]
    last_id: Option<String>,
    #[serde(default)]
    has_more: bool,
}

#[derive(Deserialize)]
struct DeleteWire {
    id: String,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

fn endpoint_from_path(path: &str) -> Option<OpenRouterBatchEndpoint> {
    match path {
        "/v1/chat/completions" => Some(OpenRouterBatchEndpoint::ChatCompletions),
        "/v1/responses" => Some(OpenRouterBatchEndpoint::Responses),
        "/v1/messages" => Some(OpenRouterBatchEndpoint::Messages),
        "/v1/embeddings" => Some(OpenRouterBatchEndpoint::Embeddings),
        _ => None,
    }
}

fn ensure_same_job(
    expected: &OpenRouterBatchJobRef,
    actual: &OpenRouterBatchJobRef,
) -> Result<(), OpenRouterBatchError> {
    if expected.batch_id != actual.batch_id
        || expected.model != actual.model
        || expected.endpoint != actual.endpoint
    {
        return Err(OpenRouterBatchError::InvalidResponse(
            "provider returned a different Batch identity".into(),
        ));
    }
    Ok(())
}

fn ensure_success(
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<(), OpenRouterBatchError> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    let request_id = headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("x-request-id")
                || name.eq_ignore_ascii_case("x-openrouter-request-id")
        })
        .map(|(_, value)| value.clone());
    let body = serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()));
    Err(OpenRouterBatchError::Provider {
        status,
        request_id,
        body,
    })
}

fn decode_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, OpenRouterBatchError> {
    serde_json::from_slice(body)
        .map_err(|error| OpenRouterBatchError::InvalidResponse(error.to_string()))
}

fn validate_model(model: &str) -> Result<(), OpenRouterBatchError> {
    if model.trim().is_empty() || model.len() > 256 || model.chars().any(char::is_control) {
        return Err(invalid(
            "model must contain 1 to 256 non-control characters",
        ));
    }
    Ok(())
}

fn validate_body(body: &OpenRouterBatchRequestBody) -> Result<(), OpenRouterBatchError> {
    match body {
        OpenRouterBatchRequestBody::ChatCompletions(body) => {
            if body.messages.is_empty()
                || body.temperature.is_some_and(|value| !value.is_finite())
                || body.top_p.is_some_and(|value| !value.is_finite())
            {
                return Err(invalid(
                    "request body is empty or contains an invalid numeric value",
                ));
            }
            for message in &body.messages {
                validate_chat_content(message.role, &message.content)?;
            }
        }
        OpenRouterBatchRequestBody::Responses(body) => match &body.input {
            OpenRouterBatchResponsesInput::Text(text) if text.trim().is_empty() => {
                return Err(invalid("Responses input must not be empty"));
            }
            OpenRouterBatchResponsesInput::Text(_) => {}
            OpenRouterBatchResponsesInput::Messages(messages) if messages.is_empty() => {
                return Err(invalid("Responses input messages must not be empty"));
            }
            OpenRouterBatchResponsesInput::Messages(messages) => {
                for message in messages {
                    validate_responses_content(message.role, &message.content)?;
                }
            }
        },
        OpenRouterBatchRequestBody::Messages(body) => {
            if body.messages.is_empty()
                || body.max_tokens == 0
                || body.temperature.is_some_and(|value| !value.is_finite())
                || body.top_p.is_some_and(|value| !value.is_finite())
            {
                return Err(invalid(
                    "request body is empty or contains an invalid numeric value",
                ));
            }
            for message in &body.messages {
                validate_messages_content(message.role, &message.content)?;
            }
        }
        OpenRouterBatchRequestBody::Embeddings(body) => {
            let valid = match &body.input {
                OpenRouterBatchEmbeddingInput::Text(text) => !text.trim().is_empty(),
                OpenRouterBatchEmbeddingInput::Texts(texts) => {
                    !texts.is_empty() && texts.iter().all(|text| !text.trim().is_empty())
                }
            };
            if !valid {
                return Err(invalid("embedding input must not be empty"));
            }
        }
    }
    Ok(())
}

fn validate_chat_content(
    role: OpenRouterBatchChatRole,
    content: &OpenRouterBatchChatContent,
) -> Result<(), OpenRouterBatchError> {
    let OpenRouterBatchChatContent::Parts(parts) = content else {
        return if matches!(content, OpenRouterBatchChatContent::Text(text) if text.trim().is_empty())
        {
            Err(invalid("message content must not be empty"))
        } else {
            Ok(())
        };
    };
    if parts.is_empty() {
        return Err(invalid("message content parts must not be empty"));
    }
    for part in parts {
        match part {
            OpenRouterBatchChatContentPart::Text { text } if text.trim().is_empty() => {
                return Err(invalid("text content part must not be empty"));
            }
            OpenRouterBatchChatContentPart::Text { .. } => {}
            OpenRouterBatchChatContentPart::ImageUrl { image_url } => {
                ensure_user_message(role, "image")?;
                validate_public_url(&image_url.url)?;
            }
            OpenRouterBatchChatContentPart::File { file } => {
                ensure_user_message(role, "file")?;
                validate_filename(&file.filename)?;
                validate_public_url(&file.file_url)?;
            }
        }
    }
    Ok(())
}

fn validate_responses_content(
    role: OpenRouterBatchResponsesRole,
    content: &OpenRouterBatchResponsesContent,
) -> Result<(), OpenRouterBatchError> {
    let OpenRouterBatchResponsesContent::Parts(parts) = content else {
        return if matches!(content, OpenRouterBatchResponsesContent::Text(text) if text.trim().is_empty())
        {
            Err(invalid("Responses message content must not be empty"))
        } else {
            Ok(())
        };
    };
    if parts.is_empty() {
        return Err(invalid("Responses message content parts must not be empty"));
    }
    for part in parts {
        match part {
            OpenRouterBatchResponsesContentPart::Text { text } if text.trim().is_empty() => {
                return Err(invalid("Responses text part must not be empty"));
            }
            OpenRouterBatchResponsesContentPart::Text { .. } => {}
            OpenRouterBatchResponsesContentPart::Image { image_url } => {
                ensure_responses_user_message(role, "image")?;
                validate_public_url(image_url)?;
            }
            OpenRouterBatchResponsesContentPart::File { file_url } => {
                ensure_responses_user_message(role, "file")?;
                validate_public_url(file_url)?;
            }
        }
    }
    Ok(())
}

fn validate_messages_content(
    role: OpenRouterBatchMessagesRole,
    content: &OpenRouterBatchMessagesContent,
) -> Result<(), OpenRouterBatchError> {
    let OpenRouterBatchMessagesContent::Parts(parts) = content else {
        return if matches!(content, OpenRouterBatchMessagesContent::Text(text) if text.trim().is_empty())
        {
            Err(invalid("Messages content must not be empty"))
        } else {
            Ok(())
        };
    };
    if parts.is_empty() {
        return Err(invalid("Messages content parts must not be empty"));
    }
    for part in parts {
        match part {
            OpenRouterBatchMessagesContentPart::Text { text } if text.trim().is_empty() => {
                return Err(invalid("Messages text part must not be empty"));
            }
            OpenRouterBatchMessagesContentPart::Text { .. } => {}
            OpenRouterBatchMessagesContentPart::Image { source }
            | OpenRouterBatchMessagesContentPart::Document { source } => {
                ensure_messages_user_message(role)?;
                validate_public_url(&source.url)?;
            }
        }
    }
    Ok(())
}

fn ensure_user_message(
    role: OpenRouterBatchChatRole,
    kind: &str,
) -> Result<(), OpenRouterBatchError> {
    if role == OpenRouterBatchChatRole::User {
        Ok(())
    } else {
        Err(invalid(format!(
            "{kind} inputs are supported only in user messages"
        )))
    }
}

fn ensure_responses_user_message(
    role: OpenRouterBatchResponsesRole,
    kind: &str,
) -> Result<(), OpenRouterBatchError> {
    if role == OpenRouterBatchResponsesRole::User {
        Ok(())
    } else {
        Err(invalid(format!(
            "{kind} inputs are supported only in user messages"
        )))
    }
}

fn ensure_messages_user_message(
    role: OpenRouterBatchMessagesRole,
) -> Result<(), OpenRouterBatchError> {
    if role == OpenRouterBatchMessagesRole::User {
        Ok(())
    } else {
        Err(invalid(
            "image and document inputs are supported only in user messages",
        ))
    }
}

fn validate_filename(filename: &str) -> Result<(), OpenRouterBatchError> {
    if filename.trim().is_empty() || filename.chars().any(char::is_control) {
        return Err(invalid(
            "file input filename must be non-empty and contain no control characters",
        ));
    }
    Ok(())
}

fn validate_public_url(value: &str) -> Result<(), OpenRouterBatchError> {
    let url = url::Url::parse(value)
        .map_err(|_| invalid("multimodal inputs require public HTTP(S) URLs"))?;
    let host = url
        .host()
        .ok_or_else(|| invalid("multimodal inputs require public HTTP(S) URLs with a host"))?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || is_obviously_non_public_host(host)
    {
        return Err(invalid(
            "multimodal inputs require public HTTP(S) URLs without credentials or local hosts",
        ));
    }
    Ok(())
}

fn is_obviously_non_public_host(host: url::Host<&str>) -> bool {
    match host {
        url::Host::Domain(domain) => {
            let normalized = domain.trim_end_matches('.').to_ascii_lowercase();
            normalized == "localhost"
                || normalized.ends_with(".localhost")
                || normalized.ends_with(".local")
                || normalized.ends_with(".internal")
        }
        url::Host::Ipv4(address) => is_obviously_non_public_ipv4(address),
        url::Host::Ipv6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return is_obviously_non_public_ipv4(mapped);
            }
            let segments = address.segments();
            address.is_loopback()
                || address.is_unspecified()
                || address.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
        }
    }
}

fn is_obviously_non_public_ipv4(address: std::net::Ipv4Addr) -> bool {
    address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_broadcast()
        || address.is_unspecified()
        || address.is_multicast()
}

fn is_provider_slug_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[derive(Default)]
struct BatchModalities {
    image: bool,
    file: bool,
}

fn validate_provider_modalities(
    endpoint: OpenRouterBatchEndpoint,
    requests: &[OpenRouterBatchLine],
    providers: &[String],
) -> Result<(), OpenRouterBatchError> {
    let mut modalities = BatchModalities::default();
    for request in requests {
        request.body.visit_modalities(&mut modalities);
    }
    if !modalities.image && !modalities.file {
        return Ok(());
    }
    for provider in providers {
        let provider_base = provider.split('/').next().unwrap_or(provider);
        let supports_image = endpoint != OpenRouterBatchEndpoint::Embeddings
            && matches!(provider_base, "openai" | "anthropic" | "xai")
            || endpoint == OpenRouterBatchEndpoint::ChatCompletions && provider_base == "deepinfra";
        let supports_file = match provider_base {
            "openai" => endpoint == OpenRouterBatchEndpoint::Responses,
            "anthropic" | "mistral" => endpoint != OpenRouterBatchEndpoint::Embeddings,
            "deepinfra" => endpoint == OpenRouterBatchEndpoint::ChatCompletions,
            _ => false,
        };
        if (modalities.image && !supports_image) || (modalities.file && !supports_file) {
            return Err(invalid(format!(
                "provider.only includes {provider}, which OpenRouter's Batch capability table does not document for these {endpoint:?} multimodal inputs"
            )));
        }
    }
    Ok(())
}

impl OpenRouterBatchRequestBody {
    fn visit_modalities(&self, modalities: &mut BatchModalities) {
        match self {
            Self::ChatCompletions(body) => {
                for message in &body.messages {
                    if let OpenRouterBatchChatContent::Parts(parts) = &message.content {
                        for part in parts {
                            match part {
                                OpenRouterBatchChatContentPart::ImageUrl { .. } => {
                                    modalities.image = true
                                }
                                OpenRouterBatchChatContentPart::File { .. } => {
                                    modalities.file = true
                                }
                                OpenRouterBatchChatContentPart::Text { .. } => {}
                            }
                        }
                    }
                }
            }
            Self::Responses(body) => {
                if let OpenRouterBatchResponsesInput::Messages(messages) = &body.input {
                    for message in messages {
                        if let OpenRouterBatchResponsesContent::Parts(parts) = &message.content {
                            for part in parts {
                                match part {
                                    OpenRouterBatchResponsesContentPart::Image { .. } => {
                                        modalities.image = true
                                    }
                                    OpenRouterBatchResponsesContentPart::File { .. } => {
                                        modalities.file = true
                                    }
                                    OpenRouterBatchResponsesContentPart::Text { .. } => {}
                                }
                            }
                        }
                    }
                }
            }
            Self::Messages(body) => {
                for message in &body.messages {
                    if let OpenRouterBatchMessagesContent::Parts(parts) = &message.content {
                        for part in parts {
                            match part {
                                OpenRouterBatchMessagesContentPart::Image { .. } => {
                                    modalities.image = true
                                }
                                OpenRouterBatchMessagesContentPart::Document { .. } => {
                                    modalities.file = true
                                }
                                OpenRouterBatchMessagesContentPart::Text { .. } => {}
                            }
                        }
                    }
                }
            }
            Self::Embeddings(_) => {}
        }
    }
}

fn valid_batch_id(value: &str) -> bool {
    value.starts_with("batch_")
        && value.len() > "batch_".len()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn parse_created_at_filter(
    value: &str,
) -> Result<chrono::DateTime<chrono::Utc>, OpenRouterBatchError> {
    if value.is_empty() || value.len() > 64 || value.chars().any(char::is_control) {
        return Err(invalid(
            "creation-time filters must be Unix seconds or an ISO-8601 date/time",
        ));
    }

    let numeric = value.bytes().all(|byte| byte.is_ascii_digit())
        || value.split_once('.').is_some_and(|(whole, fraction)| {
            !whole.is_empty()
                && !fraction.is_empty()
                && whole.bytes().all(|byte| byte.is_ascii_digit())
                && fraction.bytes().all(|byte| byte.is_ascii_digit())
        });
    if numeric {
        let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
        let seconds = seconds
            .parse::<i64>()
            .map_err(|_| invalid("Unix creation-time filter is out of range"))?;
        let nanos = if fraction.is_empty() {
            0
        } else if fraction.len() <= 9 {
            let digits = fraction.len() as u32;
            let fraction = fraction
                .parse::<u32>()
                .map_err(|_| invalid("Unix creation-time filter is invalid"))?;
            fraction * 10_u32.pow(9 - digits)
        } else {
            return Err(invalid(
                "Unix creation-time filters support at most nine fractional digits",
            ));
        };
        return chrono::DateTime::from_timestamp(seconds, nanos)
            .ok_or_else(|| invalid("Unix creation-time filter is out of range"));
    }

    if let Ok(datetime) = chrono::DateTime::parse_from_rfc3339(value) {
        return Ok(datetime.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Ok(date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is a valid time")
            .and_utc());
    }
    if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f") {
        return Ok(datetime.and_utc());
    }
    if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Ok(datetime.and_utc());
    }
    Err(invalid(
        "creation-time filters must be Unix seconds or an ISO-8601 date/time",
    ))
}

fn scope_mismatch() -> OpenRouterBatchError {
    LlmError::PermissionDenied {
        message: "OpenRouter Batch reference belongs to another provider, profile, endpoint, or account scope".into(),
    }
    .into()
}

fn invalid(message: impl Into<String>) -> OpenRouterBatchError {
    OpenRouterBatchError::InvalidInput(message.into())
}

fn mutation_transport_error(operation: &'static str, source: LlmError) -> OpenRouterBatchError {
    if matches!(
        &source,
        LlmError::Transport { .. } | LlmError::TransportTimeout { .. }
    ) {
        OpenRouterBatchError::OutcomeUnknown { operation, source }
    } else {
        OpenRouterBatchError::Llm(source)
    }
}
