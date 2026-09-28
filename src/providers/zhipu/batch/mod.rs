//! Zhipu GLM Batch jobs on the documented mainland BigModel API.
//!
//! This deliberately implements only the documented `/v4/chat/completions`
//! batch endpoint. Input files and job/result references are bound to one
//! profile, account scope, region and endpoint. International Z.AI Batch is
//! rejected because its first-party API documentation does not currently
//! publish this lifecycle.

#![doc = concat!(
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/glm-batch.md")),
    "\n\n",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/glm-batch.en.md"))
)]

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId},
    transport::{HttpExecutor, HttpRequest, HttpStreamRequest, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    pin::Pin,
    task::{Context, Poll},
};

const CHINA_BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";
const CHAT_ENDPOINT: &str = "/v4/chat/completions";
const MAX_INPUT_BYTES: usize = 200_000_000;
const MAX_INPUT_LINE_BYTES: usize = 1_000_000;
const MAX_INPUT_LINES: usize = 50_000;
const MAX_CONTROL_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Region selection is explicit. Only mainland BigModel Batch is documented
/// by the currently published first-party Batch lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmBatchRegion {
    ChinaMainland,
    International,
}

impl GlmBatchRegion {
    fn base_url(self) -> Option<&'static str> {
        match self {
            Self::ChinaMainland => Some(CHINA_BASE_URL),
            Self::International => None,
        }
    }
}

/// Connection and account identity carried by all GLM Batch references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    region: GlmBatchRegion,
    endpoint_fingerprint: String,
}

impl GlmBatchScope {
    /// Bind a profile and account to an explicitly selected GLM region.
    /// `International` currently returns [`GlmBatchError::UnsupportedRegion`]
    /// before any request can be sent.
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: GlmBatchRegion,
    ) -> Result<Self, GlmBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid("profile name and account scope must be non-empty and contain no control characters"));
        }
        let Some(base_url) = region.base_url() else {
            return Err(GlmBatchError::UnsupportedRegion { region });
        };
        Ok(Self {
            provider_id: ProviderId::from("zhipu"),
            profile_name,
            account_scope,
            region,
            endpoint_fingerprint: provider_file_endpoint_fingerprint(base_url),
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

    pub fn region(&self) -> GlmBatchRegion {
        self.region
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), GlmBatchError> {
        let Some(base_url) = self.region.base_url() else {
            return Err(GlmBatchError::UnsupportedRegion {
                region: self.region,
            });
        };
        if self.provider_id.as_str() != "zhipu"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(base_url)
        {
            return Err(invalid("GLM Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// One GLM chat message. `content` remains JSON so provider-supported text
/// and multimodal message forms can be represented without rewriting them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchMessage {
    pub role: String,
    pub content: Value,
}

/// Chat-completion body carried by one JSONL row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchChatRequest {
    pub model: String,
    pub messages: Vec<GlmBatchMessage>,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl GlmBatchChatRequest {
    pub fn new(
        model: impl Into<String>,
        messages: Vec<GlmBatchMessage>,
    ) -> Result<Self, GlmBatchError> {
        let request = Self {
            model: model.into(),
            messages,
            additional: BTreeMap::new(),
        };
        request.validate()?;
        Ok(request)
    }

    /// Preserve another provider chat-completion parameter. `model`,
    /// `messages`, and `stream` remain controlled by this typed wrapper.
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, GlmBatchError> {
        let name = name.into();
        if !valid_identity(&name) || matches!(name.as_str(), "model" | "messages" | "stream") {
            return Err(invalid(
                "parameter name is invalid, reserved, or unsupported for Batch",
            ));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    pub fn additional_parameters(&self) -> &BTreeMap<String, Value> {
        &self.additional
    }

    fn validate(&self) -> Result<(), GlmBatchError> {
        if !valid_identity(&self.model) || self.model.len() > 256 {
            return Err(invalid(
                "model must contain 1 to 256 non-control characters",
            ));
        }
        if self.messages.is_empty()
            || self
                .messages
                .iter()
                .any(|message| !valid_identity(&message.role) || message.content.is_null())
        {
            return Err(invalid(
                "chat Batch requests require messages with non-empty roles and content",
            ));
        }
        if self.additional.contains_key("stream") {
            return Err(invalid("GLM Batch requests cannot use stream"));
        }
        Ok(())
    }
}

/// One JSONL request with a caller-owned result correlation ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchLine {
    pub custom_id: String,
    pub body: GlmBatchChatRequest,
}

impl GlmBatchLine {
    pub fn new(
        custom_id: impl Into<String>,
        body: GlmBatchChatRequest,
    ) -> Result<Self, GlmBatchError> {
        let line = Self {
            custom_id: custom_id.into(),
            body,
        };
        line.validate()?;
        Ok(line)
    }

    fn validate(&self) -> Result<(), GlmBatchError> {
        if !valid_custom_id(&self.custom_id) {
            return Err(invalid(
                "custom_id must contain 1 to 256 non-control characters",
            ));
        }
        self.body.validate()
    }
}

/// Validated, single-model `/v4/chat/completions` JSONL input.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmBatchInput {
    lines: Vec<GlmBatchLine>,
}

impl GlmBatchInput {
    pub fn new(lines: Vec<GlmBatchLine>) -> Result<Self, GlmBatchError> {
        if lines.is_empty() || lines.len() > MAX_INPUT_LINES {
            return Err(invalid(
                "Batch input must contain between 1 and 50,000 rows",
            ));
        }
        let mut ids = BTreeSet::new();
        let model = &lines[0].body.model;
        for line in &lines {
            line.validate()?;
            if !ids.insert(line.custom_id.as_str()) {
                return Err(invalid("custom_id values must be unique"));
            }
            if &line.body.model != model {
                return Err(invalid("one GLM Batch file cannot mix models"));
            }
        }
        Ok(Self { lines })
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn model(&self) -> &str {
        &self.lines[0].body.model
    }

    /// Encode each row as `custom_id`, `method`, `url`, and `body` JSONL.
    pub fn encode_jsonl(&self) -> Result<Bytes, GlmBatchError> {
        let mut output = BytesMut::new();
        for line in &self.lines {
            let row = WireBatchLine {
                custom_id: &line.custom_id,
                method: "POST",
                url: CHAT_ENDPOINT,
                body: &line.body,
            };
            let encoded = serde_json::to_vec(&row)
                .map_err(|_| invalid("Batch row could not be encoded as JSON"))?;
            if encoded.len() > MAX_INPUT_LINE_BYTES {
                return Err(invalid("a Batch JSONL row exceeds 1 MB"));
            }
            if output.len().saturating_add(encoded.len()).saturating_add(1) > MAX_INPUT_BYTES {
                return Err(invalid("Batch JSONL input exceeds 200 MB"));
            }
            output.extend_from_slice(&encoded);
            output.extend_from_slice(b"\n");
        }
        Ok(output.freeze())
    }
}

#[derive(Serialize)]
struct WireBatchLine<'a> {
    custom_id: &'a str,
    method: &'static str,
    url: &'static str,
    body: &'a GlmBatchChatRequest,
}

/// Input file ID bound to its originating profile/account/region and chat
/// endpoint. The provider ID is opaque to callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchInputFileRef {
    scope: GlmBatchScope,
    endpoint_fingerprint: String,
    file_id: String,
    model: String,
}

impl GlmBatchInputFileRef {
    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn scope(&self) -> &GlmBatchScope {
        &self.scope
    }
}

/// Durable, account-scoped identity of one accepted Batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchJobRef {
    scope: GlmBatchScope,
    endpoint_fingerprint: String,
    batch_id: String,
}

impl GlmBatchJobRef {
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn scope(&self) -> &GlmBatchScope {
        &self.scope
    }
}

/// Provider-native job status. New status strings remain available in
/// `Other` rather than making the response undecodable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmBatchStatus {
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

impl GlmBatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }
}

impl Serialize for GlmBatchStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Self::Validating => "validating",
            Self::InProgress => "in_progress",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Other(value) => value,
        };
        serializer.serialize_str(value)
    }
}

impl<'de> Deserialize<'de> for GlmBatchStatus {
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
pub struct GlmBatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

/// Typed job core and preserved native response fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmBatchJob {
    pub reference: GlmBatchJobRef,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: GlmBatchStatus,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub request_counts: Option<GlmBatchRequestCounts>,
    pub native: Value,
}

/// One paginated page returned by the documented Batch list operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmBatchPage {
    pub batches: Vec<GlmBatchJob>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
    pub native: Value,
}

/// Parameters for one explicit GLM Batch list request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GlmBatchListOptions {
    after: Option<String>,
    limit: Option<u32>,
}

impl GlmBatchListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn after(mut self, id: impl Into<String>) -> Self {
        self.after = Some(id.into());
        self
    }

    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }
}

impl GlmBatchJob {
    pub fn output_ref(&self) -> Option<GlmBatchResultRef> {
        Some(self.result_ref(GlmBatchResultKind::Output, self.output_file_id.as_ref()?))
    }

    pub fn error_ref(&self) -> Option<GlmBatchResultRef> {
        Some(self.result_ref(GlmBatchResultKind::Error, self.error_file_id.as_ref()?))
    }

    fn result_ref(&self, kind: GlmBatchResultKind, file_id: &str) -> GlmBatchResultRef {
        GlmBatchResultRef {
            job: self.reference.clone(),
            result_endpoint_fingerprint: self.reference.endpoint_fingerprint.clone(),
            file_id: file_id.to_owned(),
            kind,
        }
    }
}

/// Whether a JSONL result reference points to successful output or failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmBatchResultKind {
    Output,
    Error,
}

/// Scoped reference to one provider result file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchResultRef {
    job: GlmBatchJobRef,
    result_endpoint_fingerprint: String,
    file_id: String,
    kind: GlmBatchResultKind,
}

impl GlmBatchResultRef {
    pub fn job(&self) -> &GlmBatchJobRef {
        &self.job
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn kind(&self) -> GlmBatchResultKind {
        self.kind
    }
}

/// Optional opaque metadata attached to the provider Batch record.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmBatchMetadata {
    #[serde(flatten)]
    pub values: BTreeMap<String, String>,
}

impl GlmBatchMetadata {
    fn validate(&self) -> Result<(), GlmBatchError> {
        if self.values.len() > 16
            || self.values.iter().any(|(key, value)| {
                !valid_identity(key)
                    || key.chars().count() > 64
                    || value.chars().count() > 512
                    || value.chars().any(char::is_control)
            })
        {
            return Err(invalid(
                "metadata supports at most 16 safe key/value entries (64/512 character limits)",
            ));
        }
        Ok(())
    }
}

/// Values returned when the provider does not let the client determine
/// whether a create-like request was accepted. Callers should reconcile using
/// their provider console or list endpoint; this service will not retry.
#[derive(Debug, thiserror::Error)]
pub enum GlmBatchError {
    #[error("GLM Batch operation failed: {0}")]
    Llm(#[source] Box<LlmError>),
    #[error("invalid GLM Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid GLM Batch response: {message}")]
    InvalidResponse { message: String, native: Box<Value> },
    #[error("GLM Batch API rejected the request with HTTP {status} (code {code:?})")]
    Provider {
        status: u16,
        code: Option<String>,
        request_id: Option<String>,
        body: Box<Value>,
    },
    #[error("GLM Batch is not documented for region {region:?}")]
    UnsupportedRegion { region: GlmBatchRegion },
    #[error("GLM Batch reference belongs to another endpoint, profile, region, or account")]
    ScopeMismatch,
    #[error("outcome of GLM Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        reference: Option<Box<GlmBatchJobRef>>,
        source: Box<LlmError>,
    },
    #[error("outcome of GLM Batch {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        reference: Option<Box<GlmBatchJobRef>>,
        reason: String,
    },
}

impl From<LlmError> for GlmBatchError {
    fn from(error: LlmError) -> Self {
        Self::Llm(Box::new(error))
    }
}

/// Mainland GLM Batch lifecycle. The service never retries, polls, automatically
/// paginates, or buffers the downloaded result file.
#[derive(Clone)]
pub struct GlmBatchService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: GlmBatchScope,
}

impl<'a> GlmBatchService<'a> {
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
    ) -> Result<&'o str, GlmBatchError> {
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
            return Err(invalid("Zhipu API key must be non-empty"));
        }
        Ok(credential)
    }

    pub fn new(http: &'a dyn Transport, scope: GlmBatchScope) -> Result<Self, GlmBatchError> {
        scope.validate()?;

        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &GlmBatchScope {
        &self.scope
    }

    /// Upload a validated JSONL input using multipart `purpose=batch`.
    /// Transport/response ambiguity is explicit and never retried.
    pub async fn upload_input(
        &self,
        input: &GlmBatchInput,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchInputFileRef, GlmBatchError> {
        let pinned_service = self.pin()?;
        let jsonl = input.encode_jsonl()?;
        let boundary = multipart_boundary(&jsonl)?;
        let (prefix, suffix) = multipart_edges(&boundary);
        let content_length = prefix
            .len()
            .checked_add(jsonl.len())
            .and_then(|size| size.checked_add(suffix.len()))
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| invalid("multipart upload length overflowed"))?;
        let mut chunks = vec![Bytes::from(prefix)];
        for start in (0..jsonl.len()).step_by(64 * 1024) {
            chunks.push(jsonl.slice(start..(start + 64 * 1024).min(jsonl.len())));
        }
        chunks.push(Bytes::from(suffix));
        let request = HttpStreamRequest {
            method: "POST".into(),
            url: format!("{CHINA_BASE_URL}/files"),
            headers: vec![
                pinned_service.authorization_header(request_options)?,
                (
                    "Content-Type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: futures::stream::iter(chunks.into_iter().map(Ok)).boxed(),
            content_length,
            timeout: request_options.total_timeout,
        };
        let response = HttpExecutor::new(pinned_service.http)
            .send_stream(request)
            .await
            .map_err(|source| unknown("input upload", None, source))?;
        let response = HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES))
            .await
            .map_err(|source| unknown("input upload", None, source))?;
        if !(200..300).contains(&response.status) {
            if response.status >= 500 {
                return Err(unknown_response(
                    "input upload",
                    None,
                    format!("provider returned HTTP {} after upload", response.status),
                ));
            }
            return Err(provider_error(
                response.status,
                &response.headers,
                &response.body,
            ));
        }
        let uploaded: UploadedFileWire =
            serde_json::from_slice(&response.body).map_err(|error| {
                unknown_response(
                    "input upload",
                    None,
                    format!("provider returned success but no usable file reference: {error}"),
                )
            })?;
        if uploaded
            .purpose
            .as_deref()
            .is_some_and(|purpose| purpose != "batch")
            || !valid_path_id(&uploaded.id)
        {
            return Err(unknown_response(
                "input upload",
                None,
                "provider returned success with an invalid batch file reference".into(),
            ));
        }
        Ok(GlmBatchInputFileRef {
            scope: pinned_service.scope.clone(),
            endpoint_fingerprint: pinned_service.scope.endpoint_fingerprint.clone(),
            file_id: uploaded.id,
            model: input.model().to_owned(),
        })
    }

    /// Submit the uploaded JSONL as one 24-hour chat-completion job. A
    /// transport failure, 5xx, or malformed success response returns an
    /// unknown outcome; never replay this call automatically.
    pub async fn submit(
        &self,
        input_file: &GlmBatchInputFileRef,
        metadata: &GlmBatchMetadata,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchJob, GlmBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_input_file_ref(input_file)?;
        metadata.validate()?;
        let body = SubmitWire {
            input_file_id: &input_file.file_id,
            endpoint: CHAT_ENDPOINT,
            completion_window: "24h",
            metadata: (!metadata.values.is_empty()).then_some(&metadata.values),
        };
        let response = pinned_service
            .send_json("POST", "/batches", &body, request_options)
            .await
            .map_err(|error| mutation_failure("submit", None, error))?;
        if !(200..300).contains(&response.status) {
            if response.status >= 500 {
                return Err(unknown_response(
                    "submit",
                    None,
                    format!(
                        "provider returned HTTP {} after submission",
                        response.status
                    ),
                ));
            }
            return Err(provider_error(
                response.status,
                &response.headers,
                &response.body,
            ));
        }
        let job = pinned_service
            .decode_job(&response.body)
            .map_err(|error| unknown_response("submit", None, error.to_string()))?;
        if job.endpoint != CHAT_ENDPOINT
            || job.input_file_id != input_file.file_id
            || job.completion_window != "24h"
        {
            return Err(unknown_response(
                "submit",
                Some(&job.reference),
                "provider accepted the request but returned a different input, endpoint, or window"
                    .into(),
            ));
        }
        Ok(job)
    }

    /// Fetch current state once. Callers choose whether and when to query
    /// again; this client does not poll automatically.
    pub async fn get(
        &self,
        reference: &GlmBatchJobRef,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchJob, GlmBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_job_ref(reference)?;
        let path = format!("/batches/{}", encode_path_segment(&reference.batch_id));
        let response = pinned_service
            .send_empty("GET", &path, request_options)
            .await?;
        ensure_success(&response)?;
        let job = pinned_service.decode_job(&response.body)?;
        pinned_service.ensure_same_job(reference, &job.reference)?;
        Ok(job)
    }

    /// Fetch one cursor page of tasks. The caller supplies the returned
    /// `last_id` as `after` to request a later page.
    pub async fn list(
        &self,
        options: &GlmBatchListOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchPage, GlmBatchError> {
        let pinned_service = self.pin()?;
        if options.limit == Some(0) {
            return Err(invalid("list limit must be greater than zero"));
        }
        if options
            .after
            .as_deref()
            .is_some_and(|id| !valid_path_id(id))
        {
            return Err(invalid("list cursor must be a valid Batch ID"));
        }

        let mut request = pinned_service.request("GET", "/batches", None, request_options)?;
        let mut url = url::Url::parse(&request.url)
            .map_err(|_| invalid("could not construct GLM Batch list URL"))?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(after) = options.after.as_deref() {
                query.append_pair("after", after);
            }
            if let Some(limit) = options.limit {
                query.append_pair("limit", &limit.to_string());
            }
        }
        request.url = url.into();
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await?;
        ensure_success(&response)?;
        let native: Value = serde_json::from_slice(&response.body).map_err(|error| {
            invalid_response(
                format!("could not decode Batch list response: {error}"),
                Value::Null,
            )
        })?;
        let object = native.as_object().ok_or_else(|| {
            invalid_response("Batch list response must be an object", native.clone())
        })?;
        if object.get("object").and_then(Value::as_str) != Some("list") {
            return Err(invalid_response(
                "provider returned an object other than a Batch list",
                native,
            ));
        }
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_response("Batch list data must be an array", native.clone()))?;
        let has_more = object
            .get("has_more")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                invalid_response("Batch list has_more must be a boolean", native.clone())
            })?;
        let first_id = optional_list_id(object.get("first_id"), "first_id")
            .map_err(|message| invalid_response(message, native.clone()))?;
        let last_id = optional_list_id(object.get("last_id"), "last_id")
            .map_err(|message| invalid_response(message, native.clone()))?;
        let batches = data
            .iter()
            .map(|batch| {
                let body = serde_json::to_vec(batch).map_err(|error| {
                    invalid_response(
                        format!("could not decode listed Batch: {error}"),
                        (*batch).clone(),
                    )
                })?;
                pinned_service.decode_job(&body)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen_ids = BTreeSet::new();
        if batches
            .iter()
            .any(|batch| !seen_ids.insert(batch.reference.batch_id.as_str()))
        {
            return Err(invalid_response(
                "Batch list contains duplicate IDs",
                native,
            ));
        }
        if let Some(first) = batches.first() {
            if first_id.as_deref() != Some(first.reference.batch_id.as_str())
                || last_id.as_deref()
                    != batches
                        .last()
                        .map(|batch| batch.reference.batch_id.as_str())
            {
                return Err(invalid_response(
                    "Batch list first_id or last_id did not match its data",
                    native,
                ));
            }
        } else if first_id.is_some() || last_id.is_some() {
            return Err(invalid_response(
                "empty Batch list data cannot include first_id or last_id",
                native,
            ));
        }
        if has_more {
            let Some(last_id) = last_id.as_deref() else {
                return Err(invalid_response(
                    "Batch list has_more requires a last_id cursor",
                    native,
                ));
            };
            if batches.is_empty() || options.after.as_deref() == Some(last_id) {
                return Err(invalid_response(
                    "Batch list cursor did not advance",
                    native,
                ));
            }
        }
        Ok(GlmBatchPage {
            batches,
            first_id,
            last_id,
            has_more,
            native,
        })
    }

    /// Request cancellation once. Transport ambiguity returns an unknown
    /// outcome with the original job reference; no retry is made.
    pub async fn cancel(
        &self,
        reference: &GlmBatchJobRef,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchJob, GlmBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_job_ref(reference)?;
        let path = format!(
            "/batches/{}/cancel",
            encode_path_segment(&reference.batch_id)
        );
        let response = pinned_service
            .send_empty("POST", &path, request_options)
            .await
            .map_err(|error| mutation_failure("cancel", Some(reference), error))?;
        if !(200..300).contains(&response.status) {
            if response.status >= 500 {
                return Err(unknown_response(
                    "cancel",
                    Some(reference),
                    format!(
                        "provider returned HTTP {} during cancellation",
                        response.status
                    ),
                ));
            }
            return Err(provider_error(
                response.status,
                &response.headers,
                &response.body,
            ));
        }
        let job = pinned_service
            .decode_job(&response.body)
            .map_err(|error| unknown_response("cancel", Some(reference), error.to_string()))?;
        pinned_service
            .ensure_same_job(reference, &job.reference)
            .map_err(|error| unknown_response("cancel", Some(reference), error.to_string()))?;
        Ok(job)
    }

    /// Stream the provider's JSONL output or error file. The returned bytes
    /// are not buffered, parsed, retried, or persisted by this service.
    pub async fn stream_result(
        &self,
        reference: &GlmBatchResultRef,
        request_options: &crate::RequestOptions,
    ) -> Result<GlmBatchResultStream, GlmBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_job_ref(&reference.job)?;
        if reference.result_endpoint_fingerprint != pinned_service.scope.endpoint_fingerprint
            || !valid_path_id(&reference.file_id)
        {
            return Err(GlmBatchError::ScopeMismatch);
        }
        let path = format!("/files/{}/content", encode_path_segment(&reference.file_id));
        let response = HttpExecutor::new(pinned_service.http)
            .send(pinned_service.request("GET", &path, None, request_options)?)
            .await?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES)).await?;
            ensure_success(&response)?;
            return Err(invalid_response(
                "non-success result response was not rejected",
                Value::Null,
            ));
        }
        Ok(GlmBatchResultStream {
            body: response.body,
        })
    }

    fn decode_job(&self, body: &[u8]) -> Result<GlmBatchJob, GlmBatchError> {
        let native: Value = serde_json::from_slice(body).map_err(|error| {
            invalid_response(format!("response was not JSON: {error}"), Value::Null)
        })?;
        let wire: BatchWire = serde_json::from_value(native.clone()).map_err(|error| {
            invalid_response(
                format!("response did not contain a valid Batch object: {error}"),
                native.clone(),
            )
        })?;
        if !valid_path_id(&wire.id)
            || !valid_path_id(&wire.input_file_id)
            || !valid_identity(&wire.completion_window)
            || !valid_identity(&wire.status)
            || wire.endpoint != CHAT_ENDPOINT
        {
            return Err(invalid_response(
                "provider returned an invalid Batch identity, endpoint, status, or window",
                native,
            ));
        }
        for id in [
            wire.output_file_id.as_deref(),
            wire.error_file_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !valid_path_id(id) {
                return Err(invalid_response(
                    "provider returned an invalid Batch result file ID",
                    native,
                ));
            }
        }
        Ok(GlmBatchJob {
            reference: GlmBatchJobRef {
                scope: self.scope.clone(),
                endpoint_fingerprint: self.scope.endpoint_fingerprint.clone(),
                batch_id: wire.id,
            },
            endpoint: wire.endpoint,
            input_file_id: wire.input_file_id,
            completion_window: wire.completion_window,
            status: GlmBatchStatus::from(wire.status),
            output_file_id: wire.output_file_id,
            error_file_id: wire.error_file_id,
            request_counts: wire.request_counts,
            native,
        })
    }

    fn validate_input_file_ref(
        &self,
        reference: &GlmBatchInputFileRef,
    ) -> Result<(), GlmBatchError> {
        if reference.scope != self.scope
            || reference.endpoint_fingerprint != self.scope.endpoint_fingerprint
            || !valid_path_id(&reference.file_id)
            || !valid_identity(&reference.model)
        {
            return Err(GlmBatchError::ScopeMismatch);
        }
        Ok(())
    }

    fn validate_job_ref(&self, reference: &GlmBatchJobRef) -> Result<(), GlmBatchError> {
        if reference.scope != self.scope
            || reference.endpoint_fingerprint != self.scope.endpoint_fingerprint
            || !valid_path_id(&reference.batch_id)
        {
            return Err(GlmBatchError::ScopeMismatch);
        }
        Ok(())
    }

    fn ensure_same_job(
        &self,
        expected: &GlmBatchJobRef,
        actual: &GlmBatchJobRef,
    ) -> Result<(), GlmBatchError> {
        if expected != actual {
            return Err(invalid_response(
                "provider returned a different Batch job identity",
                Value::Null,
            ));
        }
        Ok(())
    }

    async fn send_json<T: Serialize>(
        &self,
        method: &str,
        path: &str,
        body: &T,
        request_options: &crate::RequestOptions,
    ) -> Result<crate::transport::HttpResponse, GlmBatchError> {
        let body = serde_json::to_vec(body)
            .map_err(|_| invalid("Batch request could not be encoded as JSON"))?;
        Ok(HttpExecutor::new(self.http)
            .execute_bounded(
                self.request(method, path, Some(Bytes::from(body)), request_options)?,
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await?)
    }

    async fn send_empty(
        &self,
        method: &str,
        path: &str,
        request_options: &crate::RequestOptions,
    ) -> Result<crate::transport::HttpResponse, GlmBatchError> {
        Ok(HttpExecutor::new(self.http)
            .execute_bounded(
                self.request(method, path, None, request_options)?,
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await?)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<HttpRequest, GlmBatchError> {
        if !path.starts_with('/') || path.contains('?') || path.contains('#') {
            return Err(invalid("invalid internal GLM Batch route"));
        }
        let mut headers = vec![self.authorization_header(request_options)?];
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            method: method.into(),
            url: format!(
                "{}{path}",
                self.scope
                    .region
                    .base_url()
                    .ok_or(GlmBatchError::UnsupportedRegion {
                        region: self.scope.region,
                    },)?
            ),
            headers,
            body: body.unwrap_or_default(),
            timeout: request_options.total_timeout,
        })
    }

    fn authorization_header(
        &self,
        request_options: &crate::RequestOptions,
    ) -> Result<(String, String), GlmBatchError> {
        Ok((
            "Authorization".into(),
            format!("Bearer {}", self.request_credential(request_options)?),
        ))
    }
}

/// Stream of raw JSONL result bytes.
pub struct GlmBatchResultStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl Stream for GlmBatchResultStream {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().body.as_mut().poll_next(cx)
    }
}

#[derive(Deserialize)]
struct UploadedFileWire {
    id: String,
    #[serde(default)]
    purpose: Option<String>,
}

#[derive(Serialize)]
struct SubmitWire<'a> {
    input_file_id: &'a str,
    endpoint: &'static str,
    completion_window: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a BTreeMap<String, String>>,
}

#[derive(Deserialize)]
struct BatchWire {
    id: String,
    endpoint: String,
    input_file_id: String,
    completion_window: String,
    status: String,
    #[serde(default)]
    output_file_id: Option<String>,
    #[serde(default)]
    error_file_id: Option<String>,
    #[serde(default)]
    request_counts: Option<GlmBatchRequestCounts>,
}

impl From<String> for GlmBatchStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "validating" => Self::Validating,
            "in_progress" => Self::InProgress,
            "finalizing" => Self::Finalizing,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelling" => Self::Cancelling,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(value),
        }
    }
}

fn multipart_boundary(jsonl: &[u8]) -> Result<String, GlmBatchError> {
    for suffix in 0..100 {
        let boundary = format!("----lingxi-glm-batch-boundary-{suffix}");
        if !jsonl
            .windows(boundary.len())
            .any(|window| window == boundary.as_bytes())
        {
            return Ok(boundary);
        }
    }
    Err(invalid(
        "could not choose a collision-free multipart boundary",
    ))
}

fn multipart_edges(boundary: &str) -> (Vec<u8>, Vec<u8>) {
    let prefix = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"batch-input.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    let suffix = format!("\r\n--{boundary}--\r\n").into_bytes();
    (prefix, suffix)
}

fn encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(byte).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_custom_id(value: &str) -> bool {
    valid_identity(value) && value.chars().count() <= 256
}

fn valid_path_id(value: &str) -> bool {
    valid_identity(value)
        && value.len() <= 512
        && !value.contains('/')
        && !value.contains('\\')
        && !value.contains('?')
        && !value.contains('#')
}

fn optional_list_id(value: Option<&Value>, field: &str) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if valid_path_id(id) => Ok(Some(id.clone())),
        _ => Err(format!(
            "Batch list {field} must be a valid ID when present"
        )),
    }
}

fn provider_error(status: u16, headers: &[(String, String)], bytes: &[u8]) -> GlmBatchError {
    let body: Value = serde_json::from_slice(bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()));
    let code = body
        .pointer("/error/code")
        .or_else(|| body.get("code"))
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string())
        });
    let request_id = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
        .or_else(|| {
            headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("request-id"))
        })
        .map(|(_, value)| value.clone())
        .or_else(|| {
            body.get("request_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    GlmBatchError::Provider {
        status,
        code,
        request_id,
        body: Box::new(body),
    }
}

fn ensure_success(response: &crate::transport::HttpResponse) -> Result<(), GlmBatchError> {
    if (200..300).contains(&response.status) {
        Ok(())
    } else {
        Err(provider_error(
            response.status,
            &response.headers,
            &response.body,
        ))
    }
}

fn invalid(message: impl Into<String>) -> GlmBatchError {
    GlmBatchError::InvalidInput(message.into())
}

fn invalid_response(message: impl Into<String>, native: Value) -> GlmBatchError {
    GlmBatchError::InvalidResponse {
        message: message.into(),
        native: Box::new(native),
    }
}

fn unknown(
    operation: &'static str,
    reference: Option<&GlmBatchJobRef>,
    source: LlmError,
) -> GlmBatchError {
    GlmBatchError::OutcomeUnknown {
        operation,
        reference: reference.cloned().map(Box::new),
        source: Box::new(source),
    }
}

fn unknown_response(
    operation: &'static str,
    reference: Option<&GlmBatchJobRef>,
    reason: String,
) -> GlmBatchError {
    GlmBatchError::OutcomeUnknownResponse {
        operation,
        reference: reference.cloned().map(Box::new),
        reason,
    }
}

fn mutation_failure(
    operation: &'static str,
    reference: Option<&GlmBatchJobRef>,
    error: GlmBatchError,
) -> GlmBatchError {
    match error {
        GlmBatchError::Llm(source) => GlmBatchError::OutcomeUnknown {
            operation,
            reference: reference.cloned().map(Box::new),
            source,
        },
        other => other,
    }
}
