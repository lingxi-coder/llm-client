//! Typed Batch inference for Alibaba Cloud Model Studio's Qwen service.
//!
//! This module uses the documented OpenAI-compatible Batch routes in Beijing
//! and Singapore. It deliberately exposes only text chat-completion and text
//! embedding inputs. Workspace identity is retained in references for scope
//! checks; the Batch-specific documentation does not define a workspace URL or
//! request header, so it does not alter HTTP routing.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpStreamRequest, Transport},
};
use bytes::Bytes;
use futures::{stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    pin::Pin,
    task::{Context, Poll},
};

const BEIJING_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const SINGAPORE_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
const MAX_INPUT_BYTES: usize = 500_000_000;
const MAX_LINE_BYTES: usize = 1_000_000;
const MAX_INPUT_LINES: usize = 50_000;
const MAX_CONTROL_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// The Model Studio regions whose Batch endpoints are documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenBatchRegion {
    Beijing,
    Singapore,
}

impl QwenBatchRegion {
    fn base_url(self) -> &'static str {
        match self {
            Self::Beijing => BEIJING_BASE_URL,
            Self::Singapore => SINGAPORE_BASE_URL,
        }
    }
}

/// Identity that binds every uploaded file, job, and result to one Qwen
/// connection and account context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    region: QwenBatchRegion,
    workspace_id: Option<String>,
}

impl QwenBatchScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenBatchRegion,
        workspace_id: Option<String>,
    ) -> Result<Self, QwenBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if profile_name.trim().is_empty() || account_scope.trim().is_empty() {
            return Err(invalid("profile name and account scope must be non-empty"));
        }
        if workspace_id
            .as_deref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(invalid("workspace ID must be non-empty when supplied"));
        }
        Ok(Self {
            provider_id: ProviderId::from("qwen"),
            profile_name,
            account_scope,
            region,
            workspace_id,
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

    pub fn region(&self) -> QwenBatchRegion {
        self.region
    }

    pub fn workspace_id(&self) -> Option<&str> {
        self.workspace_id.as_deref()
    }

    fn endpoint_fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(self.region.base_url())
    }

    fn validate(&self) -> Result<(), QwenBatchError> {
        if self.provider_id.as_str() != "qwen"
            || self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || self
                .workspace_id
                .as_deref()
                .is_some_and(|id| id.trim().is_empty())
        {
            return Err(invalid("Qwen Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// A documented Batch endpoint. Arbitrary request URLs are not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenBatchEndpoint {
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    #[serde(rename = "/v1/embeddings")]
    Embeddings,
}

impl QwenBatchEndpoint {
    pub fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Embeddings => "/v1/embeddings",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenBatchCompletionWindow(u16);

impl QwenBatchCompletionWindow {
    /// Create a completion window from 24 through 336 whole hours.
    pub fn from_hours(hours: u16) -> Result<Self, QwenBatchError> {
        if !(24..=336).contains(&hours) {
            return Err(invalid(
                "completion window must be between 24 and 336 hours",
            ));
        }
        Ok(Self(hours))
    }

    pub fn hours(self) -> u16 {
        self.0
    }

    fn wire_value(self) -> String {
        format!("{}h", self.0)
    }
}

/// The only task metadata currently represented by the typed API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenBatchMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
}

impl QwenBatchMetadata {
    fn validate(&self) -> Result<(), QwenBatchError> {
        if self
            .name
            .as_ref()
            .is_some_and(|value| value.chars().count() > 100)
        {
            return Err(invalid("Batch metadata name exceeds 100 characters"));
        }
        if self
            .description
            .as_ref()
            .is_some_and(|value| value.chars().count() > 200)
        {
            return Err(invalid("Batch metadata description exceeds 200 characters"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenBatchChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchChatMessage {
    pub role: QwenBatchChatRole,
    pub content: String,
}

/// A typed, text-only `/v1/chat/completions` body for one JSONL row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchChatRequest {
    pub model: String,
    pub messages: Vec<QwenBatchChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// For hybrid-thinking models, this belongs at the same level as `model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QwenBatchEmbeddingInput {
    Text(String),
    Texts(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QwenBatchEncodingFormat {
    Float,
    Base64,
}

/// A typed `/v1/embeddings` body for one JSONL row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchEmbeddingRequest {
    pub model: String,
    pub input: QwenBatchEmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<QwenBatchEncodingFormat>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QwenBatchRequestBody {
    Chat(QwenBatchChatRequest),
    Embeddings(QwenBatchEmbeddingRequest),
}

impl QwenBatchRequestBody {
    fn endpoint(&self) -> QwenBatchEndpoint {
        match self {
            Self::Chat(_) => QwenBatchEndpoint::ChatCompletions,
            Self::Embeddings(_) => QwenBatchEndpoint::Embeddings,
        }
    }

    fn model(&self) -> &str {
        match self {
            Self::Chat(request) => &request.model,
            Self::Embeddings(request) => &request.model,
        }
    }

    fn thinking_mode(&self) -> Option<Option<bool>> {
        match self {
            Self::Chat(request) => Some(request.enable_thinking),
            Self::Embeddings(_) => None,
        }
    }
}

/// One request with a caller-owned stable result-matching ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchLine {
    pub custom_id: String,
    pub body: QwenBatchRequestBody,
}

/// Validated JSONL input containing one supported endpoint and one model.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenBatchInput {
    endpoint: QwenBatchEndpoint,
    lines: Vec<QwenBatchLine>,
}

impl QwenBatchInput {
    pub fn new(lines: Vec<QwenBatchLine>) -> Result<Self, QwenBatchError> {
        if lines.is_empty() || lines.len() > MAX_INPUT_LINES {
            return Err(invalid(
                "Batch input must contain between 1 and 50,000 rows",
            ));
        }
        let endpoint = lines[0].body.endpoint();
        let mut ids = BTreeSet::new();
        let mut model: Option<&str> = None;
        let mut thinking_mode: Option<Option<bool>> = None;
        for line in &lines {
            if line.custom_id.is_empty()
                || line.custom_id.chars().count() > 256
                || line.custom_id.chars().any(char::is_control)
            {
                return Err(invalid("custom_id must be 1 to 256 non-control characters"));
            }
            if !ids.insert(line.custom_id.as_str()) {
                return Err(invalid("custom_id values must be unique"));
            }
            if line.body.endpoint() != endpoint {
                return Err(invalid("one Batch file cannot mix endpoint types"));
            }
            let next_model = line.body.model();
            if next_model.trim().is_empty() {
                return Err(invalid("each request body must contain a model"));
            }
            if model.is_some_and(|previous| previous != next_model) {
                return Err(invalid("one Batch file cannot mix models"));
            }
            model = Some(next_model);
            if let Some(next_thinking_mode) = line.body.thinking_mode() {
                if thinking_mode.is_some_and(|previous| previous != next_thinking_mode) {
                    return Err(invalid("one Batch file cannot mix thinking modes"));
                }
                thinking_mode = Some(next_thinking_mode);
            }
        }
        Ok(Self { endpoint, lines })
    }

    pub fn endpoint(&self) -> QwenBatchEndpoint {
        self.endpoint
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Encode the validated input using the provider's `custom_id`/`method`/
    /// `url`/`body` JSONL row contract.
    pub fn encode_jsonl(&self) -> Result<Bytes, QwenBatchError> {
        let mut encoded = Vec::new();
        for line in &self.lines {
            let row = WireBatchLine {
                custom_id: &line.custom_id,
                method: "POST",
                url: self.endpoint.path(),
                body: &line.body,
            };
            let bytes = serde_json::to_vec(&row)
                .map_err(|_| invalid("Batch row could not be encoded as JSON"))?;
            if bytes.len() > MAX_LINE_BYTES {
                return Err(invalid("a Batch JSONL row exceeds 1 MB"));
            }
            if encoded.len().saturating_add(bytes.len()).saturating_add(1) > MAX_INPUT_BYTES {
                return Err(invalid("Batch JSONL input exceeds 500 MB"));
            }
            encoded.extend_from_slice(&bytes);
            encoded.push(b'\n');
        }
        Ok(Bytes::from(encoded))
    }
}

#[derive(Serialize)]
struct WireBatchLine<'a> {
    custom_id: &'a str,
    method: &'static str,
    url: &'static str,
    body: &'a QwenBatchRequestBody,
}

/// Provider file reference bound to this connection, endpoint, account,
/// region, and optional workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchInputFileRef {
    scope: QwenBatchScope,
    endpoint_fingerprint: String,
    endpoint: QwenBatchEndpoint,
    file_id: String,
}

impl QwenBatchInputFileRef {
    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn scope(&self) -> &QwenBatchScope {
        &self.scope
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    pub fn endpoint(&self) -> QwenBatchEndpoint {
        self.endpoint
    }
}

/// Durable identity of a submitted Qwen Batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchJobRef {
    scope: QwenBatchScope,
    endpoint_fingerprint: String,
    job_id: String,
}

impl QwenBatchJobRef {
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    pub fn scope(&self) -> &QwenBatchScope {
        &self.scope
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenBatchStatus {
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

impl QwenBatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }
}

impl Serialize for QwenBatchStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text = match self {
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
        serializer.serialize_str(text)
    }
}

impl<'de> Deserialize<'de> for QwenBatchStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Ok(match text.as_str() {
            "validating" => Self::Validating,
            "in_progress" => Self::InProgress,
            "finalizing" => Self::Finalizing,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelling" => Self::Cancelling,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(text),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenBatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

/// Typed task state returned by upload, submit, query, or cancel operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenBatchSnapshot {
    pub reference: QwenBatchJobRef,
    pub endpoint: QwenBatchEndpoint,
    pub status: QwenBatchStatus,
    pub input_file_id: String,
    pub completion_window: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub request_counts: Option<QwenBatchRequestCounts>,
    pub errors: Option<Value>,
    pub metadata: Option<BTreeMap<String, Value>>,
    pub created_at: Option<i64>,
    pub in_progress_at: Option<i64>,
    pub expires_at: Option<i64>,
    pub finalizing_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub failed_at: Option<i64>,
    pub expired_at: Option<i64>,
    pub cancelling_at: Option<i64>,
    pub cancelled_at: Option<i64>,
}

/// One cursor page returned by the documented Model Studio Batch list route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenBatchPage {
    pub batches: Vec<QwenBatchSnapshot>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
    pub native: Value,
}

/// Filters and cursor for one explicit Batch task-list request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenBatchListOptions {
    after: Option<String>,
    limit: Option<u16>,
    task_name: Option<String>,
    input_file_ids: Vec<String>,
    statuses: Vec<String>,
    create_after: Option<String>,
    create_before: Option<String>,
}

impl QwenBatchListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn after(mut self, batch_id: impl Into<String>) -> Self {
        self.after = Some(batch_id.into());
        self
    }

    pub fn limit(mut self, limit: u16) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn task_name(mut self, partial_name: impl Into<String>) -> Self {
        self.task_name = Some(partial_name.into());
        self
    }

    pub fn input_file_ids(mut self, file_ids: Vec<String>) -> Self {
        self.input_file_ids = file_ids;
        self
    }

    pub fn statuses(mut self, statuses: Vec<String>) -> Self {
        self.statuses = statuses;
        self
    }

    pub fn create_after(mut self, timestamp: impl Into<String>) -> Self {
        self.create_after = Some(timestamp.into());
        self
    }

    pub fn create_before(mut self, timestamp: impl Into<String>) -> Self {
        self.create_before = Some(timestamp.into());
        self
    }

    fn validate(&self) -> Result<(), QwenBatchError> {
        if self.after.as_deref().is_some_and(|id| !valid_job_id(id)) {
            return Err(invalid("list cursor must be a valid Batch ID"));
        }
        if self.limit.is_some_and(|limit| !(1..=100).contains(&limit)) {
            return Err(invalid("list limit must be between 1 and 100"));
        }
        if self
            .task_name
            .as_deref()
            .is_some_and(|name| name.is_empty() || name.chars().any(char::is_control))
        {
            return Err(invalid(
                "task name filter must be non-empty and contain no control characters",
            ));
        }
        if self.input_file_ids.len() > 20
            || self
                .input_file_ids
                .iter()
                .any(|id| !valid_input_file_id(id))
        {
            return Err(invalid(
                "list accepts at most 20 valid Batch input file IDs",
            ));
        }
        if self.statuses.iter().any(|status| {
            !matches!(
                status.as_str(),
                "validating"
                    | "failed"
                    | "in_progress"
                    | "finalizing"
                    | "completed"
                    | "expired"
                    | "cancelling"
                    | "cancelled"
            )
        }) {
            return Err(invalid("list contains an unsupported Batch status filter"));
        }
        for timestamp in [self.create_after.as_deref(), self.create_before.as_deref()]
            .into_iter()
            .flatten()
        {
            if timestamp.len() != 14 || !timestamp.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid("creation time filters must use yyyyMMddHHmmss"));
            }
        }
        if self
            .create_after
            .as_ref()
            .zip(self.create_before.as_ref())
            .is_some_and(|(after, before)| after > before)
        {
            return Err(invalid("create_after must not be later than create_before"));
        }
        Ok(())
    }
}

impl QwenBatchSnapshot {
    /// The documented success output file ID can be turned into a result reference.
    pub fn output_ref(&self) -> Option<QwenBatchResultRef> {
        let file_id = self.output_file_id.as_ref()?;
        if !file_id.starts_with("file-batch_output-") || !valid_file_id(file_id) {
            return None;
        }
        Some(self.result_ref(file_id))
    }

    /// The documented failed-request file ID can be turned into a result reference.
    pub fn error_ref(&self) -> Option<QwenBatchResultRef> {
        let file_id = self.error_file_id.as_ref()?;
        if !file_id.starts_with("file-batch_error-") || !valid_file_id(file_id) {
            return None;
        }
        Some(self.result_ref(file_id))
    }

    fn result_ref(&self, file_id: &str) -> QwenBatchResultRef {
        QwenBatchResultRef {
            job: self.reference.clone(),
            result_endpoint_fingerprint: self.reference.endpoint_fingerprint.clone(),
            file_id: file_id.to_owned(),
        }
    }
}

/// Scoped reference to a successful output or failed-request details file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenBatchResultRef {
    job: QwenBatchJobRef,
    result_endpoint_fingerprint: String,
    file_id: String,
}

impl QwenBatchResultRef {
    pub fn job(&self) -> &QwenBatchJobRef {
        &self.job
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenBatchSubmitOptions {
    pub completion_window: QwenBatchCompletionWindow,
    pub metadata: QwenBatchMetadata,
}

impl QwenBatchSubmitOptions {
    pub fn new(completion_window: QwenBatchCompletionWindow) -> Self {
        Self {
            completion_window,
            metadata: QwenBatchMetadata::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QwenBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Qwen Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid Qwen Batch response: {0}")]
    InvalidResponse(String),
    #[error(
        "outcome of Qwen Batch {operation} is unknown: expected task {expected_id}, provider returned {actual_id}"
    )]
    OutcomeUnknownResponse {
        operation: &'static str,
        expected_id: String,
        actual_id: String,
    },
}

/// A provider-specific Batch facade. The caller supplies the account's
/// region-appropriate API key and stable account/profile/workspace identities.
pub struct QwenBatchService<'a> {
    http: &'a dyn Transport,
    credential: Secret<String>,
    scope: QwenBatchScope,
}

impl<'a> QwenBatchService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        credential: Secret<String>,
        scope: QwenBatchScope,
    ) -> Result<Self, QwenBatchError> {
        scope.validate()?;
        if credential.expose_secret().trim().is_empty() {
            return Err(invalid("Qwen API key must be non-empty"));
        }
        Ok(Self {
            http,
            credential,
            scope,
        })
    }

    pub fn scope(&self) -> &QwenBatchScope {
        &self.scope
    }

    pub async fn upload_input(
        &self,
        input: &QwenBatchInput,
    ) -> Result<QwenBatchInputFileRef, QwenBatchError> {
        let jsonl = input.encode_jsonl()?;
        let boundary = multipart_boundary(&jsonl)?;
        let (prefix, suffix) = multipart_edges(&boundary);
        let content_length = prefix
            .len()
            .checked_add(jsonl.len())
            .and_then(|len| len.checked_add(suffix.len()))
            .and_then(|len| u64::try_from(len).ok())
            .ok_or_else(|| invalid("multipart upload length overflowed"))?;
        let mut chunks = vec![Bytes::from(prefix)];
        for start in (0..jsonl.len()).step_by(64 * 1024) {
            chunks.push(jsonl.slice(start..(start + 64 * 1024).min(jsonl.len())));
        }
        chunks.push(Bytes::from(suffix));
        let request = HttpStreamRequest {
            method: "POST".into(),
            url: format!("{}/files", self.scope.region.base_url()),
            headers: vec![
                self.authorization_header(),
                (
                    "Content-Type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: futures::stream::iter(chunks.into_iter().map(Ok)).boxed(),
            content_length,
            timeout: None,
        };
        let response = HttpExecutor::new(self.http).send_stream(request).await?;
        let response =
            HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES)).await?;
        ensure_success(response.status)?;
        let uploaded: UploadedFileWire = decode_json(&response.body)?;
        if uploaded.purpose != "batch" || uploaded.status != "processed" {
            return Err(QwenBatchError::InvalidResponse(
                "uploaded file was not accepted as a processed batch input".into(),
            ));
        }
        if !valid_input_file_id(&uploaded.id) {
            return Err(QwenBatchError::InvalidResponse(
                "provider returned an invalid Batch input file ID".into(),
            ));
        }
        Ok(QwenBatchInputFileRef {
            scope: self.scope.clone(),
            endpoint_fingerprint: self.scope.endpoint_fingerprint(),
            endpoint: input.endpoint,
            file_id: uploaded.id,
        })
    }

    pub async fn submit(
        &self,
        input_file: &QwenBatchInputFileRef,
        options: &QwenBatchSubmitOptions,
    ) -> Result<QwenBatchSnapshot, QwenBatchError> {
        self.validate_input_file_ref(input_file)?;
        options.metadata.validate()?;
        let metadata = metadata_wire(&options.metadata);
        let body = SubmitWire {
            input_file_id: &input_file.file_id,
            endpoint: input_file.endpoint,
            completion_window: options.completion_window.wire_value(),
            metadata,
        };
        let response = self.send_json("POST", "/batches", &body).await?;
        let snapshot = self.decode_snapshot(&response)?;
        if snapshot.endpoint != input_file.endpoint || snapshot.input_file_id != input_file.file_id
        {
            return Err(QwenBatchError::InvalidResponse(
                "created Batch task does not match the submitted input".into(),
            ));
        }
        if snapshot.completion_window != options.completion_window.wire_value() {
            return Err(QwenBatchError::InvalidResponse(
                "created Batch task changed its completion window".into(),
            ));
        }
        Ok(snapshot)
    }

    /// Fetch one task-list page. Use `last_id` as `after` for the next
    /// explicit request when `has_more` is true.
    pub async fn list(
        &self,
        options: &QwenBatchListOptions,
    ) -> Result<QwenBatchPage, QwenBatchError> {
        options.validate()?;
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        if let Some(after) = options.after.as_deref() {
            query.append_pair("after", after);
        }
        if let Some(limit) = options.limit {
            query.append_pair("limit", &limit.to_string());
        }
        if let Some(task_name) = options.task_name.as_deref() {
            query.append_pair("ds_name", task_name);
        }
        if !options.input_file_ids.is_empty() {
            query.append_pair("input_file_ids", &options.input_file_ids.join(","));
        }
        if !options.statuses.is_empty() {
            query.append_pair("status", &options.statuses.join(","));
        }
        if let Some(create_after) = options.create_after.as_deref() {
            query.append_pair("create_after", create_after);
        }
        if let Some(create_before) = options.create_before.as_deref() {
            query.append_pair("create_before", create_before);
        }
        let query = query.finish();
        let path = if query.is_empty() {
            "/batches".to_owned()
        } else {
            format!("/batches?{query}")
        };
        let response = self.send_empty("GET", &path).await?;
        let native: Value = decode_json(&response)?;
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("Batch list response must be an object"))?;
        if object.get("object").and_then(Value::as_str) != Some("list") {
            return Err(invalid_response(
                "provider returned an object other than a Batch list",
            ));
        }
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_response("Batch list data must be an array"))?;
        let has_more = object
            .get("has_more")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid_response("Batch list has_more must be a boolean"))?;
        let first_id = optional_list_job_id(object.get("first_id"), "first_id")?;
        let last_id = optional_list_job_id(object.get("last_id"), "last_id")?;
        let batches = data
            .iter()
            .map(|batch| {
                let body = serde_json::to_vec(batch).map_err(|error| {
                    QwenBatchError::InvalidResponse(format!(
                        "could not decode listed Batch: {error}"
                    ))
                })?;
                self.decode_snapshot(&body)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen_ids = BTreeSet::new();
        if batches
            .iter()
            .any(|batch| !seen_ids.insert(batch.reference.job_id.as_str()))
        {
            return Err(invalid_response("Batch list contains duplicate IDs"));
        }
        if let Some(first) = batches.first() {
            if first_id.as_deref() != Some(first.reference.job_id.as_str())
                || last_id.as_deref() != batches.last().map(|batch| batch.reference.job_id.as_str())
            {
                return Err(invalid_response(
                    "Batch list first_id or last_id did not match its data",
                ));
            }
        } else if first_id.is_some() || last_id.is_some() {
            return Err(invalid_response(
                "empty Batch list data cannot include first_id or last_id",
            ));
        }
        if has_more {
            let Some(last_id) = last_id.as_deref() else {
                return Err(invalid_response(
                    "Batch list has_more requires a last_id cursor",
                ));
            };
            if batches.is_empty() || options.after.as_deref() == Some(last_id) {
                return Err(invalid_response("Batch list cursor did not advance"));
            }
        }
        Ok(QwenBatchPage {
            batches,
            first_id,
            last_id,
            has_more,
            native,
        })
    }

    pub async fn query(
        &self,
        reference: &QwenBatchJobRef,
    ) -> Result<QwenBatchSnapshot, QwenBatchError> {
        self.validate_job_ref(reference)?;
        let path = format!("/batches/{}", reference.job_id);
        let response = self.send_empty("GET", &path).await?;
        let snapshot = self.decode_snapshot(&response)?;
        if snapshot.reference.job_id != reference.job_id {
            return Err(invalid_response(
                "query response task ID did not match the requested task",
            ));
        }
        Ok(snapshot)
    }

    pub async fn cancel(
        &self,
        reference: &QwenBatchJobRef,
    ) -> Result<QwenBatchSnapshot, QwenBatchError> {
        self.validate_job_ref(reference)?;
        let path = format!("/batches/{}/cancel", reference.job_id);
        let response = self.send_empty("POST", &path).await?;
        let snapshot = self.decode_snapshot(&response)?;
        if snapshot.reference.job_id != reference.job_id {
            return Err(QwenBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                expected_id: reference.job_id.clone(),
                actual_id: snapshot.reference.job_id,
            });
        }
        Ok(snapshot)
    }

    /// Start a streamed download of a completed success output or error file.
    pub async fn stream_result(
        &self,
        reference: &QwenBatchResultRef,
    ) -> Result<QwenBatchResultStream, QwenBatchError> {
        self.validate_job_ref(&reference.job)?;
        if reference.result_endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || !(valid_output_file_id(&reference.file_id)
                || valid_error_file_id(&reference.file_id))
        {
            return Err(scope_mismatch());
        }
        let path = format!("/files/{}/content", reference.file_id);
        let response = HttpExecutor::new(self.http)
            .send(self.request("GET", &path, None)?)
            .await?;
        ensure_success(response.status)?;
        Ok(QwenBatchResultStream {
            body: response.body,
        })
    }

    fn decode_snapshot(&self, body: &[u8]) -> Result<QwenBatchSnapshot, QwenBatchError> {
        let wire: TaskWire = decode_json(body)?;
        if !valid_job_id(&wire.id) || !valid_input_file_id(&wire.input_file_id) {
            return Err(QwenBatchError::InvalidResponse(
                "provider returned an invalid Batch task or input file ID".into(),
            ));
        }
        let reference = QwenBatchJobRef {
            scope: self.scope.clone(),
            endpoint_fingerprint: self.scope.endpoint_fingerprint(),
            job_id: wire.id,
        };
        Ok(QwenBatchSnapshot {
            reference,
            endpoint: wire.endpoint,
            status: QwenBatchStatus::from(wire.status),
            input_file_id: wire.input_file_id,
            completion_window: wire.completion_window,
            output_file_id: wire.output_file_id,
            error_file_id: wire.error_file_id,
            request_counts: wire.request_counts,
            errors: wire.errors,
            metadata: wire.metadata,
            created_at: wire.created_at,
            in_progress_at: wire.in_progress_at,
            expires_at: wire.expires_at,
            finalizing_at: wire.finalizing_at,
            completed_at: wire.completed_at,
            failed_at: wire.failed_at,
            expired_at: wire.expired_at,
            cancelling_at: wire.cancelling_at,
            cancelled_at: wire.cancelled_at,
        })
    }

    async fn send_json<T: Serialize>(
        &self,
        method: &str,
        path: &str,
        body: &T,
    ) -> Result<Bytes, QwenBatchError> {
        let body = serde_json::to_vec(body)
            .map_err(|_| invalid("Batch request could not be encoded as JSON"))?;
        let response = HttpExecutor::new(self.http)
            .execute(self.request(method, path, Some(Bytes::from(body)))?)
            .await?;
        ensure_success(response.status)?;
        Ok(response.body)
    }

    async fn send_empty(&self, method: &str, path: &str) -> Result<Bytes, QwenBatchError> {
        let response = HttpExecutor::new(self.http)
            .execute(self.request(method, path, None)?)
            .await?;
        ensure_success(response.status)?;
        Ok(response.body)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
    ) -> Result<HttpRequest, QwenBatchError> {
        if !path.starts_with('/') || path.contains('#') {
            return Err(invalid("invalid internal Batch route"));
        }
        let mut headers = vec![self.authorization_header()];
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            method: method.into(),
            url: format!("{}{path}", self.scope.region.base_url()),
            headers,
            body: body.unwrap_or_default(),
            timeout: None,
        })
    }

    fn authorization_header(&self) -> (String, String) {
        (
            "Authorization".into(),
            format!("Bearer {}", self.credential.expose_secret()),
        )
    }

    fn validate_input_file_ref(
        &self,
        reference: &QwenBatchInputFileRef,
    ) -> Result<(), QwenBatchError> {
        if reference.scope != self.scope
            || reference.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || self.scope.provider_id.as_str() != "qwen"
            || !valid_input_file_id(&reference.file_id)
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_job_ref(&self, reference: &QwenBatchJobRef) -> Result<(), QwenBatchError> {
        if reference.scope != self.scope
            || reference.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || self.scope.provider_id.as_str() != "qwen"
            || !valid_job_id(&reference.job_id)
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct UploadedFileWire {
    id: String,
    purpose: String,
    status: String,
}

#[derive(Serialize)]
struct SubmitWire<'a> {
    input_file_id: &'a str,
    endpoint: QwenBatchEndpoint,
    completion_window: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<MetadataWire<'a>>,
}

#[derive(Serialize)]
struct MetadataWire<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ds_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ds_description: Option<&'a str>,
}

fn metadata_wire(metadata: &QwenBatchMetadata) -> Option<MetadataWire<'_>> {
    let wire = MetadataWire {
        ds_name: metadata.name.as_deref(),
        ds_description: metadata.description.as_deref(),
    };
    (wire.ds_name.is_some() || wire.ds_description.is_some()).then_some(wire)
}

#[derive(Deserialize)]
struct TaskWire {
    id: String,
    endpoint: QwenBatchEndpoint,
    input_file_id: String,
    completion_window: String,
    status: String,
    #[serde(default)]
    output_file_id: Option<String>,
    #[serde(default)]
    error_file_id: Option<String>,
    #[serde(default)]
    request_counts: Option<QwenBatchRequestCounts>,
    #[serde(default)]
    errors: Option<Value>,
    #[serde(default)]
    metadata: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    in_progress_at: Option<i64>,
    #[serde(default)]
    expires_at: Option<i64>,
    #[serde(default)]
    finalizing_at: Option<i64>,
    #[serde(default)]
    completed_at: Option<i64>,
    #[serde(default)]
    failed_at: Option<i64>,
    #[serde(default)]
    expired_at: Option<i64>,
    #[serde(default)]
    cancelling_at: Option<i64>,
    #[serde(default)]
    cancelled_at: Option<i64>,
}

impl From<String> for QwenBatchStatus {
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

/// Successful output body stream. Result bytes remain in JSONL format and are
/// not automatically parsed, persisted, retried, or polled.
pub struct QwenBatchResultStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl Stream for QwenBatchResultStream {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().body.as_mut().poll_next(cx)
    }
}

fn multipart_boundary(jsonl: &[u8]) -> Result<String, QwenBatchError> {
    for suffix in 0..100 {
        let boundary = format!("----lingxi-qwen-batch-boundary-{suffix}");
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

fn decode_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, QwenBatchError> {
    serde_json::from_slice(body).map_err(|error| QwenBatchError::InvalidResponse(error.to_string()))
}

fn ensure_success(status: u16) -> Result<(), QwenBatchError> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    let message = format!("Model Studio Batch API returned HTTP {status}");
    let error = match status {
        401 => LlmError::Authentication { message },
        403 => LlmError::PermissionDenied { message },
        400 | 404 | 405 | 409 | 422 => LlmError::InvalidRequest { message },
        429 => LlmError::RateLimited {
            message,
            retry_after: None,
        },
        500..=599 => LlmError::ProviderInternal { message },
        _ => LlmError::Transport { message },
    };
    Err(error.into())
}

fn valid_job_id(value: &str) -> bool {
    value.starts_with("batch_") && valid_path_id(value)
}

fn optional_list_job_id(
    value: Option<&Value>,
    field: &str,
) -> Result<Option<String>, QwenBatchError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if valid_job_id(id) => Ok(Some(id.clone())),
        _ => Err(invalid_response(format!(
            "Batch list {field} must be a valid task ID when present"
        ))),
    }
}

fn valid_input_file_id(value: &str) -> bool {
    value.starts_with("file-batch-") && valid_path_id(value)
}

fn valid_output_file_id(value: &str) -> bool {
    value.starts_with("file-batch_output-") && valid_path_id(value)
}

fn valid_error_file_id(value: &str) -> bool {
    value.starts_with("file-batch_error-") && valid_path_id(value)
}

fn valid_file_id(value: &str) -> bool {
    value.starts_with("file-") && valid_path_id(value)
}

fn valid_path_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn scope_mismatch() -> QwenBatchError {
    LlmError::PermissionDenied {
        message: "Qwen Batch reference belongs to another account, profile, endpoint, region, or workspace".into(),
    }
    .into()
}

fn invalid(message: impl Into<String>) -> QwenBatchError {
    QwenBatchError::InvalidInput(message.into())
}

fn invalid_response(message: impl Into<String>) -> QwenBatchError {
    QwenBatchError::InvalidResponse(message.into())
}
