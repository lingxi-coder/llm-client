//! Typed access to Gemini's documented `generateContent` Batch API.
//!
//! The service supports inline requests and the JSONL File API workflow. Each
//! method makes one HTTP call (file upload uses its documented two-step
//! resumable protocol); there is no automatic polling, pagination, or retry.

#![doc = concat!(
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/gemini-batch.md")),
    "\n\n",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/gemini-batch.en.md"))
)]

mod embedding;
pub use embedding::*;

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{AuthStrategy, LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use futures::{stream, stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeSet,
    pin::Pin,
    task::{Context, Poll},
};
use url::Url;

const MAX_INLINE_REQUEST_BYTES: usize = 20_000_000;
const MAX_FILE_INPUT_BYTES: u64 = 2_000_000_000;
const MAX_CONTROL_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_FILE_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;
const GEMINI_FILE_ID_MAX_BYTES: usize = 40;
const GEMINI_JSONL_MEDIA_TYPE: &str = "application/jsonl";

/// Identity binding batch references to a Google profile, API endpoint, and
/// caller-defined account scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    api_base_url: String,
    endpoint_fingerprint: String,
}

impl GeminiBatchScope {
    /// Create a scope for a Gemini API root such as
    /// `https://generativelanguage.googleapis.com/v1beta`.
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        api_base_url: impl AsRef<str>,
    ) -> Result<Self, GeminiBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid_input(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        let api_base_url = normalize_api_base_url(api_base_url.as_ref())?;
        let endpoint_fingerprint = provider_file_endpoint_fingerprint(&api_base_url);
        Ok(Self {
            provider_id: ProviderId::from("google"),
            profile_name,
            account_scope,
            api_base_url,
            endpoint_fingerprint,
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

    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), GeminiBatchError> {
        let normalized = normalize_api_base_url(&self.api_base_url)?;
        if self.provider_id.as_str() != "google"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || normalized != self.api_base_url
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.api_base_url)
        {
            return Err(invalid_input("Gemini Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// A Gemini Files API resource bound to the same provider profile, endpoint,
/// and account scope as a batch service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiBatchFileRef {
    scope: GeminiBatchScope,
    file_id: String,
}

impl GeminiBatchFileRef {
    /// Bind a Files API resource name such as `files/abc-123` to a scope.
    pub fn from_resource_name(
        scope: &GeminiBatchScope,
        resource_name: impl AsRef<str>,
    ) -> Result<Self, GeminiBatchError> {
        scope.validate()?;
        let file_id = resource_name
            .as_ref()
            .strip_prefix("files/")
            .filter(|id| valid_gemini_file_id(id))
            .ok_or_else(|| invalid_input("Gemini file name must be a valid files/{id} resource"))?;
        Ok(Self {
            scope: scope.clone(),
            file_id: file_id.to_owned(),
        })
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn resource_name(&self) -> String {
        format!("files/{}", self.file_id)
    }

    pub fn scope(&self) -> &GeminiBatchScope {
        &self.scope
    }
}

/// A validated `GenerateContentRequest` value. The native request fields are
/// retained as JSON so documented Gemini request options remain available.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct GeminiBatchGenerateContentRequest(Value);

impl GeminiBatchGenerateContentRequest {
    /// Create a request with the required non-empty `contents` array.
    pub fn new(contents: Vec<Value>) -> Result<Self, GeminiBatchError> {
        Self::from_value(json!({ "contents": contents }))
    }

    /// Validate and retain a full native GenerateContent request object.
    pub fn from_value(value: Value) -> Result<Self, GeminiBatchError> {
        let Some(object) = value.as_object() else {
            return Err(invalid_input(
                "GenerateContent request must be a JSON object",
            ));
        };
        let Some(contents) = object.get("contents").and_then(Value::as_array) else {
            return Err(invalid_input(
                "GenerateContent request must contain a contents array",
            ));
        };
        if contents.is_empty() || contents.iter().any(|content| !content.is_object()) {
            return Err(invalid_input(
                "contents must contain at least one Content object",
            ));
        }
        Ok(Self(value))
    }

    /// Add one documented GenerateContent request field while preserving the
    /// rest of the native request. `contents` remains set through `new`.
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, GeminiBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name) || name == "contents" {
            return Err(invalid_input(
                "request parameter name is invalid or reserved",
            ));
        }
        let object = self.0.as_object_mut().expect("validated object");
        object.insert(name, value);
        Ok(self)
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

/// One inline GenerateContent request and optional correlation metadata.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeminiBatchRequest {
    request: GeminiBatchGenerateContentRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
}

impl GeminiBatchRequest {
    pub fn new(request: GeminiBatchGenerateContentRequest) -> Self {
        Self {
            request,
            metadata: None,
        }
    }

    /// Set an arbitrary metadata object, as supported by the REST schema.
    pub fn with_metadata(mut self, metadata: Value) -> Result<Self, GeminiBatchError> {
        if !metadata.is_object() {
            return Err(invalid_input("request metadata must be a JSON object"));
        }
        self.metadata = Some(metadata);
        Ok(self)
    }

    /// Set the guide's conventional `metadata.key` correlation value.
    pub fn with_key(self, key: impl Into<String>) -> Result<Self, GeminiBatchError> {
        let key = key.into();
        if !valid_identity(&key) {
            return Err(invalid_input("request metadata key must be non-empty"));
        }
        self.with_metadata(json!({ "key": key }))
    }

    pub fn request(&self) -> &GeminiBatchGenerateContentRequest {
        &self.request
    }

    pub fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
}

/// One row in a Gemini JSONL Batch input file. Unlike inline request metadata,
/// the JSONL correlation key is a top-level `key` field.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeminiBatchJsonlRequest {
    key: String,
    request: GeminiBatchGenerateContentRequest,
}

impl GeminiBatchJsonlRequest {
    pub fn new(
        key: impl Into<String>,
        request: GeminiBatchGenerateContentRequest,
    ) -> Result<Self, GeminiBatchError> {
        let key = key.into();
        if !valid_identity(&key) {
            return Err(invalid_input("JSONL request key must be non-empty"));
        }
        Ok(Self { key, request })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn request(&self) -> &GeminiBatchGenerateContentRequest {
        &self.request
    }
}

/// Validated JSONL rows ready for the Gemini Files API upload workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchJsonlInput {
    requests: Vec<GeminiBatchJsonlRequest>,
}

impl GeminiBatchJsonlInput {
    pub fn new(requests: Vec<GeminiBatchJsonlRequest>) -> Result<Self, GeminiBatchError> {
        if requests.is_empty() {
            return Err(invalid_input(
                "JSONL batch input must contain at least one request",
            ));
        }
        let mut keys = BTreeSet::new();
        if requests
            .iter()
            .any(|request| !keys.insert(request.key.as_str()))
        {
            return Err(invalid_input("JSONL request keys must be unique"));
        }
        Ok(Self { requests })
    }

    pub fn requests(&self) -> &[GeminiBatchJsonlRequest] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Encode one UTF-8 JSON object per line, with a trailing newline. The
    /// result respects Google's documented 2 GB input-file limit.
    pub fn to_bytes(&self) -> Result<Bytes, GeminiBatchError> {
        let size_bytes = self.encoded_size_bytes()?;
        let capacity = usize::try_from(size_bytes)
            .map_err(|_| invalid_input("JSONL input file cannot fit in memory on this target"))?;
        let mut bytes = Vec::with_capacity(capacity);
        for request in &self.requests {
            let encoded = encode_jsonl_request(request)?;
            bytes.extend_from_slice(&encoded);
            bytes.push(b'\n');
        }
        Ok(Bytes::from(bytes))
    }

    /// Calculate the exact UTF-8 JSONL size without retaining the full file.
    pub fn encoded_size_bytes(&self) -> Result<u64, GeminiBatchError> {
        let mut size_bytes = 0_u64;
        for request in &self.requests {
            let line_len = u64::try_from(encode_jsonl_request(request)?.len())
                .map_err(|_| invalid_input("JSONL line size overflowed"))?;
            size_bytes = size_bytes
                .checked_add(line_len)
                .and_then(|size| size.checked_add(1))
                .ok_or_else(|| invalid_input("JSONL input size overflowed"))?;
            if size_bytes > MAX_FILE_INPUT_BYTES {
                return Err(invalid_input("JSONL input file must not exceed 2 GB"));
            }
        }
        Ok(size_bytes)
    }

    /// Consume this input as a one-shot stream of complete JSONL lines.
    /// Serialization and the exact byte count are validated before upload.
    pub fn into_stream(
        self,
    ) -> Result<(u64, BoxStream<'static, Result<Bytes, LlmError>>), GeminiBatchError> {
        let size_bytes = self.encoded_size_bytes()?;
        let lines = self
            .requests
            .into_iter()
            .map(|request| -> Result<Bytes, LlmError> {
                let mut line =
                    encode_jsonl_request(&request).map_err(|error| LlmError::InvalidRequest {
                        message: error.to_string(),
                    })?;
                line.push(b'\n');
                Ok(Bytes::from(line))
            });
        Ok((size_bytes, stream::iter(lines).boxed()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum GeminiBatchInputSource {
    Inline(Vec<GeminiBatchRequest>),
    File(GeminiBatchFileRef),
}

/// Inline requests or a previously uploaded Gemini JSONL File resource.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchInput {
    source: GeminiBatchInputSource,
}

impl GeminiBatchInput {
    pub fn new(requests: Vec<GeminiBatchRequest>) -> Result<Self, GeminiBatchError> {
        if requests.is_empty() {
            return Err(invalid_input(
                "inline batch must contain at least one request",
            ));
        }
        Ok(Self {
            source: GeminiBatchInputSource::Inline(requests),
        })
    }

    /// Use a JSONL input file previously uploaded to Gemini Files API.
    pub fn from_file(file: GeminiBatchFileRef) -> Self {
        Self {
            source: GeminiBatchInputSource::File(file),
        }
    }

    pub fn requests(&self) -> &[GeminiBatchRequest] {
        match &self.source {
            GeminiBatchInputSource::Inline(requests) => requests,
            GeminiBatchInputSource::File(_) => &[],
        }
    }

    pub fn len(&self) -> usize {
        self.requests().len()
    }

    pub fn is_empty(&self) -> bool {
        matches!(&self.source, GeminiBatchInputSource::Inline(requests) if requests.is_empty())
    }

    pub fn file(&self) -> Option<&GeminiBatchFileRef> {
        match &self.source {
            GeminiBatchInputSource::File(file) => Some(file),
            GeminiBatchInputSource::Inline(_) => None,
        }
    }
}

/// Typed create body for `models.batchGenerateContent`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchCreateRequest {
    model: String,
    display_name: String,
    input: GeminiBatchInput,
}

impl GeminiBatchCreateRequest {
    pub fn new(
        model: impl Into<String>,
        display_name: impl Into<String>,
        input: GeminiBatchInput,
    ) -> Result<Self, GeminiBatchError> {
        let model = normalize_model_name(&model.into())?;
        let display_name = display_name.into();
        if !valid_identity(&display_name) {
            return Err(invalid_input("batch display name must be non-empty"));
        }
        Ok(Self {
            model,
            display_name,
            input,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn input(&self) -> &GeminiBatchInput {
        &self.input
    }

    fn encode(&self) -> Result<Bytes, GeminiBatchError> {
        let input_config = encode_batch_input_config(&self.input, "file_name");
        let value = json!({
            "batch": {
                "display_name": self.display_name,
                "input_config": input_config
            }
        });
        let bytes = serde_json::to_vec(&value)
            .map_err(|_| invalid_input("batch request could not be encoded"))?;
        if matches!(&self.input.source, GeminiBatchInputSource::Inline(_))
            && bytes.len() >= MAX_INLINE_REQUEST_BYTES
        {
            return Err(invalid_input(
                "inline batch request must be smaller than 20 MB",
            ));
        }
        Ok(Bytes::from(bytes))
    }
}

fn encode_batch_input_config(input: &GeminiBatchInput, file_name_field: &str) -> Value {
    match &input.source {
        GeminiBatchInputSource::Inline(requests) => {
            let mut request_list = Vec::with_capacity(requests.len());
            for request in requests {
                let mut item = Map::new();
                item.insert("request".into(), request.request.0.clone());
                if let Some(metadata) = &request.metadata {
                    item.insert("metadata".into(), metadata.clone());
                }
                request_list.push(Value::Object(item));
            }
            json!({ "requests": { "requests": request_list } })
        }
        GeminiBatchInputSource::File(file) => {
            let mut input_config = Map::new();
            input_config.insert(file_name_field.into(), json!(file.resource_name()));
            Value::Object(input_config)
        }
    }
}

/// A field from the documented GenerateContentBatch resource that can be
/// selected in `updateMask`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiBatchUpdateField {
    Model,
    DisplayName,
    InputConfig,
    Priority,
}

impl GeminiBatchUpdateField {
    fn wire_name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::DisplayName => "displayName",
            Self::InputConfig => "inputConfig",
            Self::Priority => "priority",
        }
    }
}

/// Full documented `GenerateContentBatch` resource body for a PATCH update.
/// Google requires model, displayName, and inputConfig; priority is optional.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchUpdateRequest {
    batch: GeminiBatchCreateRequest,
    priority: Option<i64>,
    update_mask: Option<Vec<GeminiBatchUpdateField>>,
}

impl GeminiBatchUpdateRequest {
    pub fn new(
        model: impl Into<String>,
        display_name: impl Into<String>,
        input: GeminiBatchInput,
    ) -> Result<Self, GeminiBatchError> {
        Ok(Self {
            batch: GeminiBatchCreateRequest::new(model, display_name, input)?,
            priority: None,
            update_mask: None,
        })
    }

    /// Set the optional int64 priority. Google allows negative values.
    pub fn with_priority(mut self, priority: i64) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Select resource fields using Google's comma-separated FieldMask names.
    pub fn with_update_mask(
        mut self,
        fields: impl IntoIterator<Item = GeminiBatchUpdateField>,
    ) -> Result<Self, GeminiBatchError> {
        let fields = fields.into_iter().collect::<Vec<_>>();
        if fields.is_empty() {
            return Err(invalid_input("updateMask must contain at least one field"));
        }
        let unique = fields
            .iter()
            .map(|field| field.wire_name())
            .collect::<BTreeSet<_>>();
        if unique.len() != fields.len() {
            return Err(invalid_input("updateMask fields must be unique"));
        }
        if fields.contains(&GeminiBatchUpdateField::Priority) && self.priority.is_none() {
            return Err(invalid_input(
                "updateMask priority requires an explicit priority value",
            ));
        }
        if self.priority.is_some() && !fields.contains(&GeminiBatchUpdateField::Priority) {
            return Err(invalid_input(
                "an explicit updateMask must include priority when a priority value is set",
            ));
        }
        self.update_mask = Some(fields);
        Ok(self)
    }

    pub fn model(&self) -> &str {
        self.batch.model()
    }

    pub fn display_name(&self) -> &str {
        self.batch.display_name()
    }

    pub fn priority(&self) -> Option<i64> {
        self.priority
    }

    fn encode(&self) -> Result<Bytes, GeminiBatchError> {
        self.validate()?;
        let mut resource = Map::new();
        resource.insert(
            "model".into(),
            json!(format!("models/{}", self.batch.model)),
        );
        resource.insert("displayName".into(), json!(self.batch.display_name));
        resource.insert(
            "inputConfig".into(),
            encode_batch_input_config(&self.batch.input, "fileName"),
        );
        if let Some(priority) = self.priority {
            resource.insert("priority".into(), json!(priority.to_string()));
        }
        let bytes = serde_json::to_vec(&Value::Object(resource))
            .map_err(|_| invalid_input("batch update request could not be encoded"))?;
        if matches!(&self.batch.input.source, GeminiBatchInputSource::Inline(_))
            && bytes.len() >= MAX_INLINE_REQUEST_BYTES
        {
            return Err(invalid_input(
                "inline batch update request must be smaller than 20 MB",
            ));
        }
        Ok(Bytes::from(bytes))
    }

    fn validate(&self) -> Result<(), GeminiBatchError> {
        if let Some(fields) = self.update_mask.as_ref() {
            let includes_priority = fields.contains(&GeminiBatchUpdateField::Priority);
            if includes_priority && self.priority.is_none() {
                return Err(invalid_input(
                    "updateMask priority requires an explicit priority value",
                ));
            }
            if self.priority.is_some() && !includes_priority {
                return Err(invalid_input(
                    "an explicit updateMask must include priority when a priority value is set",
                ));
            }
        }
        Ok(())
    }

    fn update_mask_query(&self) -> Option<String> {
        self.update_mask.as_ref().map(|fields| {
            fields
                .iter()
                .map(|field| field.wire_name())
                .collect::<Vec<_>>()
                .join(",")
        })
    }
}

/// Direct `GenerateContentBatch` resource returned by the update method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiBatchUpdateResult {
    pub reference: GeminiBatchRef,
    pub model: String,
    pub display_name: String,
    pub input_config: Value,
    pub priority: Option<i64>,
    pub native: Value,
}

/// Opaque identity for a Gemini batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum GeminiBatchRefKind {
    GenerateContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiBatchRef {
    kind: GeminiBatchRefKind,
    scope: GeminiBatchScope,
    batch_id: String,
}

impl GeminiBatchRef {
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn resource_name(&self) -> String {
        format!("batches/{}", self.batch_id)
    }

    pub fn scope(&self) -> &GeminiBatchScope {
        &self.scope
    }
}

/// Gemini's documented batch states. Unrecognized future values are retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiBatchState {
    Unspecified,
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Expired,
    Other(String),
}

impl GeminiBatchState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Expired
        )
    }

    fn parse(value: &str) -> Self {
        let suffix = value
            .strip_prefix("BATCH_STATE_")
            .or_else(|| value.strip_prefix("JOB_STATE_"))
            .unwrap_or(value);
        match suffix {
            "UNSPECIFIED" => Self::Unspecified,
            "PENDING" => Self::Pending,
            "RUNNING" => Self::Running,
            "SUCCEEDED" => Self::Succeeded,
            "FAILED" => Self::Failed,
            "CANCELLED" => Self::Cancelled,
            "EXPIRED" => Self::Expired,
            _ => Self::Other(value.into()),
        }
    }
}

/// A snapshot of the Google long-running operation returned by Batch API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiBatchSnapshot {
    pub reference: GeminiBatchRef,
    pub done: Option<bool>,
    pub state: Option<GeminiBatchState>,
    pub metadata: Option<Value>,
    pub error: Option<Value>,
    pub response: Option<Value>,
    /// File containing JSONL responses for a file-backed input batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_file: Option<GeminiBatchFileRef>,
    /// Complete operation JSON, retained for fields added by Google.
    pub native: Value,
}

/// One page from `batches.list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiBatchListPage {
    pub batches: Vec<GeminiBatchSnapshot>,
    pub next_page_token: Option<String>,
    pub native: Value,
}

/// One list request page. Pass the returned token to a later explicit call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeminiBatchListOptions {
    pub page_size: Option<u32>,
    pub page_token: Option<String>,
}

impl GeminiBatchListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_page_size(mut self, page_size: u32) -> Result<Self, GeminiBatchError> {
        if page_size == 0 {
            return Err(invalid_input("page size must be greater than zero"));
        }
        self.page_size = Some(page_size);
        Ok(self)
    }

    pub fn with_page_token(
        mut self,
        page_token: impl Into<String>,
    ) -> Result<Self, GeminiBatchError> {
        let page_token = page_token.into();
        if page_token.trim().is_empty() {
            return Err(invalid_input("page token must be non-empty"));
        }
        self.page_token = Some(page_token);
        Ok(self)
    }
}

/// One completed inline item, preserving Google response and error payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiBatchItemResult {
    pub metadata: Option<Value>,
    pub response: Option<Value>,
    pub error: Option<Value>,
    pub native: Value,
}

/// Results are available in the operation response after completion; this
/// shape distinguishes “not present yet / not inline” from an empty result set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiBatchResults {
    pub snapshot: GeminiBatchSnapshot,
    pub items: Option<Vec<GeminiBatchItemResult>>,
}

/// Byte stream for a file-backed Gemini Batch result. The body is the
/// provider's JSONL output and is not buffered or parsed by this type.
pub struct GeminiBatchResultStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl Stream for GeminiBatchResultStream {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().body.as_mut().poll_next(cx)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Gemini Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid Gemini Batch response: {0}")]
    InvalidResponse(String),
    #[error("Gemini Batch returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of Gemini Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        reference: Option<Box<GeminiBatchRef>>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Gemini Batch {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        reference: Option<Box<GeminiBatchRef>>,
        reason: String,
    },
    #[error("outcome of Gemini embedding Batch {operation} is unknown: {source}")]
    EmbeddingOutcomeUnknown {
        operation: &'static str,
        reference: Option<Box<GeminiEmbeddingBatchRef>>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Gemini embedding Batch {operation} is unknown: {reason}")]
    EmbeddingOutcomeUnknownResponse {
        operation: &'static str,
        reference: Option<Box<GeminiEmbeddingBatchRef>>,
        reason: String,
    },
}

/// Direct client for the documented Gemini Batch and Files REST methods.
/// Credentials are supplied per operation so the host retains refresh ownership.
/// Mutating calls are never replayed automatically.
#[derive(Clone)]
pub struct GeminiBatchService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: GeminiBatchScope,
}

impl<'a> GeminiBatchService<'a> {
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

    pub fn new(http: &'a dyn Transport, scope: GeminiBatchScope) -> Result<Self, GeminiBatchError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &GeminiBatchScope {
        &self.scope
    }

    /// Upload a JSONL file using Google's resumable Files API protocol. The
    /// request stream is consumed once and must match the declared byte count.
    pub async fn upload_input_stream(
        &self,
        filename: &str,
        size_bytes: u64,
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchFileRef, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        validate_jsonl_filename(filename)?;
        if size_bytes == 0 || size_bytes > MAX_FILE_INPUT_BYTES {
            return Err(invalid_input("JSONL input file must contain 1 to 2 GB"));
        }

        let upload_url = pinned_service.upload_url()?;
        let start_body = Bytes::from(
            serde_json::to_vec(&json!({ "file": { "display_name": filename } }))
                .map_err(|_| invalid_input("Gemini file upload metadata could not be encoded"))?,
        );
        let start_headers = vec![
            ("Content-Type".into(), "application/json".into()),
            ("x-goog-upload-protocol".into(), "resumable".into()),
            ("x-goog-upload-command".into(), "start".into()),
            (
                "x-goog-upload-header-content-length".into(),
                size_bytes.to_string(),
            ),
            (
                "x-goog-upload-header-content-type".into(),
                GEMINI_JSONL_MEDIA_TYPE.into(),
            ),
        ];
        let start_request = HttpRequest {
            http1_header_layout: None,
            method: "POST".into(),
            url: upload_url.into(),
            headers: start_headers,
            body: start_body,
            timeout: None,
        };
        let start = pinned_service
            .send_raw(start_request, credential)
            .await
            .map_err(|source| mutation_transport_error("upload_input", None, source))?;
        ensure_upload_success(&start, "starting the resumable session")?;
        let session_url = start
            .header("x-goog-upload-url")
            .filter(|value| same_origin(value, &pinned_service.scope.api_base_url))
            .ok_or_else(|| GeminiBatchError::OutcomeUnknownResponse {
                operation: "upload_input",
                reference: None,
                reason: "Gemini upload start did not return a same-origin upload URL".into(),
            })?;

        let request = crate::transport::HttpStreamRequest {
            http1_header_layout: None,
            method: "POST".into(),
            url: session_url.to_owned(),
            headers: vec![
                ("Content-Length".into(), size_bytes.to_string()),
                ("X-Goog-Upload-Offset".into(), "0".into()),
                ("X-Goog-Upload-Command".into(), "upload, finalize".into()),
            ],
            body: exact_length_stream(body, size_bytes),
            content_length: size_bytes,
            timeout: None,
        };
        let response = HttpExecutor::new(pinned_service.http)
            .execute_stream_bounded(request, MAX_FILE_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                mutation_transport_error(
                    "upload_input",
                    None,
                    BatchSendError::AfterDispatch(source),
                )
            })?;
        ensure_upload_success(&response, "finalizing the file upload")?;
        let native = decode_json(&response.body).map_err(|error| {
            GeminiBatchError::OutcomeUnknownResponse {
                operation: "upload_input",
                reference: None,
                reason: error.to_string(),
            }
        })?;
        let resource_name = native
            .get("file")
            .and_then(|file| file.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| GeminiBatchError::OutcomeUnknownResponse {
                operation: "upload_input",
                reference: None,
                reason: "Gemini upload response omitted file.name".into(),
            })?;
        GeminiBatchFileRef::from_resource_name(&pinned_service.scope, resource_name).map_err(
            |error| GeminiBatchError::OutcomeUnknownResponse {
                operation: "upload_input",
                reference: None,
                reason: error.to_string(),
            },
        )
    }

    /// Encode and upload validated JSONL rows through the Gemini Files API.
    pub async fn upload_input_jsonl(
        &self,
        filename: &str,
        input: GeminiBatchJsonlInput,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchFileRef, GeminiBatchError> {
        let pinned_service = self.pin()?;
        let (size_bytes, body) = input.into_stream()?;
        pinned_service
            .upload_input_stream(filename, size_bytes, body, credential)
            .await
    }

    /// Enqueue one inline or file-backed GenerateContent batch. Google documents creation as
    /// non-idempotent, so a transport or malformed success response is
    /// explicitly reported as an unknown outcome and never retried.
    pub async fn create(
        &self,
        request: &GeminiBatchCreateRequest,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchSnapshot, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        if let Some(file) = request.input.file() {
            pinned_service.validate_file_ref(file)?;
        }
        let body = request.encode()?;
        let path = format!(
            "/models/{}:batchGenerateContent",
            encode_path_segment(&request.model)
        );
        let response = pinned_service
            .send("POST", &path, Some(body), credential)
            .await
            .map_err(|source| mutation_transport_error("create", None, source))?;
        ensure_success(&response)?;
        pinned_service
            .decode_snapshot(&response.body)
            .map_err(|error| GeminiBatchError::OutcomeUnknownResponse {
                operation: "create",
                reference: None,
                reason: error.to_string(),
            })
    }

    /// Update one GenerateContent batch resource with Google's documented
    /// `updateGenerateContentBatch` PATCH method.
    pub async fn update_generate_content_batch(
        &self,
        reference: &GeminiBatchRef,
        request: &GeminiBatchUpdateRequest,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchUpdateResult, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_reference(reference)?;
        if let Some(file) = request.batch.input.file() {
            pinned_service.validate_file_ref(file)?;
        }
        let body = request.encode()?;
        let path = format!(
            "/batches/{}:updateGenerateContentBatch",
            encode_path_segment(&reference.batch_id)
        );
        let mut url = pinned_service.url(&path)?;
        if let Some(update_mask) = request.update_mask_query() {
            url.query_pairs_mut()
                .append_pair("updateMask", &update_mask);
        }
        let response = pinned_service
            .send_url("PATCH", url, Some(body), credential)
            .await
            .map_err(|source| {
                mutation_transport_error("update", Some(reference.clone()), source)
            })?;
        ensure_success(&response)?;
        let native = decode_json(&response.body).map_err(|error| {
            GeminiBatchError::OutcomeUnknownResponse {
                operation: "update",
                reference: Some(Box::new(reference.clone())),
                reason: error.to_string(),
            }
        })?;
        pinned_service
            .decode_update_result(reference, native)
            .map_err(|error| GeminiBatchError::OutcomeUnknownResponse {
                operation: "update",
                reference: Some(Box::new(reference.clone())),
                reason: error.to_string(),
            })
    }

    /// Fetch the latest state for one operation resource.
    pub async fn get(
        &self,
        reference: &GeminiBatchRef,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchSnapshot, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_reference(reference)?;
        let response = pinned_service
            .send(
                "GET",
                &format!("/batches/{}", encode_path_segment(&reference.batch_id)),
                None,
                credential,
            )
            .await?;
        ensure_success(&response)?;
        let snapshot = pinned_service.decode_snapshot(&response.body)?;
        pinned_service.ensure_same_batch(reference, &snapshot.reference)?;
        Ok(snapshot)
    }

    /// Fetch one provider-paginated page from `batches.list`.
    pub async fn list(
        &self,
        options: &GeminiBatchListOptions,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchListPage, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        let mut url = pinned_service.url("/batches")?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(page_size) = options.page_size {
                if page_size == 0 {
                    return Err(invalid_input("page size must be greater than zero"));
                }
                query.append_pair("pageSize", &page_size.to_string());
            }
            if let Some(page_token) = options.page_token.as_deref() {
                if page_token.trim().is_empty() {
                    return Err(invalid_input("page token must be non-empty"));
                }
                query.append_pair("pageToken", page_token);
            }
        }
        let response = pinned_service
            .send_url("GET", url, None, credential)
            .await?;
        ensure_success(&response)?;
        let native = decode_json(&response.body)?;
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("list response must be a JSON object"))?;
        let operations = match object.get("operations") {
            Some(Value::Array(operations)) => operations,
            Some(_) => return Err(invalid_response("operations must be an array")),
            None => {
                return Ok(GeminiBatchListPage {
                    batches: Vec::new(),
                    next_page_token: object
                        .get("nextPageToken")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    native,
                });
            }
        };
        let batches = operations
            .iter()
            .filter(|operation| !is_embedding_batch_operation(operation))
            .cloned()
            .map(|operation| pinned_service.decode_snapshot_value(operation))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GeminiBatchListPage {
            batches,
            next_page_token: object
                .get("nextPageToken")
                .and_then(Value::as_str)
                .map(str::to_owned),
            native,
        })
    }

    /// Request best-effort cancellation. A successful response means the
    /// cancellation request was accepted; call `get` separately to confirm
    /// whether the operation reached its cancelled terminal state.
    pub async fn cancel(
        &self,
        reference: &GeminiBatchRef,
        credential: &Secret<String>,
    ) -> Result<(), GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_reference(reference)?;
        let path = format!(
            "/batches/{}:cancel",
            encode_path_segment(&reference.batch_id)
        );
        let response = pinned_service
            .send("POST", &path, None, credential)
            .await
            .map_err(|source| {
                mutation_transport_error("cancel", Some(reference.clone()), source)
            })?;
        ensure_success(&response)?;
        let body = decode_json(&response.body).map_err(|error| {
            GeminiBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reference: Some(Box::new(reference.clone())),
                reason: error.to_string(),
            }
        })?;
        if body.as_object().is_none_or(|object| !object.is_empty()) {
            return Err(GeminiBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reference: Some(Box::new(reference.clone())),
                reason: "expected the documented empty JSON object".into(),
            });
        }
        Ok(())
    }

    /// Delete an operation record. Google documents this as removing client
    /// interest in the result; it does not cancel processing. Call `cancel`
    /// separately when cancellation is intended.
    pub async fn delete(
        &self,
        reference: &GeminiBatchRef,
        credential: &Secret<String>,
    ) -> Result<(), GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_reference(reference)?;
        let path = format!("/batches/{}", encode_path_segment(&reference.batch_id));
        let response = pinned_service
            .send("DELETE", &path, None, credential)
            .await
            .map_err(|source| {
                mutation_transport_error("delete", Some(reference.clone()), source)
            })?;
        ensure_success(&response)?;
        let body = decode_json(&response.body).map_err(|error| {
            GeminiBatchError::OutcomeUnknownResponse {
                operation: "delete",
                reference: Some(Box::new(reference.clone())),
                reason: error.to_string(),
            }
        })?;
        if body.as_object().is_none_or(|object| !object.is_empty()) {
            return Err(GeminiBatchError::OutcomeUnknownResponse {
                operation: "delete",
                reference: Some(Box::new(reference.clone())),
                reason: "expected the documented empty JSON object".into(),
            });
        }
        Ok(())
    }

    /// Read the current operation once and decode its inline per-item results
    /// when Google has returned them. This is a normal bounded JSON response;
    /// Gemini documents no separate REST results stream for inline batches.
    pub async fn results(
        &self,
        reference: &GeminiBatchRef,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchResults, GeminiBatchError> {
        let pinned_service = self.pin()?;
        let snapshot = pinned_service.get(reference, credential).await?;
        let items = snapshot
            .response
            .as_ref()
            .and_then(|response| response.get("output"))
            .and_then(|output| output.get("inlinedResponses"))
            .and_then(|inlined| inlined.get("inlinedResponses"))
            .map(parse_inline_results)
            .transpose()?;
        Ok(GeminiBatchResults { snapshot, items })
    }

    /// Stream the JSONL output file named by a completed file-backed batch.
    /// The caller owns buffering and line parsing; this method does not retry.
    pub async fn download_results(
        &self,
        file: &GeminiBatchFileRef,
        credential: &Secret<String>,
    ) -> Result<GeminiBatchResultStream, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_file_ref(file)?;
        let url = pinned_service.url(&format!(
            "/files/{}:download?alt=media",
            encode_path_segment(&file.file_id)
        ))?;
        let mut request = HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url: url.into(),
            headers: Vec::new(),
            body: Bytes::new(),
            timeout: None,
        };
        pinned_service
            .authenticate_request(&mut request, credential)
            .await?;
        let response = HttpExecutor::new(pinned_service.http).send(request).await?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES)).await?;
            ensure_success(&response)?;
            return Err(invalid_response(
                "non-success result response was not rejected",
            ));
        }
        Ok(GeminiBatchResultStream {
            body: response.body,
        })
    }

    fn decode_snapshot(&self, body: &[u8]) -> Result<GeminiBatchSnapshot, GeminiBatchError> {
        self.decode_snapshot_value(decode_json(body)?)
    }

    fn decode_snapshot_value(
        &self,
        native: Value,
    ) -> Result<GeminiBatchSnapshot, GeminiBatchError> {
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("operation must be a JSON object"))?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_response("operation is missing its resource name"))?;
        let batch_id = name
            .strip_prefix("batches/")
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| invalid_response("operation name is not a batches resource"))?
            .to_owned();
        let metadata = object.get("metadata").cloned();
        if metadata
            .as_ref()
            .and_then(|value| value.get("@type"))
            .and_then(Value::as_str)
            .is_some_and(is_embedding_batch_type)
        {
            return Err(invalid_response(
                "embedding Batch operation cannot be used as a generateContent batch",
            ));
        }
        let response = object.get("response").cloned();
        let output_file = response
            .as_ref()
            .and_then(|value| value.get("output"))
            .and_then(|value| value.get("responsesFile"))
            .map(|value| {
                let name = value.as_str().ok_or_else(|| {
                    invalid_response("response.output.responsesFile must be a file resource name")
                })?;
                GeminiBatchFileRef::from_resource_name(&self.scope, name)
            })
            .transpose()?;
        let state_value = metadata
            .as_ref()
            .and_then(|value| value.get("state"))
            .or_else(|| response.as_ref().and_then(|value| value.get("state")));
        let state = state_value
            .and_then(Value::as_str)
            .map(GeminiBatchState::parse);
        Ok(GeminiBatchSnapshot {
            reference: GeminiBatchRef {
                kind: GeminiBatchRefKind::GenerateContent,
                scope: self.scope.clone(),
                batch_id,
            },
            done: object.get("done").and_then(Value::as_bool),
            state,
            metadata,
            error: object.get("error").cloned(),
            response,
            output_file,
            native,
        })
    }

    fn decode_update_result(
        &self,
        reference: &GeminiBatchRef,
        native: Value,
    ) -> Result<GeminiBatchUpdateResult, GeminiBatchError> {
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("updated batch must be a JSON object"))?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_response("updated batch is missing its resource name"))?;
        if name != reference.resource_name() {
            return Err(invalid_response(
                "Google returned a different batch resource name after update",
            ));
        }
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| value.strip_prefix("models/").is_some_and(valid_resource_id))
            .ok_or_else(|| invalid_response("updated batch model must be models/{model}"))?
            .to_owned();
        let display_name = object
            .get("displayName")
            .and_then(Value::as_str)
            .filter(|value| valid_identity(value))
            .ok_or_else(|| invalid_response("updated batch displayName is invalid"))?
            .to_owned();
        let input_config = object
            .get("inputConfig")
            .filter(|value| value.is_object())
            .cloned()
            .ok_or_else(|| invalid_response("updated batch inputConfig must be an object"))?;
        let priority =
            match object.get("priority") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) => Some(value.parse::<i64>().map_err(|_| {
                    invalid_response("updated batch priority must be an int64 string")
                })?),
                Some(_) => {
                    return Err(invalid_response(
                        "updated batch priority must use the int64 string representation",
                    ));
                }
            };
        Ok(GeminiBatchUpdateResult {
            reference: reference.clone(),
            model,
            display_name,
            input_config,
            priority,
            native,
        })
    }

    fn validate_file_ref(&self, reference: &GeminiBatchFileRef) -> Result<(), GeminiBatchError> {
        if reference.scope != self.scope || !valid_gemini_file_id(&reference.file_id) {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_reference(&self, reference: &GeminiBatchRef) -> Result<(), GeminiBatchError> {
        if reference.kind != GeminiBatchRefKind::GenerateContent
            || reference.scope != self.scope
            || !valid_resource_id(&reference.batch_id)
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn ensure_same_batch(
        &self,
        expected: &GeminiBatchRef,
        actual: &GeminiBatchRef,
    ) -> Result<(), GeminiBatchError> {
        if expected != actual {
            return Err(invalid_response(
                "Google returned a different batch resource name",
            ));
        }
        Ok(())
    }

    async fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, BatchSendError> {
        let url = self.url(path).map_err(BatchSendError::BeforeDispatch)?;
        self.send_url(method, url, body, credential).await
    }

    async fn send_url(
        &self,
        method: &str,
        url: Url,
        body: Option<Bytes>,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, BatchSendError> {
        let mut headers = Vec::new();
        let body = body.unwrap_or_default();
        if !body.is_empty() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        self.send_raw(
            HttpRequest {
                http1_header_layout: None,
                method: method.into(),
                url: url.into(),
                headers,
                body,
                timeout: None,
            },
            credential,
        )
        .await
    }

    async fn send_raw(
        &self,
        mut request: HttpRequest,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, BatchSendError> {
        self.authenticate_request(&mut request, credential)
            .await
            .map_err(BatchSendError::BeforeDispatch)?;
        HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(BatchSendError::AfterDispatch)
    }

    async fn authenticate_request(
        &self,
        request: &mut HttpRequest,
        credential: &Secret<String>,
    ) -> Result<(), LlmError> {
        if let Some(binding) = &self.binding {
            let snapshot = binding.pin()?;
            let profile = snapshot
                .native_profile(binding.profile_name())
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "bound Gemini Batch profile is unavailable".into(),
                })?;
            if profile.auth != AuthStrategy::None {
                let authenticator = snapshot
                    .runtime
                    .authenticators
                    .get(&profile.auth)
                    .ok_or_else(|| LlmError::Authentication {
                        message: "bound Gemini Batch authenticator is not registered".into(),
                    })?;
                authenticator
                    .apply(request, profile, Some(credential))
                    .await?;
            }
        } else {
            request
                .headers
                .push(("x-goog-api-key".into(), credential.expose_secret().clone()));
        }
        Ok(())
    }

    fn url(&self, path: &str) -> Result<Url, LlmError> {
        Url::parse(&format!("{}{path}", self.scope.api_base_url)).map_err(|_| {
            LlmError::InvalidRequest {
                message: "Gemini Batch URL could not be constructed".into(),
            }
        })
    }

    fn upload_url(&self) -> Result<Url, GeminiBatchError> {
        let base = self
            .scope
            .api_base_url
            .strip_suffix("/v1beta")
            .unwrap_or(&self.scope.api_base_url)
            .trim_end_matches('/');
        Url::parse(&format!("{base}/upload/v1beta/files"))
            .map_err(|_| invalid_input("Gemini upload URL could not be constructed"))
    }
}

fn encode_jsonl_request(request: &GeminiBatchJsonlRequest) -> Result<Vec<u8>, GeminiBatchError> {
    let row = json!({
        "key": request.key,
        "request": request.request.as_value(),
    });
    serde_json::to_vec(&row).map_err(|_| invalid_input("JSONL request could not be encoded"))
}

fn parse_inline_results(value: &Value) -> Result<Vec<GeminiBatchItemResult>, GeminiBatchError> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid_response("inlinedResponses must be an array"))?;
    values
        .iter()
        .cloned()
        .map(|native| {
            let object = native
                .as_object()
                .ok_or_else(|| invalid_response("inline response must be a JSON object"))?;
            let response = object.get("response").cloned();
            let error = object.get("error").cloned();
            if response.is_some() == error.is_some() {
                return Err(invalid_response(
                    "inline response must contain exactly one of response or error",
                ));
            }
            Ok(GeminiBatchItemResult {
                metadata: object.get("metadata").cloned(),
                response,
                error,
                native,
            })
        })
        .collect()
}

fn ensure_success(response: &HttpResponse) -> Result<(), GeminiBatchError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let body = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    Err(GeminiBatchError::Provider {
        status: response.status,
        request_id: response
            .header("x-goog-request-id")
            .or_else(|| response.header("x-request-id"))
            .map(str::to_owned),
        body,
    })
}

fn ensure_upload_success(response: &HttpResponse, stage: &str) -> Result<(), GeminiBatchError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    if response.status >= 500 {
        return Err(GeminiBatchError::OutcomeUnknownResponse {
            operation: "upload_input",
            reference: None,
            reason: format!("Gemini returned HTTP {} while {stage}", response.status),
        });
    }
    ensure_success(response)
}

fn decode_json(bytes: &[u8]) -> Result<Value, GeminiBatchError> {
    serde_json::from_slice(bytes)
        .map_err(|error| invalid_response(format!("response JSON is invalid: {error}")))
}

fn exact_length_stream(
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    expected: u64,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::try_unfold((body, 0_u64), move |(mut body, received)| async move {
        match body.next().await {
            Some(Ok(bytes)) => {
                let next = received.checked_add(bytes.len() as u64).ok_or_else(|| {
                    LlmError::InvalidRequest {
                        message: "Gemini JSONL upload size overflowed".into(),
                    }
                })?;
                if next > expected {
                    return Err(LlmError::InvalidRequest {
                        message: "Gemini JSONL upload exceeded its declared byte count".into(),
                    });
                }
                Ok(Some((bytes, (body, next))))
            }
            Some(Err(error)) => Err(error),
            None if received == expected => Ok(None),
            None => Err(LlmError::InvalidRequest {
                message: "Gemini JSONL upload ended before its declared byte count".into(),
            }),
        }
    })
    .boxed()
}

fn same_origin(candidate: &str, base: &str) -> bool {
    let (Ok(candidate), Ok(base)) = (Url::parse(candidate), Url::parse(base)) else {
        return false;
    };
    candidate.scheme() == "https"
        && candidate.username().is_empty()
        && candidate.password().is_none()
        && candidate.origin().ascii_serialization() == base.origin().ascii_serialization()
}

fn validate_jsonl_filename(filename: &str) -> Result<(), GeminiBatchError> {
    if !filename.ends_with(".jsonl")
        || filename.chars().count() > 512
        || filename.trim().is_empty()
        || filename
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
    {
        return Err(invalid_input(
            "JSONL upload filename must be a basename ending in .jsonl and at most 512 characters",
        ));
    }
    Ok(())
}

fn valid_gemini_file_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= GEMINI_FILE_ID_MAX_BYTES
        && !value.starts_with('-')
        && !value.ends_with('-')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn normalize_api_base_url(value: &str) -> Result<String, GeminiBatchError> {
    let mut url = Url::parse(value).map_err(|_| invalid_input("Gemini API base URL is invalid"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_input(
            "Gemini API base URL must be an HTTPS URL without credentials, query, or fragment",
        ));
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&path);
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn normalize_model_name(value: &str) -> Result<String, GeminiBatchError> {
    let model = value.strip_prefix("models/").unwrap_or(value);
    if !valid_resource_id(model) {
        return Err(invalid_input("model must be a single Gemini model name"));
    }
    Ok(model.to_owned())
}

fn encode_path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_parameter_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn valid_resource_id(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value.len() <= 512
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~'))
}

fn is_embedding_batch_type(value: &str) -> bool {
    value.ends_with(".EmbedContentBatch") || value == "EmbedContentBatch"
}

fn is_generation_batch_type(value: &str) -> bool {
    value.ends_with(".GenerateContentBatch") || value == "GenerateContentBatch"
}

fn is_embedding_batch_operation(value: &Value) -> bool {
    value
        .get("metadata")
        .and_then(|metadata| metadata.get("@type"))
        .and_then(Value::as_str)
        .is_some_and(is_embedding_batch_type)
}

enum BatchSendError {
    BeforeDispatch(LlmError),
    AfterDispatch(LlmError),
}

impl From<BatchSendError> for GeminiBatchError {
    fn from(error: BatchSendError) -> Self {
        match error {
            BatchSendError::BeforeDispatch(source) | BatchSendError::AfterDispatch(source) => {
                Self::Llm(source)
            }
        }
    }
}

fn mutation_transport_error(
    operation: &'static str,
    reference: Option<GeminiBatchRef>,
    error: BatchSendError,
) -> GeminiBatchError {
    match error {
        BatchSendError::BeforeDispatch(source) => GeminiBatchError::Llm(source),
        BatchSendError::AfterDispatch(source) => GeminiBatchError::OutcomeUnknown {
            operation,
            reference: reference.map(Box::new),
            source,
        },
    }
}

fn embedding_mutation_transport_error(
    operation: &'static str,
    reference: Option<GeminiEmbeddingBatchRef>,
    error: BatchSendError,
) -> GeminiBatchError {
    match error {
        BatchSendError::BeforeDispatch(source) => GeminiBatchError::Llm(source),
        BatchSendError::AfterDispatch(source) => GeminiBatchError::EmbeddingOutcomeUnknown {
            operation,
            reference: reference.map(Box::new),
            source,
        },
    }
}

fn invalid_input(message: impl Into<String>) -> GeminiBatchError {
    GeminiBatchError::InvalidInput(message.into())
}

fn validate_credential(credential: &Secret<String>) -> Result<(), GeminiBatchError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || value.contains('\r') || value.contains('\n') {
        return Err(invalid_input(
            "Gemini API key must be non-empty and well-formed",
        ));
    }
    Ok(())
}

fn invalid_response(message: impl Into<String>) -> GeminiBatchError {
    GeminiBatchError::InvalidResponse(message.into())
}

fn scope_mismatch() -> GeminiBatchError {
    LlmError::PermissionDenied {
        message:
            "Gemini batch reference belongs to another provider, profile, endpoint, or account"
                .into(),
    }
    .into()
}
