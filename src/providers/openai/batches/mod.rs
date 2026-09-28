//! OpenAI Batch submission and scoped job lifecycle.
use crate::{
    client::RequestOptions,
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{LlmError, ProviderId, ProviderProfile, ServiceAuth, ServiceSetting},
    runtime::Deadline,
    runtime::{ClientSnapshot, ClientSource},
    transport::{HttpExecutor, HttpRequest},
};
use bytes::Bytes;
use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_INPUT: usize = 200_000_000;
const MAX_RESULT_LINE: usize = 16 * 1024 * 1024;
const BATCH_COMPLETION_WINDOW_SECONDS: i64 = 24 * 60 * 60;
const BATCH_CANCELLATION_WINDOW_SECONDS: i64 = 10 * 60;
const ATTACHMENT_EXPIRY_SAFETY_SECONDS: i64 = 5 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchApi {
    OpenAi,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRoute {
    pub api: BatchApi,
    /// Full `/v1/batches` collection URL.
    pub endpoint: String,
    /// Full `/v1/files` collection URL used to retrieve result JSONL.
    pub files_endpoint: String,
    pub auth: ServiceAuth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchEndpoint {
    Responses,
    ChatCompletions,
    Embeddings,
    Moderations,
    ImageGenerations,
    ImageEdits,
    Videos,
}
impl BatchEndpoint {
    pub fn path(self) -> &'static str {
        match self {
            Self::Responses => "/v1/responses",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Embeddings => "/v1/embeddings",
            Self::Moderations => "/v1/moderations",
            Self::ImageGenerations => "/v1/images/generations",
            Self::ImageEdits => "/v1/images/edits",
            Self::Videos => "/v1/videos",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchLine {
    pub custom_id: String,
    pub body: Value,
}

/// The documented way a request body refers to a previously uploaded file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFileUse {
    /// Responses `input_file` with a Files API `user_data` file.
    ResponsesInputFile,
    /// Responses `input_image` with a Files API `user_data` or `vision` file.
    ResponsesInputImage,
    /// Chat Completions `file` content part containing a PDF `user_data` file.
    ChatCompletionsPdf,
}

/// Caller-owned file reference used inside a Batch JSONL request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchAttachmentRef {
    file: ProviderFileRef,
    usage: BatchFileUse,
}

impl BatchAttachmentRef {
    pub fn responses_file(file: ProviderFileRef) -> Self {
        Self {
            file,
            usage: BatchFileUse::ResponsesInputFile,
        }
    }

    pub fn responses_image(file: ProviderFileRef) -> Self {
        Self {
            file,
            usage: BatchFileUse::ResponsesInputImage,
        }
    }

    pub fn chat_completions_pdf(file: ProviderFileRef) -> Self {
        Self {
            file,
            usage: BatchFileUse::ChatCompletionsPdf,
        }
    }

    pub fn file(&self) -> &ProviderFileRef {
        &self.file
    }

    pub fn usage(&self) -> BatchFileUse {
        self.usage
    }
}

/// Construct a JSONL input with unique custom IDs and one endpoint/model.
pub fn encode_jsonl(endpoint: BatchEndpoint, lines: &[BatchLine]) -> Result<Vec<u8>, BatchError> {
    reject_unbound_file_inputs(endpoint, lines)?;
    let mut output = Vec::new();
    write_jsonl_once(endpoint, lines, &mut output)?;
    Ok(output)
}

/// Encode a Batch file that contains only explicit references to caller-uploaded
/// durable Files API objects. The references remain out-of-band and must also
/// be passed to `submit_with_attachments()` for scope and expiry reconciliation.
pub fn encode_jsonl_with_attachments(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
    attachments: &[BatchAttachmentRef],
) -> Result<Vec<u8>, BatchError> {
    validate_attachment_manifest(endpoint, lines, attachments)?;
    let mut output = Vec::new();
    write_jsonl_once(endpoint, lines, &mut output)?;
    Ok(output)
}

/// Write a validated Batch input to a file or another caller-owned sink.
/// Validation completes before the first write. I/O failure may leave a
/// partial output file, which the caller should discard.
pub fn write_jsonl(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
    output: &mut impl Write,
) -> Result<u64, BatchError> {
    reject_unbound_file_inputs(endpoint, lines)?;
    write_jsonl_once(endpoint, lines, &mut std::io::sink())?;
    write_jsonl_once(endpoint, lines, output)
}

/// Stream validated JSONL to a caller-owned sink when its request bodies use
/// durable file references. Validation finishes before the first output byte.
pub fn write_jsonl_with_attachments(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
    attachments: &[BatchAttachmentRef],
    output: &mut impl Write,
) -> Result<u64, BatchError> {
    validate_attachment_manifest(endpoint, lines, attachments)?;
    write_jsonl_once(endpoint, lines, &mut std::io::sink())?;
    write_jsonl_once(endpoint, lines, output)
}

fn reject_unbound_file_inputs(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
) -> Result<(), BatchError> {
    for line in lines {
        if !collect_file_inputs(endpoint, &line.body)?.is_empty() {
            return Err(invalid(
                "Batch file inputs require an explicit durable attachment manifest",
            )
            .into());
        }
    }
    Ok(())
}

fn validate_attachment_manifest(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
    attachments: &[BatchAttachmentRef],
) -> Result<(), BatchError> {
    let mut required = BTreeMap::new();
    for line in lines {
        for (file_id, usage) in collect_file_inputs(endpoint, &line.body)? {
            if required
                .insert(file_id, usage)
                .is_some_and(|previous| previous != usage)
            {
                return Err(
                    invalid("one Batch file cannot be used as different input types").into(),
                );
            }
        }
    }
    let mut provided = BTreeMap::new();
    for attachment in attachments {
        validate_attachment_shape(endpoint, attachment)?;
        let file_id = attachment.file.file_id.as_str().to_owned();
        if provided.insert(file_id, attachment.usage).is_some() {
            return Err(invalid("Batch attachment manifest contains a duplicate file ID").into());
        }
    }
    if required != provided {
        return Err(invalid(
            "Batch file IDs in JSONL must exactly match the durable attachment manifest",
        )
        .into());
    }
    Ok(())
}

fn validate_attachment_shape(
    endpoint: BatchEndpoint,
    attachment: &BatchAttachmentRef,
) -> Result<(), BatchError> {
    let file = &attachment.file;
    let supported_endpoint = match attachment.usage {
        BatchFileUse::ResponsesInputFile | BatchFileUse::ResponsesInputImage => {
            BatchEndpoint::Responses
        }
        BatchFileUse::ChatCompletionsPdf => BatchEndpoint::ChatCompletions,
    };
    let purpose_matches = match attachment.usage {
        BatchFileUse::ResponsesInputImage => {
            matches!(file.purpose.as_deref(), Some("user_data" | "vision"))
        }
        BatchFileUse::ResponsesInputFile | BatchFileUse::ChatCompletionsPdf => {
            file.purpose.as_deref() == Some("user_data")
        }
    };
    if endpoint != supported_endpoint
        || file.provider_id.as_str() != "openai"
        || file
            .account_scope
            .as_deref()
            .is_none_or(|scope| scope.trim().is_empty())
        || file.profile_name.trim().is_empty()
        || !valid_id(&file.file_id)
        || !purpose_matches
    {
        return Err(invalid(
            "Batch attachment must be a scoped OpenAI file with the purpose required by its input type",
        )
        .into());
    }
    match attachment.usage {
        BatchFileUse::ResponsesInputImage
            if !file
                .media_type
                .as_deref()
                .is_some_and(|media_type| media_type.starts_with("image/")) =>
        {
            return Err(invalid("Responses input_image requires an image file").into());
        }
        BatchFileUse::ChatCompletionsPdf
            if file.media_type.as_deref() != Some("application/pdf") =>
        {
            return Err(invalid("Chat Completions file inputs require a PDF").into());
        }
        _ => {}
    }
    Ok(())
}

fn collect_file_inputs(
    endpoint: BatchEndpoint,
    body: &Value,
) -> Result<BTreeMap<String, BatchFileUse>, BatchError> {
    fn visit(
        endpoint: BatchEndpoint,
        value: &Value,
        found: &mut BTreeMap<String, BatchFileUse>,
    ) -> Result<(), BatchError> {
        match value {
            Value::Array(values) => {
                for value in values {
                    visit(endpoint, value, found)?;
                }
            }
            Value::Object(object) => {
                let content_type = object.get("type").and_then(Value::as_str);
                let recognized = match (endpoint, content_type) {
                    (BatchEndpoint::Responses, Some("input_file")) => {
                        validate_file_source(object, "file_data", false)?;
                        add_file_id(
                            object.get("file_id"),
                            BatchFileUse::ResponsesInputFile,
                            found,
                        )?;
                        true
                    }
                    (BatchEndpoint::Responses, Some("input_image")) => {
                        validate_file_source(object, "image_url", true)?;
                        add_file_id(
                            object.get("file_id"),
                            BatchFileUse::ResponsesInputImage,
                            found,
                        )?;
                        true
                    }
                    (BatchEndpoint::ChatCompletions, Some("file")) => {
                        let file =
                            object
                                .get("file")
                                .and_then(Value::as_object)
                                .ok_or_else(|| {
                                    invalid(
                                        "Chat Completions file content must contain a file object",
                                    )
                                })?;
                        validate_file_source(file, "file_data", false)?;
                        if file
                            .get("file_data")
                            .filter(|value| !value.is_null())
                            .and_then(Value::as_str)
                            .is_some_and(|data| !data.starts_with("data:application/pdf;base64,"))
                        {
                            return Err(invalid(
                                "Chat Completions inline file data must be a PDF data URL",
                            )
                            .into());
                        }
                        add_file_id(file.get("file_id"), BatchFileUse::ChatCompletionsPdf, found)?;
                        true
                    }
                    (BatchEndpoint::ChatCompletions, Some("image_url")) => {
                        let url = object
                            .get("image_url")
                            .and_then(Value::as_object)
                            .and_then(|image| image.get("url"))
                            .and_then(Value::as_str)
                            .ok_or_else(|| invalid("Chat Completions image_url needs a URL"))?;
                        if !url.starts_with("data:image/") || !url.contains(',') {
                            return Err(invalid("Batch remote image URLs cannot be reconciled; use an inline data URL").into());
                        }
                        false
                    }
                    _ => false,
                };
                let documented_top_level_file_id = matches!(
                    (endpoint, content_type),
                    (BatchEndpoint::Responses, Some("input_file" | "input_image"))
                );
                if object.contains_key("file_id") && !documented_top_level_file_id {
                    return Err(invalid(
                        "Batch file_id is not in a documented file input content part",
                    )
                    .into());
                }
                for (key, child) in object {
                    if key == "file_id" || (recognized && key == "file") {
                        continue;
                    }
                    visit(endpoint, child, found)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn add_file_id(
        value: Option<&Value>,
        usage: BatchFileUse,
        found: &mut BTreeMap<String, BatchFileUse>,
    ) -> Result<(), BatchError> {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            return Ok(());
        };
        let file_id = value
            .as_str()
            .filter(|id| valid_id(id))
            .ok_or_else(|| invalid("Batch file_id must be a valid uploaded file ID"))?;
        if found
            .insert(file_id.to_owned(), usage)
            .is_some_and(|previous| previous != usage)
        {
            return Err(invalid("one Batch file cannot be used as different input types").into());
        }
        Ok(())
    }

    fn validate_file_source(
        object: &serde_json::Map<String, Value>,
        inline_key: &str,
        image_url: bool,
    ) -> Result<(), BatchError> {
        let file_id = object.get("file_id").filter(|value| !value.is_null());
        let inline = object.get(inline_key).filter(|value| !value.is_null());
        if file_id.is_some() && inline.is_some() {
            return Err(invalid("Batch file input cannot mix file_id and inline data").into());
        }
        if let Some(url) = object.get("file_url").filter(|value| !value.is_null()) {
            let _ = url;
            return Err(invalid("Batch file_url inputs are not accepted because their lifetime cannot be reconciled").into());
        }
        if let Some(inline) = inline {
            if !inline
                .as_str()
                .is_some_and(|data| data.starts_with("data:") && data.contains(','))
            {
                return Err(invalid("Batch inline file data must be a data URL").into());
            }
        }
        if image_url {
            if let Some(url) = object.get("image_url").filter(|value| !value.is_null()) {
                let url = url
                    .as_str()
                    .ok_or_else(|| invalid("Batch image_url must be a string"))?;
                if !url.starts_with("data:image/") || !url.contains(',') {
                    return Err(invalid("Batch remote image URLs cannot be reconciled; use a durable file_id or an inline data URL").into());
                }
            }
        }
        if file_id.is_none() && inline.is_none() {
            return Err(invalid("Batch file input needs file_id or inline data").into());
        }
        Ok(())
    }

    let mut found = BTreeMap::new();
    visit(endpoint, body, &mut found)?;
    Ok(found)
}

fn write_jsonl_once(
    endpoint: BatchEndpoint,
    lines: &[BatchLine],
    output: &mut impl Write,
) -> Result<u64, BatchError> {
    if lines.is_empty() || lines.len() > 50_000 {
        return Err(invalid("batch input requires 1–50000 requests").into());
    }
    let mut seen = BTreeSet::new();
    let mut model: Option<&str> = None;
    let mut total = 0usize;
    for line in lines {
        if line.custom_id.trim().is_empty()
            || line.custom_id.len() > 256
            || !seen.insert(&line.custom_id)
        {
            return Err(invalid("batch custom_id is empty, too long or duplicated").into());
        }
        let body = line
            .body
            .as_object()
            .ok_or_else(|| invalid("batch body must be an object"))?;
        if body.get("stream") == Some(&Value::Bool(true)) {
            return Err(invalid("batch input cannot request streaming").into());
        }
        if let Some(next) = body.get("model") {
            let next = next
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| invalid("batch model is invalid"))?;
            if model.is_some_and(|previous| previous != next) {
                return Err(invalid("batch input mixes models").into());
            }
            model = Some(next);
        }
        let encoded = serde_json::to_vec(&json!({
            "custom_id":line.custom_id,"method":"POST","url":endpoint.path(),"body":line.body
        }))
        .map_err(|_| invalid("batch input cannot be serialized"))?;
        if total.saturating_add(encoded.len()).saturating_add(1) > MAX_INPUT {
            return Err(invalid("batch input exceeds 200 MB").into());
        }
        output.write_all(&encoded).map_err(BatchError::InputWrite)?;
        output.write_all(b"\n").map_err(BatchError::InputWrite)?;
        total += encoded.len() + 1;
    }
    Ok(total as u64)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchJobRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub job_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    Failed,
    InProgress,
    Finalizing,
    Completed,
    Expired,
    Cancelling,
    Cancelled,
    Other(String),
}
impl BatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Failed | Self::Completed | Self::Expired | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchJob {
    pub reference: BatchJobRef,
    pub result_endpoint_fingerprint: String,
    pub status: BatchStatus,
    pub endpoint: String,
    pub input_file_id: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub request_counts: Option<Value>,
    pub errors: Option<Value>,
    pub usage: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchResultKind {
    Output,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchResultRef {
    pub job: BatchJobRef,
    pub result_endpoint_fingerprint: String,
    pub file_id: String,
    pub kind: BatchResultKind,
}

impl BatchJob {
    pub fn output_ref(&self) -> Option<BatchResultRef> {
        self.output_file_id.as_ref().map(|id| BatchResultRef {
            job: self.reference.clone(),
            result_endpoint_fingerprint: self.result_endpoint_fingerprint.clone(),
            file_id: id.clone(),
            kind: BatchResultKind::Output,
        })
    }
    pub fn error_ref(&self) -> Option<BatchResultRef> {
        self.error_file_id.as_ref().map(|id| BatchResultRef {
            job: self.reference.clone(),
            result_endpoint_fingerprint: self.result_endpoint_fingerprint.clone(),
            file_id: id.clone(),
            kind: BatchResultKind::Error,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResultLine {
    pub custom_id: String,
    pub response: Option<Value>,
    pub error: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchPage {
    pub jobs: Vec<BatchJob>,
    pub has_more: bool,
    pub last_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid batch response: {0}")]
    InvalidResponse(String),
    #[error("invalid batch result file: {0}")]
    InvalidResult(String),
    #[error("could not write batch JSONL input: {0}")]
    InputWrite(#[source] std::io::Error),
    #[error("batch provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
}

/// A service handle whose live configuration is captured once per operation.
#[derive(Clone, Copy)]
pub struct BatchService<'a> {
    source: ClientSource<'a>,
}

impl<'a> BatchService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn submit(
        self,
        input: &ProviderFileRef,
        endpoint: BatchEndpoint,
        metadata: Option<&BTreeMap<String, String>>,
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        self.submit_with_attachments(input, endpoint, metadata, &[], options)
            .await
    }

    /// Submit JSONL that refers to caller-uploaded OpenAI Files API inputs.
    /// Each referenced file is retrieved immediately before submission to
    /// verify its account-bound identity, purpose, and remaining lifetime.
    /// Files are never uploaded, extended, deleted, or retried automatically.
    pub async fn submit_with_attachments(
        self,
        input: &ProviderFileRef,
        endpoint: BatchEndpoint,
        metadata: Option<&BTreeMap<String, String>>,
        attachments: &[BatchAttachmentRef],
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .submit(
                profile_name,
                input,
                endpoint,
                metadata,
                attachments,
                options,
            )
            .await
    }

    pub async fn get(
        self,
        reference: &BatchJobRef,
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .get(reference, options)
            .await
    }

    pub async fn cancel(
        self,
        reference: &BatchJobRef,
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .cancel(reference, options)
            .await
    }

    pub async fn list(
        self,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<BatchPage, BatchError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .list(profile_name, limit, after, options)
            .await
    }

    /// Collect one result JSONL file, bounded to 64 MiB. Use
    /// [`Self::stream_result`] for larger files.
    pub async fn read_result(
        self,
        reference: &BatchResultRef,
        options: &RequestOptions,
    ) -> Result<Vec<BatchResultLine>, BatchError> {
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .read_result(reference, options)
            .await
    }

    /// Stream result rows without buffering the entire JSONL file. Each line
    /// is limited to 16 MiB; dropping the stream cancels the response read.
    pub async fn stream_result(
        self,
        reference: &BatchResultRef,
        options: &RequestOptions,
    ) -> Result<BoxStream<'static, Result<BatchResultLine, BatchError>>, BatchError> {
        let snapshot = self.source.pin()?;
        PinnedBatchService { client: &snapshot }
            .stream_result(reference, options)
            .await
    }
}

#[derive(Clone, Copy)]
struct PinnedBatchService<'a> {
    client: &'a ClientSnapshot,
}

impl<'a> PinnedBatchService<'a> {
    fn resolve<'s>(
        &'s self,
        profile_name: &str,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s BatchRoute, String), BatchError> {
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown batch profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("batch profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.batches else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled batch route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| invalid("batch requires a non-secret account_scope"))?;
        Ok((profile, route, scope.into()))
    }

    fn checked<'s>(
        &'s self,
        reference: &BatchJobRef,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s BatchRoute), BatchError> {
        let (profile, route, scope) = self.resolve(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.account_scope != scope
            || !valid_id(&reference.job_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "batch belongs to another provider, profile, endpoint or account".into(),
            }
            .into());
        }
        Ok((profile, route))
    }

    async fn submit(
        self,
        profile_name: &str,
        input: &ProviderFileRef,
        endpoint: BatchEndpoint,
        metadata: Option<&BTreeMap<String, String>>,
        attachments: &[BatchAttachmentRef],
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let (profile, route, scope) = self.resolve(profile_name, options)?;
        if input.provider_id != profile.provider_id
            || input.profile_name != profile.profile_name
            || input.protocol != profile.protocol
            || input.endpoint_fingerprint != provider_file_endpoint_fingerprint(&profile.base_url)
            || input.account_scope.as_deref() != Some(scope.as_str())
            || input.purpose.as_deref() != Some("batch")
            || !valid_id(&input.file_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "batch input file belongs to another connection or purpose".into(),
            }
            .into());
        }
        if let Some(metadata) = metadata {
            if metadata.len() > 16
                || !metadata.iter().all(|(key, value)| {
                    !key.trim().is_empty() && key.len() <= 64 && value.len() <= 512
                })
            {
                return Err(invalid("batch metadata is invalid").into());
            }
        }
        let mut seen_attachments = BTreeSet::new();
        for attachment in attachments {
            validate_attachment_shape(endpoint, attachment)?;
            let file = &attachment.file;
            if file.provider_id != profile.provider_id
                || file.profile_name != profile.profile_name
                || file.protocol != profile.protocol
                || file.endpoint_fingerprint
                    != provider_file_endpoint_fingerprint(&profile.base_url)
                || file.account_scope.as_deref() != Some(scope.as_str())
            {
                return Err(LlmError::PermissionDenied {
                    message: "batch attachment belongs to another connection or account".into(),
                }
                .into());
            }
            if !seen_attachments.insert(file.file_id.as_str()) {
                return Err(
                    invalid("Batch attachment manifest contains a duplicate file ID").into(),
                );
            }
        }
        for attachment in attachments {
            let metadata = self
                .send_url(
                    route,
                    options,
                    "GET",
                    attachment_url(route, &attachment.file.file_id)?,
                    None,
                    "reconcile_attachment",
                )
                .await?;
            validate_remote_attachment(attachment, &metadata)?;
        }
        let mut body = json!({"input_file_id":input.file_id,"endpoint":endpoint.path(),"completion_window":"24h"});
        if let Some(metadata) = metadata {
            body["metadata"] = json!(metadata);
        }
        let value = self
            .send(route, options, "POST", &[], Some(body), "submit")
            .await?;
        decode_job(
            value,
            profile,
            route,
            &scope,
            None,
            Some(endpoint.path()),
            Some(&input.file_id),
        )
    }

    async fn get(
        self,
        reference: &BatchJobRef,
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let (profile, route) = self.checked(reference, options)?;
        let value = self
            .send(route, options, "GET", &[&reference.job_id], None, "get")
            .await?;
        decode_job(
            value,
            profile,
            route,
            &reference.account_scope,
            Some(&reference.job_id),
            None,
            None,
        )
    }

    async fn cancel(
        self,
        reference: &BatchJobRef,
        options: &RequestOptions,
    ) -> Result<BatchJob, BatchError> {
        let (profile, route) = self.checked(reference, options)?;
        let value = self
            .send(
                route,
                options,
                "POST",
                &[&reference.job_id, "cancel"],
                None,
                "cancel",
            )
            .await?;
        decode_job(
            value,
            profile,
            route,
            &reference.account_scope,
            Some(&reference.job_id),
            None,
            None,
        )
    }

    async fn list(
        self,
        profile_name: &str,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<BatchPage, BatchError> {
        let (profile, route, scope) = self.resolve(profile_name, options)?;
        if !(1..=100).contains(&limit) || after.is_some_and(|id| !valid_id(id)) {
            return Err(invalid("batch list limit or cursor is invalid").into());
        }
        let mut url = route_url(route, &[])?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &limit.to_string());
            if let Some(after) = after {
                query.append_pair("after", after);
            }
        }
        let value = self
            .send_url(route, options, "GET", url, None, "list")
            .await?;
        let rows = value["data"]
            .as_array()
            .ok_or_else(|| bad("batch list has no data array"))?;
        let jobs = rows
            .iter()
            .map(|row| decode_job(row.clone(), profile, route, &scope, None, None, None))
            .collect::<Result<Vec<_>, _>>()?;
        let has_more = value["has_more"]
            .as_bool()
            .ok_or_else(|| bad("batch list has no has_more boolean"))?;
        let last_id = value["last_id"].as_str().map(str::to_owned);
        if has_more && last_id.as_deref().is_none_or(|id| !valid_id(id)) {
            return Err(bad("batch list has no valid next cursor"));
        }
        Ok(BatchPage {
            jobs,
            has_more,
            last_id,
        })
    }

    /// Collect one result JSONL file, bounded to 64 MiB.
    async fn read_result(
        self,
        reference: &BatchResultRef,
        options: &RequestOptions,
    ) -> Result<Vec<BatchResultLine>, BatchError> {
        let request = self.result_request(reference, options)?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, MAX_BODY)
            .await?;
        if !(200..300).contains(&response.status) {
            let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(BatchError::Provider {
                status: response.status,
                request_id: response.header("x-request-id").map(str::to_owned),
                body,
            });
        }
        parse_result_lines(&response.body)
    }

    async fn stream_result(
        self,
        reference: &BatchResultRef,
        options: &RequestOptions,
    ) -> Result<BoxStream<'static, Result<BatchResultLine, BatchError>>, BatchError> {
        let request = self.result_request(reference, options)?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send(request)
            .await?;
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_BODY)).await?;
            let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(BatchError::Provider {
                status: response.status,
                request_id: response.header("x-request-id").map(str::to_owned),
                body,
            });
        }
        Ok(stream_result_lines(response.body))
    }

    fn result_request(
        self,
        reference: &BatchResultRef,
        options: &RequestOptions,
    ) -> Result<HttpRequest, BatchError> {
        let (_, route) = self.checked(&reference.job, options)?;
        if reference.result_endpoint_fingerprint
            != provider_file_endpoint_fingerprint(&route.files_endpoint)
            || !valid_id(&reference.file_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "batch result file belongs to another files endpoint".into(),
            }
            .into());
        }
        let mut url = url::Url::parse(&route.files_endpoint)
            .map_err(|_| invalid("invalid batch files URL"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid batch files URL"))?
            .extend([reference.file_id.as_str(), "content"]);
        let mut headers = Vec::new();
        match &route.auth {
            ServiceAuth::None => {}
            ServiceAuth::Bearer | ServiceAuth::ApiKey { .. } => {
                let secret =
                    options
                        .credential
                        .as_ref()
                        .ok_or_else(|| LlmError::Authentication {
                            message: "batch route requires a credential".into(),
                        })?;
                match &route.auth {
                    ServiceAuth::Bearer => headers.push((
                        "authorization".into(),
                        format!("Bearer {}", secret.expose_secret()),
                    )),
                    ServiceAuth::ApiKey { header } => {
                        headers.push((header.clone(), secret.expose_secret().clone()))
                    }
                    ServiceAuth::None => unreachable!(),
                }
            }
        }
        Ok(HttpRequest {
            method: "GET".into(),
            url: url.into(),
            headers,
            body: Vec::new().into(),
            timeout: Some(options.total_timeout.unwrap_or(Duration::from_secs(120))),
        })
    }

    async fn send(
        self,
        route: &BatchRoute,
        options: &RequestOptions,
        method: &'static str,
        path: &[&str],
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<Value, BatchError> {
        self.send_url(
            route,
            options,
            method,
            route_url(route, path)?,
            body,
            operation,
        )
        .await
    }
    async fn send_url(
        self,
        route: &BatchRoute,
        options: &RequestOptions,
        method: &'static str,
        url: url::Url,
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<Value, BatchError> {
        let bytes = body
            .map(|value| serde_json::to_vec(&value))
            .transpose()
            .map_err(|_| invalid("batch request cannot be serialized"))?
            .unwrap_or_default();
        if bytes.len() > MAX_BODY {
            return Err(invalid("batch request exceeds 64 MiB").into());
        }
        let mut headers = vec![("content-type".into(), "application/json".into())];
        match &route.auth {
            ServiceAuth::None => {}
            ServiceAuth::Bearer | ServiceAuth::ApiKey { .. } => {
                let secret =
                    options
                        .credential
                        .as_ref()
                        .ok_or_else(|| LlmError::Authentication {
                            message: "batch route requires a credential".into(),
                        })?;
                match &route.auth {
                    ServiceAuth::Bearer => headers.push((
                        "authorization".into(),
                        format!("Bearer {}", secret.expose_secret()),
                    )),
                    ServiceAuth::ApiKey { header } => {
                        headers.push((header.clone(), secret.expose_secret().clone()))
                    }
                    ServiceAuth::None => unreachable!(),
                }
            }
        }
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let request = HttpRequest {
            method: method.into(),
            url: url.into(),
            headers,
            body: bytes.into(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, MAX_BODY)
            .await
            .map_err(|source| {
                if matches!(operation, "submit" | "cancel")
                    && matches!(
                        source,
                        LlmError::Transport { .. } | LlmError::TransportTimeout { .. }
                    )
                {
                    BatchError::OutcomeUnknown { operation, source }
                } else {
                    BatchError::Llm(source)
                }
            })?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let value = match serde_json::from_slice(&response.body) {
            Ok(value) => value,
            Err(_) if !(200..300).contains(&response.status) => {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            }
            Err(_) => return Err(bad("batch success body is not JSON")),
        };
        if !(200..300).contains(&response.status) {
            return Err(BatchError::Provider {
                status: response.status,
                request_id,
                body: value,
            });
        }
        Ok(value)
    }
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn attachment_url(route: &BatchRoute, file_id: &str) -> Result<url::Url, LlmError> {
    if !valid_id(file_id) {
        return Err(invalid("invalid batch attachment file ID"));
    }
    let mut url =
        url::Url::parse(&route.files_endpoint).map_err(|_| invalid("invalid batch files URL"))?;
    url.path_segments_mut()
        .map_err(|_| invalid("invalid batch files URL"))?
        .push(file_id);
    Ok(url)
}

fn validate_remote_attachment(
    attachment: &BatchAttachmentRef,
    metadata: &Value,
) -> Result<(), BatchError> {
    let remote_purpose = metadata.get("purpose").and_then(Value::as_str);
    let purpose_matches = match attachment.usage {
        BatchFileUse::ResponsesInputImage => {
            matches!(remote_purpose, Some("user_data" | "vision"))
        }
        BatchFileUse::ResponsesInputFile | BatchFileUse::ChatCompletionsPdf => {
            remote_purpose == Some("user_data")
        }
    };
    if metadata.get("id").and_then(Value::as_str) != Some(attachment.file.file_id.as_str())
        || !purpose_matches
    {
        return Err(bad(
            "Files API metadata does not match the attachment ID and input purpose",
        ));
    }
    if let Some(expires_at) = metadata.get("expires_at").filter(|value| !value.is_null()) {
        let expires_at = parse_unix_timestamp(expires_at)
            .ok_or_else(|| bad("Files API expires_at is not a Unix timestamp"))?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("system clock is before the Unix epoch"))?
            .as_secs();
        let minimum_lifetime = (BATCH_COMPLETION_WINDOW_SECONDS
            + BATCH_CANCELLATION_WINDOW_SECONDS
            + ATTACHMENT_EXPIRY_SAFETY_SECONDS) as u64;
        if expires_at < now.saturating_add(minimum_lifetime) {
            return Err(invalid(
                "Batch attachment expires before the 24-hour completion window, cancellation reconciliation period, and clock/request safety margin",
            )
            .into());
        }
    }
    Ok(())
}

fn parse_unix_timestamp(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn bad(message: &str) -> BatchError {
    BatchError::InvalidResponse(message.into())
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
fn parse_result_lines(body: &[u8]) -> Result<Vec<BatchResultLine>, BatchError> {
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, raw) in body.split(|byte| *byte == b'\n').enumerate() {
        if let Some(row) = parse_result_line(raw, index + 1, &mut seen)? {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn parse_result_line(
    raw: &[u8],
    line_number: usize,
    seen: &mut BTreeSet<String>,
) -> Result<Option<BatchResultLine>, BatchError> {
    let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
    if raw.is_empty() {
        return Ok(None);
    }
    if raw.len() > MAX_RESULT_LINE {
        return Err(BatchError::InvalidResult(format!(
            "line {line_number} exceeds 16 MiB"
        )));
    }
    let value: Value = serde_json::from_slice(raw)
        .map_err(|_| BatchError::InvalidResult(format!("line {line_number} is not JSON")))?;
    let custom_id = value["custom_id"]
        .as_str()
        .filter(|id| !id.trim().is_empty() && id.len() <= 256)
        .ok_or_else(|| BatchError::InvalidResult(format!("line {line_number} has no custom_id")))?;
    if !seen.insert(custom_id.to_owned()) {
        return Err(BatchError::InvalidResult(format!(
            "duplicate custom_id on line {line_number}"
        )));
    }
    if seen.len() > 50_000 {
        return Err(BatchError::InvalidResult(
            "batch result exceeds 50000 unique custom IDs".into(),
        ));
    }
    let response = value.get("response").filter(|v| !v.is_null()).cloned();
    let error = value.get("error").filter(|v| !v.is_null()).cloned();
    if response.is_none() && error.is_none() {
        return Err(BatchError::InvalidResult(format!(
            "line {line_number} has no response or error"
        )));
    }
    Ok(Some(BatchResultLine {
        custom_id: custom_id.into(),
        response,
        error,
        native: value,
    }))
}

struct ResultStreamState {
    body: Option<BoxStream<'static, Result<Bytes, LlmError>>>,
    chunk: Bytes,
    partial: Vec<u8>,
    seen: BTreeSet<String>,
    line_number: usize,
}

fn stream_result_lines(
    body: BoxStream<'static, Result<Bytes, LlmError>>,
) -> BoxStream<'static, Result<BatchResultLine, BatchError>> {
    stream::unfold(
        ResultStreamState {
            body: Some(body),
            chunk: Bytes::new(),
            partial: Vec::new(),
            seen: BTreeSet::new(),
            line_number: 0,
        },
        |mut state| async move {
            loop {
                state.body.as_ref()?;
                if let Some(end) = state.chunk.iter().position(|byte| *byte == b'\n') {
                    let segment = state.chunk.slice(..end);
                    state.chunk = state.chunk.slice(end + 1..);
                    state.line_number += 1;
                    if state.partial.len().saturating_add(segment.len()) > MAX_RESULT_LINE + 1 {
                        state.body = None;
                        return Some((
                            Err(BatchError::InvalidResult(format!(
                                "line {} exceeds 16 MiB",
                                state.line_number
                            ))),
                            state,
                        ));
                    }
                    state.partial.extend_from_slice(&segment);
                    let raw = std::mem::take(&mut state.partial);
                    match parse_result_line(&raw, state.line_number, &mut state.seen) {
                        Ok(Some(row)) => return Some((Ok(row), state)),
                        Ok(None) => continue,
                        Err(error) => {
                            state.body = None;
                            return Some((Err(error), state));
                        }
                    }
                }
                if state.partial.len().saturating_add(state.chunk.len()) > MAX_RESULT_LINE + 1 {
                    state.body = None;
                    return Some((
                        Err(BatchError::InvalidResult(format!(
                            "line {} exceeds 16 MiB",
                            state.line_number + 1
                        ))),
                        state,
                    ));
                }
                state.partial.extend_from_slice(&state.chunk);
                state.chunk = Bytes::new();
                let body = state.body.as_mut()?;
                match body.next().await {
                    Some(Ok(chunk)) => state.chunk = chunk,
                    Some(Err(error)) => {
                        state.body = None;
                        return Some((Err(BatchError::Llm(error)), state));
                    }
                    None => {
                        state.body = None;
                        if state.partial.is_empty() {
                            return None;
                        }
                        state.line_number += 1;
                        let raw = std::mem::take(&mut state.partial);
                        return parse_result_line(&raw, state.line_number, &mut state.seen)
                            .transpose()
                            .map(|result| (result, state));
                    }
                }
            }
        },
    )
    .boxed()
}
fn route_url(route: &BatchRoute, path: &[&str]) -> Result<url::Url, LlmError> {
    let mut url = url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid batch URL"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| invalid("invalid batch URL"))?;
        for part in path {
            if !valid_id(part) {
                return Err(invalid("invalid batch path ID"));
            }
            segments.push(part);
        }
    }
    Ok(url)
}
pub fn validate_route(route: &BatchRoute) -> Result<(), LlmError> {
    let url = url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid batch route"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.path().ends_with("/batches")
    {
        return Err(invalid(
            "batch endpoint must end in /batches and contain no credentials, query or fragment",
        ));
    }
    let files =
        url::Url::parse(&route.files_endpoint).map_err(|_| invalid("invalid batch files route"))?;
    if files.scheme() != url.scheme()
        || files.host_str() != url.host_str()
        || files.port_or_known_default() != url.port_or_known_default()
        || !files.path().ends_with("/files")
        || !files.username().is_empty()
        || files.password().is_some()
        || files.query().is_some()
        || files.fragment().is_some()
    {
        return Err(invalid(
            "batch files endpoint must share the batch origin and end in /files",
        ));
    }
    if let ServiceAuth::ApiKey { header } = &route.auth {
        if header.is_empty()
            || !header
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || [
                "host",
                "content-type",
                "content-length",
                "connection",
                "transfer-encoding",
            ]
            .iter()
            .any(|reserved| header.eq_ignore_ascii_case(reserved))
        {
            return Err(invalid("invalid batch authentication header"));
        }
    }
    Ok(())
}
fn decode_job(
    value: Value,
    profile: &ProviderProfile,
    route: &BatchRoute,
    scope: &str,
    expected_id: Option<&str>,
    expected_endpoint: Option<&str>,
    expected_input: Option<&str>,
) -> Result<BatchJob, BatchError> {
    let id = value["id"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("batch has no valid ID"))?;
    if expected_id.is_some_and(|expected| expected != id) {
        return Err(bad("batch ID does not match request"));
    }
    let endpoint = value["endpoint"]
        .as_str()
        .filter(|s| s.starts_with("/v1/"))
        .ok_or_else(|| bad("batch has no valid endpoint"))?;
    if expected_endpoint.is_some_and(|expected| expected != endpoint) {
        return Err(bad("batch endpoint does not match request"));
    }
    let input_file_id = value["input_file_id"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("batch has no valid input file ID"))?;
    if expected_input.is_some_and(|expected| expected != input_file_id) {
        return Err(bad("batch input file does not match request"));
    }
    let status = match value["status"]
        .as_str()
        .ok_or_else(|| bad("batch has no status"))?
    {
        "validating" => BatchStatus::Validating,
        "failed" => BatchStatus::Failed,
        "in_progress" => BatchStatus::InProgress,
        "finalizing" => BatchStatus::Finalizing,
        "completed" => BatchStatus::Completed,
        "expired" => BatchStatus::Expired,
        "cancelling" => BatchStatus::Cancelling,
        "cancelled" => BatchStatus::Cancelled,
        other => BatchStatus::Other(other.into()),
    };
    let file_id = |field: &str| -> Result<Option<String>, BatchError> {
        match value.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(id)) if valid_id(id) => Ok(Some(id.clone())),
            _ => Err(bad("batch has invalid output or error file ID")),
        }
    };
    Ok(BatchJob {
        reference: BatchJobRef {
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
            account_scope: scope.into(),
            job_id: id.into(),
        },
        result_endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.files_endpoint),
        status,
        endpoint: endpoint.into(),
        input_file_id: input_file_id.into(),
        output_file_id: file_id("output_file_id")?,
        error_file_id: file_id("error_file_id")?,
        request_counts: value
            .get("request_counts")
            .filter(|v| !v.is_null())
            .cloned(),
        errors: value.get("errors").filter(|v| !v.is_null()).cloned(),
        usage: value.get("usage").filter(|v| !v.is_null()).cloned(),
        native: value,
    })
}
