//! Kimi API Batch task submission and result retrieval.
//!
//! This service uses the documented international Kimi API endpoint. It
//! accepts references to files already uploaded with `purpose=batch`; file
//! upload is handled by the provider Files API. The service does not poll,
//! retry, or buffer result files.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use futures::{stream::BoxStream, Stream};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{
    pin::Pin,
    task::{Context, Poll},
};

const API_BASE_URL: &str = "https://api.moonshot.ai/v1";
const MAX_CONTROL_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Identity to which durable Kimi Batch references are bound.
///
/// `account_scope` is a stable, non-secret account identity supplied by the
/// host. The API key is deliberately kept only by [`KimiBatchService`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
}

impl KimiBatchScope {
    /// Create a scope for the documented `https://api.moonshot.ai/v1` API.
    ///
    /// Arbitrary or user-controlled endpoints are rejected so a persisted
    /// reference cannot redirect a credential to another host.
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        api_base_url: impl AsRef<str>,
    ) -> Result<Self, KimiBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if profile_name.trim().is_empty()
            || profile_name.chars().any(char::is_control)
            || account_scope.trim().is_empty()
            || account_scope.chars().any(char::is_control)
        {
            return Err(invalid("profile name and account scope must be non-empty and contain no control characters"));
        }
        if api_base_url.as_ref() != API_BASE_URL {
            return Err(invalid(
                "Kimi Batch currently supports only the documented international API endpoint",
            ));
        }
        Ok(Self {
            provider_id: ProviderId::from("kimi"),
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

    fn validate(&self) -> Result<(), KimiBatchError> {
        if self.provider_id.as_str() != "kimi"
            || self.profile_name.trim().is_empty()
            || self.profile_name.chars().any(char::is_control)
            || self.account_scope.trim().is_empty()
            || self.account_scope.chars().any(char::is_control)
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(API_BASE_URL)
        {
            return Err(invalid("Kimi Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// The only endpoint supported by the current Kimi Batch guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiBatchEndpoint {
    ChatCompletions,
}

impl KimiBatchEndpoint {
    pub const fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
        }
    }
}

impl Serialize for KimiBatchEndpoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.path())
    }
}

impl<'de> Deserialize<'de> for KimiBatchEndpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "/v1/chat/completions" => Ok(Self::ChatCompletions),
            _ => Err(serde::de::Error::custom("unsupported Kimi Batch endpoint")),
        }
    }
}

/// Models listed as Batch-compatible in the current official guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiBatchModel {
    KimiK2_6,
    KimiK2_7Code,
}

impl KimiBatchModel {
    pub const fn id(self) -> &'static str {
        match self {
            Self::KimiK2_6 => "kimi-k2.6",
            Self::KimiK2_7Code => "kimi-k2.7-code",
        }
    }
}

impl Serialize for KimiBatchModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.id())
    }
}

impl<'de> Deserialize<'de> for KimiBatchModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "kimi-k2.6" => Ok(Self::KimiK2_6),
            "kimi-k2.7-code" => Ok(Self::KimiK2_7Code),
            _ => Err(serde::de::Error::custom(
                "model is not listed as Kimi Batch-compatible",
            )),
        }
    }
}

/// A Kimi Files API ID from an upload with `purpose=batch`.
///
/// Construct this only from a successful upload made under the same scope.
/// The API key and file body are not retained in this reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiBatchInputFileRef {
    scope: KimiBatchScope,
    endpoint: KimiBatchEndpoint,
    model: KimiBatchModel,
    file_id: String,
}

impl KimiBatchInputFileRef {
    pub fn new(
        scope: KimiBatchScope,
        model: KimiBatchModel,
        file_id: impl Into<String>,
    ) -> Result<Self, KimiBatchError> {
        scope.validate()?;
        let file_id = file_id.into();
        if !valid_id(&file_id) {
            return Err(invalid("Kimi Batch input file ID is invalid"));
        }
        Ok(Self {
            scope,
            endpoint: KimiBatchEndpoint::ChatCompletions,
            model,
            file_id,
        })
    }

    pub fn scope(&self) -> &KimiBatchScope {
        &self.scope
    }

    pub fn endpoint(&self) -> KimiBatchEndpoint {
        self.endpoint
    }

    pub fn model(&self) -> KimiBatchModel {
        self.model
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }
}

/// The documented Batch processing window. The current guide demonstrates
/// `24h`; this type deliberately does not imply other accepted values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiBatchCompletionWindow {
    #[serde(rename = "24h")]
    Hours24,
}

impl KimiBatchCompletionWindow {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hours24 => "24h",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KimiBatchSubmitOptions {
    pub completion_window: KimiBatchCompletionWindow,
}

impl Default for KimiBatchSubmitOptions {
    fn default() -> Self {
        Self {
            completion_window: KimiBatchCompletionWindow::Hours24,
        }
    }
}

/// Durable identity of a Kimi Batch task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiBatchJobRef {
    scope: KimiBatchScope,
    endpoint_fingerprint: String,
    endpoint: KimiBatchEndpoint,
    model: KimiBatchModel,
    input_file_id: String,
    completion_window: KimiBatchCompletionWindow,
    batch_id: String,
}

impl KimiBatchJobRef {
    pub fn scope(&self) -> &KimiBatchScope {
        &self.scope
    }

    pub fn endpoint(&self) -> KimiBatchEndpoint {
        self.endpoint
    }

    pub fn model(&self) -> KimiBatchModel {
        self.model
    }

    pub fn input_file_id(&self) -> &str {
        &self.input_file_id
    }

    pub fn completion_window(&self) -> KimiBatchCompletionWindow {
        self.completion_window
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiBatchStatus {
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

impl KimiBatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }
}

impl Serialize for KimiBatchStatus {
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

impl<'de> Deserialize<'de> for KimiBatchStatus {
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiBatchRequestCounts {
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub completed: u64,
    #[serde(default)]
    pub failed: u64,
}

/// Typed task state returned by submit, get, or cancel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBatchSnapshot {
    pub reference: KimiBatchJobRef,
    pub status: KimiBatchStatus,
    pub request_counts: Option<KimiBatchRequestCounts>,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub errors: Option<Value>,
    pub metadata: Option<Value>,
    pub created_at: Option<i64>,
    pub in_progress_at: Option<i64>,
    pub expires_at: Option<i64>,
    pub finalizing_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub failed_at: Option<i64>,
    pub cancelling_at: Option<i64>,
    pub cancelled_at: Option<i64>,
}

impl KimiBatchSnapshot {
    /// Get a scoped reference to the successful JSONL output once completed.
    pub fn output_ref(&self) -> Option<KimiBatchResultRef> {
        if self.status != KimiBatchStatus::Completed {
            return None;
        }
        let file_id = self.output_file_id.as_ref()?;
        if !valid_id(file_id) {
            return None;
        }
        Some(KimiBatchResultRef {
            job: self.reference.clone(),
            endpoint_fingerprint: self.reference.endpoint_fingerprint.clone(),
            file_id: file_id.clone(),
        })
    }
}

/// A Batch returned by the list endpoint. Kimi's list object omits the model,
/// so it keeps the provider fields and scope without inventing one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBatchListItem {
    scope: KimiBatchScope,
    batch_id: String,
    pub endpoint: KimiBatchEndpoint,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: KimiBatchStatus,
    pub request_counts: Option<KimiBatchRequestCounts>,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub native: Value,
}

impl KimiBatchListItem {
    pub fn scope(&self) -> &KimiBatchScope {
        &self.scope
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// Construct a durable reference after the caller identifies the model
    /// associated with this batch; the provider's list response omits it.
    pub fn reference_with_model(
        &self,
        model: KimiBatchModel,
    ) -> Result<KimiBatchJobRef, KimiBatchError> {
        let completion_window = match self.completion_window.as_str() {
            "24h" => KimiBatchCompletionWindow::Hours24,
            _ => return Err(bad("listed Batch has an unsupported completion window")),
        };
        Ok(KimiBatchJobRef {
            scope: self.scope.clone(),
            endpoint_fingerprint: self.scope.endpoint_fingerprint.clone(),
            endpoint: self.endpoint,
            model,
            input_file_id: self.input_file_id.clone(),
            completion_window,
            batch_id: self.batch_id.clone(),
        })
    }
}

/// One explicit page from the Kimi Batch list endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBatchListPage {
    pub batches: Vec<KimiBatchListItem>,
    pub has_more: bool,
    pub native: Value,
}

/// Cursor and page size for one Kimi Batch list call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KimiBatchListOptions {
    after: Option<String>,
    limit: Option<u32>,
}

impl KimiBatchListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn after(mut self, batch_id: impl Into<String>) -> Self {
        self.after = Some(batch_id.into());
        self
    }

    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Scoped reference to a completed successful result file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiBatchResultRef {
    job: KimiBatchJobRef,
    endpoint_fingerprint: String,
    file_id: String,
}

impl KimiBatchResultRef {
    pub fn job(&self) -> &KimiBatchJobRef {
        &self.job
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KimiBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Kimi Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid Kimi Batch response: {0}")]
    InvalidResponse(String),
    #[error("Kimi Batch provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of Kimi Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Kimi Batch {operation} is unknown because the success response was invalid: {reason}")]
    ResponseOutcomeUnknown {
        operation: &'static str,
        reason: String,
    },
}

/// Provider-specific Batch lifecycle facade.
pub struct KimiBatchService<'a> {
    http: &'a dyn Transport,
    credential: Secret<String>,
    scope: KimiBatchScope,
}

impl<'a> KimiBatchService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        credential: Secret<String>,
        scope: KimiBatchScope,
    ) -> Result<Self, KimiBatchError> {
        scope.validate()?;
        if credential.expose_secret().trim().is_empty()
            || credential
                .expose_secret()
                .chars()
                .any(|character| matches!(character, '\r' | '\n'))
        {
            return Err(invalid(
                "Kimi API key must be non-empty and contain no line breaks",
            ));
        }
        Ok(Self {
            http,
            credential,
            scope,
        })
    }

    pub fn scope(&self) -> &KimiBatchScope {
        &self.scope
    }

    /// Submit an already-uploaded JSONL file with `purpose=batch`.
    pub async fn submit(
        &self,
        input_file: &KimiBatchInputFileRef,
        options: KimiBatchSubmitOptions,
    ) -> Result<KimiBatchSnapshot, KimiBatchError> {
        self.validate_input_file(input_file)?;
        let wire = SubmitWire {
            input_file_id: &input_file.file_id,
            endpoint: input_file.endpoint,
            completion_window: options.completion_window,
        };
        let body = serde_json::to_vec(&wire)
            .map_err(|_| invalid("Kimi Batch request could not be encoded as JSON"))?;
        let response = self
            .send_control(
                "POST",
                "/batches",
                Some(Bytes::from(body)),
                "submission",
                true,
            )
            .await?;
        let wire =
            decode_batch(&response).map_err(|error| KimiBatchError::ResponseOutcomeUnknown {
                operation: "submission",
                reason: error.to_string(),
            })?;
        if wire.endpoint != input_file.endpoint
            || wire.input_file_id != input_file.file_id
            || wire.completion_window != options.completion_window.as_str()
        {
            return Err(KimiBatchError::ResponseOutcomeUnknown {
                operation: "submission",
                reason: "provider response did not match the submitted file, endpoint, or completion window".into(),
            });
        }
        self.snapshot(wire, input_file.model, options.completion_window, None)
            .map_err(|error| KimiBatchError::ResponseOutcomeUnknown {
                operation: "submission",
                reason: error.to_string(),
            })
    }

    pub async fn get(
        &self,
        reference: &KimiBatchJobRef,
    ) -> Result<KimiBatchSnapshot, KimiBatchError> {
        self.validate_job_ref(reference)?;
        let path = format!("/batches/{}", reference.batch_id);
        let response = self
            .send_control("GET", &path, None, "query", false)
            .await?;
        let wire = decode_batch(&response)?;
        self.snapshot(
            wire,
            reference.model,
            reference.completion_window,
            Some(reference),
        )
    }

    /// Fetch one page of Batch jobs. Use the last item's `batch_id` as the
    /// `after` cursor for a later explicit call when `has_more` is true.
    pub async fn list(
        &self,
        options: &KimiBatchListOptions,
    ) -> Result<KimiBatchListPage, KimiBatchError> {
        if options.limit == Some(0) {
            return Err(invalid("list limit must be greater than zero"));
        }
        if options
            .after
            .as_deref()
            .is_some_and(|batch_id| !valid_id(batch_id))
        {
            return Err(invalid("list cursor must be a valid Batch ID"));
        }
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        if let Some(after) = options.after.as_deref() {
            query.append_pair("after", after);
        }
        if let Some(limit) = options.limit {
            query.append_pair("limit", &limit.to_string());
        }
        let query = query.finish();
        let path = if query.is_empty() {
            "/batches".to_owned()
        } else {
            format!("/batches?{query}")
        };
        let response = self.send_control("GET", &path, None, "list", false).await?;
        let native: Value = serde_json::from_slice(&response)
            .map_err(|error| bad(&format!("could not decode Batch list response: {error}")))?;
        let object = native
            .as_object()
            .ok_or_else(|| bad("Batch list response must be a JSON object"))?;
        if object.get("object").and_then(Value::as_str) != Some("list") {
            return Err(bad("provider returned an object other than a Batch list"));
        }
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("Batch list data must be an array"))?;
        let has_more = object
            .get("has_more")
            .and_then(Value::as_bool)
            .ok_or_else(|| bad("Batch list has_more must be a boolean"))?;
        let batches = data
            .iter()
            .cloned()
            .map(|batch| {
                let encoded = serde_json::to_vec(&batch)
                    .map_err(|error| bad(&format!("could not encode listed Batch: {error}")))?;
                let wire = decode_batch(&encoded)?;
                if wire.object != "batch"
                    || !valid_id(&wire.id)
                    || wire.endpoint != KimiBatchEndpoint::ChatCompletions
                    || !valid_id(&wire.input_file_id)
                {
                    return Err(bad("provider returned an invalid listed Batch identity"));
                }
                Ok(KimiBatchListItem {
                    scope: self.scope.clone(),
                    batch_id: wire.id,
                    endpoint: wire.endpoint,
                    input_file_id: wire.input_file_id,
                    completion_window: wire.completion_window,
                    status: wire.status,
                    request_counts: wire.request_counts,
                    output_file_id: wire.output_file_id,
                    error_file_id: wire.error_file_id,
                    native: batch,
                })
            })
            .collect::<Result<Vec<_>, KimiBatchError>>()?;
        let mut seen_ids = std::collections::BTreeSet::new();
        if batches
            .iter()
            .any(|batch| !seen_ids.insert(batch.batch_id.as_str()))
        {
            return Err(bad("Batch list contains duplicate IDs"));
        }
        if has_more {
            let Some(last) = batches.last() else {
                return Err(bad("Batch list has_more requires a non-empty page"));
            };
            if options.after.as_deref() == Some(last.batch_id.as_str()) {
                return Err(bad("Batch list cursor did not advance"));
            }
        }
        Ok(KimiBatchListPage {
            batches,
            has_more,
            native,
        })
    }

    pub async fn cancel(
        &self,
        reference: &KimiBatchJobRef,
    ) -> Result<KimiBatchSnapshot, KimiBatchError> {
        self.validate_job_ref(reference)?;
        let path = format!("/batches/{}/cancel", reference.batch_id);
        let response = self
            .send_control("POST", &path, None, "cancellation", true)
            .await?;
        let wire =
            decode_batch(&response).map_err(|error| KimiBatchError::ResponseOutcomeUnknown {
                operation: "cancellation",
                reason: error.to_string(),
            })?;
        self.snapshot(
            wire,
            reference.model,
            reference.completion_window,
            Some(reference),
        )
        .map_err(|error| KimiBatchError::ResponseOutcomeUnknown {
            operation: "cancellation",
            reason: error.to_string(),
        })
    }

    /// Start streaming the result JSONL file without buffering it in memory.
    pub async fn stream_result(
        &self,
        reference: &KimiBatchResultRef,
    ) -> Result<KimiBatchResultStream, KimiBatchError> {
        self.validate_job_ref(&reference.job)?;
        if reference.endpoint_fingerprint != self.scope.endpoint_fingerprint
            || !valid_id(&reference.file_id)
        {
            return Err(scope_mismatch());
        }
        let path = format!("/files/{}/content", reference.file_id);
        let response = HttpExecutor::new(self.http)
            .send(self.request("GET", &path, None)?)
            .await?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error_from_stream(response).await);
        }
        Ok(KimiBatchResultStream {
            body: response.body,
        })
    }

    fn snapshot(
        &self,
        wire: BatchWire,
        model: KimiBatchModel,
        completion_window: KimiBatchCompletionWindow,
        expected: Option<&KimiBatchJobRef>,
    ) -> Result<KimiBatchSnapshot, KimiBatchError> {
        if wire.object != "batch" || !valid_id(&wire.id) || !valid_id(&wire.input_file_id) {
            return Err(bad("provider returned an invalid Batch identity"));
        }
        if let Some(expected) = expected {
            if wire.id != expected.batch_id
                || wire.endpoint != expected.endpoint
                || wire.input_file_id != expected.input_file_id
                || wire.completion_window != expected.completion_window.as_str()
            {
                return Err(bad("provider returned a different Batch identity"));
            }
        }
        for file_id in [
            wire.output_file_id.as_deref(),
            wire.error_file_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !valid_id(file_id) {
                return Err(bad("provider returned an invalid Batch result file ID"));
            }
        }
        let reference = expected.cloned().unwrap_or_else(|| KimiBatchJobRef {
            scope: self.scope.clone(),
            endpoint_fingerprint: self.scope.endpoint_fingerprint.clone(),
            endpoint: wire.endpoint,
            model,
            input_file_id: wire.input_file_id.clone(),
            completion_window,
            batch_id: wire.id.clone(),
        });
        Ok(KimiBatchSnapshot {
            reference,
            status: wire.status,
            request_counts: wire.request_counts,
            output_file_id: wire.output_file_id,
            error_file_id: wire.error_file_id,
            errors: wire.errors,
            metadata: wire.metadata,
            created_at: wire.created_at,
            in_progress_at: wire.in_progress_at,
            expires_at: wire.expires_at,
            finalizing_at: wire.finalizing_at,
            completed_at: wire.completed_at,
            failed_at: wire.failed_at,
            cancelling_at: wire.cancelling_at,
            cancelled_at: wire.cancelled_at,
        })
    }

    fn validate_input_file(&self, reference: &KimiBatchInputFileRef) -> Result<(), KimiBatchError> {
        if reference.scope != self.scope
            || reference.scope.endpoint_fingerprint != self.scope.endpoint_fingerprint
            || !valid_id(&reference.file_id)
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_job_ref(&self, reference: &KimiBatchJobRef) -> Result<(), KimiBatchError> {
        if reference.scope != self.scope
            || reference.endpoint_fingerprint != self.scope.endpoint_fingerprint
            || reference.endpoint != KimiBatchEndpoint::ChatCompletions
            || !valid_id(&reference.batch_id)
            || !valid_id(&reference.input_file_id)
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    async fn send_control(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        operation: &'static str,
        unknown_on_transport: bool,
    ) -> Result<Bytes, KimiBatchError> {
        let request = self.request(method, path, body)?;
        let response = HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if unknown_on_transport && is_ambiguous_transport(&source) {
                    KimiBatchError::OutcomeUnknown { operation, source }
                } else {
                    KimiBatchError::Llm(source)
                }
            })?;
        ensure_success(response)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
    ) -> Result<HttpRequest, KimiBatchError> {
        if !path.starts_with('/') || path.contains('#') {
            return Err(invalid("invalid internal Kimi Batch route"));
        }
        let mut headers = vec![(
            "Authorization".into(),
            format!("Bearer {}", self.credential.expose_secret()),
        )];
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            method: method.into(),
            url: format!("{API_BASE_URL}{path}"),
            headers,
            body: body.unwrap_or_default(),
            timeout: None,
        })
    }
}

#[derive(Serialize)]
struct SubmitWire<'a> {
    input_file_id: &'a str,
    endpoint: KimiBatchEndpoint,
    completion_window: KimiBatchCompletionWindow,
}

struct BatchWire {
    id: String,
    object: String,
    endpoint: KimiBatchEndpoint,
    input_file_id: String,
    completion_window: String,
    status: KimiBatchStatus,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
    request_counts: Option<KimiBatchRequestCounts>,
    errors: Option<Value>,
    metadata: Option<Value>,
    created_at: Option<i64>,
    in_progress_at: Option<i64>,
    expires_at: Option<i64>,
    finalizing_at: Option<i64>,
    completed_at: Option<i64>,
    failed_at: Option<i64>,
    cancelling_at: Option<i64>,
    cancelled_at: Option<i64>,
}

#[derive(Deserialize)]
struct BatchWireFields {
    id: String,
    object: String,
    endpoint: KimiBatchEndpoint,
    input_file_id: String,
    completion_window: String,
    status: KimiBatchStatus,
    #[serde(default)]
    output_file_id: Option<String>,
    #[serde(default)]
    error_file_id: Option<String>,
    #[serde(default)]
    request_counts: Option<KimiBatchRequestCounts>,
    #[serde(default)]
    errors: Option<Value>,
    #[serde(default)]
    metadata: Option<Value>,
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
    cancelling_at: Option<i64>,
    #[serde(default)]
    cancelled_at: Option<i64>,
}

fn decode_batch(body: &[u8]) -> Result<BatchWire, KimiBatchError> {
    let wire: BatchWireFields = serde_json::from_slice(body)
        .map_err(|error| bad(&format!("could not decode Batch object: {error}")))?;
    if wire.completion_window != "24h" {
        return Err(bad("provider returned an unsupported completion window"));
    }
    Ok(BatchWire {
        id: wire.id,
        object: wire.object,
        endpoint: wire.endpoint,
        input_file_id: wire.input_file_id,
        completion_window: wire.completion_window,
        status: wire.status,
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
        cancelling_at: wire.cancelling_at,
        cancelled_at: wire.cancelled_at,
    })
}

/// A successful result stream. Chunks remain raw UTF-8 JSONL bytes so callers
/// can parse incrementally or persist them without buffering the full file.
pub struct KimiBatchResultStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl Stream for KimiBatchResultStream {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().body.as_mut().poll_next(cx)
    }
}

async fn provider_error_from_stream(response: crate::transport::StreamResponse) -> KimiBatchError {
    let status = response.status;
    let request_id = response
        .headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("x-request-id") || name.eq_ignore_ascii_case("request-id")
        })
        .map(|(_, value)| value.clone());
    let response = HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES))
        .await
        .unwrap_or_else(|_| HttpResponse {
            status,
            headers: Vec::new(),
            body: Bytes::new(),
        });
    KimiBatchError::Provider {
        status,
        request_id: request_id.or_else(|| response.header("x-request-id").map(str::to_owned)),
        body: serde_json::from_slice(&response.body).unwrap_or_else(|_| {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }),
    }
}

fn ensure_success(response: HttpResponse) -> Result<Bytes, KimiBatchError> {
    if (200..300).contains(&response.status) {
        return Ok(response.body);
    }
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    Err(KimiBatchError::Provider {
        status: response.status,
        request_id,
        body: serde_json::from_slice(&response.body).unwrap_or_else(|_| {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }),
    })
}

fn is_ambiguous_transport(error: &LlmError) -> bool {
    matches!(
        error,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
            | LlmError::RequestTooLarge { .. }
    )
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn invalid(message: &str) -> KimiBatchError {
    KimiBatchError::InvalidInput(message.into())
}

fn bad(message: &str) -> KimiBatchError {
    KimiBatchError::InvalidResponse(message.into())
}

fn scope_mismatch() -> KimiBatchError {
    LlmError::PermissionDenied {
        message: "Kimi Batch reference belongs to another provider, profile, endpoint, or account"
            .into(),
    }
    .into()
}
