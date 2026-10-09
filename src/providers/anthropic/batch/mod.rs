//! Anthropic Messages Batch lifecycle and streamed per-request results.
//!
//! The Messages Batches API accepts requests inline and streams result records
//! from its results endpoint. This service does not create or download a local
//! JSONL file; callers own polling, result processing, and any retry decisions.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{AuthStrategy, LlmError, ProviderId},
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    pin::Pin,
    task::{Context, Poll},
};
use url::Url;

const API_PATH: &str = "/v1/messages/batches";
const API_VERSION: &str = "2023-06-01";
const BATCHES_BETA: &str = "message-batches-2024-09-24";
const MAX_REQUESTS: usize = 100_000;
const MAX_REQUEST_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONTROL_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Identity used to bind batch references to one Anthropic profile, API
/// endpoint, and caller-selected account/workspace scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    api_base_url: String,
    endpoint_fingerprint: String,
    workspace_id: Option<String>,
}

impl AnthropicBatchScope {
    /// `api_base_url` is the configured API root, such as
    /// `https://api.anthropic.com`; the service appends the documented `/v1`
    /// route. Query strings, fragments, and URL credentials are rejected.
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        api_base_url: impl AsRef<str>,
        workspace_id: Option<String>,
    ) -> Result<Self, AnthropicBatchError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid_input(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        if workspace_id
            .as_deref()
            .is_some_and(|value| !valid_identity(value))
        {
            return Err(invalid_input(
                "workspace ID must be non-empty and contain no control characters",
            ));
        }
        let api_base_url = normalize_api_base_url(api_base_url.as_ref())?;
        let endpoint_fingerprint = provider_file_endpoint_fingerprint(&api_base_url);
        Ok(Self {
            provider_id: ProviderId::from("anthropic"),
            profile_name,
            account_scope,
            api_base_url,
            endpoint_fingerprint,
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

    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    pub fn workspace_id(&self) -> Option<&str> {
        self.workspace_id.as_deref()
    }

    fn validate(&self) -> Result<(), AnthropicBatchError> {
        let normalized = normalize_api_base_url(&self.api_base_url)?;
        if self.provider_id.as_str() != "anthropic"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || self
                .workspace_id
                .as_deref()
                .is_some_and(|value| !valid_identity(value))
            || normalized != self.api_base_url
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.api_base_url)
        {
            return Err(invalid_input("Anthropic Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// One Messages API input. `content` remains a JSON value because Anthropic
/// supports text, image, document, and tool blocks with independently evolving
/// fields; the role and enclosing message shape stay typed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchMessage {
    pub role: AnthropicBatchRole,
    pub content: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicBatchRole {
    User,
    Assistant,
}

/// Typed core of one non-streaming Messages request, with additional
/// documented Messages parameters available through `with_parameter`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchParams {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicBatchMessage>,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl AnthropicBatchParams {
    pub fn new(
        model: impl Into<String>,
        max_tokens: u32,
        messages: Vec<AnthropicBatchMessage>,
    ) -> Result<Self, AnthropicBatchError> {
        let params = Self {
            model: model.into(),
            max_tokens,
            messages,
            additional: BTreeMap::new(),
        };
        params.validate()?;
        Ok(params)
    }

    /// Add another Messages API parameter, such as `system`, `tools`, or
    /// `thinking`. `stream`, `speed`, and the typed core fields cannot be
    /// overridden because Anthropic documents them as unsupported or fixed in
    /// Batch requests.
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, AnthropicBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name) {
            return Err(invalid_input(
                "additional parameter names must be non-empty and contain no control characters",
            ));
        }
        if is_reserved_param(&name) {
            return Err(invalid_input(format!(
                "Batch parameter `{name}` is represented by a typed field or is unsupported"
            )));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    pub fn additional_parameters(&self) -> &BTreeMap<String, Value> {
        &self.additional
    }

    fn validate(&self) -> Result<(), AnthropicBatchError> {
        if self.model.trim().is_empty()
            || self.model.len() > 256
            || self.model.chars().any(char::is_control)
        {
            return Err(invalid_input(
                "model must contain 1 to 256 non-control characters",
            ));
        }
        // Anthropic's Batch guide excludes max_tokens: 0 cache-warming calls.
        if self.max_tokens == 0 {
            return Err(invalid_input(
                "Messages Batch requests require max_tokens greater than zero",
            ));
        }
        if self.messages.is_empty() {
            return Err(invalid_input(
                "Messages Batch params require at least one message",
            ));
        }
        if self
            .messages
            .iter()
            .any(|message| !message.content.is_string() && !message.content.is_array())
        {
            return Err(invalid_input(
                "message content must be text or an array of content blocks",
            ));
        }
        if self.additional.keys().any(|name| is_reserved_param(name)) {
            return Err(invalid_input(
                "additional parameters contain a reserved or unsupported field",
            ));
        }
        if self
            .additional
            .get("stream")
            .is_some_and(|value| value == &Value::Bool(true))
        {
            return Err(invalid_input(
                "Messages Batch requests do not support stream: true",
            ));
        }
        Ok(())
    }
}

/// One request keyed by an Anthropic `custom_id`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchRequest {
    pub custom_id: String,
    pub params: AnthropicBatchParams,
}

impl AnthropicBatchRequest {
    pub fn new(
        custom_id: impl Into<String>,
        params: AnthropicBatchParams,
    ) -> Result<Self, AnthropicBatchError> {
        let request = Self {
            custom_id: custom_id.into(),
            params,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), AnthropicBatchError> {
        if !valid_custom_id(&self.custom_id) {
            return Err(invalid_input("custom_id must match [a-zA-Z0-9_-]{1,64}"));
        }
        self.params.validate()
    }
}

/// Inline request collection accepted by `POST /v1/messages/batches`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicBatchInput {
    requests: Vec<AnthropicBatchRequest>,
}

impl AnthropicBatchInput {
    pub fn new(requests: Vec<AnthropicBatchRequest>) -> Result<Self, AnthropicBatchError> {
        if requests.is_empty() || requests.len() > MAX_REQUESTS {
            return Err(invalid_input(
                "Message Batch requires between 1 and 100000 requests",
            ));
        }
        let mut ids = BTreeSet::new();
        for request in &requests {
            request.validate()?;
            if !ids.insert(request.custom_id.as_str()) {
                return Err(invalid_input(
                    "custom_id values must be unique within a batch",
                ));
            }
        }
        let input = Self { requests };
        let encoded = serde_json::to_vec(&SubmitWire {
            requests: &input.requests,
        })
        .map_err(|_| invalid_input("Batch input could not be encoded as JSON"))?;
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err(invalid_input("Message Batch request exceeds 256 MiB"));
        }
        Ok(input)
    }

    pub fn requests(&self) -> &[AnthropicBatchRequest] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn encode(&self) -> Result<Bytes, AnthropicBatchError> {
        let encoded = serde_json::to_vec(&SubmitWire {
            requests: &self.requests,
        })
        .map_err(|_| invalid_input("Batch input could not be encoded as JSON"))?;
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err(invalid_input("Message Batch request exceeds 256 MiB"));
        }
        Ok(Bytes::from(encoded))
    }
}

#[derive(Serialize)]
struct SubmitWire<'a> {
    requests: &'a [AnthropicBatchRequest],
}

/// Opaque, serializable provider job reference. Its private scope prevents
/// accidentally replaying an ID with another provider profile or account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicBatchRef {
    scope: AnthropicBatchScope,
    batch_id: String,
}

impl AnthropicBatchRef {
    pub fn scope(&self) -> &AnthropicBatchScope {
        &self.scope
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicBatchStatus {
    InProgress,
    Canceling,
    Ended,
    Other(String),
}

impl AnthropicBatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Ended)
    }
}

impl Serialize for AnthropicBatchStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = match self {
            Self::InProgress => "in_progress",
            Self::Canceling => "canceling",
            Self::Ended => "ended",
            Self::Other(value) => value,
        };
        serializer.serialize_str(value)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBatchRequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

/// Typed batch state and its provider-native response for forward-compatible
/// inspection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchSnapshot {
    pub reference: AnthropicBatchRef,
    pub status: AnthropicBatchStatus,
    pub request_counts: AnthropicBatchRequestCounts,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
    pub ended_at: Option<String>,
    pub cancel_initiated_at: Option<String>,
    pub archived_at: Option<String>,
    /// Retained as metadata only. Requests are always routed through the
    /// endpoint bound into `reference.scope`.
    pub results_url: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicBatchListOptions {
    limit: Option<u16>,
    before_id: Option<String>,
    after_id: Option<String>,
}

impl AnthropicBatchListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn limit(mut self, limit: u16) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn before_id(mut self, id: impl Into<String>) -> Self {
        self.before_id = Some(id.into());
        self
    }

    pub fn after_id(mut self, id: impl Into<String>) -> Self {
        self.after_id = Some(id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchPage {
    pub batches: Vec<AnthropicBatchSnapshot>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
    pub native: Value,
}

/// Confirmation returned when a completed Message Batch is deleted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBatchDeleted {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnthropicBatchItemResult {
    pub custom_id: String,
    pub outcome: AnthropicBatchItemOutcome,
    pub native: Value,
}

/// Item errors are returned as values, not stream errors. This keeps one
/// failed request from hiding other successful or canceled items.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AnthropicBatchItemOutcome {
    Succeeded { message: Value },
    Errored { error: Value },
    Canceled,
    Expired,
    Other { type_name: String, raw: Value },
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Anthropic Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid Anthropic Batch response: {0}")]
    InvalidResponse(String),
    #[error("invalid Anthropic Batch result stream: {0}")]
    InvalidResult(String),
    #[error("Anthropic Batch returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of Anthropic Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Anthropic Batch {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        reason: String,
    },
}

/// Anthropic Messages Batch client. Every operation performs exactly one HTTP
/// request; result polling and retry decisions remain caller-owned.
#[derive(Clone)]
pub struct AnthropicBatchService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: AnthropicBatchScope,
}

impl<'a> AnthropicBatchService<'a> {
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

    fn validate_account_scope(
        &self,
        request_options: &crate::RequestOptions,
    ) -> Result<(), AnthropicBatchError> {
        if request_options
            .account_scope
            .as_deref()
            .is_some_and(|account| account != self.scope.account_scope())
        {
            return Err(LlmError::PermissionDenied {
                message: "request account scope does not match the native resource scope".into(),
            }
            .into());
        }
        Ok(())
    }

    fn request_credential<'o>(
        &self,
        request_options: &'o crate::RequestOptions,
    ) -> Result<&'o str, AnthropicBatchError> {
        self.validate_account_scope(request_options)?;

        let credential = request_options
            .credential
            .as_ref()
            .map(|value| value.expose_secret().as_str())
            .unwrap_or_default();
        if credential.trim().is_empty() {
            return Err(invalid_input("Anthropic API key must be non-empty"));
        }
        Ok(credential)
    }

    pub fn new(
        http: &'a dyn Transport,
        scope: AnthropicBatchScope,
    ) -> Result<Self, AnthropicBatchError> {
        scope.validate()?;

        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &AnthropicBatchScope {
        &self.scope
    }

    /// Submit the inline `requests` array once. A transport or response-read
    /// failure may occur after Anthropic accepted it; the returned error marks
    /// that outcome as unknown and never retries the submission.
    pub async fn create(
        &self,
        input: &AnthropicBatchInput,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchSnapshot, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        let body = input.encode()?;
        let request = pinned_service
            .authenticated_request("POST", API_PATH, Some(body), request_options)
            .await?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("create", source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let snapshot = pinned_service
            .decode_snapshot(&response.body)
            .map_err(|error| AnthropicBatchError::OutcomeUnknownResponse {
                operation: "create",
                reason: error.to_string(),
            })?;
        Ok(snapshot)
    }

    pub async fn get(
        &self,
        reference: &AnthropicBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchSnapshot, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!("{API_PATH}/{}", encode_path_segment(&reference.batch_id));
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let snapshot = pinned_service.decode_snapshot(&response.body)?;
        pinned_service.ensure_same_batch(reference, &snapshot.reference)?;
        Ok(snapshot)
    }

    /// Return one cursor page. It does not automatically fetch subsequent pages.
    pub async fn list(
        &self,
        options: &AnthropicBatchListOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchPage, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        let limit = options.limit.unwrap_or(20);
        if !(1..=1000).contains(&limit) {
            return Err(invalid_input("list limit must be between 1 and 1000"));
        }
        if options.before_id.is_some() && options.after_id.is_some() {
            return Err(invalid_input(
                "provide at most one of before_id and after_id",
            ));
        }
        for cursor in [options.before_id.as_deref(), options.after_id.as_deref()]
            .into_iter()
            .flatten()
        {
            validate_batch_id(cursor)?;
        }
        let query = list_query(options, limit);
        let path = format!("{API_PATH}?{query}");
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let native: Value = serde_json::from_slice(&response.body)
            .map_err(|error| AnthropicBatchError::InvalidResponse(error.to_string()))?;
        let wire: BatchPageWire = serde_json::from_value(native.clone())
            .map_err(|error| AnthropicBatchError::InvalidResponse(error.to_string()))?;
        let batches = wire
            .data
            .into_iter()
            .map(|value| pinned_service.decode_snapshot_value(value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AnthropicBatchPage {
            batches,
            first_id: wire.first_id,
            last_id: wire.last_id,
            has_more: wire.has_more,
            native,
        })
    }

    /// Request cancellation once. A transport or response-read failure is
    /// reported as an unknown outcome; callers can use `get` to reconcile it.
    pub async fn cancel(
        &self,
        reference: &AnthropicBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchSnapshot, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!(
            "{API_PATH}/{}/cancel",
            encode_path_segment(&reference.batch_id)
        );
        let request = pinned_service
            .authenticated_request("POST", &path, None, request_options)
            .await?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("cancel", source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let snapshot = pinned_service
            .decode_snapshot(&response.body)
            .map_err(|error| AnthropicBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reason: error.to_string(),
            })?;
        if pinned_service
            .ensure_same_batch(reference, &snapshot.reference)
            .is_err()
        {
            return Err(AnthropicBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reason: "Anthropic returned a different batch identity".into(),
            });
        }
        Ok(snapshot)
    }

    /// Delete a completed Message Batch. Anthropic rejects deletion while a
    /// batch is still processing; cancel it first, then confirm a terminal
    /// state with `get`. A lost response is reported as an unknown outcome.
    pub async fn delete(
        &self,
        reference: &AnthropicBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchDeleted, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!("{API_PATH}/{}", encode_path_segment(&reference.batch_id));
        let request = pinned_service
            .authenticated_request("DELETE", &path, None, request_options)
            .await?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("delete", source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let deleted: AnthropicBatchDeleted =
            serde_json::from_slice(&response.body).map_err(|error| {
                AnthropicBatchError::OutcomeUnknownResponse {
                    operation: "delete",
                    reason: error.to_string(),
                }
            })?;
        if deleted.id != reference.batch_id
            || deleted.object_type != "message_batch_deleted"
            || validate_batch_id(&deleted.id).is_err()
        {
            return Err(AnthropicBatchError::OutcomeUnknownResponse {
                operation: "delete",
                reason: "Anthropic returned a different or invalid deleted Batch identity".into(),
            });
        }
        Ok(deleted)
    }

    /// Open the documented results endpoint and parse JSONL records as the
    /// response arrives. No whole-file buffering or local file is created.
    pub async fn stream_results(
        &self,
        reference: &AnthropicBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<AnthropicBatchResultsStream, AnthropicBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!(
            "{API_PATH}/{}/results",
            encode_path_segment(&reference.batch_id)
        );
        let request = pinned_service
            .authenticated_request("GET", &path, None, request_options)
            .await?;
        let response = HttpExecutor::new(pinned_service.http).send(request).await?;
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, None).await?;
            ensure_success(response.status, &response.headers, &response.body)?;
            return Err(AnthropicBatchError::InvalidResponse(
                "non-success Batch result response was not rejected".into(),
            ));
        }
        Ok(AnthropicBatchResultsStream::new(response.body))
    }

    fn decode_snapshot(&self, body: &[u8]) -> Result<AnthropicBatchSnapshot, AnthropicBatchError> {
        let value: Value = serde_json::from_slice(body)
            .map_err(|error| AnthropicBatchError::InvalidResponse(error.to_string()))?;
        self.decode_snapshot_value(value)
    }

    fn decode_snapshot_value(
        &self,
        value: Value,
    ) -> Result<AnthropicBatchSnapshot, AnthropicBatchError> {
        let wire: BatchWire = serde_json::from_value(value.clone())
            .map_err(|error| AnthropicBatchError::InvalidResponse(error.to_string()))?;
        if wire.object_type != "message_batch" {
            return Err(AnthropicBatchError::InvalidResponse(
                "provider returned an object other than message_batch".into(),
            ));
        }
        validate_batch_id(&wire.id).map_err(|_| {
            AnthropicBatchError::InvalidResponse("provider returned an invalid batch ID".into())
        })?;
        Ok(AnthropicBatchSnapshot {
            reference: AnthropicBatchRef {
                scope: self.scope.clone(),
                batch_id: wire.id,
            },
            status: AnthropicBatchStatus::from(wire.processing_status),
            request_counts: wire.request_counts,
            created_at: wire.created_at,
            expires_at: wire.expires_at,
            ended_at: wire.ended_at,
            cancel_initiated_at: wire.cancel_initiated_at,
            archived_at: wire.archived_at,
            results_url: wire.results_url,
            native: value,
        })
    }

    fn validate_reference(&self, reference: &AnthropicBatchRef) -> Result<(), AnthropicBatchError> {
        if reference.scope != self.scope {
            return Err(scope_mismatch());
        }
        self.scope.validate()?;
        validate_batch_id(&reference.batch_id).map_err(|_| scope_mismatch())
    }

    fn ensure_same_batch(
        &self,
        expected: &AnthropicBatchRef,
        actual: &AnthropicBatchRef,
    ) -> Result<(), AnthropicBatchError> {
        if expected.batch_id != actual.batch_id || expected.scope != actual.scope {
            return Err(AnthropicBatchError::InvalidResponse(
                "Anthropic returned a different batch identity".into(),
            ));
        }
        Ok(())
    }

    async fn send_control(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<crate::transport::HttpResponse, AnthropicBatchError> {
        let request = self
            .authenticated_request(method, path, body, request_options)
            .await?;
        Ok(HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await?)
    }

    async fn authenticated_request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<HttpRequest, AnthropicBatchError> {
        let mut request = self.request(method, path, body, request_options)?;
        if let Some(binding) = &self.binding {
            let snapshot = binding.pin()?;
            let profile = snapshot
                .native_profile(binding.profile_name())
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "bound Anthropic Batch profile is unavailable".into(),
                })?;
            let deadline = crate::runtime::Deadline::after(request_options.total_timeout);
            if profile.auth != AuthStrategy::None {
                let authenticator = snapshot
                    .runtime
                    .authenticators
                    .get(&profile.auth)
                    .ok_or_else(|| LlmError::Authentication {
                        message: "bound Anthropic Batch authenticator is not registered".into(),
                    })?;
                deadline
                    .run(authenticator.apply(
                        &mut request,
                        profile,
                        request_options.credential.as_ref(),
                    ))
                    .await??;
            }
            request.timeout = deadline.remaining()?;
        }
        Ok(request)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        request_options: &crate::RequestOptions,
    ) -> Result<HttpRequest, AnthropicBatchError> {
        if !path.starts_with('/') || path.contains('#') {
            return Err(invalid_input("invalid internal Anthropic Batch route"));
        }
        self.validate_account_scope(request_options)?;
        let mut headers = vec![
            ("anthropic-version".into(), API_VERSION.into()),
            ("anthropic-beta".into(), BATCHES_BETA.into()),
        ];
        if self.binding.is_none() {
            headers.push((
                "X-Api-Key".into(),
                self.request_credential(request_options)?.to_owned(),
            ));
        }
        if let Some(workspace_id) = self.scope.workspace_id.as_ref() {
            headers.push(("anthropic-workspace-id".into(), workspace_id.clone()));
        }
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            http1_header_layout: None,
            method: method.into(),
            url: format!("{}{path}", self.scope.api_base_url.trim_end_matches('/')),
            headers,
            body: body.unwrap_or_default(),
            timeout: request_options.total_timeout,
        })
    }
}

impl From<String> for AnthropicBatchStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "in_progress" => Self::InProgress,
            "canceling" => Self::Canceling,
            "ended" => Self::Ended,
            _ => Self::Other(value),
        }
    }
}

/// A stream of provider result records. Per-item errors are `Ok` records with
/// [`AnthropicBatchItemOutcome::Errored`]; only transport or malformed-stream
/// failures terminate the stream with `Err`.
pub struct AnthropicBatchResultsStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    current_chunk: Option<Bytes>,
    chunk_offset: usize,
    line: BytesMut,
    eof: bool,
    terminal: bool,
}

impl AnthropicBatchResultsStream {
    fn new(body: BoxStream<'static, Result<Bytes, LlmError>>) -> Self {
        Self {
            body,
            current_chunk: None,
            chunk_offset: 0,
            line: BytesMut::new(),
            eof: false,
            terminal: false,
        }
    }
}

impl Stream for AnthropicBatchResultsStream {
    type Item = Result<AnthropicBatchItemResult, AnthropicBatchError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminal {
            return Poll::Ready(None);
        }
        loop {
            if let Some(chunk) = this.current_chunk.as_ref() {
                let remaining = &chunk[this.chunk_offset..];
                let newline = remaining.iter().position(|byte| *byte == b'\n');
                let amount = newline.map_or(remaining.len(), |index| index + 1);
                if this.line.len().saturating_add(amount) > MAX_RESULT_LINE_BYTES {
                    this.terminal = true;
                    return Poll::Ready(Some(Err(AnthropicBatchError::InvalidResult(
                        "result line exceeds 16 MiB".into(),
                    ))));
                }
                this.line.extend_from_slice(&remaining[..amount]);
                this.chunk_offset += amount;
                if this.chunk_offset == chunk.len() {
                    this.current_chunk = None;
                    this.chunk_offset = 0;
                }
                if newline.is_some() {
                    let line = std::mem::take(&mut this.line);
                    match decode_result_line(line) {
                        Ok(None) => continue,
                        Ok(Some(item)) => return Poll::Ready(Some(Ok(item))),
                        Err(error) => {
                            this.terminal = true;
                            return Poll::Ready(Some(Err(error)));
                        }
                    }
                }
                continue;
            }

            if this.eof {
                this.terminal = true;
                if this.line.is_empty() {
                    return Poll::Ready(None);
                }
                let line = std::mem::take(&mut this.line);
                return Poll::Ready(decode_result_line(line).transpose());
            }

            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    this.current_chunk = Some(chunk);
                }
                Poll::Ready(Some(Err(error))) => {
                    this.terminal = true;
                    return Poll::Ready(Some(Err(AnthropicBatchError::Llm(error))));
                }
                Poll::Ready(None) => this.eof = true,
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[derive(Deserialize)]
struct BatchWire {
    id: String,
    #[serde(rename = "type")]
    object_type: String,
    processing_status: String,
    #[serde(default)]
    request_counts: AnthropicBatchRequestCounts,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    expires_at: Option<String>,
    #[serde(default)]
    ended_at: Option<String>,
    #[serde(default)]
    cancel_initiated_at: Option<String>,
    #[serde(default)]
    archived_at: Option<String>,
    #[serde(default)]
    results_url: Option<String>,
}

#[derive(Deserialize)]
struct BatchPageWire {
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
struct ResultLineWire {
    custom_id: String,
    result: Value,
}

fn decode_result_line(
    mut line: BytesMut,
) -> Result<Option<AnthropicBatchItemResult>, AnthropicBatchError> {
    if line.last() == Some(&b'\n') {
        line.truncate(line.len() - 1);
    }
    if line.last() == Some(&b'\r') {
        line.truncate(line.len() - 1);
    }
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let native: Value = serde_json::from_slice(&line)
        .map_err(|error| AnthropicBatchError::InvalidResult(error.to_string()))?;
    let wire: ResultLineWire = serde_json::from_value(native.clone())
        .map_err(|error| AnthropicBatchError::InvalidResult(error.to_string()))?;
    if !valid_custom_id(&wire.custom_id) {
        return Err(AnthropicBatchError::InvalidResult(
            "result contains an invalid custom_id".into(),
        ));
    }
    let result_type = wire
        .result
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| AnthropicBatchError::InvalidResult("result.type is missing".into()))?;
    let outcome = match result_type {
        "succeeded" => {
            let message = wire.result.get("message").cloned().ok_or_else(|| {
                AnthropicBatchError::InvalidResult("success message is missing".into())
            })?;
            AnthropicBatchItemOutcome::Succeeded { message }
        }
        "errored" => {
            let error = wire.result.get("error").cloned().ok_or_else(|| {
                AnthropicBatchError::InvalidResult("item error is missing".into())
            })?;
            AnthropicBatchItemOutcome::Errored { error }
        }
        "canceled" => AnthropicBatchItemOutcome::Canceled,
        "expired" => AnthropicBatchItemOutcome::Expired,
        other => AnthropicBatchItemOutcome::Other {
            type_name: other.into(),
            raw: wire.result,
        },
    };
    Ok(Some(AnthropicBatchItemResult {
        custom_id: wire.custom_id,
        outcome,
        native,
    }))
}

fn ensure_success(
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<(), AnthropicBatchError> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    let request_id = headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("request-id") || name.eq_ignore_ascii_case("x-request-id")
        })
        .map(|(_, value)| value.clone());
    let body = serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()));
    Err(AnthropicBatchError::Provider {
        status,
        request_id,
        body,
    })
}

fn normalize_api_base_url(value: &str) -> Result<String, AnthropicBatchError> {
    let mut url = Url::parse(value)
        .map_err(|_| invalid_input("API base URL must be an absolute HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_input(
            "API base URL must use HTTPS and cannot contain credentials, a query, or a fragment",
        ));
    }
    let path = url.path().trim_end_matches('/').to_string();
    url.set_path(if path.is_empty() { "/" } else { &path });
    let mut normalized = url.to_string();
    if normalized.ends_with('/') {
        normalized.pop();
    }
    Ok(normalized)
}

fn list_query(options: &AnthropicBatchListOptions, limit: u16) -> String {
    let mut query = url::form_urlencoded::Serializer::new(String::new());
    query.append_pair("limit", &limit.to_string());
    if let Some(id) = options.before_id.as_deref() {
        query.append_pair("before_id", id);
    }
    if let Some(id) = options.after_id.as_deref() {
        query.append_pair("after_id", id);
    }
    query.finish()
}

fn encode_path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_custom_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn valid_parameter_name(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn validate_batch_id(value: &str) -> Result<(), AnthropicBatchError> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return Err(invalid_input("batch ID is empty or invalid"));
    }
    Ok(())
}

fn is_reserved_param(name: &str) -> bool {
    matches!(name, "model" | "max_tokens" | "messages" | "speed")
}

fn scope_mismatch() -> AnthropicBatchError {
    LlmError::PermissionDenied {
        message: "Anthropic Batch reference belongs to another provider, profile, endpoint, or account scope".into(),
    }
    .into()
}

fn invalid_input(message: impl Into<String>) -> AnthropicBatchError {
    AnthropicBatchError::InvalidInput(message.into())
}

fn mutation_transport_error(operation: &'static str, source: LlmError) -> AnthropicBatchError {
    if matches!(
        &source,
        LlmError::Transport { .. } | LlmError::TransportTimeout { .. }
    ) {
        AnthropicBatchError::OutcomeUnknown { operation, source }
    } else {
        AnthropicBatchError::Llm(source)
    }
}
