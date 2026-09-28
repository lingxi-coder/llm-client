//! xAI's native Batch API over `/v1/batches`.
//!
//! The documented flows create a batch from inline request envelopes or a
//! caller-uploaded JSONL file, then read state and paged results. This module
//! does not upload or download files and does not automatically poll,
//! paginate, retry, or follow provider URLs.

use crate::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{LlmError, ProtocolFamily, ProviderId},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::time::SystemTime;
use url::Url;

const BATCHES_PATH: &str = "/batches";
const MAX_REQUESTS_PER_ADD: usize = 100_000;
const MAX_INDIVIDUAL_REQUEST_BYTES: usize = 25 * 1024 * 1024;
const MAX_ADD_BODY_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONTROL_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_PAGE_SIZE: u16 = 1000;
const MAX_XAI_BATCH_FILE_BYTES: u64 = 50_000_000;

/// Identity for keeping xAI batch handles within one provider profile,
/// endpoint/region, and caller-defined account scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiBatchScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    api_base_url: String,
    endpoint_fingerprint: String,
}

impl XaiBatchScope {
    /// Construct a scope for an xAI REST API root such as
    /// `https://api.x.ai/v1` or `https://us.api.x.ai/v1`.
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        api_base_url: impl AsRef<str>,
    ) -> Result<Self, XaiBatchError> {
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
            provider_id: ProviderId::from("xai"),
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

    fn validate(&self) -> Result<(), XaiBatchError> {
        let normalized = normalize_api_base_url(&self.api_base_url)?;
        if self.provider_id.as_str() != "xai"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || normalized != self.api_base_url
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.api_base_url)
        {
            return Err(invalid_input("xAI Batch scope identity is invalid"));
        }
        Ok(())
    }
}

/// Scoped, caller-managed xAI Files API reference for a JSONL Batch input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiBatchInputFileRef {
    scope: XaiBatchScope,
    file_id: String,
    filename: String,
    media_type: String,
    size_bytes: u64,
    expires_at: Option<String>,
    processing_status: Option<String>,
}

impl XaiBatchInputFileRef {
    /// Bind a file uploaded through `FileService` with `FilePurpose::Batch` to
    /// the exact xAI account and endpoint that will submit the Batch.
    pub fn from_uploaded_file(
        scope: &XaiBatchScope,
        file: &ProviderFileRef,
    ) -> Result<Self, XaiBatchError> {
        scope.validate()?;
        let filename = file
            .filename
            .as_deref()
            .filter(|filename| filename.ends_with(".jsonl"))
            .ok_or_else(|| invalid_input("xAI Batch input must be an uploaded .jsonl file"))?;
        let media_type = file
            .media_type
            .as_deref()
            .filter(|media_type| is_jsonl_media_type(media_type))
            .ok_or_else(|| invalid_input("xAI Batch input must have a JSONL media type"))?;
        let size_bytes = file
            .size_bytes
            .filter(|size| *size > 0 && *size <= MAX_XAI_BATCH_FILE_BYTES)
            .ok_or_else(|| {
                invalid_input(format!(
                    "xAI Batch input size must be between 1 and {MAX_XAI_BATCH_FILE_BYTES} bytes"
                ))
            })?;
        if file.provider_id.as_str() != "xai"
            || !matches!(
                file.protocol,
                ProtocolFamily::OpenAiChat | ProtocolFamily::OpenAiResponses
            )
            || file.profile_name != scope.profile_name
            || file.endpoint_fingerprint != scope.endpoint_fingerprint
            || file.account_scope.as_deref() != Some(scope.account_scope.as_str())
        {
            return Err(scope_mismatch());
        }
        if !valid_identity(&file.file_id) {
            return Err(invalid_input("xAI Batch input file ID is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            file_id: file.file_id.clone(),
            filename: filename.to_owned(),
            media_type: media_type.to_owned(),
            size_bytes,
            expires_at: file.expires_at.clone(),
            processing_status: file.processing_status.clone(),
        })
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn scope(&self) -> &XaiBatchScope {
        &self.scope
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn expires_at(&self) -> Option<&str> {
        self.expires_at.as_deref()
    }

    pub fn processing_status(&self) -> Option<&str> {
        self.processing_status.as_deref()
    }
}

/// Body for the documented `POST /batches` create operation. File-based
/// batches are sealed by xAI after creation and cannot accept inline requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct XaiBatchCreateRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_file_id: Option<String>,
    #[serde(skip)]
    input_file: Option<XaiBatchInputFileRef>,
}

impl XaiBatchCreateRequest {
    pub fn new(name: impl Into<String>) -> Result<Self, XaiBatchError> {
        let name = name.into();
        if !valid_identity(&name) {
            return Err(invalid_input("batch name must be non-empty"));
        }
        Ok(Self {
            name,
            input_file_id: None,
            input_file: None,
        })
    }

    pub fn with_input_file(mut self, input_file: XaiBatchInputFileRef) -> Self {
        self.input_file_id = Some(input_file.file_id.clone());
        self.input_file = Some(input_file);
        self
    }
}

/// Messages-compatible request data carried by the native `chat_get_completion`
/// batch request variant. Unknown Messages fields are retained in `additional`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchChatCompletion {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl XaiBatchChatCompletion {
    pub fn new(model: impl Into<String>, messages: Vec<Value>) -> Result<Self, XaiBatchError> {
        let value = Self {
            model: model.into(),
            messages,
            additional: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, XaiBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name) || matches!(name.as_str(), "model" | "messages") {
            return Err(invalid_input(
                "chat parameter name is empty, invalid, or already represented by a typed field",
            ));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        validate_text_batch_model(&self.model)?;
        if self.messages.is_empty() || self.messages.iter().any(|message| !message.is_object()) {
            return Err(invalid_input(
                "chat batch requests require at least one message object",
            ));
        }
        Ok(())
    }
}

/// Typed core for the documented `responses` request variant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchResponses {
    pub model: String,
    pub input: Value,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl XaiBatchResponses {
    pub fn new(model: impl Into<String>, input: Value) -> Result<Self, XaiBatchError> {
        let value = Self {
            model: model.into(),
            input,
            additional: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, XaiBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name) || matches!(name.as_str(), "model" | "input") {
            return Err(invalid_input(
                "Responses parameter name is empty, invalid, or already represented by a typed field",
            ));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        validate_text_batch_model(&self.model)?;
        if self.input.is_null() {
            return Err(invalid_input("Responses batch input cannot be null"));
        }
        Ok(())
    }
}

/// Core prompt/model fields shared by xAI's documented image and video batch
/// variants. Additional provider parameters remain available through the
/// explicit map instead of being guessed by this client.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchMediaPrompt {
    pub model: String,
    pub prompt: String,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl XaiBatchMediaPrompt {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Result<Self, XaiBatchError> {
        let value = Self {
            model: model.into(),
            prompt: prompt.into(),
            additional: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, XaiBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name)
            || matches!(name.as_str(), "model" | "prompt" | "video" | "duration")
        {
            return Err(invalid_input(
                "media parameter name is empty, invalid, or already represented by a typed field",
            ));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        validate_model(&self.model)?;
        if !valid_identity(&self.prompt) {
            return Err(invalid_input("media prompt must be non-empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct XaiBatchImageInput {
    pub url: String,
    #[serde(rename = "type")]
    pub input_type: String,
}

impl XaiBatchImageInput {
    pub fn image_url(url: impl Into<String>) -> Result<Self, XaiBatchError> {
        let value = Self {
            url: url.into(),
            input_type: "image_url".into(),
        };
        if !valid_identity(&value.url) {
            return Err(invalid_input("image URL must be non-empty"));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct XaiBatchVideoInput {
    pub url: String,
}

impl XaiBatchVideoInput {
    pub fn new(url: impl Into<String>) -> Result<Self, XaiBatchError> {
        let url = url.into();
        if !valid_identity(&url) {
            return Err(invalid_input("video URL must be non-empty"));
        }
        Ok(Self { url })
    }
}

/// One documented native xAI request envelope. Each request has exactly one
/// request-kind variant, matching the API's `batch_request` union.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchRequest {
    pub batch_request_id: String,
    pub batch_request: XaiBatchRequestBody,
}

impl XaiBatchRequest {
    pub fn new(
        batch_request_id: impl Into<String>,
        batch_request: XaiBatchRequestBody,
    ) -> Result<Self, XaiBatchError> {
        let value = Self {
            batch_request_id: batch_request_id.into(),
            batch_request,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        if !valid_identity(&self.batch_request_id) {
            return Err(invalid_input("batch_request_id must be non-empty"));
        }
        self.batch_request.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiBatchRequestBody {
    ChatGetCompletion(XaiBatchChatCompletion),
    Responses(XaiBatchResponses),
    ImageGeneration(XaiBatchMediaPrompt),
    ImageEdit(XaiBatchImageEdit),
    VideoGeneration(XaiBatchVideoGeneration),
    VideoExtension(XaiBatchVideoExtension),
}

impl XaiBatchRequestBody {
    fn validate(&self) -> Result<(), XaiBatchError> {
        match self {
            Self::ChatGetCompletion(request) => request.validate(),
            Self::Responses(request) => request.validate(),
            Self::ImageGeneration(request) => request.validate(),
            Self::ImageEdit(request) => request.validate(),
            Self::VideoGeneration(request) => request.validate(),
            Self::VideoExtension(request) => request.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchImageEdit {
    pub model: String,
    pub prompt: String,
    pub image: XaiBatchImageInput,
    #[serde(flatten)]
    additional: BTreeMap<String, Value>,
}

impl XaiBatchImageEdit {
    pub fn new(
        model: impl Into<String>,
        prompt: impl Into<String>,
        image: XaiBatchImageInput,
    ) -> Result<Self, XaiBatchError> {
        let value = Self {
            model: model.into(),
            prompt: prompt.into(),
            image,
            additional: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, XaiBatchError> {
        let name = name.into();
        if !valid_parameter_name(&name) || matches!(name.as_str(), "model" | "prompt" | "image") {
            return Err(invalid_input(
                "image-edit parameter name is empty, invalid, or already represented by a typed field",
            ));
        }
        self.additional.insert(name, value);
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        validate_model(&self.model)?;
        if !valid_identity(&self.prompt) {
            return Err(invalid_input("image edit prompt must be non-empty"));
        }
        if !valid_identity(&self.image.url) || self.image.input_type != "image_url" {
            return Err(invalid_input(
                "image edit requires a documented image_url input",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchVideoGeneration {
    #[serde(flatten)]
    pub request: XaiBatchMediaPrompt,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<XaiBatchVideoInput>,
}

impl XaiBatchVideoGeneration {
    pub fn new(
        request: XaiBatchMediaPrompt,
        video: Option<XaiBatchVideoInput>,
    ) -> Result<Self, XaiBatchError> {
        let value = Self { request, video };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        self.request.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchVideoExtension {
    #[serde(flatten)]
    pub request: XaiBatchMediaPrompt,
    pub video: XaiBatchVideoInput,
    pub duration: u16,
}

impl XaiBatchVideoExtension {
    pub fn new(
        request: XaiBatchMediaPrompt,
        video: XaiBatchVideoInput,
        duration: u16,
    ) -> Result<Self, XaiBatchError> {
        let value = Self {
            request,
            video,
            duration,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), XaiBatchError> {
        self.request.validate()?;
        if self.duration == 0 {
            return Err(invalid_input("video extension duration must be positive"));
        }
        Ok(())
    }
}

/// A bounded inline request set for one `POST /batches/{id}/requests` call.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiBatchInput {
    requests: Vec<XaiBatchRequest>,
}

impl XaiBatchInput {
    pub fn new(requests: Vec<XaiBatchRequest>) -> Result<Self, XaiBatchError> {
        if requests.is_empty() || requests.len() > MAX_REQUESTS_PER_ADD {
            return Err(invalid_input(
                "an add-requests call requires between 1 and 100000 requests",
            ));
        }
        let mut ids = BTreeSet::new();
        for request in &requests {
            request.validate()?;
            if !ids.insert(request.batch_request_id.as_str()) {
                return Err(invalid_input(
                    "batch_request_id values must be unique within this add call",
                ));
            }
            let size = serde_json::to_vec(request)
                .map_err(|_| invalid_input("a Batch request could not be serialized"))?
                .len();
            if size > MAX_INDIVIDUAL_REQUEST_BYTES {
                return Err(invalid_input(
                    "an individual Batch request exceeds xAI's 25 MiB payload limit",
                ));
            }
        }
        let input = Self { requests };
        let bytes = input.encode()?;
        if bytes.len() > MAX_ADD_BODY_BYTES {
            return Err(invalid_input(
                "inline add-requests JSON exceeds this client’s 256 MiB memory bound",
            ));
        }
        Ok(input)
    }

    pub fn requests(&self) -> &[XaiBatchRequest] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn encode(&self) -> Result<Bytes, XaiBatchError> {
        let bytes = serde_json::to_vec(&AddRequestsWire {
            batch_requests: &self.requests,
        })
        .map_err(|_| invalid_input("Batch requests could not be encoded as JSON"))?;
        Ok(Bytes::from(bytes))
    }
}

#[derive(Serialize)]
struct AddRequestsWire<'a> {
    batch_requests: &'a [XaiBatchRequest],
}

/// Scope-carrying reference to an xAI batch ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiBatchRef {
    scope: XaiBatchScope,
    batch_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input_file_id: Option<String>,
}

impl XaiBatchRef {
    pub fn scope(&self) -> &XaiBatchScope {
        &self.scope
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn input_file_id(&self) -> Option<&str> {
        self.input_file_id.as_deref()
    }

    pub fn is_file_based(&self) -> bool {
        self.input_file_id.is_some()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct XaiBatchState {
    #[serde(default)]
    pub num_requests: u64,
    #[serde(default)]
    pub num_pending: u64,
    #[serde(default)]
    pub num_success: u64,
    #[serde(default)]
    pub num_error: u64,
    #[serde(default)]
    pub num_cancelled: u64,
}

impl XaiBatchState {
    /// xAI documents `num_pending == 0` as finished after requests were added.
    pub fn is_terminal(&self) -> bool {
        self.num_requests > 0 && self.num_pending == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchSnapshot {
    pub reference: XaiBatchRef,
    pub name: Option<String>,
    pub state: XaiBatchState,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
    pub canceled_at: Option<String>,
    pub cancel_message: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XaiBatchPageOptions {
    limit: Option<u16>,
    pagination_token: Option<String>,
}

impl XaiBatchPageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn limit(mut self, limit: u16) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn pagination_token(mut self, token: impl Into<String>) -> Self {
        self.pagination_token = Some(token.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchListPage {
    pub batches: Vec<XaiBatchSnapshot>,
    pub pagination_token: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchRequestMetadata {
    pub batch_request_id: String,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub state: Option<String>,
    pub created_at: Option<String>,
    pub finished_at: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchRequestMetadataPage {
    pub requests: Vec<XaiBatchRequestMetadata>,
    pub pagination_token: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchResult {
    pub batch_request_id: String,
    pub outcome: XaiBatchResultOutcome,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum XaiBatchResultOutcome {
    Succeeded {
        response: Value,
    },
    Failed {
        error_message: Option<String>,
        error: Option<Value>,
    },
    Canceled,
    Pending,
    Other {
        state: Option<String>,
        raw: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct XaiBatchResultsPage {
    pub results: Vec<XaiBatchResult>,
    pub pagination_token: Option<String>,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum XaiBatchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid xAI Batch input: {0}")]
    InvalidInput(String),
    #[error("invalid xAI Batch response: {0}")]
    InvalidResponse(String),
    #[error("xAI Batch returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of xAI Batch {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        reference: Option<Box<XaiBatchRef>>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of xAI Batch {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        reference: Option<Box<XaiBatchRef>>,
        reason: String,
    },
}

/// Direct xAI Batch API client. Each method makes one request except `create`
/// and `submit`, which mirror the provider's separate create-container and add
/// requests operations. The service never retries or auto-paginates.
#[derive(Clone)]
pub struct XaiBatchService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: XaiBatchScope,
}

impl<'a> XaiBatchService<'a> {
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
    ) -> Result<&'o str, XaiBatchError> {
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
            return Err(invalid_input("xAI API key must be non-empty"));
        }
        Ok(credential)
    }

    pub fn new(http: &'a dyn Transport, scope: XaiBatchScope) -> Result<Self, XaiBatchError> {
        scope.validate()?;

        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &XaiBatchScope {
        &self.scope
    }

    /// Create the provider-side batch container. Request submission is a
    /// separate documented `submit` operation.
    pub async fn create(
        &self,
        request: &XaiBatchCreateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchSnapshot, XaiBatchError> {
        let pinned_service = self.pin()?;
        if let Some(input_file) = &request.input_file {
            pinned_service.validate_input_file(input_file)?;
            if request.input_file_id.as_deref() != Some(input_file.file_id.as_str()) {
                return Err(invalid_input(
                    "Batch request input_file_id does not match its scoped file reference",
                ));
            }
        } else if request.input_file_id.is_some() {
            return Err(invalid_input(
                "Batch input_file_id requires a scoped xAI file reference",
            ));
        }
        let body = serde_json::to_vec(request)
            .map_err(|_| invalid_input("batch create request could not be encoded"))?;
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(
                pinned_service.request(
                    "POST",
                    BATCHES_PATH,
                    Some(Bytes::from(body)),
                    request_options,
                )?,
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| mutation_transport_error("create", None, source))?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let mut snapshot = pinned_service
            .decode_snapshot(&response.body)
            .map_err(|error| XaiBatchError::OutcomeUnknownResponse {
                operation: "create",
                reference: None,
                reason: error.to_string(),
            })?;
        if let Some(input_file) = &request.input_file {
            if snapshot
                .reference
                .input_file_id
                .as_deref()
                .is_some_and(|actual| actual != input_file.file_id)
            {
                return Err(XaiBatchError::OutcomeUnknownResponse {
                    operation: "create",
                    reference: Some(Box::new(snapshot.reference)),
                    reason: "provider returned a different input_file_id".into(),
                });
            }
            snapshot.reference.input_file_id = Some(input_file.file_id.clone());
        }
        Ok(snapshot)
    }

    /// Add inline request envelopes to an existing batch. xAI documents a
    /// successful empty response; the caller can reconcile an unknown result
    /// with `get`, request metadata, or paged results. This method never
    /// replays the add operation.
    pub async fn submit(
        &self,
        reference: &XaiBatchRef,
        input: &XaiBatchInput,
        request_options: &crate::RequestOptions,
    ) -> Result<usize, XaiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        if reference.is_file_based() {
            return Err(invalid_input(
                "xAI file-based batches are sealed and cannot accept inline requests",
            ));
        }
        let body = input.encode()?;
        let path = format!(
            "{BATCHES_PATH}/{}/requests",
            encode_path_segment(&reference.batch_id)
        );
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(
                pinned_service.request("POST", &path, Some(body), request_options)?,
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| {
                mutation_transport_error("add requests", Some(reference.clone()), source)
            })?;
        ensure_success(response.status, &response.headers, &response.body)?;
        Ok(input.len())
    }

    pub async fn get(
        &self,
        reference: &XaiBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchSnapshot, XaiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!(
            "{BATCHES_PATH}/{}",
            encode_path_segment(&reference.batch_id)
        );
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let mut snapshot = pinned_service.decode_snapshot(&response.body)?;
        pinned_service.ensure_same_batch(reference, &snapshot.reference)?;
        if snapshot.reference.input_file_id.is_none() {
            snapshot.reference.input_file_id = reference.input_file_id.clone();
        }
        Ok(snapshot)
    }

    /// Fetch one page from the documented team-wide batch list.
    pub async fn list(
        &self,
        options: &XaiBatchPageOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchListPage, XaiBatchError> {
        let pinned_service = self.pin()?;
        let query = page_query(options)?;
        let path = if query.is_empty() {
            BATCHES_PATH.to_string()
        } else {
            format!("{BATCHES_PATH}?{query}")
        };
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let native: Value = decode_json_value(&response.body)?;
        let wire: BatchListWire = decode_value(native.clone())?;
        let batches = wire
            .batches
            .into_iter()
            .map(|batch| pinned_service.decode_snapshot_value(batch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(XaiBatchListPage {
            batches,
            pagination_token: wire.pagination_token,
            native,
        })
    }

    /// List per-request lifecycle metadata for one batch page.
    pub async fn list_requests(
        &self,
        reference: &XaiBatchRef,
        options: &XaiBatchPageOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchRequestMetadataPage, XaiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let query = page_query(options)?;
        let mut path = format!(
            "{BATCHES_PATH}/{}/requests",
            encode_path_segment(&reference.batch_id)
        );
        if !query.is_empty() {
            path.push('?');
            path.push_str(&query);
        }
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let native: Value = decode_json_value(&response.body)?;
        let wire: RequestMetadataPageWire = decode_value(native.clone())?;
        let requests = wire
            .batch_request_metadata
            .into_iter()
            .map(decode_request_metadata)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(XaiBatchRequestMetadataPage {
            requests,
            pagination_token: wire.pagination_token,
            native,
        })
    }

    /// Read one results page. Results are JSON pages, not a streaming file;
    /// pass the returned token to a later call to fetch the next page.
    pub async fn results(
        &self,
        reference: &XaiBatchRef,
        options: &XaiBatchPageOptions,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchResultsPage, XaiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let query = page_query(options)?;
        let mut path = format!(
            "{BATCHES_PATH}/{}/results",
            encode_path_segment(&reference.batch_id)
        );
        if !query.is_empty() {
            path.push('?');
            path.push_str(&query);
        }
        let response = pinned_service
            .send_control("GET", &path, None, request_options)
            .await?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let native: Value = decode_json_value(&response.body)?;
        let wire: ResultsPageWire = decode_value(native.clone())?;
        let results = wire
            .results
            .into_iter()
            .map(decode_result)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(XaiBatchResultsPage {
            results,
            pagination_token: wire.pagination_token,
            native,
        })
    }

    pub async fn cancel(
        &self,
        reference: &XaiBatchRef,
        request_options: &crate::RequestOptions,
    ) -> Result<XaiBatchSnapshot, XaiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        let path = format!(
            "{BATCHES_PATH}/{}:cancel",
            encode_path_segment(&reference.batch_id)
        );
        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(
                pinned_service.request("POST", &path, None, request_options)?,
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| {
                mutation_transport_error("cancel", Some(reference.clone()), source)
            })?;
        ensure_success(response.status, &response.headers, &response.body)?;
        let mut snapshot = pinned_service
            .decode_snapshot(&response.body)
            .map_err(|error| XaiBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reference: Some(Box::new(reference.clone())),
                reason: error.to_string(),
            })?;
        if pinned_service
            .ensure_same_batch(reference, &snapshot.reference)
            .is_err()
        {
            return Err(XaiBatchError::OutcomeUnknownResponse {
                operation: "cancel",
                reference: Some(Box::new(reference.clone())),
                reason: "xAI returned a different batch identity".into(),
            });
        }
        if snapshot.reference.input_file_id.is_none() {
            snapshot.reference.input_file_id = reference.input_file_id.clone();
        }
        Ok(snapshot)
    }

    fn decode_snapshot(&self, body: &[u8]) -> Result<XaiBatchSnapshot, XaiBatchError> {
        let value = decode_json_value(body)?;
        self.decode_snapshot_value(value)
    }

    fn decode_snapshot_value(&self, value: Value) -> Result<XaiBatchSnapshot, XaiBatchError> {
        let wire: BatchWire = decode_value(value.clone())?;
        validate_batch_id(&wire.batch_id).map_err(|_| {
            XaiBatchError::InvalidResponse("provider returned an invalid batch ID".into())
        })?;
        if wire
            .input_file_id
            .as_deref()
            .is_some_and(|file_id| !valid_identity(file_id))
        {
            return Err(XaiBatchError::InvalidResponse(
                "provider returned an invalid input file ID".into(),
            ));
        }
        Ok(XaiBatchSnapshot {
            reference: XaiBatchRef {
                scope: self.scope.clone(),
                batch_id: wire.batch_id,
                input_file_id: wire.input_file_id,
            },
            name: wire.name,
            state: wire.state,
            created_at: wire.create_time,
            expires_at: wire.expire_time,
            canceled_at: wire.cancel_time,
            cancel_message: wire.cancel_by_xai_message,
            native: value,
        })
    }

    fn validate_reference(&self, reference: &XaiBatchRef) -> Result<(), XaiBatchError> {
        if reference.scope != self.scope {
            return Err(scope_mismatch());
        }
        self.scope.validate()?;
        validate_batch_id(&reference.batch_id).map_err(|_| scope_mismatch())?;
        if reference
            .input_file_id
            .as_deref()
            .is_some_and(|file_id| !valid_identity(file_id))
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_input_file(&self, reference: &XaiBatchInputFileRef) -> Result<(), XaiBatchError> {
        if reference.scope != self.scope {
            return Err(scope_mismatch());
        }
        self.scope.validate()?;
        if !valid_identity(&reference.file_id)
            || !reference.filename.ends_with(".jsonl")
            || !is_jsonl_media_type(&reference.media_type)
            || reference.size_bytes == 0
            || reference.size_bytes > MAX_XAI_BATCH_FILE_BYTES
        {
            return Err(invalid_input("xAI Batch input file reference is invalid"));
        }
        crate::files::validate_file_expiration_at(
            reference.expires_at.as_deref(),
            SystemTime::now(),
        )?;
        Ok(())
    }

    fn ensure_same_batch(
        &self,
        expected: &XaiBatchRef,
        actual: &XaiBatchRef,
    ) -> Result<(), XaiBatchError> {
        if expected.batch_id != actual.batch_id
            || expected.scope != actual.scope
            || expected
                .input_file_id
                .as_ref()
                .zip(actual.input_file_id.as_ref())
                .is_some_and(|(expected, actual)| expected != actual)
        {
            return Err(XaiBatchError::InvalidResponse(
                "xAI returned a different batch identity".into(),
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
    ) -> Result<HttpResponse, XaiBatchError> {
        Ok(HttpExecutor::new(self.http)
            .execute_bounded(
                self.request(method, path, body, request_options)?,
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
    ) -> Result<HttpRequest, XaiBatchError> {
        if !path.starts_with('/') || path.contains('#') {
            return Err(invalid_input("invalid internal xAI Batch route"));
        }
        let mut headers = vec![(
            "Authorization".into(),
            format!("Bearer {}", self.request_credential(request_options)?),
        )];
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        Ok(HttpRequest {
            method: method.into(),
            url: format!("{}{path}", self.scope.api_base_url.trim_end_matches('/')),
            headers,
            body: body.unwrap_or_default(),
            timeout: request_options.total_timeout,
        })
    }
}

#[derive(Deserialize)]
struct BatchWire {
    batch_id: String,
    #[serde(default)]
    input_file_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    state: XaiBatchState,
    #[serde(default)]
    create_time: Option<String>,
    #[serde(default, alias = "expires_at")]
    expire_time: Option<String>,
    #[serde(default)]
    cancel_time: Option<String>,
    #[serde(default)]
    cancel_by_xai_message: Option<String>,
}

#[derive(Deserialize)]
struct BatchListWire {
    #[serde(default)]
    batches: Vec<Value>,
    #[serde(default)]
    pagination_token: Option<String>,
}

#[derive(Deserialize)]
struct RequestMetadataPageWire {
    #[serde(default)]
    batch_request_metadata: Vec<Value>,
    #[serde(default)]
    pagination_token: Option<String>,
}

#[derive(Deserialize)]
struct ResultsPageWire {
    #[serde(default)]
    results: Vec<Value>,
    #[serde(default)]
    pagination_token: Option<String>,
}

fn decode_request_metadata(value: Value) -> Result<XaiBatchRequestMetadata, XaiBatchError> {
    #[derive(Deserialize)]
    struct Wire {
        batch_request_id: String,
        #[serde(default)]
        endpoint: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        state: Option<String>,
        #[serde(default)]
        create_time: Option<String>,
        #[serde(default)]
        finish_time: Option<String>,
    }
    let wire: Wire = decode_value(value.clone())?;
    if !valid_identity(&wire.batch_request_id) {
        return Err(XaiBatchError::InvalidResponse(
            "request metadata contains an invalid batch_request_id".into(),
        ));
    }
    Ok(XaiBatchRequestMetadata {
        batch_request_id: wire.batch_request_id,
        endpoint: wire.endpoint,
        model: wire.model,
        state: wire.state,
        created_at: wire.create_time,
        finished_at: wire.finish_time,
        native: value,
    })
}

fn decode_result(value: Value) -> Result<XaiBatchResult, XaiBatchError> {
    #[derive(Deserialize)]
    struct Wire {
        batch_request_id: String,
        #[serde(default)]
        batch_result: Option<Value>,
        #[serde(default)]
        error_message: Option<String>,
        #[serde(default)]
        state: Option<String>,
    }
    let wire: Wire = decode_value(value.clone())?;
    if !valid_identity(&wire.batch_request_id) {
        return Err(XaiBatchError::InvalidResponse(
            "result contains an invalid batch_request_id".into(),
        ));
    }
    let batch_result = wire.batch_result.unwrap_or(Value::Null);
    let response = batch_result
        .get("response")
        .filter(|value| !value.is_null());
    let error = batch_result.get("error").cloned();
    let outcome = if let Some(response) = response {
        XaiBatchResultOutcome::Succeeded {
            response: response.clone(),
        }
    } else if wire.error_message.is_some()
        || error.is_some()
        || wire.state.as_deref() == Some("failed")
    {
        XaiBatchResultOutcome::Failed {
            error_message: wire.error_message,
            error,
        }
    } else {
        match wire.state.as_deref() {
            Some("cancelled") | Some("canceled") => XaiBatchResultOutcome::Canceled,
            Some("pending") => XaiBatchResultOutcome::Pending,
            _ => XaiBatchResultOutcome::Other {
                state: wire.state,
                raw: batch_result,
            },
        }
    };
    Ok(XaiBatchResult {
        batch_request_id: wire.batch_request_id,
        outcome,
        native: value,
    })
}

fn decode_json_value(body: &[u8]) -> Result<Value, XaiBatchError> {
    serde_json::from_slice(body).map_err(|error| XaiBatchError::InvalidResponse(error.to_string()))
}

fn decode_value<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, XaiBatchError> {
    serde_json::from_value(value).map_err(|error| XaiBatchError::InvalidResponse(error.to_string()))
}

fn ensure_success(
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<(), XaiBatchError> {
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
    Err(XaiBatchError::Provider {
        status,
        request_id,
        body,
    })
}

fn normalize_api_base_url(value: &str) -> Result<String, XaiBatchError> {
    let mut url = Url::parse(value)
        .map_err(|_| invalid_input("API base URL must be an absolute HTTPS `/v1` URL"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.path().trim_end_matches('/').ends_with("/v1")
    {
        return Err(invalid_input(
            "API base URL must use HTTPS, end in `/v1`, and contain no credentials, query, or fragment",
        ));
    }
    let path = url.path().trim_end_matches('/').to_string();
    url.set_path(&path);
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn page_query(options: &XaiBatchPageOptions) -> Result<String, XaiBatchError> {
    if options
        .limit
        .is_some_and(|limit| !(1..=MAX_PAGE_SIZE).contains(&limit))
    {
        return Err(invalid_input("page limit must be between 1 and 1000"));
    }
    if options
        .pagination_token
        .as_deref()
        .is_some_and(|token| !valid_identity(token))
    {
        return Err(invalid_input("pagination token is empty or invalid"));
    }
    let mut query = url::form_urlencoded::Serializer::new(String::new());
    if let Some(limit) = options.limit {
        query.append_pair("limit", &limit.to_string());
    }
    if let Some(token) = options.pagination_token.as_deref() {
        query.append_pair("pagination_token", token);
    }
    Ok(query.finish())
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

fn is_jsonl_media_type(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "application/x-ndjson" | "application/jsonl" | "application/json"
    )
}

fn valid_parameter_name(value: &str) -> bool {
    valid_identity(value)
}

fn validate_model(model: &str) -> Result<(), XaiBatchError> {
    if !valid_identity(model) {
        return Err(invalid_input("model must be non-empty"));
    }
    Ok(())
}

fn validate_text_batch_model(model: &str) -> Result<(), XaiBatchError> {
    validate_model(model)?;
    // These published model cards explicitly say Batch API is unavailable.
    // Unknown models remain provider-validated rather than being inferred from
    // the behavior of another Grok model.
    if matches!(
        model,
        "grok-4.5"
            | "grok-4.5-latest"
            | "grok-4.6"
            | "grok-4.7"
            | "grok-build-0.1"
            | "grok-build-latest"
            | "grok-code-fast-1"
            | "grok-code-fast"
            | "grok-code-fast-1-0825"
    ) {
        return Err(invalid_input(
            "selected xAI model does not support Batch API",
        ));
    }
    Ok(())
}

fn validate_batch_id(value: &str) -> Result<(), XaiBatchError> {
    if !valid_identity(value) {
        return Err(invalid_input("batch ID is empty or invalid"));
    }
    Ok(())
}

fn scope_mismatch() -> XaiBatchError {
    LlmError::PermissionDenied {
        message:
            "xAI Batch reference belongs to another provider, profile, endpoint, or account scope"
                .into(),
    }
    .into()
}

fn invalid_input(message: impl Into<String>) -> XaiBatchError {
    XaiBatchError::InvalidInput(message.into())
}

fn mutation_transport_error(
    operation: &'static str,
    reference: Option<XaiBatchRef>,
    source: LlmError,
) -> XaiBatchError {
    if matches!(
        &source,
        LlmError::Transport { .. } | LlmError::TransportTimeout { .. }
    ) {
        XaiBatchError::OutcomeUnknown {
            operation,
            reference: reference.map(Box::new),
            source,
        }
    } else {
        XaiBatchError::Llm(source)
    }
}
