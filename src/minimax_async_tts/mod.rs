//! MiniMax's asynchronous long-text Text-to-Audio API.
//!
//! Submission, status query, and file metadata retrieval are separate calls.
//! This client never retries an ambiguous submission, polls a task, or follows
//! MiniMax's temporary download URL. Task and file references are bound to one
//! MiniMax profile, regional API base, and account scope.

use crate::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    minimax_tts::{MiniMaxTtsPronunciationDict, MiniMaxTtsVoiceSetting},
    minimax_voices::MiniMaxVoiceRef,
    protocol::{LlmError, ProviderId, Secret},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt, time::Duration};
use thiserror::Error;
use url::Url;

pub use crate::minimax_tts::MiniMaxTtsModel as MiniMaxAsyncTtsModel;
pub use crate::minimax_tts::MiniMaxTtsRegion as MiniMaxAsyncTtsRegion;

/// Official MiniMax international async TTS service root.
pub const MINIMAX_ASYNC_TTS_INTERNATIONAL_BASE: &str = "https://api.minimax.io/v1";
/// Official MiniMax mainland async TTS service root.
pub const MINIMAX_ASYNC_TTS_CHINA_BASE: &str = "https://api.minimax.cn/v1";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_INLINE_TEXT_CHARACTERS: usize = 50_000;
const RESULT_URL_VALIDITY: Duration = Duration::from_secs(9 * 60 * 60);

/// Secret-free route and identity configuration for one MiniMax account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxAsyncTtsConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: MiniMaxAsyncTtsRegion,
    /// API base ending in `/v1`, before operation-specific paths.
    pub api_base_url: String,
    pub request_timeout: Duration,
}

impl MiniMaxAsyncTtsConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: MiniMaxAsyncTtsRegion,
    ) -> Self {
        let api_base_url = match region {
            MiniMaxAsyncTtsRegion::International => MINIMAX_ASYNC_TTS_INTERNATIONAL_BASE,
            MiniMaxAsyncTtsRegion::ChinaMainland => MINIMAX_ASYNC_TTS_CHINA_BASE,
        };
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            api_base_url: api_base_url.into(),
            request_timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Select a same-region route or a loopback endpoint for contract tests.
    pub fn with_api_base_url(mut self, api_base_url: impl Into<String>) -> Self {
        self.api_base_url = api_base_url.into();
        self
    }

    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

/// Non-secret identity carried by task and generated-file references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxAsyncTtsScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub region: MiniMaxAsyncTtsRegion,
}

/// Caller-selected TTS input. Uploaded-file IDs must come from the same
/// MiniMax profile, region, endpoint and account as the task service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiniMaxAsyncTtsInput {
    Text(String),
    TextFile(Box<ProviderFileRef>),
}

/// Async endpoint audio settings. `format` supports MP3, WAV, or FLAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxAsyncTtsAudioSetting {
    #[serde(rename = "audio_sample_rate")]
    pub sample_rate: u32,
    pub bitrate: u32,
    pub format: MiniMaxAsyncTtsAudioFormat,
    pub channel: u8,
}

impl Default for MiniMaxAsyncTtsAudioSetting {
    fn default() -> Self {
        Self {
            sample_rate: 32_000,
            bitrate: 128_000,
            format: MiniMaxAsyncTtsAudioFormat::Mp3,
            channel: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxAsyncTtsAudioFormat {
    #[default]
    Mp3,
    Wav,
    Flac,
}

impl MiniMaxAsyncTtsAudioFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Flac => "flac",
        }
    }
}

/// Typed long-text submission controls for `POST /v1/t2a_async_v2`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxAsyncTtsRequest {
    pub input: MiniMaxAsyncTtsInput,
    pub model: MiniMaxAsyncTtsModel,
    pub voice_setting: MiniMaxTtsVoiceSetting,
    pub audio_setting: Option<MiniMaxAsyncTtsAudioSetting>,
    pub pronunciation_dict: Option<MiniMaxTtsPronunciationDict>,
    pub language_boost: Option<String>,
    /// Provider-native voice effects object, retained as JSON because the
    /// documented async schema leaves its child properties open-ended.
    pub voice_modify: Option<Value>,
}

impl MiniMaxAsyncTtsRequest {
    pub fn new(text: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            input: MiniMaxAsyncTtsInput::Text(text.into()),
            model: MiniMaxAsyncTtsModel::default(),
            voice_setting: MiniMaxTtsVoiceSetting::new(voice_id),
            audio_setting: None,
            pronunciation_dict: None,
            language_boost: None,
            voice_modify: None,
        }
    }

    pub fn from_text_file(file: ProviderFileRef, voice_id: impl Into<String>) -> Self {
        Self {
            input: MiniMaxAsyncTtsInput::TextFile(Box::new(file)),
            model: MiniMaxAsyncTtsModel::default(),
            voice_setting: MiniMaxTtsVoiceSetting::new(voice_id),
            audio_setting: None,
            pronunciation_dict: None,
            language_boost: None,
            voice_modify: None,
        }
    }
}

/// Account-bound task identifier returned by MiniMax.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxAsyncTtsTaskRef {
    pub scope: MiniMaxAsyncTtsScope,
    #[serde(deserialize_with = "deserialize_numeric_id")]
    pub task_id: String,
}

/// Account-bound generated file identifier returned by submission or query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxAsyncTtsFileRef {
    pub scope: MiniMaxAsyncTtsScope,
    #[serde(deserialize_with = "deserialize_numeric_id")]
    pub file_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxAsyncTtsSubmission {
    pub task: MiniMaxAsyncTtsTaskRef,
    pub file: Option<MiniMaxAsyncTtsFileRef>,
    pub task_token: Option<Secret<String>>,
    pub usage_characters: Option<u64>,
    pub request_id: Option<String>,
    /// Provider response with the capability-like `task_token` removed.
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiniMaxAsyncTtsStatus {
    Processing,
    Success,
    Failed,
    Expired,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxAsyncTtsTask {
    pub reference: MiniMaxAsyncTtsTaskRef,
    pub status: MiniMaxAsyncTtsStatus,
    pub file: Option<MiniMaxAsyncTtsFileRef>,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Signed MiniMax download URL. `Debug` redacts it; call `as_str` to use it.
#[derive(Clone, PartialEq, Eq)]
pub struct MiniMaxAsyncTtsDownloadUrl(String);

impl MiniMaxAsyncTtsDownloadUrl {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl MiniMaxAsyncTtsStatus {
    fn from_provider(value: &str) -> Self {
        if value.eq_ignore_ascii_case("processing") {
            Self::Processing
        } else if value.eq_ignore_ascii_case("success") {
            Self::Success
        } else if value.eq_ignore_ascii_case("failed") {
            Self::Failed
        } else if value.eq_ignore_ascii_case("expired") {
            Self::Expired
        } else {
            Self::Other(value.to_owned())
        }
    }
}

impl fmt::Debug for MiniMaxAsyncTtsDownloadUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted MiniMax download URL>")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxAsyncTtsResult {
    pub file: MiniMaxAsyncTtsFileRef,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub created_at: Option<u64>,
    pub purpose: Option<String>,
    pub download_url: MiniMaxAsyncTtsDownloadUrl,
    pub download_url_valid_for: Duration,
    pub request_id: Option<String>,
    /// Provider file data, with the signed URL redacted.
    pub native: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxAsyncTtsDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum MiniMaxAsyncTtsError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid MiniMax async TTS request: {0}")]
    InvalidRequest(String),
    #[error(
        "MiniMax async TTS provider rejected the request (HTTP {http_status:?}, code {code:?}): {message}"
    )]
    Provider {
        operation: &'static str,
        http_status: Option<u16>,
        code: Option<i64>,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: MiniMaxAsyncTtsDispatch,
    },
    #[error("MiniMax async TTS returned an invalid {operation} response: {message}")]
    InvalidResponse {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("MiniMax async TTS submit outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
    #[error("MiniMax async TTS submit outcome is unknown: {message}")]
    ResponseOutcomeUnknown {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
}

impl MiniMaxAsyncTtsError {
    /// Reports whether a task submission was sent, rejected, accepted, or is
    /// ambiguous. Errors from read-only operations return `NotSent` here.
    pub fn dispatch(&self) -> MiniMaxAsyncTtsDispatch {
        match self {
            Self::Llm(_) | Self::InvalidRequest(_) => MiniMaxAsyncTtsDispatch::NotSent,
            Self::Provider {
                operation: "submit",
                dispatch,
                ..
            } => *dispatch,
            Self::InvalidResponse {
                operation: "submit",
                ..
            } => MiniMaxAsyncTtsDispatch::Accepted,
            Self::OutcomeUnknown { .. } | Self::ResponseOutcomeUnknown { .. } => {
                MiniMaxAsyncTtsDispatch::Unknown
            }
            Self::Provider { .. } | Self::InvalidResponse { .. } => {
                MiniMaxAsyncTtsDispatch::NotSent
            }
        }
    }
}

/// One-shot caller-managed async TTS lifecycle client.
pub struct MiniMaxAsyncTtsService<'a> {
    transport: &'a dyn Transport,
    config: MiniMaxAsyncTtsConfig,
    scope: MiniMaxAsyncTtsScope,
}

impl<'a> MiniMaxAsyncTtsService<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        mut config: MiniMaxAsyncTtsConfig,
    ) -> Result<Self, MiniMaxAsyncTtsError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.request_timeout.is_zero() {
            return Err(invalid("request_timeout must be positive"));
        }
        config.api_base_url = normalize_base_url(&config.api_base_url, config.region)?;
        let scope = MiniMaxAsyncTtsScope {
            provider_id: ProviderId::new("minimax"),
            profile_name: config.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&config.api_base_url),
            account_scope: config.account_scope.clone(),
            region: config.region,
        };
        Ok(Self {
            transport,
            config,
            scope,
        })
    }

    pub fn scope(&self) -> &MiniMaxAsyncTtsScope {
        &self.scope
    }

    /// Submit a task with an explicitly scoped MiniMax voice resource. The
    /// reference must belong to this profile, account, region, and `/v1` API
    /// root. Built-in IDs remain available through [`Self::submit`].
    pub async fn submit_with_voice(
        &self,
        request: &MiniMaxAsyncTtsRequest,
        voice: &MiniMaxVoiceRef,
        api_key: &Secret<String>,
    ) -> Result<MiniMaxAsyncTtsSubmission, MiniMaxAsyncTtsError> {
        self.validate_voice_reference(voice)?;
        let mut request = request.clone();
        request.voice_setting.voice_id = voice.voice_id().to_owned();
        self.submit(&request, api_key).await
    }

    fn validate_voice_reference(
        &self,
        voice: &MiniMaxVoiceRef,
    ) -> Result<(), MiniMaxAsyncTtsError> {
        let reference_scope = voice.scope();
        if MiniMaxVoiceRef::new(
            reference_scope.clone(),
            voice.kind(),
            voice.voice_id().to_owned(),
        )
        .is_err()
        {
            return Err(invalid("MiniMax voice reference is malformed"));
        }
        let canonical_api_root = self.config.api_base_url.trim_end_matches('/');
        let expected_endpoint = provider_file_endpoint_fingerprint(canonical_api_root);
        if reference_scope.provider_id != self.scope.provider_id
            || reference_scope.profile_name != self.scope.profile_name
            || reference_scope.endpoint_fingerprint != expected_endpoint
            || reference_scope.account_scope != self.scope.account_scope
            || reference_scope.region != self.scope.region
        {
            return Err(invalid(
                "MiniMax voice reference belongs to a different provider, profile, account, region, or API endpoint",
            ));
        }
        Ok(())
    }

    /// Submit one long-text or previously uploaded text-file task. If the
    /// transport or successful response is ambiguous, this returns an unknown
    /// outcome and never retries the potentially accepted request.
    pub async fn submit(
        &self,
        request: &MiniMaxAsyncTtsRequest,
        api_key: &Secret<String>,
    ) -> Result<MiniMaxAsyncTtsSubmission, MiniMaxAsyncTtsError> {
        validate_credential(api_key)?;
        validate_request(request, &self.scope)?;
        let body = request_body(request)?;
        let native = self
            .request_json(
                "POST",
                &["t2a_async_v2"],
                &[],
                Some(body),
                api_key,
                "submit",
            )
            .await?;
        let request_id = response_request_id(&native.value, native.request_id);
        let task_id = native
            .value
            .get("task_id")
            .and_then(numeric_id)
            .ok_or_else(|| {
                ambiguous_submit(
                    "success response omitted a valid task_id",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?;
        let file = native
            .value
            .get("file_id")
            .and_then(numeric_id)
            .map(|file_id| MiniMaxAsyncTtsFileRef {
                scope: self.scope.clone(),
                file_id,
            });
        let task_token = native
            .value
            .get("task_token")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(|value| Secret::new(value.to_owned()));
        let usage_characters = native.value.get("usage_characters").and_then(Value::as_u64);
        let mut safe_native = native.value;
        if let Some(object) = safe_native.as_object_mut() {
            object.remove("task_token");
        }
        Ok(MiniMaxAsyncTtsSubmission {
            task: MiniMaxAsyncTtsTaskRef {
                scope: self.scope.clone(),
                task_id,
            },
            file,
            task_token,
            usage_characters,
            request_id,
            native: safe_native,
        })
    }

    /// Query one task once. Polling pace and retry decisions stay with the
    /// caller; MiniMax documents a maximum of 10 status queries per second.
    pub async fn query(
        &self,
        task: &MiniMaxAsyncTtsTaskRef,
        api_key: &Secret<String>,
    ) -> Result<MiniMaxAsyncTtsTask, MiniMaxAsyncTtsError> {
        validate_credential(api_key)?;
        self.check_task_scope(task)?;
        let query = vec![("task_id".to_owned(), task.task_id.clone())];
        let native = self
            .request_json(
                "GET",
                &["query", "t2a_async_query_v2"],
                &query,
                None,
                api_key,
                "query",
            )
            .await?;
        let request_id = response_request_id(&native.value, native.request_id);
        let task_id = native
            .value
            .get("task_id")
            .and_then(numeric_id)
            .ok_or_else(|| {
                invalid_response(
                    "query",
                    "response omitted a valid task_id",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?;
        if task_id != task.task_id {
            return Err(invalid_response(
                "query",
                "response task_id does not match the scoped reference",
                request_id,
                native.value,
            ));
        }
        let status = native
            .value
            .get("status")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(MiniMaxAsyncTtsStatus::from_provider)
            .ok_or_else(|| {
                invalid_response(
                    "query",
                    "response omitted status",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?;
        let file = native
            .value
            .get("file_id")
            .and_then(numeric_id)
            .map(|file_id| MiniMaxAsyncTtsFileRef {
                scope: self.scope.clone(),
                file_id,
            });
        if status == MiniMaxAsyncTtsStatus::Success && file.is_none() {
            return Err(invalid_response(
                "query",
                "successful task response omitted a valid file_id",
                request_id,
                native.value,
            ));
        }
        Ok(MiniMaxAsyncTtsTask {
            reference: task.clone(),
            status,
            file,
            request_id,
            native: native.value,
        })
    }

    /// Retrieve metadata and the time-limited result URL for one generated
    /// file. This does not download or copy the result.
    pub async fn get_result(
        &self,
        file: &MiniMaxAsyncTtsFileRef,
        api_key: &Secret<String>,
    ) -> Result<MiniMaxAsyncTtsResult, MiniMaxAsyncTtsError> {
        validate_credential(api_key)?;
        self.check_file_scope(file)?;
        let query = vec![("file_id".to_owned(), file.file_id.clone())];
        let native = self
            .request_json(
                "GET",
                &["files", "retrieve"],
                &query,
                None,
                api_key,
                "get_result",
            )
            .await?;
        let request_id = response_request_id(&native.value, native.request_id);
        let file_data = native
            .value
            .get("file")
            .filter(|value| value.is_object())
            .ok_or_else(|| {
                invalid_response(
                    "get_result",
                    "response omitted file metadata",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?;
        let returned_file_id = file_data
            .get("file_id")
            .and_then(numeric_id)
            .ok_or_else(|| {
                invalid_response(
                    "get_result",
                    "file metadata omitted a valid file_id",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?;
        if returned_file_id != file.file_id {
            return Err(invalid_response(
                "get_result",
                "file metadata ID does not match the scoped reference",
                request_id,
                native.value,
            ));
        }
        let download_url = file_data
            .get("download_url")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                invalid_response(
                    "get_result",
                    "file metadata omitted download_url",
                    request_id.clone(),
                    native.value.clone(),
                )
            })?
            .to_owned();
        let filename = file_data
            .get("filename")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let size_bytes = file_data.get("bytes").and_then(Value::as_u64);
        let created_at = file_data.get("created_at").and_then(Value::as_u64);
        let purpose = file_data
            .get("purpose")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut safe_native = native.value;
        if let Some(file_object) = safe_native.get_mut("file").and_then(Value::as_object_mut) {
            file_object.remove("download_url");
        }
        Ok(MiniMaxAsyncTtsResult {
            file: file.clone(),
            filename,
            size_bytes,
            created_at,
            purpose,
            download_url: MiniMaxAsyncTtsDownloadUrl(download_url),
            download_url_valid_for: RESULT_URL_VALIDITY,
            request_id,
            native: safe_native,
        })
    }

    fn check_task_scope(&self, task: &MiniMaxAsyncTtsTaskRef) -> Result<(), MiniMaxAsyncTtsError> {
        if task.scope != self.scope || !valid_numeric_id(&task.task_id) {
            return Err(LlmError::PermissionDenied {
                message: "MiniMax TTS task belongs to another profile, endpoint, region, or account scope".into(),
            }
            .into());
        }
        Ok(())
    }

    fn check_file_scope(&self, file: &MiniMaxAsyncTtsFileRef) -> Result<(), MiniMaxAsyncTtsError> {
        if file.scope != self.scope || !valid_numeric_id(&file.file_id) {
            return Err(LlmError::PermissionDenied {
                message: "MiniMax TTS file belongs to another profile, endpoint, region, or account scope".into(),
            }
            .into());
        }
        Ok(())
    }

    async fn request_json(
        &self,
        method: &str,
        path: &[&str],
        query: &[(String, String)],
        body: Option<Value>,
        api_key: &Secret<String>,
        operation: &'static str,
    ) -> Result<DecodedResponse, MiniMaxAsyncTtsError> {
        let url = operation_url(&self.config.api_base_url, path, query)?;
        let body = match body {
            Some(value) => serde_json::to_vec(&value)
                .map_err(|_| invalid("request body cannot be serialized"))?,
            None => Vec::new(),
        };
        if body.len() > MAX_REQUEST_BYTES {
            return Err(invalid("request exceeds the 2 MiB client limit"));
        }
        let deadline = Deadline::after(Some(self.config.request_timeout));
        let request = HttpRequest {
            method: method.to_owned(),
            url: url.to_string(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", api_key.expose_secret()),
                ),
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(body),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if operation == "submit" {
                    MiniMaxAsyncTtsError::OutcomeUnknown { source }
                } else {
                    MiniMaxAsyncTtsError::Llm(source)
                }
            })?;
        decode_response(response, operation)
    }
}

struct DecodedResponse {
    value: Value,
    request_id: Option<String>,
}

fn decode_response(
    response: HttpResponse,
    operation: &'static str,
) -> Result<DecodedResponse, MiniMaxAsyncTtsError> {
    let mut request_id = response_id(&response.headers);
    let native = match serde_json::from_slice::<Value>(&response.body) {
        Ok(value) => value,
        Err(_) if !(200..300).contains(&response.status) => {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }
        Err(_) if operation == "submit" => {
            return Err(ambiguous_submit(
                "successful HTTP response was not valid JSON",
                request_id,
                Value::Null,
            ));
        }
        Err(_) => {
            return Err(invalid_response(
                operation,
                "successful HTTP response was not valid JSON",
                request_id,
                Value::Null,
            ));
        }
    };
    request_id = response_trace_id(&native).or(request_id);
    if !(200..300).contains(&response.status) {
        let dispatch =
            if operation == "submit" && (response.status == 408 || response.status >= 500) {
                MiniMaxAsyncTtsDispatch::Unknown
            } else {
                MiniMaxAsyncTtsDispatch::Rejected
            };
        return Err(provider_error(
            operation,
            Some(response.status),
            base_response_code(&native),
            base_response_message(&native).unwrap_or_else(|| "HTTP request rejected".into()),
            request_id,
            native,
            dispatch,
        ));
    }
    match base_response_code(&native) {
        Some(0) => Ok(DecodedResponse {
            value: native,
            request_id,
        }),
        Some(code) => Err(provider_error(
            operation,
            Some(response.status),
            Some(code),
            base_response_message(&native).unwrap_or_else(|| "MiniMax rejected the request".into()),
            request_id,
            native,
            MiniMaxAsyncTtsDispatch::Rejected,
        )),
        None if operation == "submit" => Err(ambiguous_submit(
            "successful response omitted base_resp.status_code",
            request_id,
            native,
        )),
        None => Err(invalid_response(
            operation,
            "successful response omitted base_resp.status_code",
            request_id,
            native,
        )),
    }
}

fn request_body(request: &MiniMaxAsyncTtsRequest) -> Result<Value, MiniMaxAsyncTtsError> {
    let mut body = json!({
        "model": request.model,
        "voice_setting": request.voice_setting,
    });
    match &request.input {
        MiniMaxAsyncTtsInput::Text(text) => body["text"] = Value::String(text.clone()),
        MiniMaxAsyncTtsInput::TextFile(file) => {
            let file_id = file.file_id.parse::<u64>().map_err(|_| {
                invalid("text-file reference must contain a positive numeric file_id")
            })?;
            body["text_file_id"] = json!(file_id);
        }
    }
    if let Some(audio_setting) = request.audio_setting {
        body["audio_setting"] = json!({
            "audio_sample_rate": audio_setting.sample_rate,
            "bitrate": audio_setting.bitrate,
            "format": audio_setting.format.as_str(),
            "channel": audio_setting.channel,
        });
    }
    if let Some(dictionary) = &request.pronunciation_dict {
        body["pronunciation_dict"] = serde_json::to_value(dictionary)
            .map_err(|_| invalid("pronunciation_dict cannot be serialized"))?;
    }
    if let Some(language) = &request.language_boost {
        body["language_boost"] = Value::String(language.clone());
    }
    if let Some(voice_modify) = &request.voice_modify {
        body["voice_modify"] = voice_modify.clone();
    }
    Ok(body)
}

fn validate_request(
    request: &MiniMaxAsyncTtsRequest,
    scope: &MiniMaxAsyncTtsScope,
) -> Result<(), MiniMaxAsyncTtsError> {
    match &request.input {
        MiniMaxAsyncTtsInput::Text(text) => {
            let characters = text.chars().count();
            if characters == 0 || text.trim().is_empty() {
                return Err(invalid("text must not be empty"));
            }
            if characters > MAX_INLINE_TEXT_CHARACTERS {
                return Err(invalid("text may contain at most 50,000 characters"));
            }
        }
        MiniMaxAsyncTtsInput::TextFile(file) => validate_text_file_ref(file, scope)?,
    }
    let voice = &request.voice_setting;
    if voice.voice_id.trim().is_empty() || has_control(&voice.voice_id) {
        return Err(invalid(
            "voice_id must be nonempty and contain no control characters",
        ));
    }
    if voice
        .speed
        .is_some_and(|speed| !speed.is_finite() || !(0.5..=2.0).contains(&speed))
    {
        return Err(invalid("voice speed must be between 0.5 and 2.0"));
    }
    if voice
        .vol
        .is_some_and(|volume| !volume.is_finite() || volume <= 0.0 || volume > 10.0)
    {
        return Err(invalid(
            "voice volume must be greater than 0 and at most 10",
        ));
    }
    if voice
        .pitch
        .is_some_and(|pitch| !(-12..=12).contains(&pitch))
    {
        return Err(invalid("voice pitch must be between -12 and 12"));
    }
    if voice
        .emotion
        .as_ref()
        .is_some_and(|emotion| emotion.trim().is_empty() || has_control(emotion))
    {
        return Err(invalid(
            "emotion must be nonempty and contain no control characters",
        ));
    }
    if let Some(audio) = request.audio_setting {
        if !matches!(
            audio.sample_rate,
            8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100
        ) {
            return Err(invalid(
                "audio_sample_rate is outside the documented T2A rates",
            ));
        }
        if !matches!(audio.bitrate, 32_000 | 64_000 | 128_000 | 256_000) {
            return Err(invalid("bitrate must be 32000, 64000, 128000, or 256000"));
        }
        if !matches!(audio.channel, 1 | 2) {
            return Err(invalid("audio channel count must be 1 or 2"));
        }
    }
    if request
        .language_boost
        .as_ref()
        .is_some_and(|language| language.trim().is_empty() || has_control(language))
    {
        return Err(invalid(
            "language_boost must be nonempty and contain no control characters",
        ));
    }
    if request
        .pronunciation_dict
        .as_ref()
        .is_some_and(|dictionary| {
            dictionary
                .tone
                .iter()
                .any(|entry| entry.trim().is_empty() || has_control(entry))
        })
    {
        return Err(invalid(
            "pronunciation entries must be nonempty and contain no control characters",
        ));
    }
    if request
        .voice_modify
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(invalid("voice_modify must be a JSON object"));
    }
    Ok(())
}

fn validate_text_file_ref(
    file: &ProviderFileRef,
    scope: &MiniMaxAsyncTtsScope,
) -> Result<(), MiniMaxAsyncTtsError> {
    if file.provider_id != scope.provider_id
        || file.profile_name != scope.profile_name
        || file.endpoint_fingerprint != scope.endpoint_fingerprint
        || file.account_scope.as_deref() != Some(scope.account_scope.as_str())
        || file.purpose.as_deref() != Some("t2a_async_input")
    {
        return Err(LlmError::PermissionDenied {
            message: "MiniMax TTS text file belongs to another profile, endpoint, or account scope, or has the wrong upload purpose".into(),
        }
        .into());
    }
    if !valid_numeric_id(&file.file_id) {
        return Err(invalid(
            "text-file reference must contain a positive numeric file_id",
        ));
    }
    Ok(())
}

fn normalize_base_url(
    value: &str,
    region: MiniMaxAsyncTtsRegion,
) -> Result<String, MiniMaxAsyncTtsError> {
    let url =
        Url::parse(value).map_err(|_| invalid("MiniMax async TTS API base URL is invalid"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != "/v1"
    {
        return Err(invalid(
            "MiniMax async TTS API base URL must end in /v1 without credentials, query, or fragment",
        ));
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let local_http = url.scheme() == "http"
        && (host == "localhost"
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| address.is_loopback()));
    if local_http {
        return Ok(url.into());
    }
    if url.scheme() != "https" || url.port_or_known_default() != Some(443) {
        return Err(invalid("MiniMax async TTS API base URL must use HTTPS"));
    }
    let matches_region = match region {
        MiniMaxAsyncTtsRegion::International => host == "api.minimax.io",
        MiniMaxAsyncTtsRegion::ChinaMainland => host == "api.minimax.cn",
    };
    if !matches_region {
        return Err(invalid(
            "MiniMax async TTS API host does not match the selected region",
        ));
    }
    Ok(url.into())
}

fn operation_url(
    base: &str,
    path: &[&str],
    query: &[(String, String)],
) -> Result<Url, MiniMaxAsyncTtsError> {
    let mut url = Url::parse(base.trim_end_matches('/'))
        .map_err(|_| invalid("MiniMax async TTS API base URL is invalid"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| invalid("MiniMax async TTS API base URL cannot accept a path"))?;
        for segment in path {
            segments.push(segment);
        }
    }
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (name, value) in query {
            pairs.append_pair(name, value);
        }
    }
    Ok(url)
}

fn valid_numeric_id(value: &str) -> bool {
    value
        .parse::<u64>()
        .is_ok_and(|number| number > 0 && number <= i64::MAX as u64)
}

fn numeric_id(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return valid_numeric_id(text).then(|| text.to_owned());
    }
    value
        .as_u64()
        .filter(|number| *number > 0 && *number <= i64::MAX as u64)
        .map(|number| number.to_string())
}

fn deserialize_numeric_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    numeric_id(&value).ok_or_else(|| serde::de::Error::custom("expected a positive int64 ID"))
}

fn validate_credential(key: &Secret<String>) -> Result<(), MiniMaxAsyncTtsError> {
    let value = key.expose_secret();
    if value.trim().is_empty() || has_control(value) {
        return Err(invalid("MiniMax API key is empty or malformed"));
    }
    Ok(())
}

fn response_id(headers: &[(String, String)]) -> Option<String> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
        .or_else(|| {
            headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("request-id"))
        })
        .map(|(_, value)| value.clone())
}

fn response_trace_id(native: &Value) -> Option<String> {
    native
        .get("trace_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn response_request_id(native: &Value, header_id: Option<String>) -> Option<String> {
    response_trace_id(native).or(header_id)
}

fn base_response_code(value: &Value) -> Option<i64> {
    let status = value.get("base_resp")?.get("status_code")?;
    status
        .as_i64()
        .or_else(|| status.as_str()?.parse::<i64>().ok())
}

fn base_response_message(value: &Value) -> Option<String> {
    value
        .get("base_resp")?
        .get("status_msg")?
        .as_str()
        .map(str::to_owned)
}

fn provider_error(
    operation: &'static str,
    http_status: Option<u16>,
    code: Option<i64>,
    message: String,
    request_id: Option<String>,
    native: Value,
    dispatch: MiniMaxAsyncTtsDispatch,
) -> MiniMaxAsyncTtsError {
    MiniMaxAsyncTtsError::Provider {
        operation,
        http_status,
        code,
        message,
        request_id,
        native: Box::new(redact_native(native, operation)),
        dispatch,
    }
}

fn invalid_response(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> MiniMaxAsyncTtsError {
    MiniMaxAsyncTtsError::InvalidResponse {
        operation,
        message: message.to_owned(),
        request_id,
        native: Box::new(redact_native(native, operation)),
    }
}

fn ambiguous_submit(
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> MiniMaxAsyncTtsError {
    MiniMaxAsyncTtsError::ResponseOutcomeUnknown {
        message: message.to_owned(),
        request_id,
        native: Box::new(redact_native(native, "submit")),
    }
}

fn redact_native(mut native: Value, operation: &str) -> Value {
    if operation == "submit" {
        if let Some(object) = native.as_object_mut() {
            object.remove("task_token");
        }
    }
    if operation == "get_result" {
        if let Some(file) = native.get_mut("file").and_then(Value::as_object_mut) {
            file.remove("download_url");
        }
    }
    native
}

fn invalid(message: &str) -> MiniMaxAsyncTtsError {
    MiniMaxAsyncTtsError::InvalidRequest(message.to_owned())
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}
