//! MiniMax's native voice clone, voice design, catalog, and delete endpoints.

use crate::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{LlmError, ProviderId, Secret},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt, time::Duration, time::SystemTime, time::UNIX_EPOCH};
use thiserror::Error;
use url::Url;

pub use crate::minimax_tts::{MiniMaxTtsModel, MiniMaxTtsRegion as MiniMaxVoicesRegion};

/// Official MiniMax international voice-management API root.
pub const MINIMAX_VOICES_INTERNATIONAL_BASE: &str = "https://api.minimax.io/v1";
/// Official MiniMax mainland voice-management API root.
pub const MINIMAX_VOICES_CHINA_BASE: &str = "https://api.minimax.cn/v1";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_VOICE_FILE_BYTES: u64 = 20_000_000;

/// Routing and account settings for one MiniMax voice-management connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxVoicesConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: MiniMaxVoicesRegion,
    pub api_base_url: String,
    pub request_timeout: Duration,
}

impl MiniMaxVoicesConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: MiniMaxVoicesRegion,
    ) -> Self {
        let api_base_url = match region {
            MiniMaxVoicesRegion::International => MINIMAX_VOICES_INTERNATIONAL_BASE,
            MiniMaxVoicesRegion::ChinaMainland => MINIMAX_VOICES_CHINA_BASE,
        };
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            api_base_url: api_base_url.into(),
            request_timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Select a same-region API root. HTTP is accepted only for loopback tests.
    pub fn with_api_base_url(mut self, api_base_url: impl Into<String>) -> Self {
        self.api_base_url = api_base_url.into();
        self
    }

    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

/// Per-call MiniMax API key. Credentials are never stored by the service.
pub struct MiniMaxVoicesCredentials {
    api_key: Secret<String>,
}

impl MiniMaxVoicesCredentials {
    pub fn new(api_key: Secret<String>) -> Self {
        Self { api_key }
    }
}

impl fmt::Debug for MiniMaxVoicesCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MiniMaxVoicesCredentials")
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Non-secret identity of one MiniMax voice-management API root and account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxVoicesScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    /// Fingerprint of the canonical `/v1` root, shared with scoped file refs.
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub region: MiniMaxVoicesRegion,
}

/// The documented catalog/deletion kind associated with a voice ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxVoiceKind {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "voice_cloning")]
    Cloned,
    #[serde(rename = "voice_generation")]
    Generated,
}

/// Account-bound reference to one MiniMax voice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxVoiceRef {
    pub scope: MiniMaxVoicesScope,
    pub kind: MiniMaxVoiceKind,
    pub voice_id: String,
}

impl MiniMaxVoiceRef {
    pub fn new(
        scope: MiniMaxVoicesScope,
        kind: MiniMaxVoiceKind,
        voice_id: impl Into<String>,
    ) -> Result<Self, MiniMaxVoicesError> {
        let voice_id = voice_id.into();
        validate_voice_reference_id(&voice_id)?;
        Ok(Self {
            scope,
            kind,
            voice_id,
        })
    }

    pub fn scope(&self) -> &MiniMaxVoicesScope {
        &self.scope
    }

    pub const fn kind(&self) -> MiniMaxVoiceKind {
        self.kind
    }

    pub fn voice_id(&self) -> &str {
        &self.voice_id
    }
}

/// A short reference recording paired with its transcript to improve a clone.
pub struct MiniMaxVoiceClonePrompt {
    pub audio: ProviderFileRef,
    pub text: String,
    /// Optional caller-declared duration used only for local validation.
    pub duration_seconds: Option<f64>,
}

impl MiniMaxVoiceClonePrompt {
    pub fn new(audio: ProviderFileRef, text: impl Into<String>) -> Self {
        Self {
            audio,
            text: text.into(),
            duration_seconds: None,
        }
    }

    pub fn with_duration_seconds(mut self, duration_seconds: f64) -> Self {
        self.duration_seconds = Some(duration_seconds);
        self
    }
}

impl fmt::Debug for MiniMaxVoiceClonePrompt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MiniMaxVoiceClonePrompt")
            .field("audio_file_id", &"<scoped file ref>")
            .field("text", &self.text)
            .field("duration_seconds", &self.duration_seconds)
            .finish()
    }
}

/// Request fields for `POST /v1/voice_clone`.
///
/// Preview fields are opt-in. No preview text is sent unless the caller sets it.
pub struct MiniMaxVoiceCloneRequest {
    pub voice_audio: ProviderFileRef,
    pub voice_audio_duration_seconds: Option<f64>,
    pub voice_id: String,
    pub clone_prompt: Option<MiniMaxVoiceClonePrompt>,
    pub text: Option<String>,
    pub model: Option<MiniMaxTtsModel>,
    pub language_boost: Option<MiniMaxVoiceLanguageBoost>,
    pub text_validation: Option<String>,
    pub accuracy: Option<f64>,
    pub need_noise_reduction: Option<bool>,
    pub need_volume_normalization: Option<bool>,
    pub aigc_watermark: Option<bool>,
}

impl MiniMaxVoiceCloneRequest {
    pub fn new(voice_audio: ProviderFileRef, voice_id: impl Into<String>) -> Self {
        Self {
            voice_audio,
            voice_audio_duration_seconds: None,
            voice_id: voice_id.into(),
            clone_prompt: None,
            text: None,
            model: None,
            language_boost: None,
            text_validation: None,
            accuracy: None,
            need_noise_reduction: None,
            need_volume_normalization: None,
            aigc_watermark: None,
        }
    }

    pub fn with_voice_audio_duration_seconds(mut self, duration_seconds: f64) -> Self {
        self.voice_audio_duration_seconds = Some(duration_seconds);
        self
    }

    pub fn with_clone_prompt(mut self, prompt: MiniMaxVoiceClonePrompt) -> Self {
        self.clone_prompt = Some(prompt);
        self
    }

    /// Opt in to paid clone-preview generation. `model` is required with text.
    pub fn with_preview(mut self, text: impl Into<String>, model: MiniMaxTtsModel) -> Self {
        self.text = Some(text.into());
        self.model = Some(model);
        self
    }

    pub fn with_language_boost(mut self, language: MiniMaxVoiceLanguageBoost) -> Self {
        self.language_boost = Some(language);
        self
    }

    pub fn with_text_validation(mut self, text: impl Into<String>) -> Self {
        self.text_validation = Some(text.into());
        self
    }

    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    pub fn with_noise_reduction(mut self, enabled: bool) -> Self {
        self.need_noise_reduction = Some(enabled);
        self
    }

    pub fn with_volume_normalization(mut self, enabled: bool) -> Self {
        self.need_volume_normalization = Some(enabled);
        self
    }

    pub fn with_aigc_watermark(mut self, enabled: bool) -> Self {
        self.aigc_watermark = Some(enabled);
        self
    }
}

impl fmt::Debug for MiniMaxVoiceCloneRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MiniMaxVoiceCloneRequest")
            .field("voice_audio", &"<scoped file ref>")
            .field(
                "voice_audio_duration_seconds",
                &self.voice_audio_duration_seconds,
            )
            .field("voice_id", &self.voice_id)
            .field("clone_prompt", &self.clone_prompt)
            .field("text", &self.text)
            .field("model", &self.model)
            .field("language_boost", &self.language_boost)
            .field("text_validation", &self.text_validation)
            .field("accuracy", &self.accuracy)
            .field("need_noise_reduction", &self.need_noise_reduction)
            .field("need_volume_normalization", &self.need_volume_normalization)
            .field("aigc_watermark", &self.aigc_watermark)
            .finish()
    }
}

/// Languages supported by MiniMax clone's `language_boost` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxVoiceLanguageBoost {
    #[serde(rename = "Chinese")]
    Chinese,
    #[serde(rename = "Chinese,Yue")]
    ChineseYue,
    #[serde(rename = "English")]
    English,
    #[serde(rename = "Arabic")]
    Arabic,
    #[serde(rename = "Russian")]
    Russian,
    #[serde(rename = "Spanish")]
    Spanish,
    #[serde(rename = "French")]
    French,
    #[serde(rename = "Portuguese")]
    Portuguese,
    #[serde(rename = "German")]
    German,
    #[serde(rename = "Turkish")]
    Turkish,
    #[serde(rename = "Dutch")]
    Dutch,
    #[serde(rename = "Ukrainian")]
    Ukrainian,
    #[serde(rename = "Vietnamese")]
    Vietnamese,
    #[serde(rename = "Indonesian")]
    Indonesian,
    #[serde(rename = "Japanese")]
    Japanese,
    #[serde(rename = "Italian")]
    Italian,
    #[serde(rename = "Korean")]
    Korean,
    #[serde(rename = "Thai")]
    Thai,
    #[serde(rename = "Polish")]
    Polish,
    #[serde(rename = "Romanian")]
    Romanian,
    #[serde(rename = "Greek")]
    Greek,
    #[serde(rename = "Czech")]
    Czech,
    #[serde(rename = "Finnish")]
    Finnish,
    #[serde(rename = "Hindi")]
    Hindi,
    #[serde(rename = "Bulgarian")]
    Bulgarian,
    #[serde(rename = "Danish")]
    Danish,
    #[serde(rename = "Hebrew")]
    Hebrew,
    #[serde(rename = "Malay")]
    Malay,
    #[serde(rename = "Persian")]
    Persian,
    #[serde(rename = "Slovak")]
    Slovak,
    #[serde(rename = "Swedish")]
    Swedish,
    #[serde(rename = "Croatian")]
    Croatian,
    #[serde(rename = "Filipino")]
    Filipino,
    #[serde(rename = "Hungarian")]
    Hungarian,
    #[serde(rename = "Norwegian")]
    Norwegian,
    #[serde(rename = "Slovenian")]
    Slovenian,
    #[serde(rename = "Catalan")]
    Catalan,
    #[serde(rename = "Nynorsk")]
    Nynorsk,
    #[serde(rename = "Tamil")]
    Tamil,
    #[serde(rename = "Afrikaans")]
    Afrikaans,
    #[serde(rename = "auto")]
    Auto,
}

/// Request body for MiniMax's voice design route. A preview is required by the
/// provider and can incur charges, so callers must provide it explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxVoiceDesignRequest {
    pub prompt: String,
    pub preview_text: String,
    pub voice_id: Option<String>,
    /// Mainland-only preview marker supported by the China API docs.
    pub aigc_watermark: Option<bool>,
}

impl MiniMaxVoiceDesignRequest {
    pub fn new(prompt: impl Into<String>, preview_text: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            preview_text: preview_text.into(),
            voice_id: None,
            aigc_watermark: None,
        }
    }

    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = Some(voice_id.into());
        self
    }

    /// Add the mainland-only marker to the generated preview audio.
    pub fn with_aigc_watermark(mut self, enabled: bool) -> Self {
        self.aigc_watermark = Some(enabled);
        self
    }
}

/// Filter accepted by `POST /v1/get_voice`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxVoiceListType {
    System,
    #[serde(rename = "voice_cloning")]
    Cloned,
    #[serde(rename = "voice_generation")]
    Generated,
    #[default]
    All,
}

impl MiniMaxVoiceListType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Cloned => "voice_cloning",
            Self::Generated => "voice_generation",
            Self::All => "all",
        }
    }
}

/// Options for a single read-only MiniMax voice catalog request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MiniMaxVoiceListRequest {
    pub voice_type: MiniMaxVoiceListType,
}

impl MiniMaxVoiceListRequest {
    pub fn new(voice_type: MiniMaxVoiceListType) -> Self {
        Self { voice_type }
    }
}

/// One item in a typed catalog category, retaining its original provider object.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoice {
    pub reference: MiniMaxVoiceRef,
    pub voice_name: Option<String>,
    pub description: Vec<String>,
    /// MiniMax returns this as a string; preserve it without assuming a date format.
    pub created_time: Option<String>,
    pub native: Value,
}

/// Typed categories returned by `POST /v1/get_voice`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoiceList {
    pub scope: MiniMaxVoicesScope,
    pub system_voice: Vec<MiniMaxVoice>,
    pub voice_cloning: Vec<MiniMaxVoice>,
    pub voice_generation: Vec<MiniMaxVoice>,
    pub base_resp: MiniMaxVoicesBaseResponse,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Typed Preview metadata documented in the clone response.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoicePreviewInfo {
    pub audio_length: Option<u64>,
    pub audio_sample_rate: Option<u64>,
    pub audio_size: Option<u64>,
    pub bitrate: Option<u64>,
    pub word_count: Option<u64>,
    pub usage_characters: Option<u64>,
    pub native: Value,
}

/// Voice clone result. `demo_audio` is returned as a raw URL and is never fetched.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoiceCloneResult {
    pub reference: MiniMaxVoiceRef,
    /// Kept as `Value`: the current reference labels this an object, but its
    /// example response uses a boolean.
    pub input_sensitive: Option<Value>,
    pub input_sensitive_type: Option<Value>,
    pub demo_audio: Option<String>,
    pub extra_info: Option<MiniMaxVoicePreviewInfo>,
    pub base_resp: MiniMaxVoicesBaseResponse,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Voice design result. `trial_audio` remains the provider's original hex string.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoiceDesignResult {
    pub reference: MiniMaxVoiceRef,
    pub trial_audio: Option<String>,
    pub base_resp: MiniMaxVoicesBaseResponse,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Receipt returned after deleting a cloned or generated voice.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxVoiceDeleteReceipt {
    pub reference: MiniMaxVoiceRef,
    pub created_time: Option<String>,
    pub base_resp: MiniMaxVoicesBaseResponse,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxVoicesBaseResponse {
    pub status_code: i64,
    pub status_msg: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxVoicesDispatch {
    NotSent,
    Rejected,
    Unknown,
}

/// Errors from MiniMax voice lifecycle operations.
#[derive(Debug, Error)]
pub enum MiniMaxVoicesError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid MiniMax voices request: {0}")]
    InvalidRequest(String),
    #[error("MiniMax voices provider rejected {operation} (HTTP {http_status:?}, code {code:?}): {message}")]
    Provider {
        operation: &'static str,
        http_status: Option<u16>,
        code: Option<i64>,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: MiniMaxVoicesDispatch,
    },
    #[error("MiniMax voices {operation} returned an invalid response: {message}")]
    InvalidResponse {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("MiniMax voices {operation} outcome is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("MiniMax voices {operation} outcome is unknown: {message}")]
    ResponseOutcomeUnknown {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
}

impl MiniMaxVoicesError {
    pub fn dispatch(&self) -> MiniMaxVoicesDispatch {
        match self {
            Self::Llm(_) | Self::InvalidRequest(_) | Self::InvalidResponse { .. } => {
                MiniMaxVoicesDispatch::NotSent
            }
            Self::Provider { dispatch, .. } => *dispatch,
            Self::OutcomeUnknown { .. } | Self::ResponseOutcomeUnknown { .. } => {
                MiniMaxVoicesDispatch::Unknown
            }
        }
    }
}

/// Caller-managed MiniMax voice lifecycle service.
pub struct MiniMaxVoicesService<'a> {
    transport: &'a dyn Transport,
    config: MiniMaxVoicesConfig,
    scope: MiniMaxVoicesScope,
}

impl<'a> MiniMaxVoicesService<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        mut config: MiniMaxVoicesConfig,
    ) -> Result<Self, MiniMaxVoicesError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.request_timeout.is_zero() {
            return Err(invalid("request_timeout must be positive"));
        }
        config.api_base_url = normalize_api_base_url(&config.api_base_url, config.region)?;
        let scope = MiniMaxVoicesScope {
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

    pub fn scope(&self) -> &MiniMaxVoicesScope {
        &self.scope
    }

    /// Clone one uploaded reference recording. No preview is generated unless
    /// `request.text` and `request.model` were explicitly supplied.
    pub async fn clone_voice(
        &self,
        request: &MiniMaxVoiceCloneRequest,
        credentials: &MiniMaxVoicesCredentials,
    ) -> Result<MiniMaxVoiceCloneResult, MiniMaxVoicesError> {
        validate_clone_request(request, &self.scope)?;
        validate_credential(&credentials.api_key)?;
        let body = clone_request_body(request)?;
        let (native, request_id) = self
            .post_json("voice_clone", "clone-voice", body, credentials, true)
            .await?;
        let response =
            decode_clone_response(native, self.scope.clone(), request_id, &request.voice_id)?;
        Ok(response)
    }

    /// Design a voice using caller-provided prompt and paid preview text.
    pub async fn design_voice(
        &self,
        request: &MiniMaxVoiceDesignRequest,
        credentials: &MiniMaxVoicesCredentials,
    ) -> Result<MiniMaxVoiceDesignResult, MiniMaxVoicesError> {
        validate_design_request(request, self.scope.region)?;
        validate_credential(&credentials.api_key)?;
        let mut body = json!({
            "prompt": request.prompt,
            "preview_text": request.preview_text,
        });
        if let Some(voice_id) = &request.voice_id {
            body["voice_id"] = Value::String(voice_id.clone());
        }
        if let Some(enabled) = request.aigc_watermark {
            body["aigc_watermark"] = Value::Bool(enabled);
        }
        let (native, request_id) = self
            .post_json("voice_design", "design-voice", body, credentials, true)
            .await?;
        decode_design_response(
            native,
            self.scope.clone(),
            request_id,
            request.voice_id.as_deref(),
        )
    }

    /// List all or one documented category of voices. This call never activates
    /// an inactive clone as a side effect.
    pub async fn list_voices(
        &self,
        request: &MiniMaxVoiceListRequest,
        credentials: &MiniMaxVoicesCredentials,
    ) -> Result<MiniMaxVoiceList, MiniMaxVoicesError> {
        validate_credential(&credentials.api_key)?;
        let body = json!({ "voice_type": request.voice_type.as_str() });
        let (native, request_id) = self
            .post_json("get_voice", "list-voices", body, credentials, false)
            .await?;
        decode_voice_list(native, self.scope.clone(), request_id)
    }

    /// Delete one cloned or generated voice. System voices are never deletable.
    pub async fn delete_voice(
        &self,
        reference: &MiniMaxVoiceRef,
        credentials: &MiniMaxVoicesCredentials,
    ) -> Result<MiniMaxVoiceDeleteReceipt, MiniMaxVoicesError> {
        self.validate_reference(reference)?;
        if !matches!(
            reference.kind,
            MiniMaxVoiceKind::Cloned | MiniMaxVoiceKind::Generated
        ) {
            return Err(invalid(
                "MiniMax delete_voice supports only cloned or generated voice references",
            ));
        }
        validate_credential(&credentials.api_key)?;
        let voice_type = match reference.kind {
            MiniMaxVoiceKind::Cloned => "voice_cloning",
            MiniMaxVoiceKind::Generated => "voice_generation",
            MiniMaxVoiceKind::System => unreachable!("system voices rejected above"),
        };
        let body = json!({ "voice_type": voice_type, "voice_id": reference.voice_id });
        let (native, request_id) = self
            .post_json("delete_voice", "delete-voice", body, credentials, true)
            .await?;
        decode_delete_response(native, reference.clone(), request_id)
    }

    fn validate_reference(&self, reference: &MiniMaxVoiceRef) -> Result<(), MiniMaxVoicesError> {
        if reference.scope != self.scope {
            return Err(invalid(
                "MiniMax voice reference belongs to another profile, endpoint, account, or region",
            ));
        }
        validate_voice_reference_id(&reference.voice_id)
    }

    async fn post_json(
        &self,
        route: &'static str,
        operation: &'static str,
        body: Value,
        credentials: &MiniMaxVoicesCredentials,
        mutation: bool,
    ) -> Result<(Value, Option<String>), MiniMaxVoicesError> {
        let serialized = serde_json::to_vec(&body)
            .map_err(|_| invalid("MiniMax voices request could not be serialized"))?;
        if serialized.len() > MAX_REQUEST_BYTES {
            return Err(invalid(
                "MiniMax voices request exceeds the 2 MiB client limit",
            ));
        }
        let url = operation_url(&self.config.api_base_url, route)?;
        let deadline = Deadline::after(Some(self.config.request_timeout));
        let request = HttpRequest {
            method: "POST".into(),
            url,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(serialized),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if mutation && is_ambiguous_transport(&source) {
                    MiniMaxVoicesError::OutcomeUnknown { operation, source }
                } else {
                    MiniMaxVoicesError::Llm(source)
                }
            })?;
        decode_response(response, operation, mutation)
    }
}

fn clone_request_body(request: &MiniMaxVoiceCloneRequest) -> Result<Value, MiniMaxVoicesError> {
    let mut body = json!({
        "file_id": request.voice_audio.file_id.parse::<i64>().map_err(|_| invalid("voice file_id is not a positive int64"))?,
        "voice_id": request.voice_id,
    });
    if let Some(prompt) = &request.clone_prompt {
        body["clone_prompt"] = json!({
            "prompt_audio": prompt.audio.file_id.parse::<i64>().map_err(|_| invalid("prompt audio file_id is not a positive int64"))?,
            "prompt_text": prompt.text,
        });
    }
    if let Some(text) = &request.text {
        body["text"] = Value::String(text.clone());
    }
    if let Some(model) = request.model {
        body["model"] = Value::String(model.as_str().to_owned());
    }
    if let Some(language) = request.language_boost {
        body["language_boost"] = serde_json::to_value(language)
            .map_err(|_| invalid("language_boost could not be serialized"))?;
    }
    if let Some(text) = &request.text_validation {
        body["text_validation"] = Value::String(text.clone());
    }
    if let Some(accuracy) = request.accuracy {
        body["accuracy"] = json!(accuracy);
    }
    if let Some(value) = request.need_noise_reduction {
        body["need_noise_reduction"] = Value::Bool(value);
    }
    if let Some(value) = request.need_volume_normalization {
        body["need_volume_normalization"] = Value::Bool(value);
    }
    if let Some(value) = request.aigc_watermark {
        body["aigc_watermark"] = Value::Bool(value);
    }
    Ok(body)
}

fn validate_clone_request(
    request: &MiniMaxVoiceCloneRequest,
    scope: &MiniMaxVoicesScope,
) -> Result<(), MiniMaxVoicesError> {
    validate_custom_voice_id(&request.voice_id)?;
    validate_voice_file(&request.voice_audio, scope, "voice_clone")?;
    if request
        .voice_audio
        .size_bytes
        .is_some_and(|size| size == 0 || size > MAX_VOICE_FILE_BYTES)
    {
        return Err(invalid(
            "MiniMax clone audio must be nonempty and no larger than 20,000,000 bytes",
        ));
    }
    if let Some(duration) = request.voice_audio_duration_seconds {
        if !duration.is_finite() || !(10.0..=300.0).contains(&duration) {
            return Err(invalid(
                "caller-declared MiniMax clone audio duration must be between 10 and 300 seconds",
            ));
        }
    }
    if request.text.is_some() != request.model.is_some() {
        return Err(invalid(
            "MiniMax clone preview requires both text and model",
        ));
    }
    if let Some(text) = &request.text {
        validate_nonempty_text(text, "clone preview text")?;
        if text.chars().count() > 1000 {
            return Err(invalid(
                "MiniMax clone preview text exceeds 1,000 characters",
            ));
        }
    }
    if let Some(text_validation) = &request.text_validation {
        validate_nonempty_text(text_validation, "text_validation")?;
        if text_validation.chars().count() > 200 {
            return Err(invalid("MiniMax text_validation exceeds 200 characters"));
        }
    }
    if request.accuracy.is_some() && request.text_validation.is_none() {
        return Err(invalid(
            "MiniMax accuracy is only used with text_validation",
        ));
    }
    if let Some(accuracy) = request.accuracy {
        if !accuracy.is_finite() || !(0.0..=1.0).contains(&accuracy) {
            return Err(invalid(
                "MiniMax accuracy must be finite and between 0 and 1",
            ));
        }
    }
    if let Some(prompt) = &request.clone_prompt {
        validate_voice_file(&prompt.audio, scope, "prompt_audio")?;
        validate_nonempty_text(&prompt.text, "clone prompt transcript")?;
        if let Some(duration) = prompt.duration_seconds {
            if !duration.is_finite() || duration <= 0.0 || duration >= 8.0 {
                return Err(invalid("caller-declared MiniMax prompt audio duration must be greater than zero and less than 8 seconds"));
            }
        }
    }
    Ok(())
}

fn validate_design_request(
    request: &MiniMaxVoiceDesignRequest,
    region: MiniMaxVoicesRegion,
) -> Result<(), MiniMaxVoicesError> {
    validate_nonempty_text(&request.prompt, "voice design prompt")?;
    validate_nonempty_text(&request.preview_text, "voice design preview_text")?;
    if request.preview_text.chars().count() > 500 {
        return Err(invalid(
            "MiniMax voice design preview_text exceeds 500 characters",
        ));
    }
    if let Some(voice_id) = &request.voice_id {
        validate_voice_reference_id(voice_id)?;
    }
    if request.aigc_watermark.is_some() && region != MiniMaxVoicesRegion::ChinaMainland {
        return Err(invalid(
            "MiniMax voice design aigc_watermark is only documented for the China Mainland API",
        ));
    }
    Ok(())
}

fn validate_voice_file(
    file: &ProviderFileRef,
    scope: &MiniMaxVoicesScope,
    purpose: &'static str,
) -> Result<(), MiniMaxVoicesError> {
    if file.provider_id != scope.provider_id
        || file.profile_name != scope.profile_name
        || file.endpoint_fingerprint != scope.endpoint_fingerprint
        || file.account_scope.as_deref() != Some(scope.account_scope.as_str())
        || file.purpose.as_deref() != Some(purpose)
    {
        return Err(LlmError::PermissionDenied {
            message: "MiniMax voice file belongs to another provider, profile, endpoint, or account, or has the wrong upload purpose".into(),
        }
        .into());
    }
    if !valid_numeric_file_id(&file.file_id) {
        return Err(invalid(
            "MiniMax voice file reference must contain a positive numeric file_id",
        ));
    }
    if let Some(size) = file.size_bytes {
        if size == 0 || size > MAX_VOICE_FILE_BYTES {
            return Err(invalid(
                "MiniMax voice audio file must be nonempty and no larger than 20,000,000 bytes",
            ));
        }
    }
    if let Some(expires_at) = &file.expires_at {
        if expiration_millis(expires_at).is_none() {
            return Err(invalid(
                "MiniMax voice file has an unrecognized expires_at value",
            ));
        }
        let expires_at = expiration_millis(expires_at).expect("validated expiration timestamp");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("system clock is before the Unix epoch"))?
            .as_millis();
        if u128::try_from(expires_at).is_ok_and(|expires| expires <= now) {
            return Err(invalid("MiniMax voice file reference is expired"));
        }
    }
    Ok(())
}

fn expiration_millis(value: &str) -> Option<i64> {
    if let Ok(seconds) = value.parse::<i64>() {
        return seconds.checked_mul(1000);
    }
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|timestamp| timestamp.timestamp_millis())
}

fn validate_custom_voice_id(value: &str) -> Result<(), MiniMaxVoicesError> {
    let bytes = value.as_bytes();
    if !(8..=256).contains(&bytes.len())
        || !bytes[0].is_ascii_alphabetic()
        || !bytes
            .last()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_'))
    {
        return Err(invalid("MiniMax clone voice_id must be 8-256 ASCII characters, start with a letter, use only letters/digits/hyphen/underscore, and end with a letter or digit"));
    }
    Ok(())
}

fn validate_voice_reference_id(value: &str) -> Result<(), MiniMaxVoicesError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        Err(invalid(
            "MiniMax voice_id must be nonempty and contain no control characters",
        ))
    } else {
        Ok(())
    }
}

fn validate_nonempty_text(value: &str, field: &str) -> Result<(), MiniMaxVoicesError> {
    if value.trim().is_empty() {
        Err(invalid(&format!("{field} must be nonempty")))
    } else {
        Ok(())
    }
}

fn validate_credential(key: &Secret<String>) -> Result<(), MiniMaxVoicesError> {
    let value = key.expose_secret();
    if value.trim().is_empty() || value.contains(['\r', '\n']) {
        return Err(invalid("MiniMax API key is empty or malformed"));
    }
    Ok(())
}

fn valid_numeric_file_id(value: &str) -> bool {
    value
        .parse::<u64>()
        .is_ok_and(|id| id > 0 && id <= i64::MAX as u64)
}

fn normalize_api_base_url(
    value: &str,
    region: MiniMaxVoicesRegion,
) -> Result<String, MiniMaxVoicesError> {
    let mut url =
        Url::parse(value).map_err(|_| invalid("MiniMax voices API base URL is invalid"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "MiniMax voices API base URL cannot contain credentials, query, or fragment",
        ));
    }
    if url.path() != "/v1" && url.path() != "/v1/" {
        return Err(invalid("MiniMax voices API base URL must end at /v1"));
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let local_http = url.scheme() == "http"
        && (host == "localhost"
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| address.is_loopback()));
    if local_http {
        url.set_path("/v1");
        return Ok(url.into());
    }
    let expected_host = match region {
        MiniMaxVoicesRegion::International => "api.minimax.io",
        MiniMaxVoicesRegion::ChinaMainland => "api.minimax.cn",
    };
    if url.scheme() != "https"
        || url.port_or_known_default() != Some(443)
        || !host.eq_ignore_ascii_case(expected_host)
    {
        return Err(invalid(
            "MiniMax voices API host does not match the selected region",
        ));
    }
    url.set_path("/v1");
    Ok(url.into())
}

fn operation_url(base: &str, route: &str) -> Result<String, MiniMaxVoicesError> {
    let mut url =
        Url::parse(base).map_err(|_| invalid("MiniMax voices API base URL is invalid"))?;
    url.path_segments_mut()
        .map_err(|_| invalid("MiniMax voices API base URL cannot accept a route"))?
        .push(route);
    Ok(url.into())
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

fn is_ambiguous_transport(error: &LlmError) -> bool {
    matches!(
        error,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    )
}

fn decode_response(
    response: HttpResponse,
    operation: &'static str,
    mutation: bool,
) -> Result<(Value, Option<String>), MiniMaxVoicesError> {
    let status = response.status;
    let mut request_id = response_id(&response.headers);
    let native = match serde_json::from_slice::<Value>(&response.body) {
        Ok(value) => value,
        Err(_) if !(200..300).contains(&status) => {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }
        Err(_) if mutation => {
            return Err(response_unknown(
                operation,
                "successful HTTP response body is not JSON",
                request_id,
                Value::String(String::from_utf8_lossy(&response.body).into_owned()),
            ));
        }
        Err(_) => {
            return Err(invalid_response(
                operation,
                "successful HTTP response body is not JSON",
                request_id,
                Value::String(String::from_utf8_lossy(&response.body).into_owned()),
            ));
        }
    };
    request_id = native
        .get("trace_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(request_id);
    if !(200..300).contains(&status) {
        let dispatch = if mutation && (status == 408 || status >= 500) {
            MiniMaxVoicesDispatch::Unknown
        } else {
            MiniMaxVoicesDispatch::Rejected
        };
        return Err(provider_error(
            operation,
            Some(status),
            base_response_code(&native),
            base_response_message(&native).unwrap_or_else(|| "MiniMax rejected the request".into()),
            request_id,
            native,
            dispatch,
        ));
    }
    match base_response_code(&native) {
        Some(0) => Ok((native, request_id)),
        Some(code) => Err(provider_error(
            operation,
            Some(status),
            Some(code),
            base_response_message(&native).unwrap_or_else(|| "MiniMax rejected the request".into()),
            request_id,
            native,
            MiniMaxVoicesDispatch::Rejected,
        )),
        None if mutation => Err(response_unknown(
            operation,
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

fn base_response_code(value: &Value) -> Option<i64> {
    value.get("base_resp")?.get("status_code")?.as_i64()
}

fn base_response_message(value: &Value) -> Option<String> {
    value
        .get("base_resp")?
        .get("status_msg")?
        .as_str()
        .map(str::to_owned)
}

fn parse_base_response(
    value: &Value,
    operation: &'static str,
    request_id: Option<String>,
) -> Result<MiniMaxVoicesBaseResponse, MiniMaxVoicesError> {
    let status_code = value
        .get("base_resp")
        .and_then(|base| base.get("status_code"))
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            invalid_response(
                operation,
                "response has no integer base_resp.status_code",
                request_id.clone(),
                value.clone(),
            )
        })?;
    let status_msg = value
        .get("base_resp")
        .and_then(|base| base.get("status_msg"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid_response(
                operation,
                "response has no string base_resp.status_msg",
                request_id,
                value.clone(),
            )
        })?
        .to_owned();
    Ok(MiniMaxVoicesBaseResponse {
        status_code,
        status_msg,
    })
}

fn decode_clone_response(
    native: Value,
    scope: MiniMaxVoicesScope,
    request_id: Option<String>,
    requested_voice_id: &str,
) -> Result<MiniMaxVoiceCloneResult, MiniMaxVoicesError> {
    let response_voice_id =
        optional_string_field(&native, "voice_id", "clone-voice", request_id.clone())
            .map_err(|error| unknown_after_success("clone-voice", error))?;
    if response_voice_id
        .as_deref()
        .is_some_and(|response_id| response_id != requested_voice_id)
    {
        return Err(response_unknown(
            "clone-voice",
            "successful response voice_id did not match the requested ID",
            request_id,
            native,
        ));
    }
    let base_resp = parse_base_response(&native, "clone-voice", request_id.clone())
        .map_err(|error| unknown_after_success("clone-voice", error))?;
    let input_sensitive = native.get("input_sensitive").cloned();
    let input_sensitive_type = native.get("input_sensitive_type").cloned();
    let demo_audio =
        optional_string_field(&native, "demo_audio", "clone-voice", request_id.clone())
            .map_err(|error| unknown_after_success("clone-voice", error))?;
    let extra_info = match native.get("extra_info") {
        None | Some(Value::Null) => None,
        Some(Value::Object(info)) => Some(
            decode_preview_info(info, &native, request_id.clone())
                .map_err(|error| unknown_after_success("clone-voice", error))?,
        ),
        Some(_) => {
            return Err(response_unknown(
                "clone-voice",
                "successful response extra_info was not an object",
                request_id,
                native,
            ));
        }
    };
    let reference = MiniMaxVoiceRef::new(scope, MiniMaxVoiceKind::Cloned, requested_voice_id)
        .map_err(|error| unknown_after_success("clone-voice", error))?;
    Ok(MiniMaxVoiceCloneResult {
        reference,
        input_sensitive,
        input_sensitive_type,
        demo_audio,
        extra_info,
        base_resp,
        request_id,
        native,
    })
}

fn decode_preview_info(
    info: &serde_json::Map<String, Value>,
    full_native: &Value,
    request_id: Option<String>,
) -> Result<MiniMaxVoicePreviewInfo, MiniMaxVoicesError> {
    let parse_optional_u64 = |field: &'static str| -> Result<Option<u64>, MiniMaxVoicesError> {
        match info.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_u64().map(Some).ok_or_else(|| {
                invalid_response(
                    "clone-voice",
                    &format!("extra_info.{field} must be a nonnegative integer"),
                    request_id.clone(),
                    full_native.clone(),
                )
            }),
        }
    };
    Ok(MiniMaxVoicePreviewInfo {
        audio_length: parse_optional_u64("audio_length")?,
        audio_sample_rate: parse_optional_u64("audio_sample_rate")?,
        audio_size: parse_optional_u64("audio_size")?,
        bitrate: parse_optional_u64("bitrate")?,
        word_count: parse_optional_u64("word_count")?,
        usage_characters: parse_optional_u64("usage_characters")?,
        native: Value::Object(info.clone()),
    })
}

fn decode_design_response(
    native: Value,
    scope: MiniMaxVoicesScope,
    request_id: Option<String>,
    requested_voice_id: Option<&str>,
) -> Result<MiniMaxVoiceDesignResult, MiniMaxVoicesError> {
    let response_voice_id = native
        .get("voice_id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| {
            response_unknown(
                "design-voice",
                "successful response omitted voice_id",
                request_id.clone(),
                native.clone(),
            )
        })?;
    if requested_voice_id.is_some_and(|requested| requested != response_voice_id) {
        return Err(response_unknown(
            "design-voice",
            "successful response voice_id did not match the requested ID",
            request_id,
            native,
        ));
    }
    let base_resp = parse_base_response(&native, "design-voice", request_id.clone())
        .map_err(|error| unknown_after_success("design-voice", error))?;
    let trial_audio =
        optional_string_field(&native, "trial_audio", "design-voice", request_id.clone())
            .map_err(|error| unknown_after_success("design-voice", error))?;
    let reference = MiniMaxVoiceRef::new(scope, MiniMaxVoiceKind::Generated, response_voice_id)
        .map_err(|error| unknown_after_success("design-voice", error))?;
    Ok(MiniMaxVoiceDesignResult {
        reference,
        trial_audio,
        base_resp,
        request_id,
        native,
    })
}

fn decode_voice_list(
    native: Value,
    scope: MiniMaxVoicesScope,
    request_id: Option<String>,
) -> Result<MiniMaxVoiceList, MiniMaxVoicesError> {
    let base_resp = parse_base_response(&native, "list-voices", request_id.clone())?;
    let system_voice = decode_voice_category(
        &native,
        "system_voice",
        MiniMaxVoiceKind::System,
        &scope,
        request_id.clone(),
    )?;
    let voice_cloning = decode_voice_category(
        &native,
        "voice_cloning",
        MiniMaxVoiceKind::Cloned,
        &scope,
        request_id.clone(),
    )?;
    let voice_generation = decode_voice_category(
        &native,
        "voice_generation",
        MiniMaxVoiceKind::Generated,
        &scope,
        request_id.clone(),
    )?;
    Ok(MiniMaxVoiceList {
        scope,
        system_voice,
        voice_cloning,
        voice_generation,
        base_resp,
        request_id,
        native,
    })
}

fn decode_voice_category(
    native: &Value,
    category: &'static str,
    kind: MiniMaxVoiceKind,
    scope: &MiniMaxVoicesScope,
    request_id: Option<String>,
) -> Result<Vec<MiniMaxVoice>, MiniMaxVoicesError> {
    let values = match native.get(category) {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(values)) => values,
        Some(_) => {
            return Err(invalid_response(
                "list-voices",
                &format!("{category} must be an array when present"),
                request_id,
                native.clone(),
            ));
        }
    };
    let mut voices = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Some(voice_id) = value
            .get("voice_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty() && !id.chars().any(char::is_control))
        else {
            return Err(invalid_response(
                "list-voices",
                &format!("{category}[{index}] has no valid voice_id"),
                request_id,
                native.clone(),
            ));
        };
        let description = match value.get("description") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut descriptions = Vec::with_capacity(items.len());
                for item in items {
                    let Some(description) = item.as_str() else {
                        return Err(invalid_response(
                            "list-voices",
                            &format!("{category}[{index}].description must contain only strings"),
                            request_id,
                            native.clone(),
                        ));
                    };
                    descriptions.push(description.to_owned());
                }
                descriptions
            }
            Some(_) => {
                return Err(invalid_response(
                    "list-voices",
                    &format!("{category}[{index}].description must be an array"),
                    request_id,
                    native.clone(),
                ));
            }
        };
        let voice_name =
            optional_string_field(value, "voice_name", "list-voices", request_id.clone())?;
        let created_time =
            optional_string_field(value, "created_time", "list-voices", request_id.clone())?;
        let reference = MiniMaxVoiceRef::new(scope.clone(), kind, voice_id).map_err(|error| {
            invalid_response(
                "list-voices",
                &error.to_string(),
                request_id.clone(),
                native.clone(),
            )
        })?;
        voices.push(MiniMaxVoice {
            reference,
            voice_name,
            description,
            created_time,
            native: value.clone(),
        });
    }
    Ok(voices)
}

fn decode_delete_response(
    native: Value,
    reference: MiniMaxVoiceRef,
    request_id: Option<String>,
) -> Result<MiniMaxVoiceDeleteReceipt, MiniMaxVoicesError> {
    let Some(response_id) = native.get("voice_id").and_then(Value::as_str) else {
        return Err(response_unknown(
            "delete-voice",
            "successful response omitted voice_id",
            request_id,
            native,
        ));
    };
    if response_id != reference.voice_id {
        return Err(response_unknown(
            "delete-voice",
            "successful response voice_id did not match the requested resource",
            request_id,
            native,
        ));
    }
    let base_resp = parse_base_response(&native, "delete-voice", request_id.clone())
        .map_err(|error| unknown_after_success("delete-voice", error))?;
    let created_time =
        optional_string_field(&native, "created_time", "delete-voice", request_id.clone())
            .map_err(|error| unknown_after_success("delete-voice", error))?;
    Ok(MiniMaxVoiceDeleteReceipt {
        reference,
        created_time,
        base_resp,
        request_id,
        native,
    })
}

fn optional_string_field(
    value: &Value,
    field: &'static str,
    operation: &'static str,
    request_id: Option<String>,
) -> Result<Option<String>, MiniMaxVoicesError> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid_response(
            operation,
            &format!("{field} must be a string when present"),
            request_id,
            value.clone(),
        )),
    }
}

fn unknown_after_success(operation: &'static str, error: MiniMaxVoicesError) -> MiniMaxVoicesError {
    match error {
        MiniMaxVoicesError::InvalidResponse {
            message,
            request_id,
            native,
            ..
        } => response_unknown(operation, &message, request_id, *native),
        other => other,
    }
}

fn provider_error(
    operation: &'static str,
    http_status: Option<u16>,
    code: Option<i64>,
    message: String,
    request_id: Option<String>,
    native: Value,
    dispatch: MiniMaxVoicesDispatch,
) -> MiniMaxVoicesError {
    MiniMaxVoicesError::Provider {
        operation,
        http_status,
        code,
        message,
        request_id,
        native: Box::new(native),
        dispatch,
    }
}

fn invalid_response(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> MiniMaxVoicesError {
    MiniMaxVoicesError::InvalidResponse {
        operation,
        message: message.to_owned(),
        request_id,
        native: Box::new(native),
    }
}

fn response_unknown(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> MiniMaxVoicesError {
    MiniMaxVoicesError::ResponseOutcomeUnknown {
        operation,
        message: message.to_owned(),
        request_id,
        native: Box::new(native),
    }
}

fn invalid(message: &str) -> MiniMaxVoicesError {
    MiniMaxVoicesError::InvalidRequest(message.to_owned())
}
