//! Qwen Audio Generation over the Model Studio Beijing workspace API.
//!
//! This is an independent non-streaming AudioGen service. It returns the
//! provider's short-lived result URL and never downloads audio, retries a
//! request, or routes through Qwen TTS or a Chat endpoint.

#![doc = concat!(
    include_str!("../../docs/qwen-audio-generation.md"),
    "\n\n",
    include_str!("../../docs/qwen-audio-generation.en.md")
)]

use crate::{
    client::RequestOptions,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, Transport},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{fmt, time::Duration};
use thiserror::Error;
use url::Url;

/// The provider-documented endpoint path for Qwen Audio Generation.
pub const QWEN_AUDIO_GENERATION_PATH: &str = "/api/v1/services/audio/tts/SpeechSynthesizer";

const MAX_TEXT_CHARACTERS: usize = 3_000;
const MAX_REFERENCES: usize = 3;
// The provider publishes a per-reference limit of 10 MB, not an exact byte
// unit. Enforce the conservative decimal cap locally.
const MAX_REFERENCE_BYTES: usize = 10_000_000;
const MAX_REFERENCE_URL_BYTES: usize = 16 * 1024;
const MAX_REQUEST_BYTES: usize = 42_000_000;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Account and Beijing workspace identity for one Qwen Audio Generation
/// service. This protocol currently has a documented Beijing workspace
/// endpoint only; no other region is inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAudioGenerationScope {
    profile_name: String,
    account_scope: String,
    workspace_id: String,
}

impl QwenAudioGenerationScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        workspace_id: impl Into<String>,
    ) -> Result<Self, QwenAudioGenerationError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            workspace_id: workspace_id.into(),
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub fn endpoint(&self) -> String {
        format!(
            "https://{}.cn-beijing.maas.aliyuncs.com{}",
            self.workspace_id, QWEN_AUDIO_GENERATION_PATH
        )
    }

    fn validate(&self) -> Result<(), QwenAudioGenerationError> {
        if !valid_identity(&self.profile_name) || !valid_identity(&self.account_scope) {
            return Err(invalid(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        validate_workspace_id(&self.workspace_id)
    }
}

/// Output encoding for the generated audio artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QwenAudioGenerationFormat {
    Wav,
    Mp3,
    Pcm,
}

impl QwenAudioGenerationFormat {
    const fn wire(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Pcm => "pcm",
        }
    }
}

/// Output sample rates published for `qwen-audio-3.1-tts-next`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenAudioGenerationSampleRate {
    #[serde(rename = "8000")]
    Hz8000,
    #[serde(rename = "16000")]
    Hz16000,
    #[serde(rename = "24000")]
    Hz24000,
    #[serde(rename = "44100")]
    Hz44100,
    #[serde(rename = "48000")]
    Hz48000,
}

impl QwenAudioGenerationSampleRate {
    const fn wire(self) -> u32 {
        match self {
            Self::Hz8000 => 8_000,
            Self::Hz16000 => 16_000,
            Self::Hz24000 => 24_000,
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
        }
    }
}

/// Output channel count accepted by the Audio Generation API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenAudioGenerationChannels {
    Mono,
    Stereo,
}

impl QwenAudioGenerationChannels {
    const fn wire(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// Container used for inline reference audio. Raw PCM references are not
/// accepted by the provider's published model contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenAudioReferenceFormat {
    Wav,
    Mp3,
    OggOpus,
}

impl QwenAudioReferenceFormat {
    const fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::OggOpus => "audio/ogg",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ReferenceSource {
    Url(String),
    Inline {
        format: QwenAudioReferenceFormat,
        bytes: Bytes,
    },
}

/// One service-readable URL or inline audio-data reference.
///
/// Inline bytes are Base64-encoded into the JSON request. URL references are
/// passed to Model Studio unchanged; this client never fetches them.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenAudioGenerationReference {
    source: ReferenceSource,
}

impl QwenAudioGenerationReference {
    /// Use an HTTP(S) audio URL that Model Studio can read.
    pub fn from_url(url: impl Into<String>) -> Result<Self, QwenAudioGenerationError> {
        let reference = Self {
            source: ReferenceSource::Url(url.into()),
        };
        reference.validate()?;
        Ok(reference)
    }

    /// Embed a WAV, MP3, or OGG Opus clip in the request as a data URI.
    pub fn from_bytes(
        format: QwenAudioReferenceFormat,
        bytes: Bytes,
    ) -> Result<Self, QwenAudioGenerationError> {
        let reference = Self {
            source: ReferenceSource::Inline { format, bytes },
        };
        reference.validate()?;
        Ok(reference)
    }

    fn validate(&self) -> Result<(), QwenAudioGenerationError> {
        match &self.source {
            ReferenceSource::Url(value) => {
                validate_audio_reference_url(value)?;
            }
            ReferenceSource::Inline { bytes, .. } => {
                if bytes.is_empty() {
                    return Err(invalid("reference audio must not be empty"));
                }
                if bytes.len() > MAX_REFERENCE_BYTES {
                    return Err(LlmError::RequestTooLarge {
                        message: "Qwen Audio Generation reference audio exceeds the local 10 MB per-clip cap".into(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    fn to_wire(&self) -> Value {
        match &self.source {
            ReferenceSource::Url(url) => json!({ "audio_url": url }),
            ReferenceSource::Inline { format, bytes } => json!({
                "audio_data": format!(
                    "data:{};base64,{}",
                    format.mime_type(),
                    BASE64.encode(bytes)
                )
            }),
        }
    }
}

impl fmt::Debug for QwenAudioGenerationReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            ReferenceSource::Url(_) => f
                .debug_struct("QwenAudioGenerationReference")
                .field("source", &"<redacted audio URL>")
                .finish(),
            ReferenceSource::Inline { format, bytes } => f
                .debug_struct("QwenAudioGenerationReference")
                .field("format", &format)
                .field("bytes", &bytes.len())
                .finish(),
        }
    }
}

/// One complete text prompt and optional reference clips for AudioGen.
///
/// Defaults remain omitted so Model Studio applies its documented defaults.
#[derive(Clone, PartialEq)]
pub struct QwenAudioGenerationRequest {
    text_prompt: String,
    references: Vec<QwenAudioGenerationReference>,
    format: Option<QwenAudioGenerationFormat>,
    sample_rate: Option<QwenAudioGenerationSampleRate>,
    channels: Option<QwenAudioGenerationChannels>,
    volume: Option<u8>,
    enable_cbr: Option<bool>,
    bit_rate: Option<u16>,
    quality: Option<u8>,
    rate: Option<f64>,
    seed: Option<i64>,
    enable_aigc_tag: Option<bool>,
}

impl QwenAudioGenerationRequest {
    pub fn new(text_prompt: impl Into<String>) -> Self {
        Self {
            text_prompt: text_prompt.into(),
            references: Vec::new(),
            format: None,
            sample_rate: None,
            channels: None,
            volume: None,
            enable_cbr: None,
            bit_rate: None,
            quality: None,
            rate: None,
            seed: None,
            enable_aigc_tag: None,
        }
    }

    pub fn with_reference(mut self, reference: QwenAudioGenerationReference) -> Self {
        self.references.push(reference);
        self
    }

    pub fn with_format(mut self, format: QwenAudioGenerationFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: QwenAudioGenerationSampleRate) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub fn with_channels(mut self, channels: QwenAudioGenerationChannels) -> Self {
        self.channels = Some(channels);
        self
    }

    pub fn with_volume(mut self, volume: u8) -> Self {
        self.volume = Some(volume);
        self
    }

    pub fn with_constant_bitrate(mut self, bit_rate_kbps: u16) -> Self {
        self.format = Some(QwenAudioGenerationFormat::Mp3);
        self.enable_cbr = Some(true);
        self.bit_rate = Some(bit_rate_kbps);
        self.quality = None;
        self
    }

    pub fn with_vbr_quality(mut self, quality: u8) -> Self {
        self.format = Some(QwenAudioGenerationFormat::Mp3);
        self.enable_cbr = Some(false);
        self.quality = Some(quality);
        self.bit_rate = None;
        self
    }

    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = Some(rate);
        self
    }

    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_aigc_tag(mut self, enabled: bool) -> Self {
        self.enable_aigc_tag = Some(enabled);
        self
    }

    pub fn text_prompt(&self) -> &str {
        &self.text_prompt
    }

    pub fn references(&self) -> &[QwenAudioGenerationReference] {
        &self.references
    }

    fn validate(&self) -> Result<(), QwenAudioGenerationError> {
        if self.text_prompt.trim().is_empty()
            || self.text_prompt.chars().count() > MAX_TEXT_CHARACTERS
            || self.text_prompt.contains('\0')
        {
            return Err(invalid(
                "text_prompt must contain 1–3000 non-NUL characters",
            ));
        }
        if self.references.len() > MAX_REFERENCES {
            return Err(invalid("at most three reference clips are accepted"));
        }
        validate_voice_markers(&self.text_prompt, self.references.len())?;
        for reference in &self.references {
            reference.validate()?;
        }
        if self.volume.is_some_and(|volume| volume > 100) {
            return Err(invalid("volume must be between 0 and 100"));
        }
        if self
            .rate
            .is_some_and(|rate| !rate.is_finite() || !(0.5..=2.0).contains(&rate))
        {
            return Err(invalid("rate must be finite and between 0.5 and 2.0"));
        }
        if self.quality.is_some_and(|quality| quality > 9) {
            return Err(invalid("MP3 VBR quality must be between 0 and 9"));
        }
        let format = self.format.unwrap_or(QwenAudioGenerationFormat::Wav);
        if self.enable_cbr == Some(true) && format != QwenAudioGenerationFormat::Mp3 {
            return Err(invalid("enable_cbr is supported only for MP3 output"));
        }
        if self.bit_rate.is_some()
            && (format != QwenAudioGenerationFormat::Mp3 || self.enable_cbr != Some(true))
        {
            return Err(invalid(
                "bit_rate requires MP3 output with constant bitrate enabled",
            ));
        }
        if self.quality.is_some()
            && (format != QwenAudioGenerationFormat::Mp3 || self.enable_cbr == Some(true))
        {
            return Err(invalid(
                "quality is supported only for MP3 variable-bitrate output",
            ));
        }
        Ok(())
    }

    fn to_wire(&self) -> Value {
        let mut input = Map::new();
        input.insert("text_prompt".into(), json!(self.text_prompt));
        if !self.references.is_empty() {
            input.insert(
                "references".into(),
                Value::Array(
                    self.references
                        .iter()
                        .map(|value| value.to_wire())
                        .collect(),
                ),
            );
        }
        if let Some(format) = self.format {
            input.insert("format".into(), json!(format.wire()));
        }
        if let Some(sample_rate) = self.sample_rate {
            input.insert("sample_rate".into(), json!(sample_rate.wire()));
        }
        if let Some(channels) = self.channels {
            input.insert("channels".into(), json!(channels.wire()));
        }
        if let Some(volume) = self.volume {
            input.insert("volume".into(), json!(volume));
        }
        if let Some(enable_cbr) = self.enable_cbr {
            input.insert("enable_cbr".into(), json!(enable_cbr));
        }
        if let Some(bit_rate) = self.bit_rate {
            input.insert("bit_rate".into(), json!(bit_rate));
        }
        if let Some(quality) = self.quality {
            input.insert("quality".into(), json!(quality));
        }
        if let Some(rate) = self.rate {
            input.insert("rate".into(), json!(rate));
        }
        if let Some(seed) = self.seed {
            input.insert("seed".into(), json!(seed));
        }
        if let Some(enable_aigc_tag) = self.enable_aigc_tag {
            input.insert("enable_aigc_tag".into(), json!(enable_aigc_tag));
        }
        json!({ "model": "qwen-audio-3.1-tts-next", "input": input })
    }
}

impl fmt::Debug for QwenAudioGenerationRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioGenerationRequest")
            .field("text_prompt", &"<redacted prompt>")
            .field("references", &self.references)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("volume", &self.volume)
            .field("enable_cbr", &self.enable_cbr)
            .field("bit_rate", &self.bit_rate)
            .field("quality", &self.quality)
            .field("rate", &self.rate)
            .field("seed", &self.seed)
            .field("enable_aigc_tag", &self.enable_aigc_tag)
            .finish()
    }
}

/// Completed non-streaming AudioGen result. The signed `audio_url` expires;
/// the caller decides whether to fetch it. Debug output redacts it and the
/// native provider payload.
#[derive(Clone, PartialEq)]
pub struct QwenAudioGenerationOutput {
    scope: QwenAudioGenerationScope,
    pub request_id: Option<String>,
    audio_url: String,
    pub audio_id: Option<String>,
    pub expires_at_unix_seconds: Option<u64>,
    pub duration_seconds: Option<f64>,
    pub finish_reason: String,
    pub usage_duration_seconds: Option<u64>,
    pub native: Value,
}

impl QwenAudioGenerationOutput {
    pub fn scope(&self) -> &QwenAudioGenerationScope {
        &self.scope
    }

    pub fn audio_url(&self) -> &str {
        &self.audio_url
    }
}

impl fmt::Debug for QwenAudioGenerationOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioGenerationOutput")
            .field("scope", &self.scope)
            .field("request_id", &self.request_id)
            .field("audio_url", &"<redacted short-lived URL>")
            .field("audio_id", &self.audio_id)
            .field("expires_at_unix_seconds", &self.expires_at_unix_seconds)
            .field("duration_seconds", &self.duration_seconds)
            .field("finish_reason", &self.finish_reason)
            .field("usage_duration_seconds", &self.usage_duration_seconds)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}

/// How far a Qwen Audio Generation request may have progressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenAudioGenerationDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

/// Errors from Qwen Audio Generation. No call is retried; `Unknown` and
/// `Accepted` may already have been processed or billed.
#[derive(Error)]
pub enum QwenAudioGenerationError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Qwen Audio Generation input: {0}")]
    InvalidInput(String),
    #[error("Qwen Audio Generation account scope does not match the request options")]
    ScopeMismatch,
    #[error("Qwen Audio Generation request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen Audio Generation was rejected with HTTP {status}: {message}")]
    Rejected {
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error(
        "Qwen Audio Generation accepted the request but returned an invalid response: {message}"
    )]
    AcceptedInvalidResponse {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
}

impl fmt::Debug for QwenAudioGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(error) => f.debug_tuple("Llm").field(error).finish(),
            Self::InvalidInput(message) => f.debug_tuple("InvalidInput").field(message).finish(),
            Self::ScopeMismatch => f.write_str("ScopeMismatch"),
            Self::OutcomeUnknown { source, request_id } => f
                .debug_struct("OutcomeUnknown")
                .field("source", source)
                .field("request_id", request_id)
                .finish(),
            Self::Rejected {
                status,
                code,
                request_id,
                ..
            } => f
                .debug_struct("Rejected")
                .field("status", status)
                .field("code", code)
                .field("message", &"<redacted provider message>")
                .field("request_id", request_id)
                .field("native", &"<redacted provider payload>")
                .finish(),
            Self::AcceptedInvalidResponse {
                message,
                request_id,
                ..
            } => f
                .debug_struct("AcceptedInvalidResponse")
                .field("message", message)
                .field("request_id", request_id)
                .field("native", &"<redacted provider payload>")
                .finish(),
        }
    }
}

impl QwenAudioGenerationError {
    pub fn dispatch(&self) -> QwenAudioGenerationDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::ScopeMismatch => {
                QwenAudioGenerationDispatch::NotSent
            }
            Self::Rejected { .. } => QwenAudioGenerationDispatch::Rejected,
            Self::OutcomeUnknown { .. } => QwenAudioGenerationDispatch::Unknown,
            Self::AcceptedInvalidResponse { .. } => QwenAudioGenerationDispatch::Accepted,
        }
    }
}

/// Independent Qwen Audio Generation API client bound to one workspace.
pub struct QwenAudioGenerationService<'a> {
    transport: &'a dyn Transport,
    scope: QwenAudioGenerationScope,
}

impl<'a> QwenAudioGenerationService<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        scope: QwenAudioGenerationScope,
    ) -> Result<Self, QwenAudioGenerationError> {
        scope.validate()?;
        Ok(Self { transport, scope })
    }

    pub fn scope(&self) -> &QwenAudioGenerationScope {
        &self.scope
    }

    /// Submit one non-streaming AudioGen request and return its native result
    /// URL. The URL is not fetched and an ambiguous POST is never repeated.
    pub async fn synthesize(
        &self,
        request: &QwenAudioGenerationRequest,
        options: &RequestOptions,
    ) -> Result<QwenAudioGenerationOutput, QwenAudioGenerationError> {
        let http_request = self.http_request(request, options)?;
        let response = HttpExecutor::new(self.transport)
            .execute_bounded(http_request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| QwenAudioGenerationError::OutcomeUnknown {
                source,
                request_id: None,
            })?;

        let native = match serde_json::from_slice::<Value>(&response.body) {
            Ok(value) => value,
            Err(_) if !(200..300).contains(&response.status) => {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            }
            Err(error) => {
                return Err(QwenAudioGenerationError::AcceptedInvalidResponse {
                    message: format!("successful response JSON could not be decoded: {error}"),
                    request_id: response.header("x-request-id").map(str::to_owned),
                    native: Box::new(Value::Null),
                });
            }
        };
        let request_id = native
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| response.header("x-request-id").map(str::to_owned));

        if !(200..300).contains(&response.status) {
            let (code, message) = error_details(&native);
            if (400..500).contains(&response.status) && response.status != 408 {
                return Err(QwenAudioGenerationError::Rejected {
                    status: response.status,
                    code,
                    message,
                    request_id,
                    native: Box::new(native),
                });
            }
            return Err(QwenAudioGenerationError::OutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "Model Studio returned HTTP {} after Audio Generation submission: {message}",
                        response.status
                    ),
                },
                request_id,
            });
        }

        decode_output(native, request_id, &self.scope)
    }

    fn http_request(
        &self,
        request: &QwenAudioGenerationRequest,
        options: &RequestOptions,
    ) -> Result<HttpRequest, QwenAudioGenerationError> {
        self.scope.validate()?;
        request.validate()?;
        if options
            .total_timeout
            .is_some_and(|timeout| timeout.is_zero())
        {
            return Err(invalid("total_timeout must be greater than zero"));
        }
        if options
            .account_scope
            .as_deref()
            .is_some_and(|value| value != self.scope.account_scope)
        {
            return Err(QwenAudioGenerationError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen Audio Generation requires the Beijing Model Studio API key".into(),
            })?;
        if credential.expose_secret().trim().is_empty()
            || credential.expose_secret().len() > 16 * 1024
            || credential.expose_secret().chars().any(char::is_control)
        {
            return Err(invalid(
                "API key must be non-empty, at most 16 KiB, and contain no control characters",
            ));
        }
        let body = serde_json::to_vec(&request.to_wire()).map_err(|error| {
            invalid(format!(
                "could not encode Audio Generation request: {error}"
            ))
        })?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: "Qwen Audio Generation request exceeds the local 42 MB JSON bound".into(),
            }
            .into());
        }
        Ok(HttpRequest {
            method: "POST".into(),
            url: self.scope.endpoint(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                ),
                ("content-type".into(), "application/json".into()),
                ("accept".into(), "application/json".into()),
            ],
            body: Bytes::from(body),
            timeout: Some(options.total_timeout.unwrap_or(DEFAULT_TIMEOUT)),
        })
    }
}

fn decode_output(
    native: Value,
    request_id: Option<String>,
    scope: &QwenAudioGenerationScope,
) -> Result<QwenAudioGenerationOutput, QwenAudioGenerationError> {
    let fail = |message: &str| QwenAudioGenerationError::AcceptedInvalidResponse {
        message: message.into(),
        request_id: request_id.clone(),
        native: Box::new(native.clone()),
    };
    let finish_reason = native
        .pointer("/output/finish_reason")
        .and_then(Value::as_str)
        .filter(|value| *value == "stop")
        .ok_or_else(|| fail("missing or non-success output.finish_reason"))?
        .to_owned();
    let audio = native
        .pointer("/output/audio")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("missing output.audio object"))?;
    let audio_url = audio
        .get("url")
        .and_then(Value::as_str)
        .filter(|value| validate_result_url(value))
        .ok_or_else(|| fail("missing or invalid output.audio.url"))?
        .to_owned();
    let duration_seconds = audio.get("duration").and_then(Value::as_f64);
    if duration_seconds.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(fail(
            "output.audio.duration must be a finite non-negative number",
        ));
    }

    Ok(QwenAudioGenerationOutput {
        scope: scope.clone(),
        request_id,
        audio_url,
        audio_id: audio.get("id").and_then(Value::as_str).map(str::to_owned),
        expires_at_unix_seconds: audio.get("expires_at").and_then(Value::as_u64),
        duration_seconds,
        finish_reason,
        usage_duration_seconds: native.pointer("/usage/duration").and_then(Value::as_u64),
        native,
    })
}

fn error_details(native: &Value) -> (Option<String>, String) {
    let code = native
        .get("code")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let message = native
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| "Model Studio rejected the Audio Generation request".into());
    (code, message)
}

fn validate_workspace_id(value: &str) -> Result<(), QwenAudioGenerationError> {
    let bytes = value.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 63
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes[bytes.len() - 1].is_ascii_alphanumeric()
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
    {
        return Err(invalid(
            "workspace ID must be one DNS label with letters, digits, and internal hyphens",
        ));
    }
    Ok(())
}

fn validate_voice_markers(
    prompt: &str,
    reference_count: usize,
) -> Result<(), QwenAudioGenerationError> {
    let bytes = prompt.as_bytes();
    let marker = b"@voice";
    let mut offset = 0;
    while let Some(relative) = bytes[offset..]
        .windows(marker.len())
        .position(|window| window == marker)
    {
        let start = offset + relative + marker.len();
        let digits_end = bytes[start..]
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .map_or(bytes.len(), |length| start + length);
        if digits_end > start {
            let index = prompt[start..digits_end]
                .parse::<usize>()
                .map_err(|_| invalid("reference voice marker index is invalid"))?;
            if !(1..=MAX_REFERENCES).contains(&index) || index > reference_count {
                return Err(invalid(
                    "@voice markers must refer to an existing reference clip from 1 to 3",
                ));
            }
        }
        offset = start.max(offset + relative + marker.len());
        if offset >= bytes.len() {
            break;
        }
    }
    Ok(())
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn validate_audio_reference_url(value: &str) -> Result<(), QwenAudioGenerationError> {
    if value.len() > MAX_REFERENCE_URL_BYTES {
        return Err(invalid(
            "reference audio URL exceeds the local 16 KiB bound",
        ));
    }
    let url = Url::parse(value).map_err(|_| invalid("reference audio URL is invalid"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "reference audio URL must be an HTTP(S) URL without user info or a fragment",
        ));
    }
    Ok(())
}

fn validate_result_url(value: &str) -> bool {
    Url::parse(value).is_ok_and(|url| {
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.fragment().is_none()
    })
}

fn invalid(message: impl Into<String>) -> QwenAudioGenerationError {
    QwenAudioGenerationError::InvalidInput(message.into())
}
