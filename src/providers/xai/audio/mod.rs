//! Native xAI batch speech-to-text and text-to-speech HTTP routes.
//!
//! xAI's `/v1/stt` endpoint accepts a multipart audio upload and returns a
//! JSON transcript. Its `/v1/tts` endpoint returns raw audio bytes by default;
//! when character timestamps are requested it instead returns a JSON envelope
//! with base64 audio. These routes are implemented directly and do not assume
//! OpenAI audio request or response shapes.

mod custom_voices;
mod url;
mod voices;

pub use custom_voices::{
    XaiCustomVoice, XaiCustomVoiceAge, XaiCustomVoiceAudioStream, XaiCustomVoiceAudioStreamError,
    XaiCustomVoiceCreateRequest, XaiCustomVoiceCursor, XaiCustomVoiceDeleteReceipt,
    XaiCustomVoiceGender, XaiCustomVoiceList, XaiCustomVoiceListRequest, XaiCustomVoicePatch,
    XaiCustomVoiceRef, XaiCustomVoiceTone, XaiCustomVoiceUseCase,
};
pub use url::XaiTranscriptionUrl;
pub use voices::{XaiVoice, XaiVoiceDetails, XaiVoiceList};

use crate::{
    files::{
        append_field, multipart_boundary, provider_file_endpoint_fingerprint, sanitize_filename,
        validate_media_type,
    },
    protocol::{LlmError, ProviderId, Secret},
    providers::openai::audio::AudioInput,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest, Transport},
};
use ::url::Url;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
    time::Duration,
};
use thiserror::Error;

/// xAI's documented inference API root.
pub const XAI_AUDIO_API_BASE_URL: &str = "https://api.x.ai/v1";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_STT_AUDIO_BYTES: u64 = 500_000_000;
const MAX_JSON_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_TTS_TEXT_CHARS: usize = 60_000;
const MAX_KEYTERMS: usize = 100;
const MAX_KEYTERM_CHARS: usize = 50;

/// Secret-free routing settings for one xAI profile and account.
///
/// The profile and account scope are copied onto each result so a host can
/// retain provenance without keeping an audio cache or storing credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiAudioConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub api_base_url: String,
    pub request_timeout: Duration,
}

impl XaiAudioConfig {
    pub fn new(profile_name: impl Into<String>, account_scope: impl Into<String>) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            api_base_url: XAI_AUDIO_API_BASE_URL.into(),
            request_timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn with_api_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.api_base_url = base_url.into();
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// xAI API key supplied by the host for every operation.
pub struct XaiAudioCredentials {
    api_key: Secret<String>,
}

impl XaiAudioCredentials {
    pub fn new(api_key: Secret<String>) -> Self {
        Self { api_key }
    }
}

impl fmt::Debug for XaiAudioCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiAudioCredentials")
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Non-secret identity of an xAI audio request's destination and owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiAudioScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub api_endpoint_fingerprint: String,
    pub account_scope: String,
}

/// Selects a published xAI speech transcription model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiTranscriptionModel {
    #[serde(rename = "grok-voice-transcribe-1.0")]
    GrokVoiceTranscribe1,
    #[serde(rename = "grok-voice-transcribe-2.0")]
    #[default]
    GrokVoiceTranscribe2,
}

impl XaiTranscriptionModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GrokVoiceTranscribe1 => "grok-voice-transcribe-1.0",
            Self::GrokVoiceTranscribe2 => "grok-voice-transcribe-2.0",
        }
    }
}

/// xAI format hints for raw/headerless STT audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiRawAudioFormat {
    Pcm,
    Mulaw,
    Alaw,
}

impl XaiRawAudioFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Mulaw => "mulaw",
            Self::Alaw => "alaw",
        }
    }
}

/// Options for the native `POST /v1/stt` multipart endpoint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiTranscriptionRequest {
    #[serde(default)]
    pub model: XaiTranscriptionModel,
    /// Set only for raw/headerless PCM, μ-law, or A-law input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_format: Option<XaiRawAudioFormat>,
    /// Required with `audio_format`; accepted rates are 8–48 kHz per xAI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Enables inverse text normalization and therefore requires `language`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub format: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub multichannel: bool,
    /// For raw multichannel audio, the interleaved channel count (2–8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u8>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub diarize: bool,
    /// Sent as repeated `keyterm` multipart fields, before the final `file`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keyterms: Vec<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub filler_words: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad_threshold: Option<f64>,
}

/// One word-level STT result, optionally carrying a diarized speaker index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiTranscriptWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Transcript for one channel when xAI multichannel transcription is enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiTranscriptChannel {
    pub index: u32,
    pub text: String,
    #[serde(default)]
    pub words: Vec<XaiTranscriptWord>,
}

/// Parsed JSON result from xAI speech-to-text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiTranscription {
    pub scope: XaiAudioScope,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(default)]
    pub words: Vec<XaiTranscriptWord>,
    #[serde(default)]
    pub channels: Vec<XaiTranscriptChannel>,
    pub request_id: Option<String>,
    /// Full xAI JSON response for fields added by the provider over time.
    pub native: Value,
}

/// Codec accepted by the xAI TTS REST route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiSpeechCodec {
    #[default]
    Mp3,
    Wav,
    Pcm,
    Mulaw,
    Alaw,
}

impl XaiSpeechCodec {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
            Self::Mulaw => "mulaw",
            Self::Alaw => "alaw",
        }
    }

    const fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
            Self::Mulaw => "audio/basic",
            Self::Alaw => "audio/alaw",
        }
    }
}

/// Requested xAI TTS codec and sample/bit rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiSpeechFormat {
    #[serde(default)]
    pub codec: XaiSpeechCodec,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(
        default = "default_bit_rate_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub bit_rate: Option<u32>,
}

impl Default for XaiSpeechFormat {
    fn default() -> Self {
        Self {
            codec: XaiSpeechCodec::default(),
            sample_rate: default_sample_rate(),
            bit_rate: Some(default_bit_rate()),
        }
    }
}

impl XaiSpeechFormat {
    pub const fn new(codec: XaiSpeechCodec) -> Self {
        Self {
            codec,
            sample_rate: 24_000,
            bit_rate: None,
        }
    }

    pub const fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub const fn with_bit_rate(mut self, bit_rate: u32) -> Self {
        self.bit_rate = Some(bit_rate);
        self
    }
}

/// Typed JSON body for xAI `POST /v1/tts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiSpeechRequest {
    pub text: String,
    /// Built-in or caller-created xAI voice identifier.
    pub voice_id: String,
    /// BCP-47 code, or `auto` for detection.
    pub language: String,
    #[serde(default, skip_serializing_if = "is_default_speech_format")]
    pub output_format: XaiSpeechFormat,
    /// Speech speed multiplier. xAI accepts 0.7 through 1.5; `None` leaves
    /// its normal-speed default in effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// xAI streaming synthesis latency level, 0 (default), 1, or 2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimize_streaming_latency: Option<u32>,
    /// Normalize written forms such as numbers and abbreviations for speech.
    #[serde(default, skip_serializing_if = "is_false")]
    pub text_normalization: bool,
    /// Replace phrases with spoken substitutions before synthesis.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub replace: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub with_timestamps: bool,
}

impl XaiSpeechRequest {
    pub fn new(
        text: impl Into<String>,
        voice_id: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            voice_id: voice_id.into(),
            language: language.into(),
            output_format: XaiSpeechFormat::default(),
            speed: None,
            optimize_streaming_latency: None,
            text_normalization: false,
            replace: BTreeMap::new(),
            with_timestamps: false,
        }
    }
}

/// Character timing metadata returned only when `with_timestamps=true`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiAudioTimestamps {
    pub graph_chars: Vec<String>,
    /// Each pair is the start/end time in seconds for the corresponding char.
    pub graph_times: Vec<[f64; 2]>,
}

/// Fully decoded xAI timestamped TTS response. Audio is raw codec bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiTimestampedSpeech {
    pub scope: XaiAudioScope,
    pub audio: Bytes,
    pub content_type: String,
    pub duration_seconds: f64,
    pub audio_timestamps: Option<XaiAudioTimestamps>,
    pub request_id: Option<String>,
}

/// Raw TTS output stream for the default non-timestamped response mode.
pub struct XaiSpeechStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    pub scope: XaiAudioScope,
    pub content_type: String,
    pub request_id: Option<String>,
    delivered_bytes: u64,
    finished: bool,
}

impl XaiSpeechStream {
    /// Return the next raw codec chunk. Dropping the stream cancels the HTTP read.
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, XaiAudioStreamError> {
        if self.finished {
            return Ok(None);
        }
        match self.body.next().await {
            Some(Ok(bytes)) => {
                self.delivered_bytes = self.delivered_bytes.saturating_add(bytes.len() as u64);
                Ok(Some(bytes))
            }
            Some(Err(source)) => {
                self.finished = true;
                Err(XaiAudioStreamError {
                    delivered_bytes: self.delivered_bytes,
                    source: Box::new(source),
                })
            }
            None => {
                self.finished = true;
                if self.delivered_bytes == 0 {
                    Err(XaiAudioStreamError {
                        delivered_bytes: 0,
                        source: Box::new(LlmError::ProviderInternal {
                            message: "xAI TTS returned an empty audio body".into(),
                        }),
                    })
                } else {
                    Ok(None)
                }
            }
        }
    }
}

#[derive(Debug, Error)]
#[error("xAI TTS audio stream interrupted after {delivered_bytes} bytes: {source}")]
pub struct XaiAudioStreamError {
    pub delivered_bytes: u64,
    #[source]
    pub source: Box<LlmError>,
}

/// Synthesis is raw bytes for ordinary calls and a JSON-decoded result when
/// character timestamps change xAI's documented response shape.
pub enum XaiSpeechOutput {
    Audio(XaiSpeechStream),
    Timestamped(XaiTimestampedSpeech),
}

impl fmt::Debug for XaiSpeechOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Audio(stream) => f
                .debug_tuple("Audio")
                .field(&format_args!("{}", stream.content_type))
                .finish(),
            Self::Timestamped(result) => f.debug_tuple("Timestamped").field(result).finish(),
        }
    }
}

#[derive(Debug, Error)]
pub enum XaiAudioError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid xAI audio request: {0}")]
    InvalidRequest(String),
    #[error("xAI {operation} returned HTTP {status}")]
    Provider {
        operation: &'static str,
        status: u16,
        request_id: Option<String>,
        body: Box<Value>,
    },
    #[error("xAI {operation} returned an invalid successful response: {message}")]
    InvalidResponse {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("xAI {operation} outcome is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
        /// Present when xAI returned a successful HTTP response but its
        /// acknowledgement could not establish the final resource state.
        response: Option<Box<XaiAudioUnknownResponse>>,
    },
}

/// A successful response retained when a mutating custom-voice operation's
/// acknowledgement is malformed or inconsistent with the requested resource.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiAudioUnknownResponse {
    pub request_id: Option<String>,
    pub native: Box<Value>,
}

/// Native xAI speech-to-text and text-to-speech HTTP client.
///
/// The service binds to an explicit API root and account scope. Credentials are
/// passed on every call, and no operation is retried or cached automatically.
#[derive(Clone)]
pub struct XaiAudioService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    config: XaiAudioConfig,
    scope: XaiAudioScope,
}

impl<'a> XaiAudioService<'a> {
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

    pub fn new(
        transport: &'a dyn Transport,
        mut config: XaiAudioConfig,
    ) -> Result<Self, XaiAudioError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.request_timeout.is_zero() {
            return Err(invalid("request_timeout must be positive"));
        }
        config.api_base_url = normalize_base_url(&config.api_base_url)?;
        let scope = XaiAudioScope {
            provider_id: ProviderId::new("xai"),
            profile_name: config.profile_name.clone(),
            api_endpoint_fingerprint: provider_file_endpoint_fingerprint(&config.api_base_url),
            account_scope: config.account_scope.clone(),
        };
        Ok(Self {
            binding: None,
            transport,
            config,
            scope,
        })
    }

    pub fn scope(&self) -> &XaiAudioScope {
        &self.scope
    }

    /// Upload one completed audio file to xAI's `/v1/stt` route.
    pub async fn transcribe(
        &self,
        input: AudioInput,
        request: &XaiTranscriptionRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiTranscription, XaiAudioError> {
        let pinned_service = self.pin()?;
        validate_transcription_request(request)?;
        validate_audio_input(&input)?;
        validate_credential(&credentials.api_key)?;

        let boundary = multipart_boundary();
        let (prefix, suffix) = transcription_multipart_parts(&boundary, &input, request);
        let content_length = input
            .size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| invalid("multipart request size overflows"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let body = multipart_stream(prefix, input.body, input.size_bytes, suffix);
        let http = HttpStreamRequest {
            method: "POST".into(),
            url: pinned_service.route_url("stt")?,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body,
            content_length,
            timeout: deadline.remaining()?,
        };
        pinned_service.dispatch_transcription(http, deadline).await
    }

    async fn dispatch_transcription(
        &self,
        http: HttpStreamRequest,
        deadline: Deadline,
    ) -> Result<XaiTranscription, XaiAudioError> {
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_stream_bounded(http, MAX_JSON_RESPONSE_BYTES)
            .await
            .map_err(|source| map_outcome("speech-to-text", source))?;
        self.decode_transcription_response(response)
    }

    fn decode_transcription_response(
        &self,
        response: HttpResponse,
    ) -> Result<XaiTranscription, XaiAudioError> {
        let request_id = response_id(&response.headers);
        if !(200..300).contains(&response.status) {
            return Err(XaiAudioError::Provider {
                operation: "speech-to-text",
                status: response.status,
                request_id,
                body: Box::new(parse_error_body(&response.body)),
            });
        }
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            XaiAudioError::InvalidResponse {
                operation: "speech-to-text",
                message: "response body is not JSON".into(),
                request_id: request_id.clone(),
                native: Box::new(Value::Null),
            }
        })?;
        decode_transcription(native, self.scope.clone(), request_id)
    }

    /// Convert text to speech using xAI's `/v1/tts` route.
    ///
    /// Normal responses remain a raw byte stream. With character timestamps,
    /// xAI changes the response to JSON containing base64 audio and timings;
    /// this branch is decoded into [`XaiTimestampedSpeech`].
    pub async fn synthesize(
        &self,
        request: &XaiSpeechRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiSpeechOutput, XaiAudioError> {
        let pinned_service = self.pin()?;
        validate_speech_request(request)?;
        validate_credential(&credentials.api_key)?;
        let body = serde_json::to_vec(&speech_body(request))
            .map_err(|_| invalid("speech request could not be serialized"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let http = HttpRequest {
            method: "POST".into(),
            url: pinned_service.route_url("tts")?,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(body),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .send(http)
            .await
            .map_err(|source| map_outcome("text-to-speech", source))?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
                .await
                .map_err(|source| map_outcome("text-to-speech", source))?;
            return Err(XaiAudioError::Provider {
                operation: "text-to-speech",
                status: response.status,
                request_id,
                body: Box::new(parse_error_body(&response.body)),
            });
        }

        if request.with_timestamps {
            let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
                .await
                .map_err(|source| map_outcome("text-to-speech", source))?;
            let value = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
                XaiAudioError::InvalidResponse {
                    operation: "text-to-speech",
                    message: "timestamped response body is not JSON".into(),
                    request_id: request_id.clone(),
                    native: Box::new(Value::Null),
                }
            })?;
            return decode_timestamped_speech(value, pinned_service.scope.clone(), request_id)
                .map(XaiSpeechOutput::Timestamped);
        }

        let content_type = response
            .header("content-type")
            .unwrap_or_else(|| request.output_format.codec.content_type())
            .to_owned();
        if !content_type.to_ascii_lowercase().starts_with("audio/") {
            return Err(XaiAudioError::InvalidResponse {
                operation: "text-to-speech",
                message: format!("expected raw audio response, received `{content_type}`"),
                request_id,
                native: Box::new(Value::Null),
            });
        }
        Ok(XaiSpeechOutput::Audio(XaiSpeechStream {
            body: response.body,
            scope: pinned_service.scope.clone(),
            content_type,
            request_id,
            delivered_bytes: 0,
            finished: false,
        }))
    }

    fn route_url(&self, route: &str) -> Result<String, XaiAudioError> {
        let mut url = Url::from_str(&self.config.api_base_url)
            .map_err(|_| invalid("configured API base URL is invalid"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("configured API base URL cannot accept route paths"))?
            .pop_if_empty()
            .push(route);
        Ok(url.into())
    }
}

fn transcription_multipart_parts(
    boundary: &str,
    input: &AudioInput,
    request: &XaiTranscriptionRequest,
) -> (Bytes, Bytes) {
    let mut prefix = transcription_option_fields(boundary, request);
    let filename = sanitize_filename(&input.filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            input.media_type
        )
        .as_bytes(),
    );
    let suffix = Bytes::from(format!("\r\n--{boundary}--\r\n").into_bytes());
    (prefix.freeze(), suffix)
}

fn transcription_option_fields(boundary: &str, request: &XaiTranscriptionRequest) -> BytesMut {
    let mut prefix = BytesMut::new();
    append_field(&mut prefix, boundary, "model", request.model.as_str());
    if let Some(audio_format) = request.audio_format {
        append_field(&mut prefix, boundary, "audio_format", audio_format.as_str());
    }
    if let Some(sample_rate) = request.sample_rate {
        append_field(
            &mut prefix,
            boundary,
            "sample_rate",
            &sample_rate.to_string(),
        );
    }
    if let Some(language) = &request.language {
        append_field(&mut prefix, boundary, "language", language);
    }
    if request.format {
        append_field(&mut prefix, boundary, "format", "true");
    }
    if request.multichannel {
        append_field(&mut prefix, boundary, "multichannel", "true");
    }
    if let Some(channels) = request.channels {
        append_field(&mut prefix, boundary, "channels", &channels.to_string());
    }
    if request.diarize {
        append_field(&mut prefix, boundary, "diarize", "true");
    }
    for keyterm in &request.keyterms {
        append_field(&mut prefix, boundary, "keyterm", keyterm);
    }
    if request.filler_words {
        append_field(&mut prefix, boundary, "filler_words", "true");
    }
    if let Some(vad_threshold) = request.vad_threshold {
        append_field(
            &mut prefix,
            boundary,
            "vad_threshold",
            &vad_threshold.to_string(),
        );
    }
    prefix
}

#[derive(Clone, Copy)]
enum MultipartPhase {
    Prefix,
    Audio,
    Done,
}

struct MultipartState {
    prefix: Option<Bytes>,
    audio: BoxStream<'static, Result<Bytes, LlmError>>,
    suffix: Option<Bytes>,
    phase: MultipartPhase,
    declared_audio_bytes: u64,
    actual_audio_bytes: u64,
}

fn multipart_stream(
    prefix: Bytes,
    audio: BoxStream<'static, Result<Bytes, LlmError>>,
    declared_audio_bytes: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::unfold(
        MultipartState {
            prefix: Some(prefix),
            audio,
            suffix: Some(suffix),
            phase: MultipartPhase::Prefix,
            declared_audio_bytes,
            actual_audio_bytes: 0,
        },
        |mut state| async move {
            match state.phase {
                MultipartPhase::Prefix => {
                    state.phase = MultipartPhase::Audio;
                    Some((Ok(state.prefix.take().expect("prefix has bytes")), state))
                }
                MultipartPhase::Audio => match state.audio.next().await {
                    Some(Ok(chunk)) => {
                        let Some(total) = state.actual_audio_bytes.checked_add(chunk.len() as u64)
                        else {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "xAI STT input byte count overflowed".into(),
                                }),
                                state,
                            ));
                        };
                        if total > state.declared_audio_bytes {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "xAI STT input exceeded its declared length".into(),
                                }),
                                state,
                            ));
                        }
                        state.actual_audio_bytes = total;
                        Some((Ok(chunk), state))
                    }
                    Some(Err(_)) => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "xAI STT input stream was interrupted".into(),
                            }),
                            state,
                        ))
                    }
                    None if state.actual_audio_bytes != state.declared_audio_bytes => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "xAI STT input ended before its declared length".into(),
                            }),
                            state,
                        ))
                    }
                    None => {
                        state.phase = MultipartPhase::Done;
                        Some((Ok(state.suffix.take().expect("suffix has bytes")), state))
                    }
                },
                MultipartPhase::Done => None,
            }
        },
    )
    .boxed()
}

fn decode_transcription(
    native: Value,
    scope: XaiAudioScope,
    request_id: Option<String>,
) -> Result<XaiTranscription, XaiAudioError> {
    let text = native.get("text").and_then(Value::as_str).ok_or_else(|| {
        invalid_response(
            "speech-to-text",
            "transcript has no text",
            request_id.clone(),
            native.clone(),
        )
    })?;
    let words: Vec<XaiTranscriptWord> = decode_words(
        native.get("words"),
        "words",
        request_id.clone(),
        native.clone(),
    )?;
    let channels = native
        .get("channels")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|channel| {
                    let index = channel
                        .get("index")
                        .and_then(Value::as_u64)
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| {
                            invalid_response(
                                "speech-to-text",
                                "channel has no valid index",
                                request_id.clone(),
                                native.clone(),
                            )
                        })?;
                    let text = channel
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            invalid_response(
                                "speech-to-text",
                                "channel has no text",
                                request_id.clone(),
                                native.clone(),
                            )
                        })?
                        .to_owned();
                    let words = decode_words(
                        channel.get("words"),
                        "channel words",
                        request_id.clone(),
                        native.clone(),
                    )?;
                    Ok(XaiTranscriptChannel { index, text, words })
                })
                .collect::<Result<Vec<_>, XaiAudioError>>()
        })
        .transpose()?
        .unwrap_or_default();
    let language = native
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let duration_seconds = native.get("duration").and_then(Value::as_f64);
    if duration_seconds.is_some_and(|duration| !duration.is_finite() || duration < 0.0) {
        return Err(invalid_response(
            "speech-to-text",
            "duration is not finite and nonnegative",
            request_id,
            native,
        ));
    }
    Ok(XaiTranscription {
        scope,
        text: text.to_owned(),
        language,
        duration_seconds,
        words,
        channels,
        request_id,
        native,
    })
}

fn decode_words(
    raw: Option<&Value>,
    field: &'static str,
    request_id: Option<String>,
    native: Value,
) -> Result<Vec<XaiTranscriptWord>, XaiAudioError> {
    let Some(values) = raw else {
        return Ok(Vec::new());
    };
    let values = values.as_array().ok_or_else(|| {
        invalid_response(
            "speech-to-text",
            &format!("{field} is not an array"),
            request_id.clone(),
            native.clone(),
        )
    })?;
    values
        .iter()
        .map(|word| {
            let text = word
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid_response(
                        "speech-to-text",
                        &format!("{field} item has no text"),
                        request_id.clone(),
                        native.clone(),
                    )
                })?
                .to_owned();
            let start = word.get("start").and_then(Value::as_f64).ok_or_else(|| {
                invalid_response(
                    "speech-to-text",
                    &format!("{field} item has no start time"),
                    request_id.clone(),
                    native.clone(),
                )
            })?;
            let end = word.get("end").and_then(Value::as_f64).ok_or_else(|| {
                invalid_response(
                    "speech-to-text",
                    &format!("{field} item has no end time"),
                    request_id.clone(),
                    native.clone(),
                )
            })?;
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end < start {
                return Err(invalid_response(
                    "speech-to-text",
                    &format!("{field} item has invalid timestamps"),
                    request_id.clone(),
                    native.clone(),
                ));
            }
            let speaker = word
                .get("speaker")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok());
            Ok(XaiTranscriptWord {
                text,
                start,
                end,
                speaker,
            })
        })
        .collect()
}

fn decode_timestamped_speech(
    native: Value,
    scope: XaiAudioScope,
    request_id: Option<String>,
) -> Result<XaiTimestampedSpeech, XaiAudioError> {
    let encoded = native.get("audio").and_then(Value::as_str).ok_or_else(|| {
        invalid_response(
            "text-to-speech",
            "timestamped response has no base64 audio",
            request_id.clone(),
            native.clone(),
        )
    })?;
    let audio = STANDARD.decode(encoded).map_err(|_| {
        invalid_response(
            "text-to-speech",
            "timestamped audio is not valid base64",
            request_id.clone(),
            native.clone(),
        )
    })?;
    if audio.is_empty() {
        return Err(invalid_response(
            "text-to-speech",
            "timestamped response contains empty audio",
            request_id,
            native,
        ));
    }
    let content_type = native
        .get("content_type")
        .and_then(Value::as_str)
        .filter(|value| value.starts_with("audio/"))
        .ok_or_else(|| {
            invalid_response(
                "text-to-speech",
                "timestamped response has no audio content_type",
                request_id.clone(),
                native.clone(),
            )
        })?
        .to_owned();
    let duration_seconds = native
        .get("duration")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| {
            invalid_response(
                "text-to-speech",
                "timestamped response has invalid duration",
                request_id.clone(),
                native.clone(),
            )
        })?;
    let audio_timestamps = native
        .get("audio_timestamps")
        .map(|value| {
            serde_json::from_value::<XaiAudioTimestamps>(value.clone()).map_err(|_| {
                invalid_response(
                    "text-to-speech",
                    "timestamp metadata has an invalid shape",
                    request_id.clone(),
                    native.clone(),
                )
            })
        })
        .transpose()?;
    if let Some(timestamps) = &audio_timestamps {
        if timestamps.graph_chars.len() != timestamps.graph_times.len()
            || timestamps.graph_times.iter().any(|[start, end]| {
                !start.is_finite() || !end.is_finite() || *start < 0.0 || *end < *start
            })
        {
            return Err(invalid_response(
                "text-to-speech",
                "timestamp metadata is inconsistent",
                request_id,
                native,
            ));
        }
    }
    Ok(XaiTimestampedSpeech {
        scope,
        audio: Bytes::from(audio),
        content_type,
        duration_seconds,
        audio_timestamps,
        request_id,
    })
}

fn speech_body(request: &XaiSpeechRequest) -> Value {
    let mut body = json!({
        "text": request.text,
        "voice_id": request.voice_id,
        "language": request.language,
        "output_format": {
            "codec": request.output_format.codec.as_str(),
            "sample_rate": request.output_format.sample_rate,
        },
    });
    if let Some(bit_rate) = request.output_format.bit_rate {
        body["output_format"]["bit_rate"] = json!(bit_rate);
    }
    if let Some(speed) = request.speed {
        body["speed"] = json!(speed);
    }
    if let Some(level) = request.optimize_streaming_latency {
        body["optimize_streaming_latency"] = json!(level);
    }
    if request.text_normalization {
        body["text_normalization"] = Value::Bool(true);
    }
    if !request.replace.is_empty() {
        body["replace"] = json!(&request.replace);
    }
    if request.with_timestamps {
        body["with_timestamps"] = Value::Bool(true);
    }
    body
}

fn validate_transcription_request(request: &XaiTranscriptionRequest) -> Result<(), XaiAudioError> {
    if request.format && request.language.as_deref().is_none_or(str::is_empty) {
        return Err(invalid("format=true requires a language"));
    }
    if request
        .language
        .as_ref()
        .is_some_and(|value| invalid_form_value(value))
    {
        return Err(invalid(
            "language must be nonempty and contain no control characters",
        ));
    }
    match (request.audio_format, request.sample_rate) {
        (Some(_), Some(rate)) if supported_sample_rate(rate) => {}
        (Some(_), _) => return Err(invalid("raw audio requires a supported sample_rate")),
        (None, Some(_)) => {
            return Err(invalid(
                "sample_rate is only accepted with raw audio_format",
            ))
        }
        (None, None) => {}
    }
    match (request.multichannel, request.channels) {
        (false, Some(_)) => return Err(invalid("channels requires multichannel=true")),
        (true, Some(channels)) if (2..=8).contains(&channels) => {}
        (true, Some(_)) => return Err(invalid("multichannel channels must be between 2 and 8")),
        (true, None) if request.audio_format.is_some() => {
            return Err(invalid("raw multichannel audio requires channels"));
        }
        _ => {}
    }
    if request.keyterms.len() > MAX_KEYTERMS
        || request.keyterms.iter().any(|term| {
            term.is_empty() || term.chars().count() > MAX_KEYTERM_CHARS || invalid_form_value(term)
        })
    {
        return Err(invalid(
            "keyterms must contain at most 100 nonempty values of at most 50 characters",
        ));
    }
    if request
        .vad_threshold
        .is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
    {
        return Err(invalid("vad_threshold must be between 0 and 1"));
    }
    Ok(())
}

fn validate_audio_input(input: &AudioInput) -> Result<(), XaiAudioError> {
    if input.size_bytes == 0 {
        return Err(invalid("audio input must not be empty"));
    }
    if input.size_bytes > MAX_STT_AUDIO_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: "xAI STT audio exceeds the documented 500 MB upload limit".into(),
        }
        .into());
    }
    if input.filename.trim().is_empty() || input.filename.chars().any(char::is_control) {
        return Err(invalid(
            "audio filename must be nonempty and contain no control characters",
        ));
    }
    validate_media_type(&input.media_type)?;
    if !input.media_type.starts_with("audio/")
        && !matches!(
            input.media_type.as_str(),
            "application/ogg" | "video/mp4" | "video/x-matroska"
        )
    {
        return Err(invalid(
            "xAI STT input must use an audio or supported audio-container media type",
        ));
    }
    Ok(())
}

fn validate_speech_request(request: &XaiSpeechRequest) -> Result<(), XaiAudioError> {
    if request.text.trim().is_empty() {
        return Err(invalid("speech text must be nonempty"));
    }
    if request.text.chars().count() > MAX_TTS_TEXT_CHARS {
        return Err(invalid("speech text exceeds xAI's 60,000 character limit"));
    }
    if request.voice_id.trim().is_empty() || invalid_form_value(&request.voice_id) {
        return Err(invalid(
            "voice_id must be nonempty and contain no control characters",
        ));
    }
    if request.language.trim().is_empty() || invalid_form_value(&request.language) {
        return Err(invalid(
            "language must be nonempty and contain no control characters",
        ));
    }
    if request
        .speed
        .is_some_and(|speed| !speed.is_finite() || !(0.7..=1.5).contains(&speed))
    {
        return Err(invalid("speed must be finite and between 0.7 and 1.5"));
    }
    if request
        .optimize_streaming_latency
        .is_some_and(|level| !matches!(level, 0..=2))
    {
        return Err(invalid("optimize_streaming_latency must be 0, 1, or 2"));
    }
    validate_speech_replacements(&request.replace).map_err(invalid)?;
    if !supported_sample_rate(request.output_format.sample_rate) {
        return Err(invalid(
            "TTS sample_rate must be one of xAI's documented rates",
        ));
    }
    if let Some(bit_rate) = request.output_format.bit_rate {
        if request.output_format.codec != XaiSpeechCodec::Mp3
            || !matches!(bit_rate, 32_000 | 64_000 | 96_000 | 128_000 | 192_000)
        {
            return Err(invalid(
                "bit_rate is available for MP3 at 32000, 64000, 96000, 128000, or 192000 bps",
            ));
        }
    }
    Ok(())
}

/// Validate the xAI TTS and WebSocket pronunciation-replacement map limits.
/// The provider remains authoritative for matching and post-substitution size.
pub(crate) fn validate_speech_replacements(
    replacements: &BTreeMap<String, String>,
) -> Result<(), &'static str> {
    if replacements.len() > 200 {
        return Err("replace supports at most 200 entries");
    }

    let mut normalized_keys = BTreeSet::new();
    for (key, value) in replacements {
        if key.trim().is_empty() {
            return Err("replace keys must not be blank");
        }
        if key.chars().count() > 100 {
            return Err("replace keys cannot exceed 100 characters");
        }
        if value.chars().count() > 128 {
            return Err("replace values cannot exceed 128 characters");
        }
        if !key
            .chars()
            .all(|character| character.is_alphanumeric() || matches!(character, '\'' | ' '))
        {
            return Err("replace keys may contain only letters, digits, apostrophes, and spaces");
        }
        let normalized = key
            .chars()
            .filter(|character| !character.is_whitespace())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if !normalized_keys.insert(normalized) {
            return Err("replace keys must be distinct ignoring case and whitespace");
        }
    }
    Ok(())
}

fn validate_credential(credential: &Secret<String>) -> Result<(), XaiAudioError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || value.contains(['\r', '\n']) {
        return Err(invalid("xAI API key is empty or malformed"));
    }
    Ok(())
}

fn normalize_base_url(value: &str) -> Result<String, XaiAudioError> {
    let mut url = Url::parse(value).map_err(|_| invalid("API base URL is invalid"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "API base URL cannot contain credentials, query parameters, or a fragment",
        ));
    }
    let local_http = url.scheme() == "http"
        && url.host_str().is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        });
    if url.scheme() != "https" && !local_http {
        return Err(invalid(
            "API base URL must use HTTPS (HTTP is allowed only on loopback)",
        ));
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(if path.is_empty() { "/" } else { &path });
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn supported_sample_rate(rate: u32) -> bool {
    matches!(rate, 8_000 | 16_000 | 22_050 | 24_000 | 44_100 | 48_000)
}

fn invalid_form_value(value: &str) -> bool {
    value.is_empty() || value.chars().any(char::is_control)
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

fn parse_error_body(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()))
}

fn map_outcome(operation: &'static str, source: LlmError) -> XaiAudioError {
    if matches!(
        source,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    ) {
        XaiAudioError::OutcomeUnknown {
            operation,
            source,
            response: None,
        }
    } else {
        XaiAudioError::Llm(source)
    }
}

fn invalid_response(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> XaiAudioError {
    XaiAudioError::InvalidResponse {
        operation,
        message: message.to_owned(),
        request_id,
        native: Box::new(native),
    }
}

fn invalid(message: &str) -> XaiAudioError {
    XaiAudioError::InvalidRequest(message.to_owned())
}

const fn default_sample_rate() -> u32 {
    24_000
}

const fn default_bit_rate() -> u32 {
    128_000
}

fn default_bit_rate_option() -> Option<u32> {
    Some(default_bit_rate())
}

fn is_false(value: &bool) -> bool {
    !value
}

fn is_default_speech_format(value: &XaiSpeechFormat) -> bool {
    *value == XaiSpeechFormat::default()
}
