//! MiniMax's native Text-to-Audio HTTP service (`POST /v1/t2a_v2`).
//!
//! The request and response types follow MiniMax's provider-native schema.
//! Ordinary calls return hexadecimal audio inside JSON; `output_format=url`
//! returns a temporary provider URL. Streaming calls expose native SSE events
//! and decoded hex chunks. This module does not download audio URLs, retry a
//! synthesis request, or use the separate asynchronous task API.

use crate::{
    files::provider_file_endpoint_fingerprint,
    framing::sse::SseFrameSplitter,
    protocol::{LlmError, ProviderId, Secret},
    providers::minimax::voices::MiniMaxVoiceRef,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, Transport, MAX_ERROR_BODY_SIZE},
};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::VecDeque, fmt, time::Duration};
use thiserror::Error;
use url::Url;

/// Official MiniMax international T2A HTTP route.
pub const MINIMAX_TTS_INTERNATIONAL_ENDPOINT: &str = "https://api.minimax.io/v1/t2a_v2";
/// Official MiniMax mainland China T2A HTTP route.
pub const MINIMAX_TTS_CHINA_ENDPOINT: &str = "https://api.minimax.cn/v1/t2a_v2";
/// Official international endpoint documented for reduced time-to-first-audio.
pub const MINIMAX_TTS_INTERNATIONAL_FAST_ENDPOINT: &str = "https://api-uw.minimax.io/v1/t2a_v2";
/// Mainland China backup endpoint published in MiniMax's regional reference.
pub const MINIMAX_TTS_CHINA_BACKUP_ENDPOINT: &str = "https://api-bj.minimaxi.com/v1/t2a_v2";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT_CHARACTERS: usize = 10_000;
const URL_AUDIO_VALIDITY: Duration = Duration::from_secs(24 * 60 * 60);

/// MiniMax account route. Choose this explicitly because credentials and API
/// availability are region-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTtsRegion {
    International,
    ChinaMainland,
}

impl MiniMaxTtsRegion {
    const fn endpoint(self) -> &'static str {
        match self {
            Self::International => MINIMAX_TTS_INTERNATIONAL_ENDPOINT,
            Self::ChinaMainland => MINIMAX_TTS_CHINA_ENDPOINT,
        }
    }
}

/// Secret-free settings for one MiniMax profile, account, and regional route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxTtsConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: MiniMaxTtsRegion,
    pub endpoint: String,
    pub request_timeout: Duration,
}

impl MiniMaxTtsConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: MiniMaxTtsRegion,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            endpoint: region.endpoint().to_owned(),
            request_timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Select one of MiniMax's documented same-region T2A endpoints.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Per-call MiniMax API key. The service never stores or serializes credentials.
pub struct MiniMaxTtsCredentials {
    api_key: Secret<String>,
}

impl MiniMaxTtsCredentials {
    pub fn new(api_key: Secret<String>) -> Self {
        Self { api_key }
    }
}

impl fmt::Debug for MiniMaxTtsCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxTtsCredentials")
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Non-secret identity attached to generated audio and URL results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTtsScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub api_endpoint_fingerprint: String,
    pub account_scope: String,
    pub region: MiniMaxTtsRegion,
}

/// T2A model identifiers currently listed by MiniMax's HTTP reference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxTtsModel {
    #[default]
    #[serde(rename = "speech-2.8-hd")]
    Speech28Hd,
    #[serde(rename = "speech-2.8-turbo")]
    Speech28Turbo,
    #[serde(rename = "speech-2.6-hd")]
    Speech26Hd,
    #[serde(rename = "speech-2.6-turbo")]
    Speech26Turbo,
    #[serde(rename = "speech-02-hd")]
    Speech02Hd,
    #[serde(rename = "speech-02-turbo")]
    Speech02Turbo,
    #[serde(rename = "speech-01-hd")]
    Speech01Hd,
    #[serde(rename = "speech-01-turbo")]
    Speech01Turbo,
}

impl MiniMaxTtsModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Speech28Hd => "speech-2.8-hd",
            Self::Speech28Turbo => "speech-2.8-turbo",
            Self::Speech26Hd => "speech-2.6-hd",
            Self::Speech26Turbo => "speech-2.6-turbo",
            Self::Speech02Hd => "speech-02-hd",
            Self::Speech02Turbo => "speech-02-turbo",
            Self::Speech01Hd => "speech-01-hd",
            Self::Speech01Turbo => "speech-01-turbo",
        }
    }
}

/// Provider-native codec placed in `audio_setting.format`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTtsAudioFormat {
    #[default]
    Mp3,
    Wav,
    Pcm,
    Flac,
}

impl MiniMaxTtsAudioFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
            Self::Flac => "flac",
        }
    }
}

/// Output representation of MiniMax's non-streaming T2A response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTtsOutputFormat {
    /// `data.audio` is a hex-encoded audio payload.
    #[default]
    Hex,
    /// `data.audio` is a temporary provider-hosted URL, valid for 24 hours.
    Url,
}

impl MiniMaxTtsOutputFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Hex => "hex",
            Self::Url => "url",
        }
    }
}

/// Timestamp granularity for HTTP T2A subtitles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTtsSubtitleType {
    #[default]
    Sentence,
    Word,
    /// Streaming-optimized word timestamps. Accepted only by
    /// [`MiniMaxTtsService::synthesize_stream`].
    WordStreaming,
}

impl MiniMaxTtsSubtitleType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Sentence => "sentence",
            Self::Word => "word",
            Self::WordStreaming => "word_streaming",
        }
    }
}

/// Voice controls sent inside MiniMax's `voice_setting` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTtsVoiceSetting {
    pub voice_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vol: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<i8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
}

impl MiniMaxTtsVoiceSetting {
    pub fn new(voice_id: impl Into<String>) -> Self {
        Self {
            voice_id: voice_id.into(),
            speed: None,
            vol: None,
            pitch: None,
            emotion: None,
        }
    }
}

/// Audio controls sent inside MiniMax's `audio_setting` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTtsAudioSetting {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_bitrate")]
    pub bitrate: u32,
    #[serde(default)]
    pub format: MiniMaxTtsAudioFormat,
    #[serde(default = "default_channel")]
    pub channel: u8,
}

impl Default for MiniMaxTtsAudioSetting {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            bitrate: default_bitrate(),
            format: MiniMaxTtsAudioFormat::default(),
            channel: default_channel(),
        }
    }
}

/// MiniMax's inline pronunciation dictionary structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTtsPronunciationDict {
    #[serde(default)]
    pub tone: Vec<String>,
}

/// Typed request for MiniMax T2A HTTP synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTtsRequest {
    pub model: MiniMaxTtsModel,
    pub text: String,
    pub voice_setting: MiniMaxTtsVoiceSetting,
    #[serde(default)]
    pub audio_setting: MiniMaxTtsAudioSetting,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_boost: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pronunciation_dict: Option<MiniMaxTtsPronunciationDict>,
    #[serde(default)]
    pub output_format: MiniMaxTtsOutputFormat,
    #[serde(default, skip_serializing_if = "is_false")]
    pub aigc_watermark: bool,
    /// Ask the provider to return subtitle metadata with generated audio.
    #[serde(default, skip_serializing_if = "is_false")]
    pub subtitle_enable: bool,
    /// Subtitle timestamp granularity when subtitle output is enabled.
    #[serde(default)]
    pub subtitle_type: MiniMaxTtsSubtitleType,
}

impl MiniMaxTtsRequest {
    pub fn new(text: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            model: MiniMaxTtsModel::default(),
            text: text.into(),
            voice_setting: MiniMaxTtsVoiceSetting::new(voice_id),
            audio_setting: MiniMaxTtsAudioSetting::default(),
            language_boost: None,
            pronunciation_dict: None,
            output_format: MiniMaxTtsOutputFormat::default(),
            aigc_watermark: false,
            subtitle_enable: false,
            subtitle_type: MiniMaxTtsSubtitleType::default(),
        }
    }
}

/// Optional provider statistics returned in `extra_info`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MiniMaxTtsExtraInfo {
    #[serde(default)]
    pub audio_length: Option<u64>,
    #[serde(default)]
    pub audio_sample_rate: Option<u32>,
    #[serde(default)]
    pub audio_size: Option<u64>,
    #[serde(default)]
    pub bitrate: Option<u32>,
    #[serde(default)]
    pub word_count: Option<u64>,
    #[serde(default)]
    pub invisible_character_ratio: Option<f64>,
    #[serde(default)]
    pub usage_characters: Option<u64>,
    #[serde(default)]
    pub audio_format: Option<String>,
    #[serde(default)]
    pub audio_channel: Option<u8>,
}

/// Successfully generated audio bytes, decoded from MiniMax's hexadecimal JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxTtsAudio {
    pub scope: MiniMaxTtsScope,
    pub bytes: Bytes,
    pub format: MiniMaxTtsAudioFormat,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
    pub extra_info: Option<MiniMaxTtsExtraInfo>,
    /// Full provider JSON for fields added after this client version.
    pub native: Value,
}

/// Temporary MiniMax-hosted audio location returned by `output_format=url`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxTtsUrl {
    pub scope: MiniMaxTtsScope,
    pub url: String,
    pub valid_for: Duration,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
    pub extra_info: Option<MiniMaxTtsExtraInfo>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MiniMaxTtsOutput {
    Audio(MiniMaxTtsAudio),
    Url(MiniMaxTtsUrl),
}

/// One native JSON event from MiniMax's HTTP T2A SSE response.
///
/// Audio is decoded from `data.audio` hex when present. The complete native
/// event remains available so callers can inspect provider-added fields and
/// decide how to handle a possible terminal audio value themselves. This
/// client does not concatenate or deduplicate event audio.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxTtsStreamDataEvent {
    pub status: Option<i64>,
    pub audio: Option<Bytes>,
    pub native: Value,
}

/// An event returned by [`MiniMaxTtsEventStream`].
#[derive(Debug, Clone, PartialEq)]
pub enum MiniMaxTtsStreamEvent {
    Data(MiniMaxTtsStreamDataEvent),
    /// MiniMax's SSE sentinel. It is exposed to callers and ends iteration.
    Done,
}

impl MiniMaxTtsStreamEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done) || matches!(self, Self::Data(event) if event.status == Some(2))
    }
}

/// Incremental, request-scoped MiniMax HTTP T2A SSE events.
///
/// Each event is returned once as supplied; audio is never coalesced and the
/// terminal event's audio is not assumed to be a delta or an aggregate.
pub struct MiniMaxTtsEventStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_error: Option<LlmError>,
    ended: bool,
    terminal: bool,
    terminal_event_received: bool,
    received_audio: bool,
    bytes_delivered: u64,
    pub scope: MiniMaxTtsScope,
    pub request_id: Option<String>,
}

#[derive(Debug, Error)]
pub enum MiniMaxTtsStreamError {
    #[error("MiniMax TTS stream interrupted after {bytes_delivered} audio bytes: {source}")]
    Interrupted {
        bytes_delivered: u64,
        request_id: Option<String>,
        #[source]
        source: LlmError,
    },
    #[error("MiniMax TTS stream provider error after {bytes_delivered} audio bytes (code {code:?}): {message}")]
    Provider {
        bytes_delivered: u64,
        request_id: Option<String>,
        code: Option<i64>,
        message: String,
        native: Box<Value>,
        dispatch: MiniMaxTtsDispatch,
    },
    #[error("invalid MiniMax TTS stream event after {bytes_delivered} audio bytes: {message}")]
    InvalidEvent {
        bytes_delivered: u64,
        request_id: Option<String>,
        message: String,
        raw_data: Bytes,
        native: Option<Box<Value>>,
    },
}

impl MiniMaxTtsStreamError {
    pub fn bytes_delivered(&self) -> u64 {
        match self {
            Self::Interrupted {
                bytes_delivered, ..
            }
            | Self::Provider {
                bytes_delivered, ..
            }
            | Self::InvalidEvent {
                bytes_delivered, ..
            } => *bytes_delivered,
        }
    }

    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Interrupted { request_id, .. }
            | Self::Provider { request_id, .. }
            | Self::InvalidEvent { request_id, .. } => request_id.as_deref(),
        }
    }

    pub fn dispatch(&self) -> MiniMaxTtsDispatch {
        match self {
            Self::Interrupted {
                bytes_delivered, ..
            } => {
                if *bytes_delivered > 0 {
                    MiniMaxTtsDispatch::Accepted
                } else {
                    MiniMaxTtsDispatch::Unknown
                }
            }
            Self::Provider {
                bytes_delivered,
                dispatch,
                ..
            } => {
                if *bytes_delivered > 0 {
                    MiniMaxTtsDispatch::Accepted
                } else {
                    *dispatch
                }
            }
            Self::InvalidEvent { .. } => MiniMaxTtsDispatch::Accepted,
        }
    }
}

impl MiniMaxTtsEventStream {
    /// Read one event. The stream ends at status 2, `[DONE]`, or clean EOF
    /// after at least one audio chunk. Dropping it cancels the HTTP body.
    /// `terminal_event_received()` distinguishes the two explicit terminal
    /// markers from clean EOF accepted after audio.
    pub fn terminal_event_received(&self) -> bool {
        self.terminal_event_received
    }

    pub async fn next_event(
        &mut self,
    ) -> Result<Option<MiniMaxTtsStreamEvent>, MiniMaxTtsStreamError> {
        if self.terminal {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                match self.decode_event(frame) {
                    Ok(event) => {
                        self.terminal = event.is_terminal();
                        self.terminal_event_received |= event.is_terminal();
                        return Ok(Some(event));
                    }
                    Err(error) => {
                        self.terminal = true;
                        return Err(error);
                    }
                }
            }
            if let Some(source) = self.pending_error.take() {
                self.terminal = true;
                return Err(MiniMaxTtsStreamError::Interrupted {
                    bytes_delivered: self.bytes_delivered,
                    request_id: self.request_id.clone(),
                    source,
                });
            }
            if self.ended {
                self.terminal = true;
                if self.received_audio {
                    return Ok(None);
                }
                return Err(self.invalid_event(
                    "stream ended without audio data".into(),
                    Bytes::new(),
                    None,
                ));
            }
            match self.body.next().await {
                Some(Ok(bytes)) => {
                    let (frames, error) = self.splitter.push_batch(&bytes);
                    self.ready.extend(frames);
                    self.pending_error = error;
                }
                Some(Err(source)) => self.pending_error = Some(source),
                None => {
                    self.ended = true;
                    match self.splitter.finish() {
                        Ok(Some(frame)) => self.ready.push_back(frame),
                        Ok(None) => {}
                        Err(source) => self.pending_error = Some(source),
                    }
                }
            }
        }
    }

    fn decode_event(
        &mut self,
        raw_data: Vec<u8>,
    ) -> Result<MiniMaxTtsStreamEvent, MiniMaxTtsStreamError> {
        let raw_data = Bytes::from(raw_data);
        if raw_data.as_ref() == b"[DONE]" {
            if !self.received_audio {
                return Err(self.invalid_event(
                    "stream ended without audio data".into(),
                    raw_data,
                    None,
                ));
            }
            return Ok(MiniMaxTtsStreamEvent::Done);
        }
        let native: Value = serde_json::from_slice(&raw_data).map_err(|error| {
            self.invalid_event(
                format!("event data is not valid JSON: {error}"),
                raw_data.clone(),
                None,
            )
        })?;
        if !native.is_object() {
            return Err(self.invalid_event(
                "event data must be a JSON object".into(),
                raw_data,
                Some(native),
            ));
        }
        if let Some(trace) = trace_id(&native) {
            self.request_id = Some(trace);
        }
        let code = validate_stream_base_response(&native).map_err(|message| {
            self.invalid_event(message.into(), raw_data.clone(), Some(native.clone()))
        })?;
        if let Some(code) = code {
            if code != 0 {
                let message = base_response_message(&native)
                    .unwrap_or_else(|| "provider rejected streamed synthesis".into());
                let dispatch = if self.bytes_delivered == 0 {
                    MiniMaxTtsDispatch::Rejected
                } else {
                    MiniMaxTtsDispatch::Accepted
                };
                return Err(MiniMaxTtsStreamError::Provider {
                    bytes_delivered: self.bytes_delivered,
                    request_id: self.request_id.clone(),
                    code: Some(code),
                    message,
                    native: Box::new(native),
                    dispatch,
                });
            }
        }
        if native.get("error").is_some_and(|error| !error.is_null()) {
            return Err(MiniMaxTtsStreamError::Provider {
                bytes_delivered: self.bytes_delivered,
                request_id: self.request_id.clone(),
                code,
                message: base_response_message(&native).unwrap_or_else(|| {
                    native["error"]
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| "provider returned an in-band stream error".into())
                }),
                native: Box::new(native),
                dispatch: if self.bytes_delivered == 0 {
                    MiniMaxTtsDispatch::Rejected
                } else {
                    MiniMaxTtsDispatch::Accepted
                },
            });
        }
        let data = native.get("data");
        let status = match data.and_then(|data| data.get("status")) {
            Some(value) if value.is_null() => None,
            Some(value) => Some(value.as_i64().ok_or_else(|| {
                self.invalid_event(
                    "data.status must be an integer when present".into(),
                    raw_data.clone(),
                    Some(native.clone()),
                )
            })?),
            None => None,
        };
        let audio = match data.and_then(|data| data.get("audio")) {
            Some(value) if value.is_null() => None,
            Some(value) => match value.as_str() {
                Some("") => None,
                Some(hex) => match decode_hex(hex) {
                    Ok(decoded) => Some(Bytes::from(decoded)),
                    Err(message) => {
                        return Err(self.invalid_event(message.into(), raw_data, Some(native)));
                    }
                },
                None => {
                    return Err(self.invalid_event(
                        "data.audio must be hexadecimal text when present".into(),
                        raw_data,
                        Some(native),
                    ));
                }
            },
            None => None,
        };
        if let Some(audio) = &audio {
            self.received_audio = true;
            self.bytes_delivered = self.bytes_delivered.saturating_add(audio.len() as u64);
        }
        if status == Some(2) && !self.received_audio {
            return Err(self.invalid_event(
                "terminal status arrived without audio data".into(),
                raw_data,
                Some(native),
            ));
        }
        Ok(MiniMaxTtsStreamEvent::Data(MiniMaxTtsStreamDataEvent {
            status,
            audio,
            native,
        }))
    }

    fn invalid_event(
        &self,
        message: String,
        raw_data: Bytes,
        native: Option<Value>,
    ) -> MiniMaxTtsStreamError {
        MiniMaxTtsStreamError::InvalidEvent {
            bytes_delivered: self.bytes_delivered,
            request_id: self.request_id.clone(),
            message,
            raw_data,
            native: native.map(Box::new),
        }
    }
}

/// Retry safety signal for one MiniMax synthesis attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxTtsDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum MiniMaxTtsError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid MiniMax TTS request: {0}")]
    InvalidRequest(String),
    #[error("MiniMax TTS rejected the request (HTTP {http_status:?}, code {code:?}): {message}")]
    Provider {
        http_status: Option<u16>,
        code: Option<i64>,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: MiniMaxTtsDispatch,
    },
    #[error("MiniMax TTS returned an invalid successful response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("MiniMax TTS outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

impl MiniMaxTtsError {
    pub fn dispatch(&self) -> MiniMaxTtsDispatch {
        match self {
            Self::Llm(_) | Self::InvalidRequest(_) => MiniMaxTtsDispatch::NotSent,
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => MiniMaxTtsDispatch::Accepted,
            Self::OutcomeUnknown { .. } => MiniMaxTtsDispatch::Unknown,
        }
    }
}

/// MiniMax's native T2A HTTP client.
///
/// Select one account's official `.io` or `.cn` route when constructing it;
/// pass that account's API key to each request. Calls are one-shot and are not
/// retried, polled, or downloaded through a returned URL.
#[derive(Clone)]
pub struct MiniMaxTtsService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    config: MiniMaxTtsConfig,
    scope: MiniMaxTtsScope,
}

impl<'a> MiniMaxTtsService<'a> {
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
        mut config: MiniMaxTtsConfig,
    ) -> Result<Self, MiniMaxTtsError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.request_timeout.is_zero() {
            return Err(invalid("request_timeout must be positive"));
        }
        config.endpoint = normalize_endpoint(&config.endpoint, config.region)?;
        let scope = MiniMaxTtsScope {
            provider_id: ProviderId::new("minimax"),
            profile_name: config.profile_name.clone(),
            api_endpoint_fingerprint: provider_file_endpoint_fingerprint(&config.endpoint),
            account_scope: config.account_scope.clone(),
            region: config.region,
        };
        Ok(Self {
            binding: None,
            transport,
            config,
            scope,
        })
    }

    pub fn scope(&self) -> &MiniMaxTtsScope {
        &self.scope
    }

    /// Synthesize text once and return either decoded audio bytes or MiniMax's
    /// short-lived URL. URL output remains caller-managed and is never fetched.
    pub async fn synthesize(
        &self,
        request: &MiniMaxTtsRequest,
        credentials: &MiniMaxTtsCredentials,
    ) -> Result<MiniMaxTtsOutput, MiniMaxTtsError> {
        let pinned_service = self.pin()?;
        validate_request(request, false)?;
        validate_credential(&credentials.api_key)?;
        let body = serde_json::to_vec(&request_body(request, false))
            .map_err(|_| invalid("request cannot be serialized"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let http = HttpRequest {
            method: "POST".into(),
            url: pinned_service.config.endpoint.clone(),
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
            .execute_bounded(http, MAX_RESPONSE_BYTES)
            .await
            .map_err(map_outcome)?;
        let request_id = response_id(&response.headers);
        let native = match serde_json::from_slice::<Value>(&response.body) {
            Ok(value) => value,
            Err(_) if !(200..300).contains(&response.status) => {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            }
            Err(_) => {
                return Err(invalid_response(
                    "successful response body is not valid JSON",
                    request_id,
                    Value::Null,
                ));
            }
        };
        let request_id = trace_id(&native).or(request_id);
        if !(200..300).contains(&response.status) {
            return Err(provider_error(
                Some(response.status),
                base_response_code(&native),
                base_response_message(&native).unwrap_or_else(|| "HTTP request rejected".into()),
                request_id,
                native,
                MiniMaxTtsDispatch::Rejected,
            ));
        }
        if let Some(code) = base_response_code(&native) {
            if code != 0 {
                return Err(provider_error(
                    Some(response.status),
                    Some(code),
                    base_response_message(&native)
                        .unwrap_or_else(|| "provider rejected request".into()),
                    request_id,
                    native,
                    MiniMaxTtsDispatch::Rejected,
                ));
            }
        } else {
            return Err(invalid_response(
                "successful response is missing base_resp.status_code",
                request_id,
                native,
            ));
        }
        decode_success(native, request_id, request, pinned_service.scope.clone())
    }

    /// Start one HTTP T2A SSE request and return its native events incrementally.
    ///
    /// Streaming is limited to MiniMax's documented MP3/hex mode. The stream
    /// preserves each JSON event and decoded `data.audio` chunk; it does not
    /// coalesce or deduplicate audio because the provider does not document
    /// whether status-2 audio is a final delta or an aggregate copy.
    pub async fn synthesize_stream(
        &self,
        request: &MiniMaxTtsRequest,
        credentials: &MiniMaxTtsCredentials,
    ) -> Result<MiniMaxTtsEventStream, MiniMaxTtsError> {
        let pinned_service = self.pin()?;
        validate_request(request, true)?;
        validate_credential(&credentials.api_key)?;
        let body = serde_json::to_vec(&request_body(request, true))
            .map_err(|_| invalid("request cannot be serialized"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .send(HttpRequest {
                method: "POST".into(),
                url: pinned_service.config.endpoint.clone(),
                headers: vec![
                    (
                        "authorization".into(),
                        format!("Bearer {}", credentials.api_key.expose_secret()),
                    ),
                    ("content-type".into(), "application/json".into()),
                    ("accept".into(), "text/event-stream".into()),
                ],
                body: Bytes::from(body),
                timeout: deadline.remaining()?,
            })
            .await
            .map_err(map_outcome)?;
        let initial_request_id = response_id(&response.headers);
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE))
                .await
                .map_err(map_outcome)?;
            let native = serde_json::from_slice::<Value>(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            let request_id = trace_id(&native).or(initial_request_id);
            return Err(provider_error(
                Some(response.status),
                base_response_code(&native),
                base_response_message(&native).unwrap_or_else(|| "HTTP request rejected".into()),
                request_id,
                native,
                MiniMaxTtsDispatch::Rejected,
            ));
        }
        let content_type = response.header("content-type").unwrap_or_default();
        if !content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
        {
            let response = HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE))
                .await
                .map_err(map_outcome)?;
            let native = serde_json::from_slice::<Value>(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            let request_id = trace_id(&native).or(initial_request_id);
            let code = base_response_code(&native);
            if code.is_some_and(|code| code != 0)
                || native.get("error").is_some_and(|error| !error.is_null())
            {
                return Err(provider_error(
                    Some(response.status),
                    code.filter(|code| *code != 0),
                    base_response_message(&native).unwrap_or_else(|| {
                        native["error"]
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| "provider rejected streamed synthesis".into())
                    }),
                    request_id,
                    native,
                    MiniMaxTtsDispatch::Rejected,
                ));
            }
            return Err(invalid_response(
                "streaming request did not return text/event-stream",
                request_id,
                native,
            ));
        }
        Ok(MiniMaxTtsEventStream {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_error: None,
            ended: false,
            terminal: false,
            terminal_event_received: false,
            received_audio: false,
            bytes_delivered: 0,
            scope: pinned_service.scope.clone(),
            request_id: initial_request_id,
        })
    }

    /// Synthesize with an explicitly scoped MiniMax voice resource. The
    /// reference must come from the same profile, account, region, and `/v1`
    /// API root as this T2A service. Built-in voice IDs remain available via
    /// [`Self::synthesize`] without creating a reference.
    pub async fn synthesize_with_voice(
        &self,
        request: &MiniMaxTtsRequest,
        voice: &MiniMaxVoiceRef,
        credentials: &MiniMaxTtsCredentials,
    ) -> Result<MiniMaxTtsOutput, MiniMaxTtsError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_voice_reference(voice)?;
        let mut request = request.clone();
        request.voice_setting.voice_id = voice.voice_id().to_owned();
        pinned_service.synthesize(&request, credentials).await
    }

    /// Stream synthesis through a voice reference scoped to this account.
    pub async fn synthesize_stream_with_voice(
        &self,
        request: &MiniMaxTtsRequest,
        voice: &MiniMaxVoiceRef,
        credentials: &MiniMaxTtsCredentials,
    ) -> Result<MiniMaxTtsEventStream, MiniMaxTtsError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_voice_reference(voice)?;
        let mut request = request.clone();
        request.voice_setting.voice_id = voice.voice_id().to_owned();
        pinned_service
            .synthesize_stream(&request, credentials)
            .await
    }

    fn validate_voice_reference(&self, voice: &MiniMaxVoiceRef) -> Result<(), MiniMaxTtsError> {
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
        let api_root = self
            .config
            .endpoint
            .strip_suffix("/t2a_v2")
            .ok_or_else(|| invalid("MiniMax T2A endpoint has no canonical API root"))?;
        let expected_endpoint = provider_file_endpoint_fingerprint(api_root);
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
}

fn request_body(request: &MiniMaxTtsRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": request.model.as_str(),
        "text": request.text,
        "stream": stream,
        "voice_setting": {
            "voice_id": request.voice_setting.voice_id,
        },
        "audio_setting": {
            "sample_rate": request.audio_setting.sample_rate,
            "format": request.audio_setting.format.as_str(),
            "channel": request.audio_setting.channel,
        },
        "output_format": if stream { "hex" } else { request.output_format.as_str() },
        "subtitle_enable": request.subtitle_enable,
        "subtitle_type": request.subtitle_type.as_str(),
    });
    if let Some(speed) = request.voice_setting.speed {
        body["voice_setting"]["speed"] = json!(speed);
    }
    if request.audio_setting.format == MiniMaxTtsAudioFormat::Mp3 {
        body["audio_setting"]["bitrate"] = json!(request.audio_setting.bitrate);
    }
    if let Some(vol) = request.voice_setting.vol {
        body["voice_setting"]["vol"] = json!(vol);
    }
    if let Some(pitch) = request.voice_setting.pitch {
        body["voice_setting"]["pitch"] = json!(pitch);
    }
    if let Some(emotion) = &request.voice_setting.emotion {
        body["voice_setting"]["emotion"] = json!(emotion);
    }
    if let Some(language_boost) = &request.language_boost {
        body["language_boost"] = json!(language_boost);
    }
    if let Some(pronunciation_dict) = &request.pronunciation_dict {
        body["pronunciation_dict"] = json!(pronunciation_dict);
    }
    if request.aigc_watermark && !stream {
        body["aigc_watermark"] = Value::Bool(true);
    }
    body
}

fn decode_success(
    native: Value,
    request_id: Option<String>,
    request: &MiniMaxTtsRequest,
    scope: MiniMaxTtsScope,
) -> Result<MiniMaxTtsOutput, MiniMaxTtsError> {
    let data = native
        .get("data")
        .filter(|data| !data.is_null())
        .ok_or_else(|| {
            invalid_response(
                "response data is missing or null",
                request_id.clone(),
                native.clone(),
            )
        })?;
    let audio_value = data
        .get("audio")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            invalid_response(
                "response data has no audio value",
                request_id.clone(),
                native.clone(),
            )
        })?;
    let trace_id = trace_id(&native);
    let request_id = trace_id.clone().or(request_id);
    let extra_info = native
        .get("extra_info")
        .cloned()
        .map(serde_json::from_value::<MiniMaxTtsExtraInfo>)
        .transpose()
        .map_err(|_| {
            invalid_response(
                "extra_info has an invalid shape",
                request_id.clone(),
                native.clone(),
            )
        })?;

    match request.output_format {
        MiniMaxTtsOutputFormat::Hex => {
            let bytes = decode_hex(audio_value)
                .map_err(|message| invalid_response(message, request_id.clone(), native.clone()))?;
            if bytes.is_empty() {
                return Err(invalid_response("hex audio is empty", request_id, native));
            }
            if extra_info
                .as_ref()
                .and_then(|info| info.audio_size)
                .is_some_and(|declared| declared != bytes.len() as u64)
            {
                return Err(invalid_response(
                    "decoded audio length does not match extra_info.audio_size",
                    request_id,
                    native,
                ));
            }
            Ok(MiniMaxTtsOutput::Audio(MiniMaxTtsAudio {
                scope,
                bytes: Bytes::from(bytes),
                format: request.audio_setting.format,
                trace_id,
                request_id,
                extra_info,
                native,
            }))
        }
        MiniMaxTtsOutputFormat::Url => {
            validate_audio_url(audio_value)
                .map_err(|message| invalid_response(message, request_id.clone(), native.clone()))?;
            Ok(MiniMaxTtsOutput::Url(MiniMaxTtsUrl {
                scope,
                url: audio_value.to_owned(),
                valid_for: URL_AUDIO_VALIDITY,
                trace_id,
                request_id,
                extra_info,
                native,
            }))
        }
    }
}

fn validate_request(request: &MiniMaxTtsRequest, streaming: bool) -> Result<(), MiniMaxTtsError> {
    let characters = request.text.chars().count();
    if characters == 0 || request.text.trim().is_empty() {
        return Err(invalid("text must not be empty"));
    }
    if characters >= MAX_TEXT_CHARACTERS {
        return Err(invalid("text must be shorter than 10,000 characters"));
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
        .is_some_and(|vol| !vol.is_finite() || vol <= 0.0 || vol > 10.0)
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
    let audio = request.audio_setting;
    if !matches!(
        audio.sample_rate,
        8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100
    ) {
        return Err(invalid(
            "audio sample rate must be one of MiniMax's documented rates",
        ));
    }
    if audio.channel != 1 && audio.channel != 2 {
        return Err(invalid("audio channel count must be 1 or 2"));
    }
    if audio.format == MiniMaxTtsAudioFormat::Mp3
        && !matches!(audio.bitrate, 32_000 | 64_000 | 128_000 | 256_000)
    {
        return Err(invalid(
            "MP3 bitrate must be 32000, 64000, 128000, or 256000",
        ));
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
    if let Some(dictionary) = &request.pronunciation_dict {
        if dictionary
            .tone
            .iter()
            .any(|entry| entry.trim().is_empty() || has_control(entry))
        {
            return Err(invalid(
                "pronunciation entries must be nonempty and contain no control characters",
            ));
        }
    }
    if streaming {
        if request.audio_setting.format != MiniMaxTtsAudioFormat::Mp3 {
            return Err(invalid("MiniMax HTTP streaming supports MP3 audio only"));
        }
        if request.output_format != MiniMaxTtsOutputFormat::Hex {
            return Err(invalid(
                "MiniMax HTTP streaming supports hexadecimal output only",
            ));
        }
        if request.aigc_watermark {
            return Err(invalid(
                "AIGC watermark is available only for non-streaming TTS",
            ));
        }
    } else if request.subtitle_type == MiniMaxTtsSubtitleType::WordStreaming {
        return Err(invalid("word_streaming subtitles require stream=true"));
    }
    Ok(())
}

fn normalize_endpoint(value: &str, region: MiniMaxTtsRegion) -> Result<String, MiniMaxTtsError> {
    let url = Url::parse(value).map_err(|_| invalid("T2A endpoint URL is invalid"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/v1/t2a_v2"
    {
        return Err(invalid(
            "T2A endpoint must be a bare /v1/t2a_v2 URL without credentials, query, or fragment",
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
        return Err(invalid("MiniMax T2A endpoint must use HTTPS"));
    }
    let allowed = match region {
        MiniMaxTtsRegion::International => {
            matches!(host.as_str(), "api.minimax.io" | "api-uw.minimax.io")
        }
        MiniMaxTtsRegion::ChinaMainland => {
            matches!(host.as_str(), "api.minimax.cn" | "api-bj.minimaxi.com")
        }
    };
    if !allowed {
        return Err(invalid(
            "T2A endpoint host does not match the selected MiniMax region",
        ));
    }
    Ok(url.into())
}

fn validate_credential(credential: &Secret<String>) -> Result<(), MiniMaxTtsError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || has_control(value) {
        return Err(invalid("MiniMax API key is empty or malformed"));
    }
    Ok(())
}

fn validate_audio_url(value: &str) -> Result<(), &'static str> {
    let url = Url::parse(value).map_err(|_| "provider audio URL is invalid")?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
    {
        return Err("provider audio URL must be an HTTPS URL without user information");
    }
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, &'static str> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return Err("hex audio has an invalid length");
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let high = hex_nibble(pair[0]).ok_or("hex audio contains a non-hexadecimal character")?;
        let low = hex_nibble(pair[1]).ok_or("hex audio contains a non-hexadecimal character")?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn base_response_code(value: &Value) -> Option<i64> {
    value.get("base_resp")?.get("status_code")?.as_i64()
}

fn validate_stream_base_response(value: &Value) -> Result<Option<i64>, &'static str> {
    let Some(base_response) = value.get("base_resp") else {
        return Ok(None);
    };
    if base_response.is_null() {
        return Ok(None);
    }
    let Some(base_response) = base_response.as_object() else {
        return Err("base_resp must be an object when present");
    };
    match base_response.get("status_code") {
        Some(status_code) => status_code
            .as_i64()
            .map(Some)
            .ok_or("base_resp.status_code must be an integer when present"),
        None => Ok(None),
    }
}

fn base_response_message(value: &Value) -> Option<String> {
    value
        .get("base_resp")?
        .get("status_msg")?
        .as_str()
        .map(str::to_owned)
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

fn trace_id(value: &Value) -> Option<String> {
    value
        .get("trace_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn provider_error(
    http_status: Option<u16>,
    code: Option<i64>,
    message: String,
    request_id: Option<String>,
    native: Value,
    dispatch: MiniMaxTtsDispatch,
) -> MiniMaxTtsError {
    MiniMaxTtsError::Provider {
        http_status,
        code,
        message,
        request_id,
        native: Box::new(native),
        dispatch,
    }
}

fn invalid_response(message: &str, request_id: Option<String>, native: Value) -> MiniMaxTtsError {
    MiniMaxTtsError::InvalidResponse {
        message: message.to_owned(),
        request_id,
        native: Box::new(native),
    }
}

fn map_outcome(source: LlmError) -> MiniMaxTtsError {
    if matches!(
        source,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    ) {
        MiniMaxTtsError::OutcomeUnknown { source }
    } else {
        MiniMaxTtsError::Llm(source)
    }
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}

fn invalid(message: &str) -> MiniMaxTtsError {
    MiniMaxTtsError::InvalidRequest(message.to_owned())
}

const fn default_sample_rate() -> u32 {
    32_000
}

const fn default_bitrate() -> u32 {
    128_000
}

const fn default_channel() -> u8 {
    1
}

fn is_false(value: &bool) -> bool {
    !value
}
