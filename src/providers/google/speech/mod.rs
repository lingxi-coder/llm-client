//! Unary and SSE text-to-speech through Google's Gemini Developer API Interactions.
//!
//! This is separate from the Live API: it takes a transcript and returns a
//! completed audio artifact. It supports the two documented GA Gemini 3.8 TTS
//! models, retains the native Interaction response, and borrows the API key
//! for each operation.

#![doc = concat!(
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/gemini-speech.md")),
    "\n\n",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/gemini-speech.en.md"))
)]

mod voices;
pub use voices::*;

use crate::{
    files::provider_file_endpoint_fingerprint,
    framing::sse::SseFrameSplitter,
    protocol::{LlmError, ProviderId, Secret},
    transport::{
        HttpExecutor, HttpRequest, HttpResponse, StreamResponse, Transport, MAX_ERROR_BODY_SIZE,
    },
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::time::Duration;
use thiserror::Error;
use url::Url;

/// Google's documented Interactions resource used by Gemini TTS.
pub const GEMINI_SPEECH_ENDPOINT: &str =
    "https://generativelanguage.googleapis.com/v1beta/interactions";

const API_HOST: &str = "generativelanguage.googleapis.com";
const INTERACTIONS_PATH: &str = "/v1beta/interactions";
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
/// Maximum bytes read from one streaming Interaction response, including SSE framing.
pub const MAX_GEMINI_SPEECH_STREAM_BYTES: usize = 96 * 1024 * 1024;
/// Maximum decoded audio yielded by one streaming Interaction.
pub const MAX_GEMINI_SPEECH_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT_CHARS: usize = 500_000;
const MAX_STYLE_CHARS: usize = 2_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Supported GA TTS models documented by Google.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeminiSpeechModel {
    #[serde(rename = "gemini-3.8-flash-tts")]
    Gemini38FlashTts,
    #[serde(rename = "gemini-3.8-flash-lite-tts")]
    Gemini38FlashLiteTts,
}

impl GeminiSpeechModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gemini38FlashTts => "gemini-3.8-flash-tts",
            Self::Gemini38FlashLiteTts => "gemini-3.8-flash-lite-tts",
        }
    }
}

/// Audio encodings currently documented by the Gemini 3.8 TTS guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeminiSpeechFormat {
    #[serde(rename = "audio/wav")]
    Wav,
    #[serde(rename = "audio/l16")]
    LinearPcm16,
    #[serde(rename = "audio/mulaw")]
    MuLaw,
    #[serde(rename = "audio/alaw")]
    ALaw,
}

impl GeminiSpeechFormat {
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::LinearPcm16 => "audio/l16",
            Self::MuLaw => "audio/mulaw",
            Self::ALaw => "audio/alaw",
        }
    }
}

/// Sample rates shown by Google's TTS API documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiSpeechSampleRate {
    Hz8000,
    Hz16000,
    Hz24000,
}

impl GeminiSpeechSampleRate {
    pub const fn as_hz(self) -> u32 {
        match self {
            Self::Hz8000 => 8_000,
            Self::Hz16000 => 16_000,
            Self::Hz24000 => 24_000,
        }
    }
}

/// Account and exact endpoint identity for one Gemini Developer API project.
/// The endpoint must be Google's HTTPS Interactions resource; it is retained
/// and fingerprinted so account-scoped results cannot be confused with other
/// providers or routes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiSpeechScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint: String,
    endpoint_fingerprint: String,
}

impl GeminiSpeechScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        endpoint: impl AsRef<str>,
    ) -> Result<Self, GeminiSpeechError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        let endpoint = normalize_endpoint(endpoint.as_ref())?;
        let endpoint_fingerprint = provider_file_endpoint_fingerprint(&endpoint);
        Ok(Self {
            provider_id: ProviderId::from("google"),
            profile_name,
            account_scope,
            endpoint,
            endpoint_fingerprint,
        })
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), GeminiSpeechError> {
        let endpoint = normalize_endpoint(&self.endpoint)?;
        if self.provider_id.as_str() != "google"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || endpoint != self.endpoint
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.endpoint)
        {
            return Err(invalid("Gemini speech scope identity is invalid"));
        }
        Ok(())
    }
}

/// One speaker configured for Gemini's single-request two-speaker TTS mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiSpeechSpeaker {
    /// Label used by the corresponding [`GeminiSpeechTurn`] values.
    pub speaker: String,
    /// A documented prebuilt voice name.
    pub voice: String,
}

impl GeminiSpeechSpeaker {
    pub fn new(speaker: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            voice: voice.into(),
        }
    }
}

/// One text segment in a multi-speaker TTS request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiSpeechTurn {
    /// Must match one of the configured [`GeminiSpeechSpeaker::speaker`] labels.
    pub speaker: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

impl GeminiSpeechTurn {
    pub fn new(speaker: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            text: text.into(),
            style: None,
        }
    }

    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.style = Some(style.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeminiSpeechMultiSpeakerInput {
    speakers: Vec<GeminiSpeechSpeaker>,
    turns: Vec<GeminiSpeechTurn>,
}

/// A single- or multi-speaker TTS request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiSpeechRequest {
    model: GeminiSpeechModel,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    text: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    voice: String,
    style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    multi_speaker: Option<GeminiSpeechMultiSpeakerInput>,
    format: GeminiSpeechFormat,
    sample_rate: GeminiSpeechSampleRate,
}

impl std::fmt::Debug for GeminiSpeechRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let voice = if self.voice.starts_with("voicekey_") {
            "<redacted>"
        } else {
            self.voice.as_str()
        };
        f.debug_struct("GeminiSpeechRequest")
            .field("model", &self.model)
            .field("text", &self.text)
            .field("voice", &voice)
            .field("style", &self.style)
            .field("multi_speaker", &self.multi_speaker)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

impl GeminiSpeechRequest {
    pub fn new(
        model: GeminiSpeechModel,
        text: impl Into<String>,
        voice: impl Into<String>,
    ) -> Result<Self, GeminiSpeechError> {
        let request = Self {
            model,
            text: text.into(),
            voice: voice.into(),
            style: None,
            multi_speaker: None,
            format: GeminiSpeechFormat::Wav,
            sample_rate: GeminiSpeechSampleRate::Hz24000,
        };
        request.validate()?;
        Ok(request)
    }

    /// Construct a two-speaker request using prebuilt voices and explicitly
    /// labeled turns, as documented by Gemini Interactions TTS.
    pub fn new_multi_speaker(
        model: GeminiSpeechModel,
        speakers: Vec<GeminiSpeechSpeaker>,
        turns: Vec<GeminiSpeechTurn>,
    ) -> Result<Self, GeminiSpeechError> {
        let request = Self {
            model,
            text: String::new(),
            voice: String::new(),
            style: None,
            multi_speaker: Some(GeminiSpeechMultiSpeakerInput { speakers, turns }),
            format: GeminiSpeechFormat::Wav,
            sample_rate: GeminiSpeechSampleRate::Hz24000,
        };
        request.validate()?;
        Ok(request)
    }

    /// Construct a request with Gemini 3.8's documented streaming default:
    /// headerless Linear PCM (`audio/l16`) at 24 kHz.
    pub fn new_streaming(
        model: GeminiSpeechModel,
        text: impl Into<String>,
        voice: impl Into<String>,
    ) -> Result<Self, GeminiSpeechError> {
        let mut request = Self::new(model, text, voice)?;
        request.format = GeminiSpeechFormat::LinearPcm16;
        Ok(request)
    }

    /// Construct a multi-speaker request with Gemini 3.8's documented
    /// streaming default: headerless Linear PCM (`audio/l16`) at 24 kHz.
    pub fn new_multi_speaker_streaming(
        model: GeminiSpeechModel,
        speakers: Vec<GeminiSpeechSpeaker>,
        turns: Vec<GeminiSpeechTurn>,
    ) -> Result<Self, GeminiSpeechError> {
        let mut request = Self::new_multi_speaker(model, speakers, turns)?;
        request.format = GeminiSpeechFormat::LinearPcm16;
        Ok(request)
    }

    pub fn with_style(mut self, style: impl Into<String>) -> Result<Self, GeminiSpeechError> {
        if self.multi_speaker.is_some() {
            return Err(invalid(
                "multi-speaker style belongs to each GeminiSpeechTurn, not the whole request",
            ));
        }
        self.style = Some(style.into());
        self.validate()?;
        Ok(self)
    }

    pub fn with_format(mut self, format: GeminiSpeechFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: GeminiSpeechSampleRate) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn model(&self) -> GeminiSpeechModel {
        self.model
    }

    pub fn format(&self) -> GeminiSpeechFormat {
        self.format
    }

    pub fn sample_rate(&self) -> GeminiSpeechSampleRate {
        self.sample_rate
    }

    fn validate(&self) -> Result<(), GeminiSpeechError> {
        if let Some(multi_speaker) = &self.multi_speaker {
            if !self.text.is_empty() || !self.voice.is_empty() || self.style.is_some() {
                return Err(invalid(
                    "multi-speaker TTS cannot also contain single-speaker fields",
                ));
            }
            if multi_speaker.speakers.len() != 2 {
                return Err(invalid(
                    "Gemini multi-speaker TTS requires exactly two configured speakers",
                ));
            }
            if multi_speaker.turns.is_empty() {
                return Err(invalid(
                    "Gemini multi-speaker TTS requires at least one dialogue turn",
                ));
            }
            let mut speaker_names = HashSet::with_capacity(multi_speaker.speakers.len());
            for speaker in &multi_speaker.speakers {
                if !valid_identity(&speaker.speaker) {
                    return Err(invalid(
                        "Gemini speaker labels must be non-empty and contain no control characters",
                    ));
                }
                if !speaker_names.insert(speaker.speaker.as_str()) {
                    return Err(invalid("Gemini multi-speaker labels must be unique"));
                }
                if !valid_prebuilt_voice(&speaker.voice) {
                    return Err(invalid(
                        "Gemini multi-speaker TTS requires valid prebuilt voice names",
                    ));
                }
            }
            let mut total_chars = 0usize;
            for turn in &multi_speaker.turns {
                if turn.text.trim().is_empty() || turn.text.contains('\0') {
                    return Err(invalid(
                        "Gemini dialogue turn text must be non-empty and contain no NUL characters",
                    ));
                }
                if !speaker_names.contains(turn.speaker.as_str()) {
                    return Err(invalid(
                        "each Gemini dialogue turn must name a configured speaker",
                    ));
                }
                total_chars = total_chars.saturating_add(turn.text.chars().count());
                if let Some(style) = &turn.style {
                    validate_style(style)?;
                }
            }
            if total_chars > MAX_TEXT_CHARS {
                return Err(invalid(
                    "combined Gemini dialogue text exceeds the 500000-character local limit",
                ));
            }
        } else {
            if self.text.trim().is_empty()
                || self.text.chars().count() > MAX_TEXT_CHARS
                || self.text.contains('\0')
            {
                return Err(invalid("TTS text must contain 1–500000 non-NUL characters"));
            }
            if !valid_voice(&self.voice) {
                return Err(invalid(
                    "voice must be a prebuilt name or documented custom voice ID",
                ));
            }
            if let Some(style) = &self.style {
                validate_style(style)?;
            }
        }
        Ok(())
    }
}

/// An Interaction ID bound to the Google profile, endpoint, and account that
/// created it. Retain this only when the caller intends to resume a stored
/// stream or reconcile its outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiSpeechInteractionRef {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
    id: String,
}

impl GeminiSpeechInteractionRef {
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

    pub fn id(&self) -> &str {
        &self.id
    }
}

/// How far a synthesis request may have progressed. This service never retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiSpeechDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum GeminiSpeechError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Gemini speech input: {0}")]
    InvalidInput(String),
    #[error("Gemini speech returned HTTP {status}: {message}")]
    Provider {
        status: u16,
        message: String,
        request_id: Option<String>,
        dispatch: GeminiSpeechDispatch,
        native: Box<Value>,
    },
    #[error("Gemini accepted the speech request but returned an invalid response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("Gemini speech request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

impl GeminiSpeechError {
    pub fn dispatch(&self) -> GeminiSpeechDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) => GeminiSpeechDispatch::NotSent,
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => GeminiSpeechDispatch::Accepted,
            Self::OutcomeUnknown { .. } => GeminiSpeechDispatch::Unknown,
        }
    }
}

#[derive(Debug, Error)]
pub enum GeminiSpeechStreamError {
    #[error(
        "Gemini speech stream was interrupted after {audio_bytes_received} audio bytes: {source}"
    )]
    Interrupted {
        #[source]
        source: LlmError,
        interaction_id: Option<String>,
        reference: Option<GeminiSpeechInteractionRef>,
        last_event_id: Option<String>,
        audio_bytes_received: usize,
        interaction_completed: bool,
    },
    #[error("invalid Gemini speech stream event: {message}")]
    InvalidEvent {
        message: String,
        interaction_id: Option<String>,
        reference: Option<GeminiSpeechInteractionRef>,
        last_event_id: Option<String>,
        audio_bytes_received: usize,
        native: Box<Value>,
    },
    #[error("Gemini speech stream reported an error: {message}")]
    ProviderEvent {
        message: String,
        code: Option<String>,
        interaction_id: Option<String>,
        reference: Option<GeminiSpeechInteractionRef>,
        event_id: Option<String>,
        audio_bytes_received: usize,
        native: Box<Value>,
    },
    #[error("Gemini speech stream exceeded the {limit}-byte {kind} limit")]
    LimitExceeded {
        kind: &'static str,
        limit: usize,
        interaction_id: Option<String>,
        reference: Option<GeminiSpeechInteractionRef>,
        last_event_id: Option<String>,
        audio_bytes_received: usize,
        native: Option<Box<Value>>,
    },
}

/// One native Interactions SSE event. Audio deltas are also decoded into
/// `audio_chunk`, while their original base64 and metadata remain in `native`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiSpeechStreamEvent {
    pub event_type: String,
    pub event_id: Option<String>,
    pub interaction_id: Option<String>,
    pub reference: Option<GeminiSpeechInteractionRef>,
    pub step_index: Option<u64>,
    pub audio_chunk: Option<Bytes>,
    pub audio_mime_type: Option<String>,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
    pub native: Value,
}

/// A single bounded Gemini TTS SSE response. Each `next_event` call yields at
/// most one native event; audio chunks are decoded incrementally and are not
/// accumulated into an unbounded artifact.
pub struct GeminiSpeechStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_error: Option<LlmError>,
    ended: bool,
    done: bool,
    interaction_completed: bool,
    scope: GeminiSpeechScope,
    interaction_id: Option<String>,
    reference: Option<GeminiSpeechInteractionRef>,
    last_event_id: Option<String>,
    wire_bytes_received: usize,
    audio_bytes_received: usize,
    requested_format: GeminiSpeechFormat,
    requested_sample_rate: GeminiSpeechSampleRate,
    request_id: Option<String>,
}

impl GeminiSpeechStream {
    fn new(
        response: StreamResponse,
        scope: GeminiSpeechScope,
        requested_format: GeminiSpeechFormat,
        requested_sample_rate: GeminiSpeechSampleRate,
    ) -> Self {
        let request_id = response.header("x-goog-request-id").map(str::to_owned);
        Self {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_error: None,
            ended: false,
            done: false,
            interaction_completed: false,
            scope,
            interaction_id: None,
            reference: None,
            last_event_id: None,
            wire_bytes_received: 0,
            audio_bytes_received: 0,
            requested_format,
            requested_sample_rate,
            request_id,
        }
    }

    pub fn interaction_id(&self) -> Option<&str> {
        self.interaction_id.as_deref()
    }

    pub fn reference(&self) -> Option<&GeminiSpeechInteractionRef> {
        self.reference.as_ref()
    }

    /// Latest JSON `event_id` cursor, suitable for a caller-managed resume.
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    pub fn audio_bytes_received(&self) -> usize {
        self.audio_bytes_received
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn requested_format(&self) -> GeminiSpeechFormat {
        self.requested_format
    }

    pub fn requested_sample_rate(&self) -> GeminiSpeechSampleRate {
        self.requested_sample_rate
    }

    pub async fn next_event(
        &mut self,
    ) -> Result<Option<GeminiSpeechStreamEvent>, GeminiSpeechStreamError> {
        if self.done {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                if frame == b"[DONE]" {
                    self.done = true;
                    if !self.interaction_completed {
                        return Err(self.invalid_event(
                            "stream ended with [DONE] before interaction.completed",
                            Value::String("[DONE]".into()),
                        ));
                    }
                    if self.interaction_id.is_none() {
                        return Err(self.invalid_event(
                            "stream completed without an Interaction ID",
                            Value::String("[DONE]".into()),
                        ));
                    }
                    return Ok(None);
                }
                let native: Value = match serde_json::from_slice(&frame) {
                    Ok(native) => native,
                    Err(_) => {
                        self.done = true;
                        return Err(self.invalid_event(
                            "SSE data was not valid JSON",
                            Value::String(String::from_utf8_lossy(&frame).into_owned()),
                        ));
                    }
                };
                let Some(event_type) = native.get("event_type").and_then(Value::as_str) else {
                    self.done = true;
                    return Err(self.invalid_event("event was missing event_type", native));
                };
                let event_type = event_type.to_owned();
                if self.interaction_completed {
                    self.done = true;
                    return Err(self.invalid_event(
                        "provider emitted another event after interaction.completed",
                        native,
                    ));
                }

                let event_id = native
                    .get("event_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let Some(id) = &event_id {
                    if !valid_event_id(id) {
                        self.done = true;
                        return Err(self.invalid_event("event_id was invalid", native));
                    }
                    self.last_event_id = Some(id.clone());
                }

                let incoming_id = native
                    .get("interaction")
                    .and_then(|interaction| interaction.get("id"))
                    .and_then(Value::as_str)
                    .or_else(|| native.get("interaction_id").and_then(Value::as_str));
                if let Some(id) = incoming_id {
                    if !valid_interaction_id(id)
                        || self
                            .interaction_id
                            .as_deref()
                            .is_some_and(|existing| existing != id)
                    {
                        self.done = true;
                        return Err(self.invalid_event(
                            "Interaction ID was invalid or changed during the stream",
                            native,
                        ));
                    }
                    self.interaction_id = Some(id.into());
                    self.reference = Some(interaction_ref(&self.scope, id));
                }

                if event_type == "error" {
                    self.done = true;
                    return Err(GeminiSpeechStreamError::ProviderEvent {
                        message: native
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("provider emitted an error event")
                            .to_owned(),
                        code: native
                            .pointer("/error/code")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        interaction_id: self.interaction_id.clone(),
                        reference: self.reference.clone(),
                        event_id,
                        audio_bytes_received: self.audio_bytes_received,
                        native: Box::new(native),
                    });
                }

                if event_type == "interaction.completed" {
                    if let Some(status) = native
                        .pointer("/interaction/status")
                        .and_then(Value::as_str)
                        .filter(|status| *status != "completed")
                    {
                        self.done = true;
                        return Err(GeminiSpeechStreamError::ProviderEvent {
                            message: format!("Interaction terminated with status `{status}`"),
                            code: None,
                            interaction_id: self.interaction_id.clone(),
                            reference: self.reference.clone(),
                            event_id,
                            audio_bytes_received: self.audio_bytes_received,
                            native: Box::new(native),
                        });
                    }
                    self.interaction_completed = true;
                }

                let step_index = native.get("index").and_then(Value::as_u64);
                let mut audio_chunk = None;
                let mut audio_mime_type = None;
                let mut audio_sample_rate = None;
                let mut audio_channels = None;
                if native.pointer("/delta/type").and_then(Value::as_str) == Some("audio") {
                    let Some(encoded) = native
                        .pointer("/delta/data")
                        .and_then(Value::as_str)
                        .filter(|data| !data.is_empty())
                    else {
                        self.done = true;
                        return Err(
                            self.invalid_event("audio delta was missing base64 data", native)
                        );
                    };
                    let decoded = match BASE64.decode(encoded) {
                        Ok(bytes) if !bytes.is_empty() => bytes,
                        _ => {
                            self.done = true;
                            return Err(self.invalid_event(
                                "audio delta data was empty or invalid standard base64",
                                native,
                            ));
                        }
                    };
                    let Some(total) = self.audio_bytes_received.checked_add(decoded.len()) else {
                        self.done = true;
                        return Err(self.audio_limit_error(native));
                    };
                    if total > MAX_GEMINI_SPEECH_AUDIO_BYTES {
                        self.done = true;
                        return Err(self.audio_limit_error(native));
                    }
                    self.audio_bytes_received = total;
                    audio_chunk = Some(Bytes::from(decoded));
                    audio_mime_type = native
                        .pointer("/delta/mime_type")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    audio_sample_rate = native
                        .pointer("/delta/sample_rate")
                        .and_then(Value::as_u64)
                        .and_then(|rate| u32::try_from(rate).ok());
                    audio_channels = native
                        .pointer("/delta/channels")
                        .and_then(Value::as_u64)
                        .and_then(|channels| u32::try_from(channels).ok());
                }
                return Ok(Some(GeminiSpeechStreamEvent {
                    event_type,
                    event_id,
                    interaction_id: self.interaction_id.clone(),
                    reference: self.reference.clone(),
                    step_index,
                    audio_chunk,
                    audio_mime_type,
                    audio_sample_rate,
                    audio_channels,
                    native,
                }));
            }

            if let Some(source) = self.pending_error.take() {
                self.done = true;
                return Err(self.interrupted(source));
            }
            if self.ended {
                self.done = true;
                return Err(self.interrupted(LlmError::StreamInterrupted {
                    message: if self.interaction_completed {
                        "Gemini TTS stream ended after interaction.completed but before [DONE]"
                    } else {
                        "Gemini TTS stream ended before [DONE]"
                    }
                    .into(),
                }));
            }

            match self.body.next().await {
                Some(Ok(bytes)) => {
                    let Some(total) = self.wire_bytes_received.checked_add(bytes.len()) else {
                        self.done = true;
                        return Err(self.wire_limit_error());
                    };
                    if total > MAX_GEMINI_SPEECH_STREAM_BYTES {
                        self.done = true;
                        return Err(self.wire_limit_error());
                    }
                    self.wire_bytes_received = total;
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

    fn invalid_event(&self, message: &str, native: Value) -> GeminiSpeechStreamError {
        GeminiSpeechStreamError::InvalidEvent {
            message: message.into(),
            interaction_id: self.interaction_id.clone(),
            reference: self.reference.clone(),
            last_event_id: self.last_event_id.clone(),
            audio_bytes_received: self.audio_bytes_received,
            native: Box::new(native),
        }
    }

    fn interrupted(&self, source: LlmError) -> GeminiSpeechStreamError {
        GeminiSpeechStreamError::Interrupted {
            source,
            interaction_id: self.interaction_id.clone(),
            reference: self.reference.clone(),
            last_event_id: self.last_event_id.clone(),
            audio_bytes_received: self.audio_bytes_received,
            interaction_completed: self.interaction_completed,
        }
    }

    fn audio_limit_error(&self, native: Value) -> GeminiSpeechStreamError {
        GeminiSpeechStreamError::LimitExceeded {
            kind: "decoded audio",
            limit: MAX_GEMINI_SPEECH_AUDIO_BYTES,
            interaction_id: self.interaction_id.clone(),
            reference: self.reference.clone(),
            last_event_id: self.last_event_id.clone(),
            audio_bytes_received: self.audio_bytes_received,
            native: Some(Box::new(native)),
        }
    }

    fn wire_limit_error(&self) -> GeminiSpeechStreamError {
        GeminiSpeechStreamError::LimitExceeded {
            kind: "wire response",
            limit: MAX_GEMINI_SPEECH_STREAM_BYTES,
            interaction_id: self.interaction_id.clone(),
            reference: self.reference.clone(),
            last_event_id: self.last_event_id.clone(),
            audio_bytes_received: self.audio_bytes_received,
            native: None,
        }
    }
}

/// Audio bytes plus provider and account provenance. The complete native
/// Interaction response is retained, including usage and model output steps.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiSpeechResponse {
    pub audio: Bytes,
    pub requested_format: GeminiSpeechFormat,
    pub requested_sample_rate: GeminiSpeechSampleRate,
    pub response_mime_type: Option<String>,
    pub response_sample_rate: Option<u32>,
    pub interaction_id: Option<String>,
    pub model: Option<String>,
    pub scope: GeminiSpeechScope,
    pub native: Value,
}

/// Typed adapter for unary Gemini 3.8 Flash TTS and Flash-Lite TTS.
#[derive(Clone)]
pub struct GeminiSpeechService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: GeminiSpeechScope,
}

impl<'a> GeminiSpeechService<'a> {
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
        http: &'a dyn Transport,
        scope: GeminiSpeechScope,
    ) -> Result<Self, GeminiSpeechError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &GeminiSpeechScope {
        &self.scope
    }

    /// Create one non-streaming TTS interaction. Output audio is base64 in the
    /// documented `model_output` step and is decoded into `audio`; the full
    /// response remains available as `native`.
    pub async fn synthesize(
        &self,
        credential: &Secret<String>,
        request: &GeminiSpeechRequest,
    ) -> Result<GeminiSpeechResponse, GeminiSpeechError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_credential(credential)?;
        request.validate()?;

        let body = speech_body(request, false);
        let body = serde_json::to_vec(&body)
            .map_err(|_| invalid("could not encode Gemini TTS request"))?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(invalid("Gemini TTS request exceeds the 2 MiB local limit"));
        }

        let response = HttpExecutor::new(pinned_service.http)
            .execute_bounded(
                HttpRequest {
                    http1_header_layout: None,
                    method: "POST".into(),
                    url: pinned_service.scope.endpoint.clone(),
                    headers: vec![
                        ("x-goog-api-key".into(), credential.expose_secret().clone()),
                        ("content-type".into(), "application/json".into()),
                    ],
                    body: Bytes::from(body),
                    timeout: Some(REQUEST_TIMEOUT),
                },
                MAX_RESPONSE_BYTES,
            )
            .await
            .map_err(outcome_unknown)?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(provider_error(response, request_id));
        }

        let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
            GeminiSpeechError::InvalidResponse {
                message: "response was not valid JSON".into(),
                request_id: request_id.clone(),
                native: Box::new(Value::Null),
            }
        })?;
        decode_speech_response(pinned_service.scope.clone(), request, request_id, native)
    }

    /// Open one caller-managed streaming TTS Interaction. The returned stream
    /// decodes audio deltas incrementally and retains every native event. It
    /// never reconnects or retries; after interruption, callers can inspect
    /// the Interaction reference and latest event cursor and decide whether
    /// to resume through the documented Interactions GET route.
    pub async fn synthesize_stream(
        &self,
        credential: &Secret<String>,
        request: &GeminiSpeechRequest,
    ) -> Result<GeminiSpeechStream, GeminiSpeechError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_credential(credential)?;
        request.validate()?;

        let body = speech_body(request, true);
        let body = serde_json::to_vec(&body)
            .map_err(|_| invalid("could not encode Gemini TTS stream request"))?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(invalid("Gemini TTS request exceeds the 2 MiB local limit"));
        }

        let response = HttpExecutor::new(pinned_service.http)
            .send(HttpRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url: pinned_service.scope.endpoint.clone(),
                headers: vec![
                    ("x-goog-api-key".into(), credential.expose_secret().clone()),
                    ("content-type".into(), "application/json".into()),
                    ("accept".into(), "text/event-stream".into()),
                ],
                body: Bytes::from(body),
                timeout: Some(REQUEST_TIMEOUT),
            })
            .await
            .map_err(outcome_unknown)?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE))
                .await
                .map_err(outcome_unknown)?;
            return Err(provider_error(response, request_id));
        }
        Ok(GeminiSpeechStream::new(
            response,
            pinned_service.scope.clone(),
            request.format,
            request.sample_rate,
        ))
    }
}

fn speech_body(request: &GeminiSpeechRequest, stream: bool) -> Value {
    let content = if let Some(multi_speaker) = &request.multi_speaker {
        multi_speaker
            .turns
            .iter()
            .map(|turn| {
                let mut speech_metadata = serde_json::Map::new();
                speech_metadata.insert("type".into(), json!("speech_metadata"));
                speech_metadata.insert("speaker".into(), json!(turn.speaker));
                if let Some(style) = &turn.style {
                    speech_metadata.insert("style".into(), json!(style));
                }
                json!({
                    "type": "text",
                    "text": turn.text,
                    "annotations": [Value::Object(speech_metadata)],
                })
            })
            .collect::<Vec<_>>()
    } else {
        let mut text_part = serde_json::Map::new();
        text_part.insert("type".into(), json!("text"));
        text_part.insert("text".into(), json!(request.text));
        if let Some(style) = &request.style {
            text_part.insert(
                "annotations".into(),
                json!([{"type": "speech_metadata", "style": style}]),
            );
        }
        vec![Value::Object(text_part)]
    };
    let speech_config = if let Some(multi_speaker) = &request.multi_speaker {
        json!({
            "mode": "conversational",
            "speakers": multi_speaker.speakers.iter().map(|speaker| json!({
                "speaker": speaker.speaker,
                "voice": speaker.voice,
            })).collect::<Vec<_>>(),
        })
    } else {
        json!([{"voice": request.voice}])
    };
    json!({
        "model": request.model.as_str(),
        "input": [{
            "type": "user_input",
            "content": content,
        }],
        "response_format": {
            "type": "audio",
            "mime_type": request.format.mime_type(),
            "sample_rate": request.sample_rate.as_hz(),
        },
        "generation_config": {
            "speech_config": speech_config,
        },
        "stream": stream,
    })
}

fn decode_speech_response(
    scope: GeminiSpeechScope,
    request: &GeminiSpeechRequest,
    request_id: Option<String>,
    native: Value,
) -> Result<GeminiSpeechResponse, GeminiSpeechError> {
    let audio_part = native
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|step| step.get("type").and_then(Value::as_str) == Some("model_output"))
        .flat_map(|step| {
            step.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .rfind(|part| part.get("type").and_then(Value::as_str) == Some("audio"))
        .ok_or_else(|| {
            invalid_response(
                "response contained no audio output step",
                request_id.clone(),
                &native,
            )
        })?;
    let encoded_audio = audio_part
        .get("data")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty())
        .ok_or_else(|| {
            invalid_response(
                "audio output had no inline data",
                request_id.clone(),
                &native,
            )
        })?;
    let audio = BASE64.decode(encoded_audio).map_err(|_| {
        invalid_response(
            "audio output data was not valid standard base64",
            request_id.clone(),
            &native,
        )
    })?;
    if audio.is_empty() {
        return Err(invalid_response(
            "audio output decoded to an empty byte sequence",
            request_id,
            &native,
        ));
    }
    Ok(GeminiSpeechResponse {
        audio: Bytes::from(audio),
        requested_format: request.format,
        requested_sample_rate: request.sample_rate,
        response_mime_type: audio_part
            .get("mime_type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        response_sample_rate: audio_part
            .get("sample_rate")
            .and_then(Value::as_u64)
            .and_then(|rate| u32::try_from(rate).ok()),
        interaction_id: native.get("id").and_then(Value::as_str).map(str::to_owned),
        model: native
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        scope,
        native,
    })
}

fn provider_error(response: HttpResponse, header_request_id: Option<String>) -> GeminiSpeechError {
    let native = parse_body(&response.body);
    let status = response.status;
    let message = native
        .pointer("/error/message")
        .or_else(|| native.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("provider returned an error")
        .to_owned();
    let request_id = header_request_id.or_else(|| {
        native
            .pointer("/error/details")
            .and_then(Value::as_array)
            .and_then(|details| {
                details.iter().find_map(|detail| {
                    detail
                        .get("requestId")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
            })
    });
    let dispatch = if (400..500).contains(&status) {
        GeminiSpeechDispatch::Rejected
    } else {
        GeminiSpeechDispatch::Unknown
    };
    GeminiSpeechError::Provider {
        status,
        message,
        request_id,
        dispatch,
        native: Box::new(native),
    }
}

fn invalid_response(
    message: &str,
    request_id: Option<String>,
    native: &Value,
) -> GeminiSpeechError {
    GeminiSpeechError::InvalidResponse {
        message: message.into(),
        request_id,
        native: Box::new(native.clone()),
    }
}

fn normalize_endpoint(endpoint: &str) -> Result<String, GeminiSpeechError> {
    let mut url = Url::parse(endpoint)
        .map_err(|_| invalid("Gemini speech endpoint must be an absolute HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str() != Some(API_HOST)
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != INTERACTIONS_PATH
    {
        return Err(invalid(
            "endpoint must be Google's HTTPS /v1beta/interactions resource without query or fragment",
        ));
    }
    url.set_path(INTERACTIONS_PATH);
    Ok(url.into())
}

fn interaction_ref(scope: &GeminiSpeechScope, id: &str) -> GeminiSpeechInteractionRef {
    GeminiSpeechInteractionRef {
        provider_id: scope.provider_id.clone(),
        profile_name: scope.profile_name.clone(),
        account_scope: scope.account_scope.clone(),
        endpoint_fingerprint: scope.endpoint_fingerprint.clone(),
        id: id.to_owned(),
    }
}

fn valid_interaction_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn valid_event_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 4096 && !id.chars().any(char::is_control)
}

fn validate_credential(credential: &Secret<String>) -> Result<(), GeminiSpeechError> {
    let key = credential.expose_secret();
    if key.trim().is_empty() || key.contains('\r') || key.contains('\n') {
        return Err(invalid(
            "Google API key must be non-empty and contain no CR/LF",
        ));
    }
    Ok(())
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_voice(value: &str) -> bool {
    valid_voice_id(value) || valid_voice_key(value)
}

fn valid_voice_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn valid_voice_key(value: &str) -> bool {
    value.starts_with("voicekey_")
        && value.len() > "voicekey_".len()
        && value.len() <= 4_096
        && !value.chars().any(char::is_control)
}

fn valid_prebuilt_voice(value: &str) -> bool {
    valid_voice_id(value) && !value.starts_with("voice_") && !value.starts_with("voicekey_")
}

fn validate_style(style: &str) -> Result<(), GeminiSpeechError> {
    if style.chars().count() > MAX_STYLE_CHARS || style.chars().any(char::is_control) {
        return Err(invalid(
            "speech style must contain at most 2000 characters and no control characters",
        ));
    }
    Ok(())
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn parse_body(body: &[u8]) -> Value {
    serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()))
}

fn outcome_unknown(source: LlmError) -> GeminiSpeechError {
    if matches!(
        &source,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    ) {
        GeminiSpeechError::OutcomeUnknown { source }
    } else {
        GeminiSpeechError::Llm(source)
    }
}

fn invalid(message: &str) -> GeminiSpeechError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
    .into()
}
