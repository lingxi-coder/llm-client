//! Vertex AI Gemini-TTS over the documented publisher-model REST API.
//!
//! The caller chooses the Google Cloud project, location, model, and bearer
//! access token. This service does not fetch or refresh credentials, download
//! or play audio, retry requests, or route through the Cloud Text-to-Speech
//! API.

#![doc = concat!(
    include_str!("../../docs/vertex-speech.md"),
    "\n\n",
    include_str!("../../docs/vertex-speech.en.md")
)]

use crate::{
    client::RequestOptions,
    framing::sse::SseFrameSplitter,
    protocol::{LlmError, Secret},
    transport::{
        HttpExecutor, HttpRequest, HttpResponse, StreamResponse, Transport, MAX_ERROR_BODY_SIZE,
    },
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, StreamExt};
use serde_json::{json, Value};
use std::{collections::VecDeque, fmt, time::Duration};
use thiserror::Error;

/// API version used by Google's current Vertex Gemini-TTS REST examples.
pub const VERTEX_SPEECH_API_VERSION: &str = "v1beta1";
/// Global endpoint. Regional calls use `{location}-aiplatform.googleapis.com`.
pub const VERTEX_SPEECH_GLOBAL_ENDPOINT: &str = "https://aiplatform.googleapis.com";

const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_STREAM_BYTES: usize = 96 * 1024 * 1024;
const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Gemini-TTS model identifiers currently listed in Google's Vertex guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexSpeechModel {
    Gemini31FlashTtsPreview,
    Gemini25FlashTts,
    Gemini25FlashLitePreviewTts,
    Gemini25ProTts,
}

impl VertexSpeechModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gemini31FlashTtsPreview => "gemini-3.1-flash-tts-preview",
            Self::Gemini25FlashTts => "gemini-2.5-flash-tts",
            Self::Gemini25FlashLitePreviewTts => "gemini-2.5-flash-lite-preview-tts",
            Self::Gemini25ProTts => "gemini-2.5-pro-tts",
        }
    }

    const fn supports_multi_speaker(self) -> bool {
        !matches!(self, Self::Gemini25FlashLitePreviewTts)
    }

    fn supports_location(self, location: &str) -> bool {
        if location == "global" {
            return true;
        }
        match location {
            "northamerica-northeast1" => matches!(
                self,
                Self::Gemini25FlashTts | Self::Gemini25FlashLitePreviewTts
            ),
            "europe-central2" | "europe-north1" | "europe-southwest1" | "europe-west1"
            | "europe-west4" | "us-central1" | "us-east1" | "us-east4" | "us-east5"
            | "us-south1" | "us-west1" | "us-west4" => {
                !matches!(self, Self::Gemini31FlashTtsPreview)
            }
            _ => false,
        }
    }
}

/// Explicit caller-owned Google Cloud project and regional account scope.
///
/// The service derives the HTTPS authority from `location` and does not accept
/// an arbitrary URL. The `account_scope` is a stable local identity; it is not
/// sent to Google. `project_id` is sent in both the publisher-model resource
/// name and `x-goog-user-project` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexSpeechScope {
    profile_name: String,
    account_scope: String,
    project_id: String,
    location: String,
}

impl VertexSpeechScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self, VertexSpeechError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            project_id: project_id.into(),
            location: location.into(),
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

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    fn validate(&self) -> Result<(), VertexSpeechError> {
        if !valid_identity(&self.profile_name) || !valid_identity(&self.account_scope) {
            return Err(invalid(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        if !valid_path_label(&self.project_id) {
            return Err(invalid(
                "Google Cloud project ID or number must be a safe path label",
            ));
        }
        if !valid_location(&self.location) {
            return Err(invalid(
                "location must be global or a documented Vertex Gemini-TTS region",
            ));
        }
        Ok(())
    }

    fn endpoint(&self, model: VertexSpeechModel, streaming: bool) -> String {
        let host = if self.location == "global" {
            "aiplatform.googleapis.com".to_owned()
        } else {
            format!("{}-aiplatform.googleapis.com", self.location)
        };
        let action = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "https://{host}/{}/projects/{}/locations/{}/publishers/google/models/{}:{action}",
            VERTEX_SPEECH_API_VERSION,
            self.project_id,
            self.location,
            model.as_str(),
        )
    }
}

/// One speaker configured for a Vertex Gemini-TTS dialogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexSpeechSpeaker {
    pub speaker: String,
    pub voice_name: String,
}

impl VertexSpeechSpeaker {
    pub fn new(speaker: impl Into<String>, voice_name: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            voice_name: voice_name.into(),
        }
    }
}

/// One line in a Vertex Gemini-TTS multi-speaker dialogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexSpeechTurn {
    pub speaker: String,
    pub text: String,
}

impl VertexSpeechTurn {
    pub fn new(speaker: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            text: text.into(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum SpeechMode {
    Single {
        text: String,
        voice_name: String,
    },
    Multi {
        speakers: Vec<VertexSpeechSpeaker>,
        turns: Vec<VertexSpeechTurn>,
    },
}

/// Single- or two-speaker synthesis input for one supported Vertex TTS model.
#[derive(Clone, PartialEq, Eq)]
pub struct VertexSpeechRequest {
    model: VertexSpeechModel,
    language_code: String,
    prompt: Option<String>,
    mode: SpeechMode,
}

impl VertexSpeechRequest {
    /// Build a single-speaker request. Supply an optional style prompt with
    /// [`with_prompt`](Self::with_prompt).
    pub fn single(
        model: VertexSpeechModel,
        text: impl Into<String>,
        language_code: impl Into<String>,
        voice_name: impl Into<String>,
    ) -> Self {
        Self {
            model,
            language_code: language_code.into(),
            prompt: None,
            mode: SpeechMode::Single {
                text: text.into(),
                voice_name: voice_name.into(),
            },
        }
    }

    /// Build a two-speaker request. Dialogue turns are encoded as speaker-
    /// labeled lines in Vertex's documented `contents` text field.
    pub fn multi_speaker(
        model: VertexSpeechModel,
        language_code: impl Into<String>,
        speakers: Vec<VertexSpeechSpeaker>,
        turns: Vec<VertexSpeechTurn>,
    ) -> Self {
        Self {
            model,
            language_code: language_code.into(),
            prompt: None,
            mode: SpeechMode::Multi { speakers, turns },
        }
    }

    /// Add the natural-language style prompt used by Gemini-TTS.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn model(&self) -> VertexSpeechModel {
        self.model
    }

    pub fn language_code(&self) -> &str {
        &self.language_code
    }

    fn validate(&self, scope: &VertexSpeechScope) -> Result<(), VertexSpeechError> {
        if !self.model.supports_location(&scope.location) {
            return Err(invalid(format!(
                "model {} is not listed for Vertex location {}",
                self.model.as_str(),
                scope.location
            )));
        }
        if !valid_language(&self.language_code) {
            return Err(invalid(
                "language code must be a non-empty locale tag using ASCII letters, digits, and hyphens",
            ));
        }
        if self
            .prompt
            .as_deref()
            .is_some_and(|prompt| prompt.trim().is_empty() || has_disallowed_control(prompt))
        {
            return Err(invalid(
                "style prompt must be non-empty and contain no control characters",
            ));
        }

        match &self.mode {
            SpeechMode::Single { text, voice_name } => {
                if text.trim().is_empty() || has_disallowed_control(text) {
                    return Err(invalid(
                        "speech text must be non-empty and contain no NUL/control characters",
                    ));
                }
                if !valid_voice_name(voice_name) {
                    return Err(invalid("prebuilt voice name is invalid"));
                }
            }
            SpeechMode::Multi { speakers, turns } => {
                if !self.model.supports_multi_speaker() {
                    return Err(invalid(format!(
                        "model {} is documented for single-speaker synthesis only",
                        self.model.as_str()
                    )));
                }
                if speakers.len() != 2 {
                    return Err(invalid(
                        "Vertex Gemini-TTS multi-speaker requests require exactly two speakers",
                    ));
                }
                if turns.is_empty() {
                    return Err(invalid("multi-speaker dialogue needs at least one turn"));
                }
                let mut names = std::collections::HashSet::new();
                for speaker in speakers {
                    if !valid_speaker(&speaker.speaker) || !names.insert(&speaker.speaker) {
                        return Err(invalid(
                            "speaker labels must be unique, non-empty ASCII alphanumeric identifiers",
                        ));
                    }
                    if !valid_voice_name(&speaker.voice_name) {
                        return Err(invalid("prebuilt voice name is invalid"));
                    }
                }
                for turn in turns {
                    if !names.contains(&turn.speaker) {
                        return Err(invalid(
                            "every dialogue turn must name a configured speaker",
                        ));
                    }
                    if turn.text.trim().is_empty() || has_disallowed_control(&turn.text) {
                        return Err(invalid("dialogue turn text must be non-empty and contain no NUL/control characters"));
                    }
                }
            }
        }

        if self.contents_text().len() > 8_000 {
            return Err(invalid(
                "Vertex Gemini-TTS contents exceed the documented 8000-byte input limit",
            ));
        }
        Ok(())
    }

    fn contents_text(&self) -> String {
        let text = match &self.mode {
            SpeechMode::Single { text, .. } => text.clone(),
            SpeechMode::Multi { turns, .. } => turns
                .iter()
                .map(|turn| format!("{}: {}", turn.speaker, turn.text))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        match self.prompt.as_deref() {
            Some(prompt) => format!("{prompt}: {text}"),
            None => text,
        }
    }

    fn body(&self) -> Value {
        let mut speech_config = serde_json::Map::new();
        speech_config.insert("languageCode".into(), json!(self.language_code));
        match &self.mode {
            SpeechMode::Single { voice_name, .. } => {
                speech_config.insert(
                    "voiceConfig".into(),
                    json!({ "prebuiltVoiceConfig": { "voiceName": voice_name } }),
                );
            }
            SpeechMode::Multi { speakers, .. } => {
                speech_config.insert(
                    "multiSpeakerVoiceConfig".into(),
                    json!({
                        "speakerVoiceConfigs": speakers.iter().map(|speaker| json!({
                            "speaker": speaker.speaker,
                            "voiceConfig": {
                                "prebuiltVoiceConfig": { "voiceName": speaker.voice_name }
                            }
                        })).collect::<Vec<_>>()
                    }),
                );
            }
        }
        json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": self.contents_text() }]
            }],
            "generationConfig": { "speechConfig": speech_config }
        })
    }
}

impl fmt::Debug for VertexSpeechRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = match &self.mode {
            SpeechMode::Single { voice_name, .. } => format!("single voice={voice_name}"),
            SpeechMode::Multi { speakers, turns } => {
                format!("multi speakers={} turns={}", speakers.len(), turns.len())
            }
        };
        f.debug_struct("VertexSpeechRequest")
            .field("model", &self.model)
            .field("language_code", &self.language_code)
            .field("prompt", &self.prompt.as_ref().map(|_| "<redacted>"))
            .field("contents", &"<redacted speech text>")
            .field("mode", &mode)
            .finish()
    }
}

/// Unary synthesis result. Vertex Gemini-TTS returns headerless PCM16 mono at
/// 24 kHz; the complete native GenerateContent response remains available.
#[derive(Clone, PartialEq)]
pub struct VertexSpeechResponse {
    pub audio: Bytes,
    pub mime_type: Option<String>,
    pub pcm_sample_rate_hz: u32,
    pub pcm_channels: u8,
    pub pcm_bits_per_sample: u8,
    /// Candidate finish reason, when supplied. Non-`STOP` reasons remain
    /// visible so callers can identify output that may be partial.
    pub finish_reason: Option<String>,
    pub model: VertexSpeechModel,
    pub scope: VertexSpeechScope,
    pub native: Value,
}

impl fmt::Debug for VertexSpeechResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexSpeechResponse")
            .field("audio_bytes", &self.audio.len())
            .field("mime_type", &self.mime_type)
            .field("pcm_sample_rate_hz", &self.pcm_sample_rate_hz)
            .field("pcm_channels", &self.pcm_channels)
            .field("pcm_bits_per_sample", &self.pcm_bits_per_sample)
            .field("finish_reason", &self.finish_reason)
            .field("model", &self.model)
            .field("scope", &self.scope)
            .field("native", &"<redacted native audio response>")
            .finish()
    }
}

/// How far the one-shot synthesis request may have progressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexSpeechDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum VertexSpeechError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Vertex speech input: {0}")]
    InvalidInput(String),
    #[error("Vertex speech account scope does not match request options")]
    ScopeMismatch,
    #[error("Vertex speech provider rejected the request with HTTP {status}: {message}")]
    ProviderRejected {
        status: u16,
        message: String,
        request_id: Option<String>,
    },
    #[error("Vertex accepted the speech request but returned an invalid response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
    },
    #[error("Vertex speech request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
}

impl VertexSpeechError {
    pub fn dispatch(&self) -> VertexSpeechDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::ScopeMismatch => {
                VertexSpeechDispatch::NotSent
            }
            Self::ProviderRejected { .. } => VertexSpeechDispatch::Rejected,
            Self::InvalidResponse { .. } => VertexSpeechDispatch::Accepted,
            Self::OutcomeUnknown { .. } => VertexSpeechDispatch::Unknown,
        }
    }
}

/// HTTP service for the Vertex Gemini-TTS publisher-model route.
pub struct VertexSpeechService<'a> {
    transport: &'a dyn Transport,
    scope: VertexSpeechScope,
}

impl<'a> VertexSpeechService<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        scope: VertexSpeechScope,
    ) -> Result<Self, VertexSpeechError> {
        scope.validate()?;
        Ok(Self { transport, scope })
    }

    pub fn scope(&self) -> &VertexSpeechScope {
        &self.scope
    }

    /// Synthesize one complete input and decode the provider's PCM audio.
    /// The bearer token and optional total deadline are taken from
    /// `RequestOptions`; this call is never retried.
    pub async fn synthesize(
        &self,
        request: &VertexSpeechRequest,
        options: &RequestOptions,
    ) -> Result<VertexSpeechResponse, VertexSpeechError> {
        let http_request = self.http_request(request, options, false)?;
        let response = HttpExecutor::new(self.transport)
            .execute_bounded(http_request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| VertexSpeechError::OutcomeUnknown {
                source,
                request_id: None,
            })?;
        let request_id = response_request_id(&response.headers);
        if !(200..300).contains(&response.status) {
            return Err(provider_error(response, request_id));
        }
        let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
            VertexSpeechError::InvalidResponse {
                message: "response was not valid JSON".into(),
                request_id: request_id.clone(),
            }
        })?;
        decode_response(self.scope.clone(), request.model, request_id, native)
    }

    /// Start one HTTP server-streaming generation request. The stream yields
    /// one event for every native GenerateContent response, with any decoded
    /// audio and the original response JSON retained together.
    pub async fn synthesize_stream(
        &self,
        request: &VertexSpeechRequest,
        options: &RequestOptions,
    ) -> Result<VertexSpeechStream, VertexSpeechError> {
        let http_request = self.http_request(request, options, true)?;
        let response = HttpExecutor::new(self.transport)
            .send(http_request)
            .await
            .map_err(|source| VertexSpeechError::OutcomeUnknown {
                source,
                request_id: None,
            })?;
        let request_id = response_request_id(&response.headers);
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE))
                .await
                .map_err(|source| VertexSpeechError::OutcomeUnknown {
                    source,
                    request_id: request_id.clone(),
                })?;
            return Err(provider_error(response, request_id));
        }
        if response.header("content-type").is_some_and(|value| {
            !value
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
        }) {
            return Err(VertexSpeechError::InvalidResponse {
                message: "streamGenerateContent did not return text/event-stream".into(),
                request_id,
            });
        }
        Ok(VertexSpeechStream::new(
            response,
            self.scope.clone(),
            request.model,
            request_id,
        ))
    }

    fn http_request(
        &self,
        request: &VertexSpeechRequest,
        options: &RequestOptions,
        streaming: bool,
    ) -> Result<HttpRequest, VertexSpeechError> {
        self.scope.validate()?;
        request.validate(&self.scope)?;
        if options
            .account_scope
            .as_deref()
            .is_some_and(|value| value != self.scope.account_scope)
        {
            return Err(VertexSpeechError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message:
                    "Vertex speech requires a caller-supplied Google Cloud bearer access token"
                        .into(),
            })?;
        validate_bearer_token(credential)?;
        let body = serde_json::to_vec(&request.body())
            .map_err(|_| invalid("could not encode Vertex speech request"))?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(invalid(
                "Vertex speech request exceeds the local 64 KiB limit",
            ));
        }
        Ok(HttpRequest {
            method: "POST".into(),
            url: self.scope.endpoint(request.model, streaming),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                ),
                ("x-goog-user-project".into(), self.scope.project_id.clone()),
                ("content-type".into(), "application/json".into()),
                (
                    "accept".into(),
                    if streaming {
                        "text/event-stream"
                    } else {
                        "application/json"
                    }
                    .into(),
                ),
            ],
            body: Bytes::from(body),
            timeout: Some(options.total_timeout.unwrap_or(DEFAULT_TIMEOUT)),
        })
    }
}

/// One streamed Vertex `GenerateContentResponse`; `audio` is empty when that
/// response contains no audio part, and `native` always retains the JSON.
#[derive(Clone, PartialEq)]
pub struct VertexSpeechStreamEvent {
    pub audio: Bytes,
    pub mime_type: Option<String>,
    /// Candidate finish reason, when this response marks generation complete.
    pub finish_reason: Option<String>,
    /// Prompt block reason for the documented no-candidate blocked response.
    pub prompt_block_reason: Option<String>,
    pub native: Value,
}

impl fmt::Debug for VertexSpeechStreamEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexSpeechStreamEvent")
            .field("audio_bytes", &self.audio.len())
            .field("mime_type", &self.mime_type)
            .field("finish_reason", &self.finish_reason)
            .field("prompt_block_reason", &self.prompt_block_reason)
            .field("native", &"<redacted native audio response>")
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum VertexSpeechStreamError {
    #[error(
        "Vertex speech stream was interrupted after {audio_bytes_received} audio bytes: {source}"
    )]
    Interrupted {
        #[source]
        source: LlmError,
        audio_bytes_received: usize,
        request_id: Option<String>,
    },
    #[error("invalid Vertex speech stream response: {message}")]
    InvalidResponse {
        message: String,
        audio_bytes_received: usize,
        request_id: Option<String>,
    },
    #[error("Vertex speech stream ended before a terminal GenerateContent response after {audio_bytes_received} audio bytes")]
    PrematureEof {
        audio_bytes_received: usize,
        request_id: Option<String>,
    },
}

/// Incremental decoder for the REST `streamGenerateContent?alt=sse` response.
pub struct VertexSpeechStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    scope: VertexSpeechScope,
    model: VertexSpeechModel,
    request_id: Option<String>,
    stream_bytes_received: usize,
    audio_bytes_received: usize,
    terminal_received: bool,
    completion_reason: Option<String>,
    source_done: bool,
    done: bool,
}

impl VertexSpeechStream {
    fn new(
        response: StreamResponse,
        scope: VertexSpeechScope,
        model: VertexSpeechModel,
        request_id: Option<String>,
    ) -> Self {
        Self {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            scope,
            model,
            request_id,
            stream_bytes_received: 0,
            audio_bytes_received: 0,
            terminal_received: false,
            completion_reason: None,
            source_done: false,
            done: false,
        }
    }

    pub fn scope(&self) -> &VertexSpeechScope {
        &self.scope
    }

    pub fn model(&self) -> VertexSpeechModel {
        self.model
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn audio_bytes_received(&self) -> usize {
        self.audio_bytes_received
    }

    /// The terminal finish or prompt-block reason, if the stream supplied one.
    pub fn completion_reason(&self) -> Option<&str> {
        self.completion_reason.as_deref()
    }

    /// Yield one decoded native response event at a time. Dropping the stream
    /// drops the underlying HTTP response body and does not reconnect.
    pub async fn next_event(
        &mut self,
    ) -> Result<Option<VertexSpeechStreamEvent>, VertexSpeechStreamError> {
        if self.done {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                return self.decode_frame(frame).map(Some);
            }
            if self.source_done {
                let final_frame = match self.splitter.finish() {
                    Ok(frame) => frame,
                    Err(source) => {
                        self.done = true;
                        return Err(self.interrupted(source));
                    }
                };
                if let Some(frame) = final_frame {
                    return self.decode_frame(frame).map(Some);
                }
                self.done = true;
                if self.terminal_received {
                    return Ok(None);
                }
                return Err(VertexSpeechStreamError::PrematureEof {
                    audio_bytes_received: self.audio_bytes_received,
                    request_id: self.request_id.clone(),
                });
            }
            match self.body.next().await {
                Some(Ok(chunk)) => {
                    self.stream_bytes_received = self
                        .stream_bytes_received
                        .checked_add(chunk.len())
                        .ok_or_else(|| {
                            self.done = true;
                            self.invalid_event("stream byte count overflowed")
                        })?;
                    if self.stream_bytes_received > MAX_STREAM_BYTES {
                        self.done = true;
                        return Err(
                            self.invalid_event("stream exceeded the local 96 MiB wire limit")
                        );
                    }
                    let frames = match self.splitter.push(&chunk) {
                        Ok(frames) => frames,
                        Err(source) => {
                            self.done = true;
                            return Err(self.interrupted(source));
                        }
                    };
                    self.ready.extend(frames);
                }
                Some(Err(source)) => {
                    self.done = true;
                    return Err(self.interrupted(source));
                }
                None => self.source_done = true,
            }
        }
    }

    fn decode_frame(
        &mut self,
        frame: Vec<u8>,
    ) -> Result<VertexSpeechStreamEvent, VertexSpeechStreamError> {
        let native: Value = match serde_json::from_slice(&frame) {
            Ok(native) => native,
            Err(_) => {
                self.done = true;
                return Err(
                    self.invalid_event("SSE data was not a valid GenerateContent JSON response")
                );
            }
        };
        if let Some(error) = native.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("provider returned an in-stream error")
                .chars()
                .take(1_000)
                .collect::<String>();
            self.done = true;
            return Err(self.invalid_event(&format!("provider in-stream error: {message}")));
        }
        let (audio, mime_type) = match decode_audio_parts(&native, self.audio_bytes_received) {
            Ok(audio) => audio,
            Err(message) => {
                self.done = true;
                return Err(self.invalid_event(&message));
            }
        };
        self.audio_bytes_received = self
            .audio_bytes_received
            .checked_add(audio.len())
            .ok_or_else(|| {
                self.done = true;
                self.invalid_event("audio byte count overflowed")
            })?;
        if self.audio_bytes_received > MAX_AUDIO_BYTES {
            self.done = true;
            return Err(self.invalid_event("decoded audio exceeded the local 64 MiB limit"));
        }
        let (finish_reason, prompt_block_reason, terminal) = stream_completion(&native);
        if terminal {
            self.terminal_received = true;
            self.completion_reason = finish_reason.as_deref().map(str::to_owned).or_else(|| {
                prompt_block_reason
                    .as_deref()
                    .map(|reason| format!("PROMPT_BLOCKED:{reason}"))
            });
        }
        Ok(VertexSpeechStreamEvent {
            audio,
            mime_type,
            finish_reason,
            prompt_block_reason,
            native,
        })
    }

    fn interrupted(&self, source: LlmError) -> VertexSpeechStreamError {
        VertexSpeechStreamError::Interrupted {
            source,
            audio_bytes_received: self.audio_bytes_received,
            request_id: self.request_id.clone(),
        }
    }

    fn invalid_event(&self, message: &str) -> VertexSpeechStreamError {
        VertexSpeechStreamError::InvalidResponse {
            message: message.into(),
            audio_bytes_received: self.audio_bytes_received,
            request_id: self.request_id.clone(),
        }
    }
}

fn decode_response(
    scope: VertexSpeechScope,
    model: VertexSpeechModel,
    request_id: Option<String>,
    native: Value,
) -> Result<VertexSpeechResponse, VertexSpeechError> {
    let (finish_reason, prompt_block_reason, _) = stream_completion(&native);
    let (audio, mime_type) =
        decode_audio_parts(&native, 0).map_err(|message| VertexSpeechError::InvalidResponse {
            message,
            request_id: request_id.clone(),
        })?;
    if audio.is_empty() {
        return Err(VertexSpeechError::InvalidResponse {
            message: prompt_block_reason.map_or_else(
                || "successful response contained no audio bytes".into(),
                |reason| format!("request was blocked ({reason}) and returned no audio bytes"),
            ),
            request_id,
        });
    }
    Ok(VertexSpeechResponse {
        audio,
        mime_type,
        pcm_sample_rate_hz: 24_000,
        pcm_channels: 1,
        pcm_bits_per_sample: 16,
        finish_reason,
        model,
        scope,
        native,
    })
}

fn decode_audio_parts(
    native: &Value,
    already_received: usize,
) -> Result<(Bytes, Option<String>), String> {
    let mut audio = BytesMut::new();
    let mut mime_type = None;
    let Some(candidates) = native.get("candidates") else {
        return Ok((audio.freeze(), mime_type));
    };
    let candidates = candidates
        .as_array()
        .ok_or_else(|| "response candidates field was not an array".to_owned())?;
    let Some(candidate) = candidates.first() else {
        return Ok((audio.freeze(), mime_type));
    };
    let Some(content) = candidate.get("content") else {
        return Ok((audio.freeze(), mime_type));
    };
    let Some(parts) = content.get("parts") else {
        return Ok((audio.freeze(), mime_type));
    };
    let parts = parts
        .as_array()
        .ok_or_else(|| "candidate content parts field was not an array".to_owned())?;
    for part in parts {
        let inline = part.get("inlineData").or_else(|| part.get("inline_data"));
        let Some(inline) = inline else {
            continue;
        };
        let mime = inline
            .get("mimeType")
            .or_else(|| inline.get("mime_type"))
            .and_then(Value::as_str);
        if let Some(mime) = mime {
            validate_pcm_mime(mime)?;
        }
        let data = inline
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| "audio inlineData was missing its base64 data field".to_owned())?;
        let encoded_bound = MAX_AUDIO_BYTES
            .saturating_sub(already_received)
            .saturating_sub(audio.len());
        if data.len()
            > encoded_bound
                .saturating_mul(4)
                .div_ceil(3)
                .saturating_add(4)
        {
            return Err("encoded audio exceeded the local decoded audio limit".into());
        }
        let decoded = BASE64
            .decode(data)
            .map_err(|_| "audio inlineData contained invalid base64".to_owned())?;
        if decoded.len() > encoded_bound {
            return Err("decoded audio exceeded the local 64 MiB limit".into());
        }
        audio.extend_from_slice(&decoded);
        if mime_type.is_none() {
            mime_type = mime.map(str::to_owned);
        }
    }
    let audio = audio.freeze();
    let mime_type = if audio.is_empty() { None } else { mime_type };
    Ok((audio, mime_type))
}

fn validate_pcm_mime(mime: &str) -> Result<(), String> {
    let mut parameters = mime.split(';');
    let media_type = parameters.next().unwrap_or_default().trim();
    if !media_type.eq_ignore_ascii_case("audio/pcm")
        && !media_type.eq_ignore_ascii_case("audio/l16")
    {
        return Err(format!(
            "Vertex Gemini-TTS returned {media_type:?}; expected documented raw linear PCM"
        ));
    }
    for parameter in parameters {
        let Some((key, value)) = parameter.trim().split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        if key.trim().eq_ignore_ascii_case("rate") && value != "24000" {
            return Err("Vertex Gemini-TTS audio MIME declared a non-24000 Hz sample rate".into());
        }
        if key.trim().eq_ignore_ascii_case("codec") && !value.eq_ignore_ascii_case("pcm") {
            return Err("Vertex Gemini-TTS audio MIME declared a non-PCM codec".into());
        }
        if key.trim().eq_ignore_ascii_case("channels")
            && !value.eq_ignore_ascii_case("mono")
            && value != "1"
        {
            return Err("Vertex Gemini-TTS audio MIME declared non-mono audio".into());
        }
        if (key.trim().eq_ignore_ascii_case("bits") || key.trim().eq_ignore_ascii_case("bit-depth"))
            && value != "16"
        {
            return Err("Vertex Gemini-TTS audio MIME declared a non-16-bit sample format".into());
        }
    }
    Ok(())
}

fn stream_completion(native: &Value) -> (Option<String>, Option<String>, bool) {
    let candidates = native.get("candidates").and_then(Value::as_array);
    let finish_reason = candidates
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| {
            candidate
                .get("finishReason")
                .or_else(|| candidate.get("finish_reason"))
        })
        .and_then(Value::as_str)
        .filter(|reason| {
            !reason.is_empty() && !reason.eq_ignore_ascii_case("FINISH_REASON_UNSPECIFIED")
        })
        .map(str::to_owned);
    if finish_reason.is_some() {
        return (finish_reason, None, true);
    }

    let prompt_block_reason = if candidates.is_none_or(Vec::is_empty) {
        native
            .pointer("/promptFeedback/blockReason")
            .or_else(|| native.pointer("/prompt_feedback/block_reason"))
            .and_then(Value::as_str)
            .filter(|reason| {
                !reason.is_empty() && !reason.eq_ignore_ascii_case("BLOCKED_REASON_UNSPECIFIED")
            })
            .map(str::to_owned)
    } else {
        None
    };
    let terminal = prompt_block_reason.is_some();
    (None, prompt_block_reason, terminal)
}

fn provider_error(response: HttpResponse, request_id: Option<String>) -> VertexSpeechError {
    let code = serde_json::from_slice::<Value>(&response.body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/code")
                .or_else(|| value.get("code"))
                .map(|code| match code {
                    Value::String(value) => value.clone(),
                    other => other.to_string(),
                })
        });
    let message = serde_json::from_slice::<Value>(&response.body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
                .map(|message| message.chars().take(1_000).collect::<String>())
        })
        .unwrap_or_else(|| "Google Cloud rejected the speech request".into());
    let message = match code {
        Some(code) if !code.is_empty() => format!("{code}: {message}"),
        _ => message,
    };
    if (400..500).contains(&response.status) && response.status != 408 {
        VertexSpeechError::ProviderRejected {
            status: response.status,
            message,
            request_id,
        }
    } else {
        VertexSpeechError::OutcomeUnknown {
            source: LlmError::ProviderInternal {
                message: format!("Google Cloud returned HTTP {}: {message}", response.status),
            },
            request_id,
        }
    }
}

fn response_request_id(headers: &[(String, String)]) -> Option<String> {
    headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("x-goog-request-id")
                || name.eq_ignore_ascii_case("x-request-id")
        })
        .map(|(_, value)| value.clone())
}

fn validate_bearer_token(credential: &Secret<String>) -> Result<(), VertexSpeechError> {
    let token = credential.expose_secret();
    if token.trim().is_empty() || token.chars().any(char::is_control) {
        return Err(LlmError::Authentication {
            message: "Google Cloud bearer access token must be non-empty and contain no control characters".into(),
        }
        .into());
    }
    Ok(())
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_path_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value.as_bytes()[value.len() - 1].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn valid_location(value: &str) -> bool {
    matches!(
        value,
        "global"
            | "northamerica-northeast1"
            | "europe-central2"
            | "europe-north1"
            | "europe-southwest1"
            | "europe-west1"
            | "europe-west4"
            | "us-central1"
            | "us-east1"
            | "us-east4"
            | "us-east5"
            | "us-south1"
            | "us-west1"
            | "us-west4"
    )
}

fn valid_language(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 35
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn valid_voice_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn valid_speaker(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn has_disallowed_control(value: &str) -> bool {
    value
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
}

fn invalid(message: impl Into<String>) -> VertexSpeechError {
    VertexSpeechError::InvalidInput(message.into())
}
