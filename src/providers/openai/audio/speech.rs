//! OpenAI text-to-speech with raw binary or server-sent-event output streaming.
use super::*;
use crate::{
    framing::sse::SseFrameSplitter,
    transport::{HttpRequest, MAX_ERROR_BODY_SIZE},
};
use std::collections::VecDeque;

// `SseFrameSplitter` rejects an event after this same 8 MiB limit. Keep a
// local bound for Base64 before asking the decoder to allocate its output.
const MAX_SPEECH_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;
pub(super) const OPENAI_SPEECH_ENDPOINT: &str = "https://api.openai.com/v1/audio/speech";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechModel {
    Gpt4oMiniTts,
    Gpt4oMiniTts2025_12_15,
    Tts1,
    Tts1Hd,
}
impl SpeechModel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Gpt4oMiniTts => "gpt-4o-mini-tts",
            Self::Gpt4oMiniTts2025_12_15 => "gpt-4o-mini-tts-2025-12-15",
            Self::Tts1 => "tts-1",
            Self::Tts1Hd => "tts-1-hd",
        }
    }
    fn is_gpt(self) -> bool {
        matches!(self, Self::Gpt4oMiniTts | Self::Gpt4oMiniTts2025_12_15)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,
    Shimmer,
    Verse,
    Marin,
    Cedar,
    /// A custom voice reference scoped to an OpenAI profile and project.
    Custom(CustomVoiceRef),
}
impl SpeechVoice {
    fn as_str(&self) -> Option<&'static str> {
        match self {
            Self::Alloy => Some("alloy"),
            Self::Ash => Some("ash"),
            Self::Ballad => Some("ballad"),
            Self::Coral => Some("coral"),
            Self::Echo => Some("echo"),
            Self::Fable => Some("fable"),
            Self::Nova => Some("nova"),
            Self::Onyx => Some("onyx"),
            Self::Sage => Some("sage"),
            Self::Shimmer => Some("shimmer"),
            Self::Verse => Some("verse"),
            Self::Marin => Some("marin"),
            Self::Cedar => Some("cedar"),
            Self::Custom(_) => None,
        }
    }
    fn legacy_supported(&self) -> bool {
        matches!(
            self,
            Self::Alloy
                | Self::Ash
                | Self::Coral
                | Self::Echo
                | Self::Fable
                | Self::Nova
                | Self::Onyx
                | Self::Sage
                | Self::Shimmer
        )
    }

    fn wire_value(&self) -> Value {
        match self {
            Self::Custom(reference) => serde_json::json!({"id":reference.id()}),
            _ => Value::String(self.as_str().expect("builtin voice has name").into()),
        }
    }
}

pub(super) fn validate_voice_id(id: &str) -> Result<(), LlmError> {
    if id.trim().is_empty()
        || id != id.trim()
        || id.chars().any(|ch| ch.is_control())
        || !id.starts_with("voice_")
    {
        return Err(invalid("custom speech voice requires a valid voice_ ID"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}
impl SpeechFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub model: SpeechModel,
    pub input: String,
    pub voice: SpeechVoice,
    #[serde(default)]
    pub format: SpeechFormat,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub speed: Option<f64>,
}

/// Raw audio chunks. Dropping the stream closes the HTTP body.
pub struct SpeechStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    pub format: SpeechFormat,
    pub request_id: Option<String>,
    /// PCM output is signed 16-bit little-endian mono at 24 kHz.
    pub pcm_sample_rate_hz: Option<u32>,
    pub pcm_channels: Option<u8>,
    pub pcm_bits_per_sample: Option<u8>,
    delivered: u64,
    finished: bool,
}
#[derive(Debug, thiserror::Error)]
#[error("speech audio stream interrupted after {bytes_delivered} bytes: {source}")]
pub struct SpeechStreamError {
    pub bytes_delivered: u64,
    #[source]
    pub source: Box<LlmError>,
}
impl SpeechStream {
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, SpeechStreamError> {
        if self.finished {
            return Ok(None);
        }
        match self.body.next().await {
            Some(Ok(bytes)) => {
                self.delivered = self.delivered.saturating_add(bytes.len() as u64);
                Ok(Some(bytes))
            }
            Some(Err(source)) => {
                self.finished = true;
                Err(SpeechStreamError {
                    bytes_delivered: self.delivered,
                    source: Box::new(source),
                })
            }
            None => {
                self.finished = true;
                if self.delivered == 0 {
                    Err(SpeechStreamError {
                        bytes_delivered: 0,
                        source: Box::new(LlmError::ProviderInternal {
                            message: "speech provider returned no audio bytes".into(),
                        }),
                    })
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// One recognized or unrecognized event from the OpenAI Speech SSE stream.
/// `raw_data` retains the exact bytes after SSE `data:` framing for callers
/// that need fields this client does not interpret.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechEvent {
    /// Decoded audio from a documented `speech.audio.delta` event.
    AudioDelta {
        audio: Bytes,
        native: Value,
        raw_data: Bytes,
    },
    /// The documented terminal event. Usage stays native JSON after its
    /// required integer counters are validated.
    AudioDone {
        usage: Value,
        native: Value,
        raw_data: Bytes,
    },
    /// An event whose shape is not part of the documented success contract.
    /// This includes future event types and any undocumented in-band errors.
    Unknown {
        event_type: Option<String>,
        native: Option<Value>,
        raw_data: Bytes,
    },
}

impl SpeechEvent {
    /// Whether this event completes the speech stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::AudioDone { .. })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpeechEventStreamError {
    #[error("speech event stream interrupted: {source}")]
    Interrupted {
        #[source]
        source: LlmError,
    },
    #[error("invalid speech event: {message}")]
    InvalidEvent { message: String, native: Value },
}

/// A Speech API SSE response. The stream is fused after `speech.audio.done`;
/// dropping it early cancels the HTTP body and does not resubmit the request.
pub struct SpeechEventStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_error: Option<LlmError>,
    ended: bool,
    terminal: bool,
    pub request_id: Option<String>,
}

impl SpeechEventStream {
    pub async fn next_event(&mut self) -> Result<Option<SpeechEvent>, SpeechEventStreamError> {
        if self.terminal {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                match decode_speech_event(frame) {
                    Ok(event) => {
                        self.terminal = event.is_terminal();
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
                return Err(SpeechEventStreamError::Interrupted { source });
            }
            if self.ended {
                self.terminal = true;
                return Err(SpeechEventStreamError::Interrupted {
                    source: LlmError::StreamInterrupted {
                        message: "speech stream ended before speech.audio.done".into(),
                    },
                });
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
}

impl AudioService<'_> {
    /// Generate speech independently of Chat audio. Returns binary chunks as received.
    pub async fn synthesize(
        self,
        request: &SpeechRequest,
        options: &RequestOptions,
    ) -> Result<SpeechStream, AudioError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .synthesize(profile_name, request, options)
            .await
    }

    /// Generate speech as documented Server-Sent Events. `speech.audio.done`
    /// is required for a successful stream; undocumented events are preserved.
    pub async fn synthesize_stream(
        self,
        request: &SpeechRequest,
        options: &RequestOptions,
    ) -> Result<SpeechEventStream, AudioError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .synthesize_stream(profile_name, request, options)
            .await
    }
}
impl Pinned<'_> {
    async fn synthesize(
        &self,
        profile_name: &str,
        request: &SpeechRequest,
        options: &RequestOptions,
    ) -> Result<SpeechStream, AudioError> {
        let response = self
            .speech_response(profile_name, request, options, "audio", false)
            .await?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        Ok(SpeechStream {
            body: response.body,
            format: request.format,
            request_id,
            pcm_sample_rate_hz: (request.format == SpeechFormat::Pcm).then_some(24_000),
            pcm_channels: (request.format == SpeechFormat::Pcm).then_some(1),
            pcm_bits_per_sample: (request.format == SpeechFormat::Pcm).then_some(16),
            delivered: 0,
            finished: false,
        })
    }

    async fn synthesize_stream(
        &self,
        profile_name: &str,
        request: &SpeechRequest,
        options: &RequestOptions,
    ) -> Result<SpeechEventStream, AudioError> {
        if !request.model.is_gpt() {
            return Err(invalid("Speech SSE streaming requires a GPT TTS model").into());
        }
        let response = self
            .speech_response(profile_name, request, options, "sse", true)
            .await?;
        let content_type = response.header("content-type").unwrap_or_default();
        if !content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
        {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE)).await?;
            let native = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(AudioError::InvalidResponse {
                message: "Speech SSE request did not return text/event-stream".into(),
                native,
            });
        }
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        Ok(SpeechEventStream {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_error: None,
            ended: false,
            terminal: false,
            request_id,
        })
    }

    async fn speech_response(
        &self,
        profile_name: &str,
        request: &SpeechRequest,
        options: &RequestOptions,
        stream_format: &str,
        accept_sse: bool,
    ) -> Result<crate::transport::StreamResponse, AudioError> {
        validate_speech(request)?;
        let route = self.route(profile_name)?;
        let endpoint =
            route
                .speech_endpoint
                .as_deref()
                .ok_or_else(|| LlmError::UnsupportedCapability {
                    message: "profile has no speech route".into(),
                })?;
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown audio profile"))?;
        if let SpeechVoice::Custom(reference) = &request.voice {
            if profile.provider_id.as_str() != "openai" || endpoint != OPENAI_SPEECH_ENDPOINT {
                return Err(LlmError::UnsupportedCapability {
                    message: "custom speech voices require the official OpenAI Speech route".into(),
                }
                .into());
            }
            reference.validate_for(
                profile.provider_id.as_str(),
                profile_name,
                endpoint,
                options.account_scope.as_deref(),
            )?;
        }
        if accept_sse
            && (profile.provider_id.as_str() != "openai" || endpoint != OPENAI_SPEECH_ENDPOINT)
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Speech SSE streaming requires the official OpenAI Speech route".into(),
            }
            .into());
        }
        let secret = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "speech route requires a credential".into(),
            })?;
        let mut body = serde_json::json!({
            "model": request.model.as_str(),
            "input": request.input,
            "voice": request.voice.wire_value(),
            "response_format": request.format.as_str(),
            "stream_format": stream_format,
        });
        if let Some(instructions) = &request.instructions {
            body["instructions"] = Value::String(instructions.clone());
        }
        if let Some(speed) = request.speed {
            body["speed"] = Value::from(speed);
        }
        let bytes = serde_json::to_vec(&body)
            .map_err(|_| invalid("speech request cannot be serialized"))?;
        let deadline = Deadline::after(options.total_timeout);
        let mut headers = vec![
            (
                "authorization".into(),
                format!("Bearer {}", secret.expose_secret()),
            ),
            ("content-type".into(), "application/json".into()),
        ];
        if accept_sse {
            headers.push(("accept".into(), "text/event-stream".into()));
        }
        let http = HttpRequest {
            http1_header_layout: None,
            method: "POST".into(),
            url: endpoint.into(),
            headers,
            body: bytes.into(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send(http)
            .await
            .map_err(|source| match source {
                LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. } => AudioError::OutcomeUnknown { source },
                other => AudioError::Llm(other),
            })?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE)).await?;
            let request_id = response
                .header("x-request-id")
                .or_else(|| response.header("request-id"))
                .map(str::to_owned);
            let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(AudioError::Provider {
                status: response.status,
                request_id,
                body,
            });
        }
        Ok(response)
    }
}

fn decode_speech_event(raw_data: Vec<u8>) -> Result<SpeechEvent, SpeechEventStreamError> {
    let raw_data = Bytes::from(raw_data);
    let Ok(native) = serde_json::from_slice::<Value>(&raw_data) else {
        return Ok(SpeechEvent::Unknown {
            event_type: None,
            native: None,
            raw_data,
        });
    };
    let Some(event_type) = native.get("type").and_then(Value::as_str) else {
        return Ok(SpeechEvent::Unknown {
            event_type: None,
            native: Some(native),
            raw_data,
        });
    };
    match event_type {
        "speech.audio.delta" => {
            let audio = native.get("audio").and_then(Value::as_str).ok_or_else(|| {
                SpeechEventStreamError::InvalidEvent {
                    message: "speech.audio.delta is missing Base64 `audio`".into(),
                    native: native.clone(),
                }
            })?;
            if audio.len() > MAX_SPEECH_SSE_EVENT_BYTES {
                return Err(SpeechEventStreamError::InvalidEvent {
                    message: "speech.audio.delta Base64 exceeds the SSE event limit".into(),
                    native,
                });
            }
            let decoded =
                BASE64
                    .decode(audio)
                    .map_err(|_| SpeechEventStreamError::InvalidEvent {
                        message: "speech.audio.delta `audio` is not valid Base64".into(),
                        native: native.clone(),
                    })?;
            Ok(SpeechEvent::AudioDelta {
                audio: Bytes::from(decoded),
                native,
                raw_data,
            })
        }
        "speech.audio.done" => {
            let usage = native
                .get("usage")
                .filter(|usage| usage.is_object())
                .ok_or_else(|| SpeechEventStreamError::InvalidEvent {
                    message: "speech.audio.done is missing object `usage`".into(),
                    native: native.clone(),
                })?;
            for field in ["input_tokens", "output_tokens", "total_tokens"] {
                if !usage
                    .get(field)
                    .is_some_and(|value| value.is_i64() || value.is_u64())
                {
                    return Err(SpeechEventStreamError::InvalidEvent {
                        message: format!("speech.audio.done `usage.{field}` must be an integer"),
                        native: native.clone(),
                    });
                }
            }
            Ok(SpeechEvent::AudioDone {
                usage: usage.clone(),
                native,
                raw_data,
            })
        }
        _ => Ok(SpeechEvent::Unknown {
            event_type: Some(event_type.to_owned()),
            native: Some(native),
            raw_data,
        }),
    }
}
fn validate_speech(request: &SpeechRequest) -> Result<(), LlmError> {
    let length = request.input.chars().count();
    if !(1..=4096).contains(&length) {
        return Err(invalid("speech input must contain 1–4096 characters"));
    }
    if request
        .instructions
        .as_ref()
        .is_some_and(|instructions| instructions.trim().is_empty() || instructions.len() > 10_000)
    {
        return Err(invalid("speech instructions must contain 1–10000 bytes"));
    }
    if let SpeechVoice::Custom(reference) = &request.voice {
        validate_voice_id(reference.id())?;
    }
    if !request.model.is_gpt()
        && (request.instructions.is_some() || !request.voice.legacy_supported())
    {
        return Err(invalid(
            "tts-1 and tts-1-hd do not support instructions or this voice",
        ));
    }
    if request
        .speed
        .is_some_and(|speed| !speed.is_finite() || !(0.25..=4.0).contains(&speed))
    {
        return Err(invalid("speech speed must be between 0.25 and 4"));
    }
    Ok(())
}
