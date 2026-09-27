//! xAI's native realtime speech-to-text WebSocket endpoint.
//!
//! This is a transcription-only protocol at `/v1/stt`; it is intentionally
//! separate from xAI's speech-to-speech Realtime conversation adapter.

use crate::{
    protocol::Secret,
    realtime::{
        RealtimeAudioFormat, RealtimeClose, RealtimeCodec, RealtimeConnectRequest,
        RealtimeConnection, RealtimeControl, RealtimeError, RealtimeEvent, RealtimeEvents,
        RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeSink,
        RealtimeTransport,
    },
    runtime::Deadline,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{future::select, stream, Future, StreamExt};
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, VecDeque},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use thiserror::Error;
use url::Url;

/// First-party xAI realtime speech-to-text WebSocket route.
pub const XAI_STT_WEBSOCKET_ENDPOINT: &str = "wss://api.x.ai/v1/stt";

const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
const DEFAULT_ENDPOINTING_MS: u32 = 400;
const MAX_READY_PREFACE_EVENTS: usize = 16;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Documented realtime speech-to-text model slugs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum XaiSttModel {
    /// Original transcription model.
    GrokVoiceTranscribe1,
    /// Current default transcription model.
    #[default]
    GrokVoiceTranscribe2,
}

impl XaiSttModel {
    fn as_str(self) -> &'static str {
        match self {
            Self::GrokVoiceTranscribe1 => "grok-voice-transcribe-1.0",
            Self::GrokVoiceTranscribe2 => "grok-voice-transcribe-2.0",
        }
    }
}

/// Raw audio encoding declared on xAI's STT WebSocket query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum XaiSttEncoding {
    /// Raw PCM audio.
    #[default]
    Pcm,
    /// Raw G.711 mu-law audio.
    MuLaw,
    /// Raw G.711 A-law audio.
    ALaw,
    /// Raw Opus packets, exactly one packet per binary WebSocket frame.
    Opus,
}

impl XaiSttEncoding {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::MuLaw => "mulaw",
            Self::ALaw => "alaw",
            Self::Opus => "opus",
        }
    }

    fn realtime_format(self, sample_rate_hz: u32) -> RealtimeAudioFormat {
        match self {
            Self::Pcm => RealtimeAudioFormat::Pcm16 { sample_rate_hz },
            Self::MuLaw => RealtimeAudioFormat::G711MuLaw,
            Self::ALaw => RealtimeAudioFormat::G711ALaw,
            Self::Opus => RealtimeAudioFormat::Encoded {
                mime_type: "audio/opus".into(),
            },
        }
    }
}

/// Options encoded as query parameters on the official STT WebSocket URL.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiSttConfig {
    pub model: XaiSttModel,
    pub encoding: XaiSttEncoding,
    /// Omit to use the provider's default (16 kHz). Must be omitted for Opus.
    pub sample_rate_hz: Option<u32>,
    pub interim_results: bool,
    /// Silence endpointing threshold in milliseconds; 0–5000.
    pub endpointing_ms: u32,
    /// Formatting language. Supported formatting languages are listed in the
    /// xAI Speech to Text guide.
    pub language: Option<String>,
    /// Omit to leave diarization at the provider default.
    pub diarize: Option<bool>,
    pub filler_words: bool,
    pub multichannel: bool,
    /// 1–8 interleaved channels. Values above one require `multichannel`.
    pub channels: u8,
    /// Repeated `keyterm` query parameters (up to 100 terms, 50 chars each).
    pub keyterms: Vec<String>,
    /// Optional Smart Turn confidence threshold in the 0–1 range.
    pub smart_turn: Option<f64>,
    /// Optional silence timeout for Smart Turn, in the 1–5000 ms range.
    pub smart_turn_timeout_ms: Option<u32>,
    /// Optional voice activity threshold in the 0–1 range.
    pub vad_threshold: Option<f64>,
    /// One deadline for the WebSocket handshake and the server-ready event.
    pub connect_timeout: Duration,
}

impl Default for XaiSttConfig {
    fn default() -> Self {
        Self {
            model: XaiSttModel::GrokVoiceTranscribe2,
            encoding: XaiSttEncoding::Pcm,
            sample_rate_hz: None,
            interim_results: false,
            endpointing_ms: DEFAULT_ENDPOINTING_MS,
            language: None,
            diarize: None,
            filler_words: false,
            multichannel: false,
            channels: 1,
            keyterms: Vec::new(),
            smart_turn: None,
            smart_turn_timeout_ms: None,
            vad_threshold: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl XaiSttConfig {
    fn validate(&self) -> Result<(), RealtimeError> {
        if self.connect_timeout.is_zero() {
            return Err(invalid_config("connect_timeout must be nonzero"));
        }
        if self.encoding == XaiSttEncoding::Opus && self.sample_rate_hz.is_some() {
            return Err(invalid_config(
                "sample_rate_hz must be omitted when encoding is Opus",
            ));
        }
        if self
            .sample_rate_hz
            .is_some_and(|rate| !matches!(rate, 8_000 | 16_000 | 22_050 | 24_000 | 44_100 | 48_000))
        {
            return Err(invalid_config(
                "sample_rate_hz must be one of 8000, 16000, 22050, 24000, 44100, or 48000",
            ));
        }
        if self.endpointing_ms > 5000 {
            return Err(invalid_config("endpointing_ms must be between 0 and 5000"));
        }
        if !(1..=8).contains(&self.channels) {
            return Err(invalid_config("channels must be between 1 and 8"));
        }
        if self.multichannel && self.channels < 2 {
            return Err(invalid_config(
                "multichannel requires at least two channels",
            ));
        }
        if !self.multichannel && self.channels != 1 {
            return Err(invalid_config(
                "channels greater than one require multichannel=true",
            ));
        }
        if self.encoding == XaiSttEncoding::Opus && self.multichannel {
            return Err(invalid_config("Opus does not support multichannel mode"));
        }
        if self
            .language
            .as_deref()
            .is_some_and(|language| language.trim().is_empty() || has_control(language))
        {
            return Err(invalid_config(
                "language must be nonempty and contain no control characters",
            ));
        }
        if self.keyterms.len() > 100
            || self.keyterms.iter().any(|term| {
                term.trim().is_empty() || term.chars().count() > 50 || has_control(term)
            })
        {
            return Err(invalid_config(
                "keyterms must contain at most 100 nonempty terms of at most 50 characters",
            ));
        }
        if self
            .smart_turn
            .is_some_and(|threshold| !in_unit_interval(threshold))
        {
            return Err(invalid_config("smart_turn must be between 0 and 1"));
        }
        if self
            .smart_turn_timeout_ms
            .is_some_and(|timeout| !(1..=5000).contains(&timeout))
        {
            return Err(invalid_config(
                "smart_turn_timeout_ms must be between 1 and 5000",
            ));
        }
        if self.smart_turn_timeout_ms.is_some() && self.smart_turn.is_none() {
            return Err(invalid_config(
                "smart_turn_timeout_ms requires smart_turn to be enabled",
            ));
        }
        if self
            .vad_threshold
            .is_some_and(|threshold| !in_unit_interval(threshold))
        {
            return Err(invalid_config("vad_threshold must be between 0 and 1"));
        }
        Ok(())
    }

    fn endpoint(&self) -> Result<String, RealtimeError> {
        let mut endpoint = Url::parse(XAI_STT_WEBSOCKET_ENDPOINT)
            .map_err(|_| invalid_config("the built-in xAI STT WebSocket URL is invalid"))?;
        {
            let mut query = endpoint.query_pairs_mut();
            query.append_pair("model", self.model.as_str());
            query.append_pair("encoding", self.encoding.as_str());
            if let Some(sample_rate_hz) = self.sample_rate_hz {
                query.append_pair("sample_rate", &sample_rate_hz.to_string());
            }
            query.append_pair("interim_results", bool_text(self.interim_results));
            query.append_pair("endpointing", &self.endpointing_ms.to_string());
            if let Some(language) = self.language.as_deref() {
                query.append_pair("language", language);
            }
            if let Some(diarize) = self.diarize {
                query.append_pair("diarize", bool_text(diarize));
            }
            query.append_pair("filler_words", bool_text(self.filler_words));
            query.append_pair("multichannel", bool_text(self.multichannel));
            if self.multichannel {
                query.append_pair("channels", &self.channels.to_string());
            }
            for keyterm in &self.keyterms {
                query.append_pair("keyterm", keyterm);
            }
            if let Some(smart_turn) = self.smart_turn {
                query.append_pair("smart_turn", &smart_turn.to_string());
            }
            if let Some(timeout) = self.smart_turn_timeout_ms {
                query.append_pair("smart_turn_timeout", &timeout.to_string());
            }
            if let Some(vad_threshold) = self.vad_threshold {
                query.append_pair("vad_threshold", &vad_threshold.to_string());
            }
        }
        Ok(endpoint.into())
    }

    fn realtime_audio_format(&self) -> RealtimeAudioFormat {
        self.encoding
            .realtime_format(self.sample_rate_hz.unwrap_or(DEFAULT_SAMPLE_RATE_HZ))
    }
}

/// Connection or provider rejection while waiting for xAI's ready event.
#[derive(Debug, Error)]
pub enum XaiSttError {
    #[error(transparent)]
    Realtime(#[from] RealtimeError),
    #[error("xAI STT rejected the WebSocket session: {message}")]
    Provider { message: String, native: Value },
}

/// Events from xAI's streaming transcription endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum XaiSttEvent {
    /// The provider sent `transcript.created`; audio input is now accepted.
    Ready {
        native: Value,
    },
    TranscriptPartial {
        text: String,
        words: Value,
        is_final: bool,
        speech_final: bool,
        start: f64,
        duration: f64,
        channel_index: Option<u8>,
        end_of_turn_confidence: Option<f64>,
        native: Value,
    },
    TranscriptDone {
        text: Option<String>,
        words: Option<Value>,
        duration: f64,
        channel_index: Option<u8>,
        native: Value,
    },
    ProviderError {
        message: String,
        native: Value,
    },
    /// An unrecognized or malformed text event retained as provider data.
    ProviderEvent {
        name: String,
        native: Value,
    },
    /// Underlying WebSocket transport failed before xAI's normal final close.
    ConnectionInterrupted {
        message: String,
    },
    /// xAI closed normally after every expected `transcript.done` event.
    Completed {
        expected_channels: u8,
    },
    /// A locally requested WebSocket close. It does not imply a final transcript.
    LocalClose {
        code: u16,
        reason: String,
    },
}

/// Connected xAI transcription session. The connect call waits for
/// `transcript.created`, so returned controls may immediately send audio.
pub struct XaiSttSession {
    control: XaiSttControl,
    events: XaiSttEvents,
}

impl XaiSttSession {
    /// Connect using one request-scoped credential and wait for the provider's
    /// ready event. `connect_timeout` bounds both WebSocket setup and readiness.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        credential: Secret<String>,
        config: XaiSttConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, XaiSttDriver), XaiSttError> {
        config.validate()?;
        if credential.expose_secret().trim().is_empty() {
            return Err(invalid_config("xAI STT requires a nonempty API key").into());
        }
        let endpoint = config.endpoint()?;
        let deadline = Deadline::after(Some(config.connect_timeout));
        let request = RealtimeConnectRequest {
            endpoint,
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let terminal = Arc::new(AtomicBool::new(false));
        let audio_done_sent = Arc::new(AtomicBool::new(false));
        let remote_eof = Arc::new(AtomicBool::new(false));
        let done_channels = Arc::new(Mutex::new(BTreeSet::new()));
        let codec = Arc::new(XaiSttCodec {
            config: config.clone(),
            terminal: terminal.clone(),
            done_channels: done_channels.clone(),
            audio_done_sent: audio_done_sent.clone(),
        });
        let tracking_transport = XaiSttTrackingTransport {
            inner: transport,
            audio_done_sent: audio_done_sent.clone(),
            remote_eof: remote_eof.clone(),
        };
        let connected = deadline
            .run(RealtimeSession::connect(
                &tracking_transport,
                request,
                codec,
                limits,
            ))
            .await
            .map_err(|_| connect_timeout_error())??;
        let (session, driver) = connected;
        let (control, events) = session.into_parts();
        let driver_future: DriverFuture = Box::pin(driver.run());
        let (driver_future, prefetched) = deadline
            .run(wait_for_ready(events, driver_future, config.channels))
            .await
            .map_err(|_| connect_timeout_error())??;

        let state = Arc::new(Mutex::new(InputState::Open));
        let control = XaiSttControl {
            inner: control,
            ready: Arc::new(AtomicBool::new(true)),
            state,
            audio_format: config.realtime_audio_format(),
        };
        let (inner_events, pending) = prefetched;
        let events = XaiSttEvents {
            inner: inner_events,
            pending,
            terminal: terminal.clone(),
            audio_done_sent: audio_done_sent.clone(),
            remote_eof: remote_eof.clone(),
            channels: config.channels,
            delivered_done_channels: BTreeSet::new(),
        };
        Ok((
            Self { control, events },
            XaiSttDriver {
                inner: driver_future,
                terminal,
                audio_done_sent,
                remote_eof,
            },
        ))
    }

    pub fn into_parts(self) -> (XaiSttControl, XaiSttEvents) {
        (self.control, self.events)
    }
}

type DriverFuture = Pin<Box<dyn Future<Output = Result<(), RealtimeError>> + Send + 'static>>;

async fn wait_for_ready(
    mut events: RealtimeEvents,
    mut driver: DriverFuture,
    channels: u8,
) -> Result<(DriverFuture, (RealtimeEvents, VecDeque<XaiSttEvent>)), XaiSttError> {
    let mut pending = VecDeque::new();
    let audio_done_sent = AtomicBool::new(false);
    let remote_eof = AtomicBool::new(false);
    let terminal = AtomicBool::new(false);
    let mut delivered_done_channels = BTreeSet::new();
    for _ in 0..MAX_READY_PREFACE_EVENTS {
        let (event, pending_driver) = {
            match select(Box::pin(events.next()), driver).await {
                futures::future::Either::Left((event, pending_driver)) => (event, pending_driver),
                futures::future::Either::Right((driver_result, pending_events)) => {
                    drop(pending_events);
                    return Err(driver_result
                        .err()
                        .unwrap_or(RealtimeError::UnexpectedRemoteClose)
                        .into());
                }
            }
        };
        driver = pending_driver;
        let Some(event) = event else {
            return Err(RealtimeError::UnexpectedRemoteClose.into());
        };
        match event {
            RealtimeEvent::ProviderEvent { name, native } if name == "transcript.created" => {
                pending.push_back(XaiSttEvent::Ready { native });
                return Ok((driver, (events, pending)));
            }
            RealtimeEvent::ProviderEvent { name, native }
                if name == "error" && native.get("message").and_then(Value::as_str).is_some() =>
            {
                let message = native["message"].as_str().unwrap().to_owned();
                return Err(XaiSttError::Provider { message, native });
            }
            RealtimeEvent::ConnectionInterrupted { message } => {
                return Err(RealtimeError::Transport { message }.into());
            }
            RealtimeEvent::Closed { code, reason } => {
                return Err(RealtimeError::Transport {
                    message: format!("xAI STT closed before transcript.created ({code}: {reason})"),
                }
                .into());
            }
            event => {
                pending.push_back(convert_event(
                    event,
                    channels,
                    &audio_done_sent,
                    &remote_eof,
                    &terminal,
                    &mut delivered_done_channels,
                ));
            }
        }
    }
    Err(RealtimeError::Transport {
        message: "xAI STT did not send transcript.created within the ready-event bound".into(),
    }
    .into())
}

struct XaiSttTrackingTransport {
    inner: Arc<dyn RealtimeTransport>,
    audio_done_sent: Arc<AtomicBool>,
    remote_eof: Arc<AtomicBool>,
}

#[async_trait]
impl RealtimeTransport for XaiSttTrackingTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let connection = self.inner.connect(request).await?;
        let inbound_eof = self.remote_eof.clone();
        let inbound = stream::unfold(connection.inbound, move |mut inbound| {
            let inbound_eof = inbound_eof.clone();
            async move {
                match inbound.next().await {
                    Some(frame) => Some((frame, inbound)),
                    None => {
                        inbound_eof.store(true, Ordering::Release);
                        None
                    }
                }
            }
        })
        .boxed();
        Ok(RealtimeConnection {
            outbound: Box::new(XaiSttTrackingSink {
                inner: connection.outbound,
                audio_done_sent: self.audio_done_sent.clone(),
            }),
            inbound,
        })
    }
}

struct XaiSttTrackingSink {
    inner: Box<dyn RealtimeSink>,
    audio_done_sent: Arc<AtomicBool>,
}

#[async_trait]
impl RealtimeSink for XaiSttTrackingSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let is_audio_done = match &frame {
            RealtimeFrame::Text(bytes) => {
                serde_json::from_slice::<Value>(bytes)
                    .ok()
                    .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
                    .as_deref()
                    == Some("audio.done")
            }
            _ => false,
        };
        self.inner.send(frame).await?;
        if is_audio_done {
            self.audio_done_sent.store(true, Ordering::Release);
        }
        Ok(())
    }

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.close(close).await
    }
}

/// Typed bounded control handle for one STT session.
#[derive(Clone)]
pub struct XaiSttControl {
    inner: RealtimeControl,
    ready: Arc<AtomicBool>,
    state: Arc<Mutex<InputState>>,
    audio_format: RealtimeAudioFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputState {
    Open,
    AudioDoneQueued,
    LocallyClosed,
}

impl XaiSttControl {
    /// Queue one raw audio chunk as one binary WebSocket frame.
    pub fn send_audio(&self, data: impl Into<Bytes>) -> Result<(), RealtimeError> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(invalid_input(
                "wait for transcript.created before sending audio",
            ));
        }
        let data = data.into();
        if data.is_empty() {
            return Err(invalid_input("audio frame must not be empty"));
        }
        let state = self.state.lock().unwrap();
        if *state != InputState::Open {
            return Err(RealtimeError::Closed);
        }
        self.inner.send(RealtimeInput::Audio {
            data,
            format: self.audio_format.clone(),
        })
    }

    /// Finalize the current utterance while keeping the WebSocket open.
    pub fn finalize_utterance(&self) -> Result<(), RealtimeError> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(invalid_input(
                "wait for transcript.created before finalizing audio",
            ));
        }
        let state = self.state.lock().unwrap();
        if *state != InputState::Open {
            return Err(RealtimeError::Closed);
        }
        self.inner.send(RealtimeInput::CommitAudio)
    }

    /// Signal that no more audio will be sent. The caller should keep reading
    /// events until every `transcript.done` has arrived and the server closes.
    pub fn finish_audio(&self) -> Result<(), RealtimeError> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(invalid_input(
                "wait for transcript.created before finishing audio",
            ));
        }
        let mut state = self.state.lock().unwrap();
        if *state != InputState::Open {
            return Err(RealtimeError::Closed);
        }
        self.inner.send(RealtimeInput::Interrupt)?;
        *state = InputState::AudioDoneQueued;
        Ok(())
    }

    /// Close the WebSocket locally. This aborts any unfinished transcription;
    /// it does not send `audio.done` or guarantee `transcript.done`.
    pub async fn close(&self) -> Result<(), RealtimeError> {
        {
            let mut state = self.state.lock().unwrap();
            if *state == InputState::LocallyClosed {
                return Err(RealtimeError::Closed);
            }
            *state = InputState::LocallyClosed;
        }
        self.inner
            .close(crate::realtime::RealtimeClose::normal(
                "xAI STT closed locally",
            ))
            .await
    }
}

/// Receiver for typed xAI STT events; unknown events retain their raw JSON.
pub struct XaiSttEvents {
    inner: RealtimeEvents,
    pending: VecDeque<XaiSttEvent>,
    terminal: Arc<AtomicBool>,
    audio_done_sent: Arc<AtomicBool>,
    remote_eof: Arc<AtomicBool>,
    channels: u8,
    delivered_done_channels: BTreeSet<u8>,
}

impl XaiSttEvents {
    pub async fn next(&mut self) -> Option<XaiSttEvent> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        let event = self.inner.next().await?;
        Some(convert_event(
            event,
            self.channels,
            &self.audio_done_sent,
            &self.remote_eof,
            &self.terminal,
            &mut self.delivered_done_channels,
        ))
    }
}

/// Driver wrapper that recognizes only xAI's documented final transcript plus
/// subsequent remote EOF as a successful session end.
pub struct XaiSttDriver {
    inner: DriverFuture,
    terminal: Arc<AtomicBool>,
    audio_done_sent: Arc<AtomicBool>,
    remote_eof: Arc<AtomicBool>,
}

impl XaiSttDriver {
    pub async fn run(self) -> Result<(), RealtimeError> {
        match self.inner.await {
            Err(RealtimeError::UnexpectedRemoteClose)
                if self.terminal.load(Ordering::Acquire)
                    && self.audio_done_sent.load(Ordering::Acquire)
                    && self.remote_eof.load(Ordering::Acquire) =>
            {
                Ok(())
            }
            result => result,
        }
    }
}

struct XaiSttCodec {
    config: XaiSttConfig,
    terminal: Arc<AtomicBool>,
    done_channels: Arc<Mutex<BTreeSet<u8>>>,
    audio_done_sent: Arc<AtomicBool>,
}

impl RealtimeCodec for XaiSttCodec {
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let frame = match input {
            RealtimeInput::Audio { data, format } => {
                if data.is_empty() {
                    return Err(invalid_input("audio frame must not be empty"));
                }
                if *format != self.config.realtime_audio_format() {
                    return Err(invalid_input(
                        "audio format does not match the xAI STT session query",
                    ));
                }
                RealtimeFrame::binary(data.clone())
            }
            RealtimeInput::CommitAudio => {
                RealtimeFrame::text(json!({"type":"finalize"}).to_string())
            }
            RealtimeInput::Interrupt => {
                RealtimeFrame::text(json!({"type":"audio.done"}).to_string())
            }
            RealtimeInput::RetrieveItem { .. }
            | RealtimeInput::DeleteItem { .. }
            | RealtimeInput::TruncateAudio { .. } => {
                return Err(invalid_input(
                    "xAI STT does not support conversation item retrieval, deletion, or truncation",
                ));
            }
            _ => {
                return Err(invalid_input(
                    "xAI STT accepts audio, utterance finalization, and audio.done only",
                ));
            }
        };
        Ok(vec![frame])
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let value = match frame {
            RealtimeFrame::Text(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(value) => value,
                Err(_) => match String::from_utf8(bytes.to_vec()) {
                    Ok(text) => Value::String(text),
                    Err(error) => json!({
                        "raw_text_base64": BASE64.encode(error.into_bytes()),
                    }),
                },
            },
            RealtimeFrame::Binary(bytes) => json!({
                "frame": "binary",
                "data_base64": BASE64.encode(bytes),
            }),
        };
        let event_type = value.get("type").and_then(Value::as_str);
        let mut event_name = event_type.map(str::to_owned);
        let done_event = if event_type == Some("transcript.done")
            && self.audio_done_sent.load(Ordering::Acquire)
        {
            parse_done(&value, self.config.channels)
        } else {
            None
        };
        if let Some(done_event) = done_event {
            let channel = done_event.channel_index.unwrap_or(0);
            let mut done = self.done_channels.lock().unwrap();
            if done.insert(channel) {
                event_name = Some("__xai_stt_valid_transcript_done".into());
                if done.len() == usize::from(self.config.channels) {
                    self.terminal.store(true, Ordering::Release);
                }
            }
        }
        let name = event_name.unwrap_or_else(|| {
            if value.is_string() {
                "raw_text".into()
            } else if value.get("frame").and_then(Value::as_str) == Some("binary") {
                "raw_binary".into()
            } else if value.get("raw_text_base64").is_some() {
                "raw_text_base64".into()
            } else {
                "unknown".into()
            }
        });
        Ok(vec![RealtimeEvent::ProviderEvent {
            name,
            native: value,
        }])
    }
}

fn convert_event(
    event: RealtimeEvent,
    channels: u8,
    audio_done_sent: &AtomicBool,
    remote_eof: &AtomicBool,
    terminal: &AtomicBool,
    delivered_done_channels: &mut BTreeSet<u8>,
) -> XaiSttEvent {
    match event {
        RealtimeEvent::ProviderEvent { name, native } => match name.as_str() {
            "transcript.created" => XaiSttEvent::Ready { native },
            "transcript.partial" => match parse_partial(&native, channels) {
                Some(partial) => XaiSttEvent::TranscriptPartial {
                    text: partial.text,
                    words: partial.words,
                    is_final: partial.is_final,
                    speech_final: partial.speech_final,
                    start: partial.start,
                    duration: partial.duration,
                    channel_index: partial.channel_index,
                    end_of_turn_confidence: partial.end_of_turn_confidence,
                    native,
                },
                None => XaiSttEvent::ProviderEvent { name, native },
            },
            "__xai_stt_valid_transcript_done"
                if native.get("type").and_then(Value::as_str) == Some("transcript.done") =>
            {
                if audio_done_sent.load(Ordering::Acquire) {
                    if let Some(done_event) = parse_done(&native, channels) {
                        let channel = done_event.channel_index.unwrap_or(0);
                        if delivered_done_channels.insert(channel) {
                            return XaiSttEvent::TranscriptDone {
                                text: done_event.text,
                                words: done_event.words,
                                duration: done_event.duration,
                                channel_index: done_event.channel_index,
                                native,
                            };
                        }
                    }
                }
                XaiSttEvent::ProviderEvent {
                    name: "transcript.done".into(),
                    native,
                }
            }
            "__xai_stt_valid_transcript_done" => XaiSttEvent::ProviderEvent {
                name: native
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("__xai_stt_valid_transcript_done")
                    .to_owned(),
                native,
            },
            "transcript.done" => XaiSttEvent::ProviderEvent { name, native },
            "error" => match native.get("message").and_then(Value::as_str) {
                Some(message) => XaiSttEvent::ProviderError {
                    message: message.to_owned(),
                    native,
                },
                None => XaiSttEvent::ProviderEvent { name, native },
            },
            _ => XaiSttEvent::ProviderEvent { name, native },
        },
        RealtimeEvent::ConnectionInterrupted { message: _ }
            if remote_eof.load(Ordering::Acquire)
                && audio_done_sent.load(Ordering::Acquire)
                && terminal.load(Ordering::Acquire) =>
        {
            XaiSttEvent::Completed {
                expected_channels: channels,
            }
        }
        RealtimeEvent::ConnectionInterrupted { message } => {
            XaiSttEvent::ConnectionInterrupted { message }
        }
        RealtimeEvent::Closed { code, reason } => XaiSttEvent::LocalClose { code, reason },
        RealtimeEvent::ProviderError { message, .. } => XaiSttEvent::ProviderError {
            message,
            native: Value::Null,
        },
        other => XaiSttEvent::ProviderEvent {
            name: "realtime_event".into(),
            native: json!({"event": format!("{other:?}")}),
        },
    }
}

struct ParsedPartial {
    text: String,
    words: Value,
    is_final: bool,
    speech_final: bool,
    start: f64,
    duration: f64,
    channel_index: Option<u8>,
    end_of_turn_confidence: Option<f64>,
}

struct ParsedDone {
    text: Option<String>,
    words: Option<Value>,
    duration: f64,
    channel_index: Option<u8>,
}

fn parse_partial(value: &Value, channels: u8) -> Option<ParsedPartial> {
    let text = value.get("text")?.as_str()?.to_owned();
    let words = value.get("words")?;
    if !words.is_array() {
        return None;
    }
    let is_final = value.get("is_final")?.as_bool()?;
    let speech_final = value.get("speech_final")?.as_bool()?;
    let start = finite_nonnegative(value.get("start")?)?;
    let duration = finite_nonnegative(value.get("duration")?)?;
    let channel_index = parse_channel_index(value, channels)?;
    let end_of_turn_confidence = match value.get("end_of_turn_confidence") {
        Some(confidence) => {
            let confidence = confidence.as_f64()?;
            if !in_unit_interval(confidence) {
                return None;
            }
            Some(confidence)
        }
        None => None,
    };
    Some(ParsedPartial {
        text,
        words: words.clone(),
        is_final,
        speech_final,
        start,
        duration,
        channel_index,
        end_of_turn_confidence,
    })
}

fn parse_done(value: &Value, channels: u8) -> Option<ParsedDone> {
    let duration = finite_nonnegative(value.get("duration")?)?;
    let text = match value.get("text") {
        Some(text) => Some(text.as_str()?.to_owned()),
        None => None,
    };
    let words = match value.get("words") {
        Some(words) if words.is_array() => Some(words.clone()),
        Some(_) => return None,
        None => None,
    };
    let channel_index = parse_channel_index(value, channels)?;
    Some(ParsedDone {
        text,
        words,
        duration,
        channel_index,
    })
}

fn parse_channel_index(value: &Value, channels: u8) -> Option<Option<u8>> {
    let Some(index) = value.get("channel_index") else {
        return if channels == 1 { Some(None) } else { None };
    };
    let index = index.as_u64()?;
    if index >= u64::from(channels) {
        return None;
    }
    Some(Some(index as u8))
}

fn finite_nonnegative(value: &Value) -> Option<f64> {
    let number = value.as_f64()?;
    (number.is_finite() && number >= 0.0).then_some(number)
}

fn in_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}

fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn invalid_config(message: &str) -> RealtimeError {
    RealtimeError::InvalidConfig {
        message: message.into(),
    }
}

fn invalid_input(message: &str) -> RealtimeError {
    RealtimeError::InvalidInput {
        message: message.into(),
    }
}

fn connect_timeout_error() -> XaiSttError {
    RealtimeError::Transport {
        message: "xAI STT connect or transcript.created timed out".into(),
    }
    .into()
}
