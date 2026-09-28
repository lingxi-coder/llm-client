//! xAI's bidirectional streaming text-to-speech WebSocket protocol.
//!
//! This module is separate from the HTTP TTS service in [`crate::providers::xai::audio`]
//! and from xAI's speech-to-speech conversation adapter.

use crate::{
    protocol::Secret,
    providers::xai::audio::{validate_speech_replacements, XaiSpeechCodec, XaiSpeechFormat},
    realtime::{
        validate_frame, validate_limits, RealtimeClose, RealtimeConnectRequest, RealtimeConnection,
        RealtimeError, RealtimeFrame, RealtimeLimits, RealtimeSink, RealtimeTransport,
    },
    runtime::Deadline,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    future::{select, Either},
    FutureExt, SinkExt, StreamExt,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use thiserror::Error;
use url::Url;

/// First-party xAI streaming TTS WebSocket endpoint.
pub const XAI_STREAMING_TTS_WEBSOCKET_ENDPOINT: &str = "wss://api.x.ai/v1/tts";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_TEXT_DELTA_CHARS: usize = 60_000;

/// Query settings for xAI's streaming `/v1/tts` WebSocket route.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiStreamingTtsConfig {
    /// Required BCP-47 language code or `auto`.
    pub language: String,
    /// Built-in or custom voice ID. Omission uses the provider default `eve`.
    pub voice: Option<String>,
    /// Output codec and audio rates. The HTTP and streaming routes share these
    /// documented format fields while using distinct request protocols.
    pub output_format: XaiSpeechFormat,
    /// Speech speed multiplier from 0.7 through 1.5.
    pub speed: f64,
    /// WSS guide levels: 0 (quality), 1 (moderate), 2 (aggressive).
    pub optimize_streaming_latency: u8,
    pub text_normalization: bool,
    pub with_timestamps: bool,
    /// Deadline for the WebSocket handshake. The server sends no ready event.
    pub connect_timeout: Duration,
}

impl XaiStreamingTtsConfig {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            voice: None,
            output_format: XaiSpeechFormat::default(),
            speed: 1.0,
            optimize_streaming_latency: 0,
            text_normalization: false,
            with_timestamps: false,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        if self.language.trim().is_empty() || has_control(&self.language) {
            return Err(invalid_config(
                "language must be nonempty and contain no control characters",
            ));
        }
        if self
            .voice
            .as_deref()
            .is_some_and(|voice| voice.trim().is_empty() || has_control(voice))
        {
            return Err(invalid_config(
                "voice must be nonempty and contain no control characters",
            ));
        }
        if !supported_sample_rate(self.output_format.sample_rate) {
            return Err(invalid_config(
                "sample_rate must be one of 8000, 16000, 22050, 24000, 44100, or 48000",
            ));
        }
        if let Some(bit_rate) = self.output_format.bit_rate {
            if self.output_format.codec != XaiSpeechCodec::Mp3
                || !matches!(bit_rate, 32_000 | 64_000 | 96_000 | 128_000 | 192_000)
            {
                return Err(invalid_config(
                    "bit_rate is available for MP3 at 32000, 64000, 96000, 128000, or 192000 bps",
                ));
            }
        }
        if !self.speed.is_finite() || !(0.7..=1.5).contains(&self.speed) {
            return Err(invalid_config("speed must be between 0.7 and 1.5"));
        }
        if self.optimize_streaming_latency > 2 {
            return Err(invalid_config(
                "optimize_streaming_latency must be 0, 1, or 2",
            ));
        }
        if self.connect_timeout.is_zero() {
            return Err(invalid_config("connect_timeout must be nonzero"));
        }
        Ok(())
    }

    fn endpoint(&self) -> Result<String, RealtimeError> {
        let mut endpoint = Url::parse(XAI_STREAMING_TTS_WEBSOCKET_ENDPOINT)
            .map_err(|_| invalid_config("the built-in xAI streaming TTS URL is invalid"))?;
        {
            let mut query = endpoint.query_pairs_mut();
            query.append_pair("language", &self.language);
            if let Some(voice) = &self.voice {
                query.append_pair("voice", voice);
            }
            query.append_pair("codec", speech_codec_name(self.output_format.codec));
            query.append_pair("sample_rate", &self.output_format.sample_rate.to_string());
            if let Some(bit_rate) = self.output_format.bit_rate {
                query.append_pair("bit_rate", &bit_rate.to_string());
            }
            query.append_pair("speed", &self.speed.to_string());
            query.append_pair(
                "optimize_streaming_latency",
                &self.optimize_streaming_latency.to_string(),
            );
            query.append_pair("text_normalization", bool_text(self.text_normalization));
            query.append_pair("with_timestamps", bool_text(self.with_timestamps));
        }
        Ok(endpoint.into())
    }
}

/// A streaming TTS session. The host runs the returned driver while sending
/// text and consuming audio events.
pub struct XaiStreamingTtsSession {
    control: XaiStreamingTtsControl,
    events: XaiStreamingTtsEvents,
}

impl XaiStreamingTtsSession {
    /// Establish a request-scoped WebSocket connection. Upgrade success is the
    /// readiness signal; this protocol has no server-created event.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        credential: Secret<String>,
        config: XaiStreamingTtsConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, XaiStreamingTtsDriver), XaiStreamingTtsError> {
        config.validate()?;
        if credential.expose_secret().trim().is_empty()
            || credential.expose_secret().contains(['\r', '\n'])
        {
            return Err(invalid_config("xAI API key is empty or malformed").into());
        }
        validate_limits(limits)?;
        let endpoint = config.endpoint()?;
        let request = RealtimeConnectRequest {
            endpoint,
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let deadline = Deadline::after(Some(config.connect_timeout));
        let connection = deadline
            .run(transport.connect(request))
            .await
            .map_err(|_| connect_timeout_error())??;

        // futures mpsc gives a single sender one private slot; subtract it to
        // preserve RealtimeLimits' exact caller-visible queue capacity.
        let (outbound, outbound_receiver) = mpsc::channel(limits.outbound_capacity - 1);
        let (events_sender, events_receiver) = mpsc::channel(limits.event_capacity - 1);
        let sent_text_done = Arc::new(AtomicU64::new(0));
        let sent_text_clear = Arc::new(AtomicU64::new(0));
        let turn = Arc::new(Mutex::new(TtsTurnState::Idle));
        let outbound = Arc::new(Mutex::new(outbound));
        let codec = XaiStreamingTtsCodec {
            turn: turn.clone(),
            sent_text_done: sent_text_done.clone(),
            sent_text_clear: sent_text_clear.clone(),
            received_text_done: 0,
            received_text_clear: 0,
        };
        let connection = RealtimeConnection {
            outbound: Box::new(TrackedTtsSink {
                inner: connection.outbound,
                sent_text_done,
                sent_text_clear,
            }),
            inbound: connection.inbound,
        };
        let control = XaiStreamingTtsControl {
            turn,
            outbound,
            max_frame_bytes: limits.max_frame_bytes,
        };
        Ok((
            Self {
                control,
                events: XaiStreamingTtsEvents {
                    receiver: events_receiver,
                },
            },
            XaiStreamingTtsDriver {
                outbound: outbound_receiver,
                events: events_sender,
                codec,
                connection,
                max_frame_bytes: limits.max_frame_bytes,
                prefer_inbound: true,
            },
        ))
    }

    pub fn into_parts(self) -> (XaiStreamingTtsControl, XaiStreamingTtsEvents) {
        (self.control, self.events)
    }
}

/// Errors while validating or opening an xAI streaming TTS session.
#[derive(Debug, Error)]
pub enum XaiStreamingTtsError {
    #[error(transparent)]
    Realtime(#[from] RealtimeError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TtsTurnState {
    Idle,
    TextStreaming,
    AwaitingAudio,
    Clearing,
    LocallyClosed,
}

enum TtsCommand {
    Frame(RealtimeFrame),
    Close(RealtimeClose, oneshot::Sender<Result<(), RealtimeError>>),
}

enum DriverStep {
    Outbound(Option<TtsCommand>),
    Inbound(Option<Result<RealtimeFrame, RealtimeError>>),
}

/// Cloneable bounded sender for one streaming TTS connection.
#[derive(Clone)]
pub struct XaiStreamingTtsControl {
    turn: Arc<Mutex<TtsTurnState>>,
    outbound: Arc<Mutex<mpsc::Sender<TtsCommand>>>,
    max_frame_bytes: usize,
}

impl XaiStreamingTtsControl {
    /// Queue one text chunk. A chunk may contain at most 60,000 characters.
    pub fn send_text_delta(&self, delta: impl Into<String>) -> Result<(), RealtimeError> {
        let delta = delta.into();
        if delta.is_empty() {
            return Err(invalid_input("text.delta must not be empty"));
        }
        if delta.chars().count() > MAX_TEXT_DELTA_CHARS {
            return Err(invalid_input(
                "text.delta exceeds xAI's 60,000-character per-message limit",
            ));
        }
        let frame = RealtimeFrame::text(json!({"type":"text.delta","delta":delta}).to_string());
        self.enqueue_stateful_frame(
            frame,
            &[TtsTurnState::Idle, TtsTurnState::TextStreaming],
            TtsTurnState::TextStreaming,
        )
    }

    /// End the current text utterance and wait for its `audio.done` event.
    pub fn finish_utterance(&self) -> Result<(), RealtimeError> {
        let frame = RealtimeFrame::text(json!({"type":"text.done"}).to_string());
        self.enqueue_stateful_frame(
            frame,
            &[TtsTurnState::TextStreaming],
            TtsTurnState::AwaitingAudio,
        )
    }

    /// Cancel the current utterance. New text is accepted after `audio.clear`.
    /// xAI also permits cancelling while idle, in which case it acknowledges
    /// with `audio.clear` immediately.
    pub fn clear_utterance(&self) -> Result<(), RealtimeError> {
        let frame = RealtimeFrame::text(json!({"type":"text.clear"}).to_string());
        self.enqueue_stateful_frame(
            frame,
            &[
                TtsTurnState::Idle,
                TtsTurnState::TextStreaming,
                TtsTurnState::AwaitingAudio,
            ],
            TtsTurnState::Clearing,
        )
    }

    /// Update the provider's phrase replacement map. The service applies a
    /// change to the next utterance that begins; it does not rewrite client
    /// text. The active map changes only when `session.updated` arrives.
    pub fn update_replacements(
        &self,
        replacements: &BTreeMap<String, String>,
    ) -> Result<(), RealtimeError> {
        validate_speech_replacements(replacements).map_err(invalid_input)?;
        let frame = RealtimeFrame::text(
            json!({"type":"session.update","replace":replacements}).to_string(),
        );
        let turn = self.turn.lock().unwrap();
        if *turn == TtsTurnState::LocallyClosed {
            return Err(RealtimeError::Closed);
        }
        let mut outbound = self.outbound.lock().unwrap();
        enqueue_frame(&mut outbound, frame, self.max_frame_bytes)
    }

    /// Close the WebSocket locally. This does not imply a completed utterance.
    pub async fn close(&self) -> Result<(), RealtimeError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        {
            let mut turn = self.turn.lock().unwrap();
            if *turn == TtsTurnState::LocallyClosed {
                return Err(RealtimeError::Closed);
            }
            let close = RealtimeClose::normal("xAI streaming TTS closed locally");
            let mut outbound = self.outbound.lock().unwrap();
            enqueue_command(&mut outbound, TtsCommand::Close(close, ack_tx))?;
            *turn = TtsTurnState::LocallyClosed;
        }
        ack_rx.await.map_err(|_| RealtimeError::Closed)?
    }

    fn enqueue_stateful_frame(
        &self,
        frame: RealtimeFrame,
        allowed_states: &[TtsTurnState],
        next_state: TtsTurnState,
    ) -> Result<(), RealtimeError> {
        let mut turn = self.turn.lock().unwrap();
        if !allowed_states.contains(&*turn) {
            return if *turn == TtsTurnState::LocallyClosed {
                Err(RealtimeError::Closed)
            } else {
                Err(invalid_input(
                    "current TTS utterance state rejects this command",
                ))
            };
        }
        let mut outbound = self.outbound.lock().unwrap();
        enqueue_frame(&mut outbound, frame, self.max_frame_bytes)?;
        *turn = next_state;
        Ok(())
    }
}

/// Typed receiver for xAI streaming TTS events. Audio is returned as raw codec
/// bytes; the host owns playback and clears stale buffered audio on `AudioClear`.
pub struct XaiStreamingTtsEvents {
    receiver: mpsc::Receiver<XaiStreamingTtsEvent>,
}

impl XaiStreamingTtsEvents {
    pub async fn next(&mut self) -> Option<XaiStreamingTtsEvent> {
        self.receiver.next().await
    }
}

/// Events emitted by the streaming TTS protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum XaiStreamingTtsEvent {
    AudioDelta {
        audio: Bytes,
        native: Value,
    },
    AudioDone {
        trace_id: String,
        native: Value,
    },
    AudioClear {
        native: Value,
    },
    SessionUpdated {
        replace: Value,
        native: Value,
    },
    ProviderError {
        message: String,
        native: Value,
    },
    /// Unknown or malformed provider JSON, preserved without interpretation.
    ProviderEvent {
        name: String,
        native: Value,
    },
    ConnectionInterrupted {
        message: String,
    },
    LocalClose {
        code: u16,
        reason: String,
    },
}

/// Driver for the xAI streaming TTS WebSocket session.
pub struct XaiStreamingTtsDriver {
    outbound: mpsc::Receiver<TtsCommand>,
    events: mpsc::Sender<XaiStreamingTtsEvent>,
    codec: XaiStreamingTtsCodec,
    connection: RealtimeConnection,
    max_frame_bytes: usize,
    prefer_inbound: bool,
}

impl XaiStreamingTtsDriver {
    /// Run until the host closes the session, the transport fails, or the
    /// provider closes unexpectedly. xAI keeps successful multi-turn sessions
    /// open, so remote EOF is always reported as an interruption.
    pub async fn run(mut self) -> Result<(), RealtimeError> {
        loop {
            let step = {
                let outbound = self.outbound.next().fuse();
                let inbound = self.connection.inbound.next().fuse();
                futures::pin_mut!(outbound, inbound);
                if self.prefer_inbound {
                    match select(inbound, outbound).await {
                        Either::Left((frame, _)) => DriverStep::Inbound(frame),
                        Either::Right((command, _)) => DriverStep::Outbound(command),
                    }
                } else {
                    match select(outbound, inbound).await {
                        Either::Left((command, _)) => DriverStep::Outbound(command),
                        Either::Right((frame, _)) => DriverStep::Inbound(frame),
                    }
                }
            };
            self.prefer_inbound = !self.prefer_inbound;
            match step {
                DriverStep::Outbound(Some(TtsCommand::Frame(frame))) => {
                    if let Err(error) = validate_frame(&frame, self.max_frame_bytes) {
                        let _ = self
                            .emit(XaiStreamingTtsEvent::ConnectionInterrupted {
                                message: error.to_string(),
                            })
                            .await;
                        return Err(error);
                    }
                    if let Err(error) = self.connection.outbound.send(frame).await {
                        let _ = self
                            .emit(XaiStreamingTtsEvent::ConnectionInterrupted {
                                message: error.to_string(),
                            })
                            .await;
                        return Err(error);
                    }
                }
                DriverStep::Outbound(Some(TtsCommand::Close(close, ack))) => {
                    let result = self.connection.outbound.close(close.clone()).await;
                    let _ = ack.send(result.clone());
                    if result.is_ok() {
                        self.emit(XaiStreamingTtsEvent::LocalClose {
                            code: close.code,
                            reason: close.reason,
                        })
                        .await?;
                    }
                    return result;
                }
                DriverStep::Outbound(None) => {
                    let close = RealtimeClose::normal("xAI streaming TTS control dropped");
                    let result = self.connection.outbound.close(close.clone()).await;
                    if result.is_ok() {
                        self.emit(XaiStreamingTtsEvent::LocalClose {
                            code: close.code,
                            reason: close.reason,
                        })
                        .await?;
                    }
                    return result;
                }
                DriverStep::Inbound(Some(Ok(frame))) => {
                    if let Err(error) = validate_frame(&frame, self.max_frame_bytes) {
                        let _ = self
                            .emit(XaiStreamingTtsEvent::ConnectionInterrupted {
                                message: error.to_string(),
                            })
                            .await;
                        return Err(error);
                    }
                    let event = self.codec.decode(frame);
                    self.emit(event).await?;
                }
                DriverStep::Inbound(Some(Err(error))) => {
                    let _ = self
                        .emit(XaiStreamingTtsEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
                DriverStep::Inbound(None) => {
                    let error = RealtimeError::UnexpectedRemoteClose;
                    let _ = self
                        .emit(XaiStreamingTtsEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
            }
        }
    }

    async fn emit(&mut self, event: XaiStreamingTtsEvent) -> Result<(), RealtimeError> {
        self.events
            .send(event)
            .await
            .map_err(|_| RealtimeError::EventReceiverDropped)
    }
}

struct XaiStreamingTtsCodec {
    turn: Arc<Mutex<TtsTurnState>>,
    sent_text_done: Arc<AtomicU64>,
    sent_text_clear: Arc<AtomicU64>,
    received_text_done: u64,
    received_text_clear: u64,
}

impl XaiStreamingTtsCodec {
    fn decode(&mut self, frame: RealtimeFrame) -> XaiStreamingTtsEvent {
        let native = match frame {
            RealtimeFrame::Text(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(value) => value,
                Err(_) => match String::from_utf8(bytes.to_vec()) {
                    Ok(text) => Value::String(text),
                    Err(error) => json!({"raw_text_base64": BASE64.encode(error.into_bytes())}),
                },
            },
            RealtimeFrame::Binary(bytes) => json!({
                "frame":"binary",
                "data_base64":BASE64.encode(bytes),
            }),
        };
        let Some(event_type) = native.get("type").and_then(Value::as_str) else {
            return XaiStreamingTtsEvent::ProviderEvent {
                name: raw_frame_name(&native),
                native,
            };
        };

        match event_type {
            "audio.delta" => self.decode_audio_delta(native),
            "audio.done" => self.decode_audio_done(native),
            "audio.clear" => self.decode_audio_clear(native),
            "session.updated" => match native.get("replace") {
                Some(replace) if replace.is_object() => XaiStreamingTtsEvent::SessionUpdated {
                    replace: replace.clone(),
                    native,
                },
                _ => XaiStreamingTtsEvent::ProviderEvent {
                    name: event_type.into(),
                    native,
                },
            },
            "error" => match native.get("message").and_then(Value::as_str) {
                Some(message) => XaiStreamingTtsEvent::ProviderError {
                    message: message.to_owned(),
                    native,
                },
                None => XaiStreamingTtsEvent::ProviderEvent {
                    name: event_type.into(),
                    native,
                },
            },
            _ => XaiStreamingTtsEvent::ProviderEvent {
                name: event_type.into(),
                native,
            },
        }
    }

    fn decode_audio_delta(&self, native: Value) -> XaiStreamingTtsEvent {
        let allowed = matches!(
            *self.turn.lock().unwrap(),
            TtsTurnState::TextStreaming | TtsTurnState::AwaitingAudio | TtsTurnState::Clearing
        );
        let delta = native.get("delta").and_then(Value::as_str);
        match (allowed, delta.and_then(|value| BASE64.decode(value).ok())) {
            (true, Some(audio)) => XaiStreamingTtsEvent::AudioDelta {
                audio: Bytes::from(audio),
                native,
            },
            _ => XaiStreamingTtsEvent::ProviderEvent {
                name: "audio.delta".into(),
                native,
            },
        }
    }

    fn decode_audio_done(&mut self, native: Value) -> XaiStreamingTtsEvent {
        let trace_id = match native.get("trace_id").and_then(Value::as_str) {
            Some(trace_id) if !trace_id.is_empty() => trace_id.to_owned(),
            _ => {
                return XaiStreamingTtsEvent::ProviderEvent {
                    name: "audio.done".into(),
                    native,
                }
            }
        };
        let sent = self.sent_text_done.load(Ordering::Acquire);
        let mut turn = self.turn.lock().unwrap();
        if *turn != TtsTurnState::AwaitingAudio || sent <= self.received_text_done {
            return XaiStreamingTtsEvent::ProviderEvent {
                name: "audio.done".into(),
                native,
            };
        }
        self.received_text_done = self.received_text_done.saturating_add(1);
        *turn = TtsTurnState::Idle;
        XaiStreamingTtsEvent::AudioDone { trace_id, native }
    }

    fn decode_audio_clear(&mut self, native: Value) -> XaiStreamingTtsEvent {
        let sent = self.sent_text_clear.load(Ordering::Acquire);
        let mut turn = self.turn.lock().unwrap();
        if *turn != TtsTurnState::Clearing || sent <= self.received_text_clear {
            return XaiStreamingTtsEvent::ProviderEvent {
                name: "audio.clear".into(),
                native,
            };
        }
        self.received_text_clear = self.received_text_clear.saturating_add(1);
        self.received_text_done = self.sent_text_done.load(Ordering::Acquire);
        *turn = TtsTurnState::Idle;
        XaiStreamingTtsEvent::AudioClear { native }
    }
}

struct TrackedTtsSink {
    inner: Box<dyn RealtimeSink>,
    sent_text_done: Arc<AtomicU64>,
    sent_text_clear: Arc<AtomicU64>,
}

#[async_trait]
impl RealtimeSink for TrackedTtsSink {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        let event_type = match &frame {
            RealtimeFrame::Text(bytes) => serde_json::from_slice::<Value>(bytes)
                .ok()
                .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned)),
            RealtimeFrame::Binary(_) => None,
        };
        self.inner.send(frame).await?;
        match event_type.as_deref() {
            Some("text.done") => {
                self.sent_text_done.fetch_add(1, Ordering::Release);
            }
            Some("text.clear") => {
                self.sent_text_clear.fetch_add(1, Ordering::Release);
            }
            _ => {}
        }
        Ok(())
    }

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.close(close).await
    }
}

fn enqueue_frame(
    outbound: &mut mpsc::Sender<TtsCommand>,
    frame: RealtimeFrame,
    max_frame_bytes: usize,
) -> Result<(), RealtimeError> {
    validate_frame(&frame, max_frame_bytes)?;
    enqueue_command(outbound, TtsCommand::Frame(frame))
}

fn enqueue_command(
    outbound: &mut mpsc::Sender<TtsCommand>,
    command: TtsCommand,
) -> Result<(), RealtimeError> {
    outbound.try_send(command).map_err(|error| {
        if error.is_full() {
            RealtimeError::QueueFull
        } else {
            RealtimeError::Closed
        }
    })
}

fn speech_codec_name(codec: XaiSpeechCodec) -> &'static str {
    match codec {
        XaiSpeechCodec::Mp3 => "mp3",
        XaiSpeechCodec::Wav => "wav",
        XaiSpeechCodec::Pcm => "pcm",
        XaiSpeechCodec::Mulaw => "mulaw",
        XaiSpeechCodec::Alaw => "alaw",
    }
}

fn supported_sample_rate(sample_rate: u32) -> bool {
    matches!(
        sample_rate,
        8_000 | 16_000 | 22_050 | 24_000 | 44_100 | 48_000
    )
}

fn raw_frame_name(native: &Value) -> String {
    if native.is_string() {
        "raw_text".into()
    } else if native.get("frame").and_then(Value::as_str) == Some("binary") {
        "raw_binary".into()
    } else if native.get("raw_text_base64").is_some() {
        "raw_text_base64".into()
    } else {
        "unknown".into()
    }
}

fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
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

fn connect_timeout_error() -> RealtimeError {
    RealtimeError::Transport {
        message: "xAI streaming TTS WebSocket handshake timed out".into(),
    }
}
