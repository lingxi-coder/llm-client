//! Provider-neutral, executor-neutral primitives for bidirectional sessions.
//!
//! Hosts can supply a [`RealtimeTransport`] or enable the optional
//! `realtime-websocket` feature for the Rustls-backed `RustlsWebSocketTransport`.
//! The host runs the returned [`RealtimeDriver`] on its executor; the built-in
//! WebSocket transport specifically requires a Tokio runtime.

#[cfg(feature = "realtime-websocket")]
mod websocket;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    future::{poll_fn, select, Either},
    stream::BoxStream,
    FutureExt, Sink, SinkExt, StreamExt,
};
use serde_json::Value;
use std::{
    fmt,
    pin::Pin,
    sync::{Arc, Mutex},
    task::Poll,
};

const MAX_QUEUE_CAPACITY: usize = 65_536;
/// A local upper bound for one queued batch of function outputs. This is a
/// memory/latency guard, not a provider limit.
pub const MAX_REALTIME_TOOL_RESULTS: usize = 128;

#[cfg(feature = "realtime-websocket")]
pub use websocket::RustlsWebSocketTransport;

/// A WebSocket-style data frame. Ping/Pong/Close stay outside provider data
/// frames: callers may request Ping through [`RealtimeSink::ping`], while the
/// concrete transport owns WebSocket framing and automatic Pong replies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeFrame {
    Text(Bytes),
    Binary(Bytes),
}

impl RealtimeFrame {
    pub fn text(value: impl Into<Bytes>) -> Self {
        Self::Text(value.into())
    }

    pub fn binary(value: impl Into<Bytes>) -> Self {
        Self::Binary(value.into())
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Text(bytes) | Self::Binary(bytes) => bytes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Connection details passed to an injected transport. Header values and the
/// endpoint are intentionally omitted from `Debug` because either can contain
/// a credential.
#[derive(Clone)]
pub struct RealtimeConnectRequest {
    pub endpoint: String,
    pub headers: Vec<(String, String)>,
    pub max_frame_bytes: usize,
}

impl fmt::Debug for RealtimeConnectRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeConnectRequest")
            .field("endpoint", &"<redacted>")
            .field(
                "header_names",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("max_frame_bytes", &self.max_frame_bytes)
            .finish()
    }
}

/// Bounds for the session's queues and provider frames. A queued frame is at
/// most `max_frame_bytes`, so the outbound queue has a finite upper bound of
/// `outbound_capacity * max_frame_bytes` plus channel bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeLimits {
    pub outbound_capacity: usize,
    pub event_capacity: usize,
    pub max_frame_bytes: usize,
}

impl Default for RealtimeLimits {
    fn default() -> Self {
        Self {
            outbound_capacity: 8,
            event_capacity: 32,
            max_frame_bytes: 1024 * 1024,
        }
    }
}

/// A provider-agnostic audio encoding declaration. Codec implementations may
/// reject formats they do not support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeAudioFormat {
    Pcm16 { sample_rate_hz: u32 },
    G711MuLaw,
    G711ALaw,
    Encoded { mime_type: String },
}

/// Events callers can enqueue. Audio should be split into small chunks by the
/// host before enqueueing so the driver can keep latency and memory bounded.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeInput {
    Text(String),
    Audio {
        data: Bytes,
        format: RealtimeAudioFormat,
    },
    /// One image or video frame. Provider codecs enforce their own MIME and
    /// sequencing requirements; callers retain frame capture and pacing.
    Image {
        data: Bytes,
        mime_type: String,
    },
    /// Request the server's current representation of a conversation item.
    /// Provider codecs may reject this operation when the protocol lacks it.
    RetrieveItem {
        item_id: String,
    },
    /// Remove a conversation item from provider history when supported.
    DeleteItem {
        item_id: String,
    },
    /// Truncate assistant audio and synchronize the provider transcript to
    /// the point the host has played, when supported.
    TruncateAudio {
        item_id: String,
        content_index: u32,
        audio_end_ms: u32,
    },
    CommitAudio,
    /// Clear uncommitted input audio, when the provider protocol supports it.
    ClearAudio,
    /// Finish the provider session and wait for its final lifecycle event.
    FinishSession,
    ToolResult {
        call_id: String,
        output: Value,
    },
    /// Submit several function outputs atomically as one bounded queue entry.
    /// Providers may require an explicit continuation after all outputs.
    ToolResults {
        results: Vec<RealtimeToolResult>,
    },
    /// Ask a provider to continue after submitted tool results. The host can
    /// delay this input until playback or other turn-specific work is done.
    ContinueResponse,
    Interrupt,
}

/// One caller-executed function result. `call_id` must be the ID sent by the
/// provider's function-call event; `output` is serialized by the codec.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeToolResult {
    pub call_id: String,
    pub output: Value,
}

/// Normalized provider events. Provider details remain available in
/// `ProviderEvent` for capabilities that are not represented here yet.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeEvent {
    SessionReady,
    TurnStarted {
        turn_id: Option<String>,
    },
    TurnCompleted {
        turn_id: Option<String>,
        status: Option<String>,
    },
    AudioDelta {
        data: Bytes,
        format: RealtimeAudioFormat,
        item_id: Option<String>,
    },
    TextDelta {
        text: String,
        item_id: Option<String>,
        final_chunk: bool,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: Value,
    },
    UserSpeechStarted,
    Interrupted,
    ProviderError {
        code: Option<String>,
        message: String,
    },
    ProviderEvent {
        name: String,
        native: Value,
    },
    ConnectionInterrupted {
        message: String,
    },
    Closed {
        code: u16,
        reason: String,
    },
}

/// A close frame to send through the transport. The default is a normal close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeClose {
    pub code: u16,
    pub reason: String,
}

impl RealtimeClose {
    pub fn new(code: u16, reason: impl Into<String>) -> Result<Self, RealtimeError> {
        let reason = reason.into();
        let standard_code = (1000..=1014).contains(&code) && !matches!(code, 1004..=1006);
        let application_code = (3000..=4999).contains(&code);
        if !standard_code && !application_code {
            return Err(RealtimeError::InvalidCloseCode { code });
        }
        if reason.len() > 123 {
            return Err(RealtimeError::CloseReasonTooLong {
                actual: reason.len(),
            });
        }
        Ok(Self { code, reason })
    }

    pub fn normal(reason: impl Into<String>) -> Self {
        Self {
            code: 1000,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealtimeError {
    #[error("invalid realtime configuration: {message}")]
    InvalidConfig { message: String },
    #[error("invalid realtime input: {message}")]
    InvalidInput { message: String },
    #[error("realtime provider codec failed: {message}")]
    Codec { message: String },
    #[error("realtime transport failed: {message}")]
    Transport { message: String },
    #[error("realtime frame is {actual} bytes; configured maximum is {max}")]
    FrameTooLarge { actual: usize, max: usize },
    #[error("realtime outbound queue is full")]
    QueueFull,
    #[error("realtime session is closed")]
    Closed,
    #[error("realtime event receiver was dropped")]
    EventReceiverDropped,
    #[error("realtime connection ended before a local close")]
    UnexpectedRemoteClose,
    #[error("invalid WebSocket close code {code}")]
    InvalidCloseCode { code: u16 },
    #[error("WebSocket close reason is {actual} bytes; maximum is 123")]
    CloseReasonTooLong { actual: usize },
}

/// An injected bidirectional transport. Implementations own authentication,
/// TLS, WebSocket framing and ping/pong handling; they must not retry or
/// reconnect implicitly. Dropping the returned halves must release the live
/// connection promptly.
#[async_trait]
pub trait RealtimeTransport: Send + Sync + 'static {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError>;
}

/// The writable half of an injected connection.
#[async_trait]
pub trait RealtimeSink: Send + 'static {
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError>;

    /// Send a WebSocket Ping control frame when this transport supports it.
    ///
    /// The default keeps injected transports source-compatible and reports
    /// unsupported capability explicitly. Transports remain responsible for
    /// WebSocket Pong handling.
    async fn ping(&mut self, _payload: Bytes) -> Result<(), RealtimeError> {
        Err(RealtimeError::InvalidInput {
            message: "realtime transport does not support explicit WebSocket Ping frames".into(),
        })
    }

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError>;
}

/// Split transport connection. The incoming stream must yield one complete
/// message at a time; frames are checked against the configured size limit
/// before provider decoding.
pub struct RealtimeConnection {
    pub outbound: Box<dyn RealtimeSink>,
    pub inbound: BoxStream<'static, Result<RealtimeFrame, RealtimeError>>,
}

/// Maps provider-neutral input and raw frames to normalized realtime events.
/// `initial_frames` are sent synchronously after connecting and before the
/// session becomes visible to its caller.
pub trait RealtimeCodec: Send + Sync + 'static {
    fn initial_frames(&self) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        Ok(Vec::new())
    }

    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError>;

    /// Called after a fully encoded input passes frame bounds and is accepted
    /// by the outbound queue. This is not a provider acknowledgement. Codecs
    /// may release local lookup state already captured in the queued frames.
    /// Failed validation, a full queue, and a closed queue do not invoke it.
    fn input_queued(&self, _input: &RealtimeInput) {}

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError>;
}

enum Command {
    Frames(Vec<RealtimeFrame>),
    Close(RealtimeClose, oneshot::Sender<Result<(), RealtimeError>>),
}

/// Control and receive handles for a connected session. Run the companion
/// [`RealtimeDriver`] concurrently on the host's executor.
pub struct RealtimeSession {
    control: RealtimeControl,
    events: RealtimeEvents,
}

impl RealtimeSession {
    pub async fn connect(
        transport: &dyn RealtimeTransport,
        mut request: RealtimeConnectRequest,
        codec: Arc<dyn RealtimeCodec>,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        request.max_frame_bytes = limits.max_frame_bytes;
        let initial = codec.initial_frames()?;
        // Validate the entire initialization before opening a remote session
        // or sending any part of its configuration.
        for frame in &initial {
            validate_frame(frame, limits.max_frame_bytes)?;
        }
        let mut connection = transport.connect(request).await?;
        for frame in initial {
            connection_send_initial(connection.outbound.as_mut(), frame).await?;
        }

        // futures mpsc gives every sender a private slot in addition to the
        // shared buffer. There is exactly one channel sender for each of these
        // channels, so subtract that slot to get the caller's exact limit.
        let (outbound_tx, outbound_rx) = mpsc::channel(limits.outbound_capacity - 1);
        let (event_tx, event_rx) = mpsc::channel(limits.event_capacity - 1);
        let control = RealtimeControl {
            outbound: Arc::new(Mutex::new(outbound_tx)),
            codec: codec.clone(),
            limits,
        };
        let driver = RealtimeDriver {
            outbound: outbound_rx,
            events: event_tx,
            codec,
            connection,
            max_frame_bytes: limits.max_frame_bytes,
        };
        Ok((
            Self {
                control,
                events: RealtimeEvents { receiver: event_rx },
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (RealtimeControl, RealtimeEvents) {
        (self.control, self.events)
    }
}

/// Cloneable bounded sender for one realtime session.
#[derive(Clone)]
pub struct RealtimeControl {
    outbound: Arc<Mutex<mpsc::Sender<Command>>>,
    codec: Arc<dyn RealtimeCodec>,
    limits: RealtimeLimits,
}

impl RealtimeControl {
    /// Queue one provider-native command with the same frame and queue bounds.
    pub(crate) fn send_provider_frame(&self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        validate_frame(&frame, self.limits.max_frame_bytes)?;
        self.outbound
            .lock()
            .unwrap()
            .try_send(Command::Frames(vec![frame]))
            .map_err(|error| {
                if error.is_full() {
                    RealtimeError::QueueFull
                } else {
                    RealtimeError::Closed
                }
            })
    }

    /// Encode and enqueue one logical input without waiting. Returns
    /// [`RealtimeError::QueueFull`] when the bounded queue cannot accept it.
    pub fn send(&self, input: RealtimeInput) -> Result<(), RealtimeError> {
        preflight_input(&input, self.limits.max_frame_bytes)?;
        let frames = self.codec.encode(&input)?;
        if frames.is_empty() {
            return Err(RealtimeError::InvalidInput {
                message: "codec encoded input to no frames".into(),
            });
        }
        let mut total_bytes = 0usize;
        for frame in &frames {
            validate_frame(frame, self.limits.max_frame_bytes)?;
            total_bytes = total_bytes.saturating_add(frame.len());
        }
        if total_bytes > self.limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: total_bytes,
                max: self.limits.max_frame_bytes,
            });
        }
        self.outbound
            .lock()
            .unwrap()
            .try_send(Command::Frames(frames))
            .map_err(|error| {
                if error.is_full() {
                    RealtimeError::QueueFull
                } else {
                    RealtimeError::Closed
                }
            })?;
        self.codec.input_queued(&input);
        Ok(())
    }

    pub fn interrupt(&self) -> Result<(), RealtimeError> {
        self.send(RealtimeInput::Interrupt)
    }

    /// Close the remote connection after already queued messages are sent.
    /// This waits for the driver to perform and acknowledge the close.
    pub async fn close(&self, close: RealtimeClose) -> Result<(), RealtimeError> {
        let close = RealtimeClose::new(close.code, close.reason)?;
        let (ack_tx, ack_rx) = oneshot::channel();
        let outbound = self.outbound.clone();
        let mut command = Some(Command::Close(close, ack_tx));
        poll_fn(move |cx| {
            let mut sender = outbound.lock().unwrap();
            match Pin::new(&mut *sender).poll_ready(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(_)) => Poll::Ready(Err(RealtimeError::Closed)),
                Poll::Ready(Ok(())) => {
                    let command = command.take().expect("close is queued once");
                    Poll::Ready(
                        Pin::new(&mut *sender)
                            .start_send(command)
                            .map_err(|_| RealtimeError::Closed),
                    )
                }
            }
        })
        .await?;
        ack_rx.await.map_err(|_| RealtimeError::Closed)?
    }
}

/// Bounded stream of normalized provider events.
pub struct RealtimeEvents {
    receiver: mpsc::Receiver<RealtimeEvent>,
}

impl RealtimeEvents {
    pub async fn next(&mut self) -> Option<RealtimeEvent> {
        self.receiver.next().await
    }
}

/// Executor-neutral pump. Poll [`RealtimeDriver::run`] on the same executor
/// that runs the host session; dropping it releases the connection immediately.
pub struct RealtimeDriver {
    outbound: mpsc::Receiver<Command>,
    events: mpsc::Sender<RealtimeEvent>,
    codec: Arc<dyn RealtimeCodec>,
    connection: RealtimeConnection,
    max_frame_bytes: usize,
}

impl RealtimeDriver {
    pub async fn run(mut self) -> Result<(), RealtimeError> {
        loop {
            let outbound = self.outbound.next().fuse();
            let inbound = self.connection.inbound.next().fuse();
            futures::pin_mut!(outbound, inbound);
            match select(outbound, inbound).await {
                Either::Left((Some(Command::Frames(frames)), _)) => {
                    for frame in frames {
                        if let Err(error) = validate_frame(&frame, self.max_frame_bytes) {
                            let _ = self
                                .emit(RealtimeEvent::ProviderError {
                                    code: Some("frame_too_large".into()),
                                    message: error.to_string(),
                                })
                                .await;
                            return Err(error);
                        }
                        if let Err(error) = self.connection.outbound.send(frame).await {
                            let _ = self
                                .emit(RealtimeEvent::ConnectionInterrupted {
                                    message: error.to_string(),
                                })
                                .await;
                            return Err(error);
                        }
                    }
                }
                Either::Left((Some(Command::Close(close, ack)), _)) => {
                    let result = self.connection.outbound.close(close.clone()).await;
                    let _ = ack.send(result.clone());
                    if result.is_ok() {
                        self.emit(RealtimeEvent::Closed {
                            code: close.code,
                            reason: close.reason,
                        })
                        .await?;
                    }
                    return result;
                }
                Either::Left((None, _)) => {
                    let close = RealtimeClose::normal("realtime control dropped");
                    self.connection.outbound.close(close.clone()).await?;
                    self.emit(RealtimeEvent::Closed {
                        code: close.code,
                        reason: close.reason,
                    })
                    .await?;
                    return Ok(());
                }
                Either::Right((Some(Ok(frame)), _)) => {
                    if let Err(error) = validate_frame(&frame, self.max_frame_bytes) {
                        let _ = self
                            .emit(RealtimeEvent::ProviderError {
                                code: Some("frame_too_large".into()),
                                message: error.to_string(),
                            })
                            .await;
                        return Err(error);
                    }
                    let events = match self.codec.decode(frame) {
                        Ok(events) => events,
                        Err(error) => {
                            let _ = self
                                .emit(RealtimeEvent::ProviderError {
                                    code: Some("invalid_provider_frame".into()),
                                    message: error.to_string(),
                                })
                                .await;
                            return Err(error);
                        }
                    };
                    for event in events {
                        self.emit(event).await?;
                    }
                }
                Either::Right((Some(Err(error)), _)) => {
                    let _ = self
                        .emit(RealtimeEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
                Either::Right((None, _)) => {
                    let error = RealtimeError::UnexpectedRemoteClose;
                    let _ = self
                        .emit(RealtimeEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
            }
        }
    }

    async fn emit(&mut self, event: RealtimeEvent) -> Result<(), RealtimeError> {
        self.events
            .send(event)
            .await
            .map_err(|_| RealtimeError::EventReceiverDropped)
    }
}

async fn connection_send_initial(
    sink: &mut dyn RealtimeSink,
    frame: RealtimeFrame,
) -> Result<(), RealtimeError> {
    sink.send(frame).await
}

pub(crate) fn validate_limits(limits: RealtimeLimits) -> Result<(), RealtimeError> {
    if limits.outbound_capacity == 0
        || limits.event_capacity == 0
        || limits.outbound_capacity > MAX_QUEUE_CAPACITY
        || limits.event_capacity > MAX_QUEUE_CAPACITY
        || limits.max_frame_bytes == 0
    {
        return Err(RealtimeError::InvalidConfig {
            message: format!(
                "queue capacities must be between 1 and {MAX_QUEUE_CAPACITY}; max_frame_bytes must be non-zero"
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_frame(frame: &RealtimeFrame, max: usize) -> Result<(), RealtimeError> {
    if frame.len() > max {
        return Err(RealtimeError::FrameTooLarge {
            actual: frame.len(),
            max,
        });
    }
    Ok(())
}

fn preflight_input(input: &RealtimeInput, max: usize) -> Result<(), RealtimeError> {
    let raw_len = match input {
        RealtimeInput::Text(text) => text.len(),
        RealtimeInput::Audio { data, .. } => data.len(),
        RealtimeInput::Image { data, mime_type } => data.len().saturating_add(mime_type.len()),
        RealtimeInput::RetrieveItem { item_id } | RealtimeInput::DeleteItem { item_id } => {
            item_id.len()
        }
        RealtimeInput::TruncateAudio { item_id, .. } => item_id.len(),
        RealtimeInput::ToolResult { call_id, output } => call_id.len() + output.to_string().len(),
        RealtimeInput::ToolResults { results } => {
            if results.is_empty() {
                return Err(RealtimeError::InvalidInput {
                    message: "tool result batch must contain at least one result".into(),
                });
            }
            if results.len() > MAX_REALTIME_TOOL_RESULTS {
                return Err(RealtimeError::InvalidInput {
                    message: format!(
                        "tool result batch exceeds the local limit of {MAX_REALTIME_TOOL_RESULTS} results"
                    ),
                });
            }
            results.iter().fold(0usize, |total, result| {
                total
                    .saturating_add(result.call_id.len())
                    .saturating_add(result.output.to_string().len())
            })
        }
        RealtimeInput::CommitAudio
        | RealtimeInput::ClearAudio
        | RealtimeInput::FinishSession
        | RealtimeInput::ContinueResponse
        | RealtimeInput::Interrupt => 0,
    };
    if raw_len > max {
        return Err(RealtimeError::FrameTooLarge {
            actual: raw_len,
            max,
        });
    }
    Ok(())
}
