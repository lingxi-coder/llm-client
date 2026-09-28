//! MiniMax's native bidirectional T2A WebSocket session.
//!
//! The host supplies a [`RealtimeTransport`] and schedules optional WebSocket
//! pings. This module does not create a timer, reconnect, retry, or play audio.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{ProviderId, Secret},
    providers::minimax::streaming_tts::MiniMaxStreamingTtsRequest,
    providers::minimax::voices::MiniMaxVoiceRef,
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
    runtime::Deadline,
};
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    future::{select, Either},
    lock::Mutex as AsyncMutex,
    StreamExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use thiserror::Error;

pub use crate::providers::minimax::streaming_tts::{
    MiniMaxStreamingTtsAudioFormat as MiniMaxBidiTtsAudioFormat,
    MiniMaxStreamingTtsAudioSetting as MiniMaxBidiTtsAudioSetting,
    MiniMaxStreamingTtsEmotion as MiniMaxBidiTtsEmotion,
    MiniMaxStreamingTtsLanguageBoost as MiniMaxBidiTtsLanguageBoost,
    MiniMaxStreamingTtsModel as MiniMaxBidiTtsModel,
    MiniMaxStreamingTtsRequest as MiniMaxBidiTtsParameters,
    MiniMaxStreamingTtsSoundEffect as MiniMaxBidiTtsSoundEffect,
    MiniMaxStreamingTtsSubtitleType as MiniMaxBidiTtsSubtitleType,
    MiniMaxStreamingTtsTimbreWeight as MiniMaxBidiTtsTimbreWeight,
    MiniMaxStreamingTtsVoiceModify as MiniMaxBidiTtsVoiceModify,
    MiniMaxStreamingTtsVoiceSetting as MiniMaxBidiTtsVoiceSetting,
};
pub use crate::providers::minimax::tts::MiniMaxTtsRegion as MiniMaxBidiTtsRegion;

/// Official International MiniMax bidirectional T2A WebSocket route.
pub const MINIMAX_BIDI_TTS_INTERNATIONAL_ENDPOINT: &str = "wss://api.minimax.io/ws/v1/t2a_v2_bidi";
/// Official Mainland MiniMax bidirectional T2A WebSocket route.
pub const MINIMAX_BIDI_TTS_CHINA_ENDPOINT: &str = "wss://api.minimax.cn/ws/v1/t2a_v2_bidi";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const HARD_MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
const HARD_MAX_TEXT_CHARS: usize = 10_000;
const HARD_MAX_AUDIO_BYTES: usize = 512 * 1024 * 1024;
const HARD_MAX_SETUP_EVENTS: usize = 128;
const MAX_WEBSOCKET_PING_BYTES: usize = 125;
const CLOSE_TIMEOUT: Duration = Duration::from_secs(2);

/// Account, region, and ready-deadline selection for one Bidi T2A connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxBidiTtsConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: MiniMaxBidiTtsRegion,
    pub connect_timeout: Duration,
}

impl MiniMaxBidiTtsConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: MiniMaxBidiTtsRegion,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// Bound the WebSocket handshake, `connected_success`, and `task_started`.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
}

/// Per-connection API key. The service never stores credentials after connect.
pub struct MiniMaxBidiTtsCredentials {
    api_key: Secret<String>,
}

impl MiniMaxBidiTtsCredentials {
    pub fn new(api_key: Secret<String>) -> Self {
        Self { api_key }
    }
}

impl fmt::Debug for MiniMaxBidiTtsCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsCredentials")
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Non-secret identity attached to one Bidi session and its events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxBidiTtsScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub region: MiniMaxBidiTtsRegion,
}

/// Bidi task-start parameters plus its optional client-generated session ID.
///
/// `parameters` uses the shared typed MiniMax T2A WebSocket setting schema.
/// `session_id` is only echoed by the provider and is omitted when unset.
#[derive(Clone, PartialEq)]
pub struct MiniMaxBidiTtsRequest {
    pub parameters: MiniMaxStreamingTtsRequest,
    pub session_id: Option<String>,
}

impl MiniMaxBidiTtsRequest {
    pub fn new(voice_id: impl Into<String>) -> Self {
        Self {
            parameters: MiniMaxStreamingTtsRequest::new(voice_id),
            session_id: None,
        }
    }

    pub fn with_parameters(mut self, parameters: MiniMaxStreamingTtsRequest) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

impl fmt::Debug for MiniMaxBidiTtsRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsRequest")
            .field("parameters", &self.parameters)
            .field("session_id", &self.session_id)
            .finish()
    }
}

/// Caller-selected frame, text, output, and setup-event memory bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniMaxBidiTtsLimits {
    pub max_frame_bytes: usize,
    pub max_text_chars: usize,
    pub max_audio_bytes_per_session: usize,
    pub max_setup_events: usize,
}

impl Default for MiniMaxBidiTtsLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 1024 * 1024,
            max_text_chars: HARD_MAX_TEXT_CHARS,
            max_audio_bytes_per_session: 64 * 1024 * 1024,
            max_setup_events: 16,
        }
    }
}

impl MiniMaxBidiTtsLimits {
    fn validate(self) -> Result<(), MiniMaxBidiTtsError> {
        if self.max_frame_bytes == 0 || self.max_frame_bytes > HARD_MAX_FRAME_BYTES {
            return Err(invalid("max_frame_bytes is outside the supported bounds"));
        }
        if self.max_text_chars > HARD_MAX_TEXT_CHARS {
            return Err(invalid(
                "max_text_chars cannot exceed 10,000 Unicode characters",
            ));
        }
        if self.max_audio_bytes_per_session == 0
            || self.max_audio_bytes_per_session > HARD_MAX_AUDIO_BYTES
        {
            return Err(invalid(
                "max_audio_bytes_per_session is outside the supported bounds",
            ));
        }
        if self.max_setup_events > HARD_MAX_SETUP_EVENTS {
            return Err(invalid(
                "max_setup_events cannot exceed the local hard limit",
            ));
        }
        Ok(())
    }
}

/// Raw provider events consumed while establishing a session.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxBidiTtsMetadata {
    pub scope: MiniMaxBidiTtsScope,
    pub connected_native: Value,
    pub task_started_native: Value,
    pub setup_events: Vec<Value>,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub connect_id: Option<String>,
}

impl fmt::Debug for MiniMaxBidiTtsMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsMetadata")
            .field("scope", &self.scope)
            .field("connected_native", &"<preserved>")
            .field("task_started_native", &"<preserved>")
            .field("setup_event_count", &self.setup_events.len())
            .field("session_id", &self.session_id)
            .field("trace_id", &self.trace_id)
            .field("connect_id", &self.connect_id)
            .finish()
    }
}

/// One provider-native Bidi event projected to a typed kind.
#[derive(Clone, PartialEq)]
pub struct MiniMaxBidiTtsEvent {
    pub kind: MiniMaxBidiTtsEventKind,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub connect_id: Option<String>,
    /// Full provider JSON for forward-compatible replay and diagnostics.
    pub native: Option<Value>,
    /// Exact bytes when the frame could not be decoded as JSON.
    pub raw_frame: Option<Bytes>,
}

impl fmt::Debug for MiniMaxBidiTtsEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsEvent")
            .field("kind", &self.kind)
            .field("session_id", &self.session_id)
            .field("trace_id", &self.trace_id)
            .field("connect_id", &self.connect_id)
            .field("native", &"<preserved>")
            .field("raw_frame_bytes", &self.raw_frame.as_ref().map(Bytes::len))
            .finish()
    }
}

/// Event projection. `is_final` closes an audio response; `SentenceEnd` marks
/// one server-built sentence; `TaskFinished` marks the entire session.
#[derive(Clone, PartialEq)]
pub enum MiniMaxBidiTtsEventKind {
    AudioDelta {
        audio: Bytes,
        is_final: bool,
        extra_info: Option<Value>,
    },
    SentenceStart,
    SentenceEnd,
    TaskCanceled,
    TaskFlushed,
    TaskFinished,
    TaskFailed {
        status_code: Option<i64>,
        status_message: Option<String>,
    },
    /// Soft provider errors 2204/2205 keep this session open; the host chooses
    /// whether to explicitly send another `task_continue`.
    SoftError {
        status_code: i64,
        status_message: Option<String>,
    },
    Native {
        event: Option<String>,
    },
    Malformed {
        event: Option<String>,
        reason: String,
    },
}

impl fmt::Debug for MiniMaxBidiTtsEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioDelta {
                audio,
                is_final,
                extra_info,
            } => f
                .debug_struct("AudioDelta")
                .field("audio_bytes", &audio.len())
                .field("is_final", is_final)
                .field("extra_info", &extra_info.as_ref().map(|_| "<preserved>"))
                .finish(),
            Self::SentenceStart => f.write_str("SentenceStart"),
            Self::SentenceEnd => f.write_str("SentenceEnd"),
            Self::TaskCanceled => f.write_str("TaskCanceled"),
            Self::TaskFlushed => f.write_str("TaskFlushed"),
            Self::TaskFinished => f.write_str("TaskFinished"),
            Self::TaskFailed {
                status_code,
                status_message,
            } => f
                .debug_struct("TaskFailed")
                .field("status_code", status_code)
                .field("status_message", status_message)
                .finish(),
            Self::SoftError {
                status_code,
                status_message,
            } => f
                .debug_struct("SoftError")
                .field("status_code", status_code)
                .field("status_message", status_message)
                .finish(),
            Self::Native { event } => f.debug_tuple("Native").field(event).finish(),
            Self::Malformed { event, reason } => f
                .debug_struct("Malformed")
                .field("event", event)
                .field("reason", reason)
                .finish(),
        }
    }
}

/// One established task. Poll [`MiniMaxBidiTtsEvents::next`] while sending
/// text or control events through the input half.
pub struct MiniMaxBidiTtsSession {
    input: MiniMaxBidiTtsInput,
    events: MiniMaxBidiTtsEvents,
}

impl fmt::Debug for MiniMaxBidiTtsSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsSession")
            .field("metadata", &self.input.metadata)
            .finish_non_exhaustive()
    }
}

impl MiniMaxBidiTtsSession {
    pub fn metadata(&self) -> &MiniMaxBidiTtsMetadata {
        &self.input.metadata
    }

    pub fn into_parts(self) -> (MiniMaxBidiTtsInput, MiniMaxBidiTtsEvents) {
        (self.input, self.events)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionStatus {
    Ready,
    SendingContinue,
    SendingFlush,
    FlushAckedWhileSending,
    FlushPending,
    SendingCancel,
    CancelAckedWhileSending,
    CancelPending,
    SendingFinish,
    FinishAckedWhileSending,
    FinishPending,
    FinishAcked,
    Finished,
    Failed,
    Aborted,
}

struct SharedState {
    status: SessionStatus,
    audio_bytes: usize,
    finish_send_result: Option<oneshot::Receiver<bool>>,
}

struct SharedSession {
    state: AsyncMutex<SharedState>,
    sink: AsyncMutex<Option<Box<dyn RealtimeSink>>>,
    terminal: AtomicBool,
    writer_active: AtomicBool,
    close_signal: mpsc::UnboundedSender<()>,
    close_receiver: AsyncMutex<mpsc::UnboundedReceiver<()>>,
    reader_signal: mpsc::UnboundedSender<()>,
}

impl SharedSession {
    fn request_terminal(&self) {
        if !self.terminal.swap(true, Ordering::AcqRel) {
            let _ = self.close_signal.unbounded_send(());
            let _ = self.reader_signal.unbounded_send(());
        }
    }

    async fn close_sink(&self, close: RealtimeClose) -> Result<(), MiniMaxBidiTtsError> {
        self.request_terminal();
        let mut slot = self.sink.lock().await;
        close_sink_slot(&mut slot, close).await;
        Ok(())
    }

    async fn close_sink_if_idle(&self, close: RealtimeClose) {
        self.request_terminal();
        if self.writer_active.load(Ordering::Acquire) {
            return;
        }
        let Some(mut slot) = self.sink.try_lock() else {
            return;
        };
        if !self.writer_active.load(Ordering::Acquire) {
            close_sink_slot(&mut slot, close).await;
        }
    }
}

struct PendingWriteGuard {
    shared: Arc<SharedSession>,
    armed: bool,
}

impl PendingWriteGuard {
    fn new(shared: Arc<SharedSession>) -> Self {
        Self {
            shared,
            armed: true,
        }
    }

    fn complete(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingWriteGuard {
    fn drop(&mut self) {
        if self.armed {
            // A canceled future cannot report whether the provider received
            // the frame. Fail closed and wake any active sink write.
            self.shared.request_terminal();
        }
    }
}

struct WriterActivityGuard(Arc<SharedSession>);

impl WriterActivityGuard {
    fn new(shared: Arc<SharedSession>) -> Self {
        shared.writer_active.store(true, Ordering::Release);
        Self(shared)
    }
}

impl Drop for WriterActivityGuard {
    fn drop(&mut self) {
        self.0.writer_active.store(false, Ordering::Release);
    }
}

async fn close_sink_slot(slot: &mut Option<Box<dyn RealtimeSink>>, close: RealtimeClose) {
    if let Some(mut sink) = slot.take() {
        close_sink_ref(sink.as_mut(), close).await;
    }
}

async fn close_sink_ref(sink: &mut dyn RealtimeSink, close: RealtimeClose) {
    let _ = Deadline::after(Some(CLOSE_TIMEOUT))
        .run(sink.close(close))
        .await;
}

/// Immediate, bounded writes to one MiniMax Bidi task.
pub struct MiniMaxBidiTtsInput {
    shared: Arc<SharedSession>,
    limits: MiniMaxBidiTtsLimits,
    metadata: Arc<MiniMaxBidiTtsMetadata>,
}

impl fmt::Debug for MiniMaxBidiTtsInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsInput")
            .field("metadata", &self.metadata)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl MiniMaxBidiTtsInput {
    pub fn metadata(&self) -> &MiniMaxBidiTtsMetadata {
        &self.metadata
    }

    /// Send one exact text delta. The provider buffers sentences across calls.
    /// The client does not trim whitespace, add punctuation, or split text.
    pub async fn continue_text(&mut self, text: &str) -> Result<(), MiniMaxBidiTtsError> {
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(MiniMaxBidiTtsError::Closed);
        }
        if text.chars().count() > self.limits.max_text_chars {
            return Err(invalid(
                "task_continue text exceeds the configured character limit",
            ));
        }
        let frame = json_frame(
            &json!({"event":"task_continue","text":text}),
            self.limits.max_frame_bytes,
        )?;
        {
            let mut state = self.shared.state.lock().await;
            ensure_input_ready(&self.shared, state.status)?;
            state.status = SessionStatus::SendingContinue;
        }
        let mut pending_write = PendingWriteGuard::new(self.shared.clone());
        let send_result = self.send_frame(frame).await;
        let mut state = self.shared.state.lock().await;
        match send_result {
            Ok(()) => {
                if state.status == SessionStatus::Failed
                    || self.shared.terminal.load(Ordering::Acquire)
                {
                    state.status = SessionStatus::Failed;
                    pending_write.complete();
                    Err(MiniMaxBidiTtsError::Closed)
                } else {
                    if state.status == SessionStatus::SendingContinue {
                        state.status = SessionStatus::Ready;
                    }
                    pending_write.complete();
                    Ok(())
                }
            }
            Err(source) => {
                state.status = SessionStatus::Failed;
                self.shared.request_terminal();
                pending_write.complete();
                Err(MiniMaxBidiTtsError::OutcomeUnknown {
                    operation: "task_continue",
                    reason: source.to_string(),
                })
            }
        }
    }

    /// Flush server-buffered tail text without ending the session. Text input
    /// remains gated until `task_flushed` has been read from the event stream.
    pub async fn flush(&mut self) -> Result<(), MiniMaxBidiTtsError> {
        self.send_control("task_flush", "task_flush", ControlSend::Flush)
            .await
    }

    /// Interrupt current synthesis and discard buffered text. This is allowed
    /// while a flush is pending; wait for `task_canceled` before continuing.
    pub async fn cancel(&mut self) -> Result<(), MiniMaxBidiTtsError> {
        self.send_control("task_cancel", "task_cancel", ControlSend::Cancel)
            .await
    }

    /// Flush remaining text and close the task. Success is confirmed only by
    /// `task_finished` followed by an actual clean WebSocket EOF.
    pub async fn finish(&mut self) -> Result<(), MiniMaxBidiTtsError> {
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(MiniMaxBidiTtsError::Closed);
        }
        let frame = json_frame(&json!({"event":"task_finish"}), self.limits.max_frame_bytes)?;
        let (notify_send, notify_receive) = oneshot::channel();
        {
            let mut state = self.shared.state.lock().await;
            ensure_input_ready(&self.shared, state.status)?;
            state.status = SessionStatus::SendingFinish;
            state.finish_send_result = Some(notify_receive);
        }
        let mut pending_write = PendingWriteGuard::new(self.shared.clone());

        let send_result = self.send_frame(frame).await;
        let mut state = self.shared.state.lock().await;
        match send_result {
            Ok(()) => {
                match state.status {
                    SessionStatus::SendingFinish => state.status = SessionStatus::FinishPending,
                    SessionStatus::FinishAckedWhileSending => {
                        state.status = SessionStatus::FinishAcked
                    }
                    SessionStatus::Failed | SessionStatus::Aborted => {}
                    _ => {}
                }
                state.finish_send_result.take();
                let _ = notify_send.send(true);
                pending_write.complete();
                if state.status == SessionStatus::Failed
                    || self.shared.terminal.load(Ordering::Acquire)
                {
                    Err(MiniMaxBidiTtsError::Closed)
                } else {
                    Ok(())
                }
            }
            Err(source) => {
                state.status = SessionStatus::Failed;
                self.shared.request_terminal();
                state.finish_send_result.take();
                let _ = notify_send.send(false);
                pending_write.complete();
                Err(MiniMaxBidiTtsError::OutcomeUnknown {
                    operation: "task_finish",
                    reason: source.to_string(),
                })
            }
        }
    }

    /// Send a host-scheduled WebSocket Ping control frame. This never emits a
    /// JSON provider event and never starts an automatic keepalive timer.
    pub async fn ping(&mut self, payload: Bytes) -> Result<(), MiniMaxBidiTtsError> {
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(MiniMaxBidiTtsError::Closed);
        }
        if payload.len() > MAX_WEBSOCKET_PING_BYTES {
            return Err(invalid("WebSocket Ping payload cannot exceed 125 bytes"));
        }
        {
            let state = self.shared.state.lock().await;
            if self.shared.terminal.load(Ordering::Acquire) {
                return Err(MiniMaxBidiTtsError::Closed);
            }
            if !matches!(
                state.status,
                SessionStatus::Ready
                    | SessionStatus::SendingContinue
                    | SessionStatus::FlushPending
                    | SessionStatus::CancelPending
            ) {
                return Err(MiniMaxBidiTtsError::Closed);
            }
        }
        let _writer_active = WriterActivityGuard::new(self.shared.clone());
        let mut sink_slot = self.shared.sink.lock().await;
        let ping_result = {
            let sink = sink_slot.as_mut().ok_or(MiniMaxBidiTtsError::Closed)?;
            let mut close_receiver = self.shared.close_receiver.lock().await;
            match select(
                Box::pin(sink.ping(payload)),
                Box::pin(close_receiver.next()),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right((_, _)) => Err(RealtimeError::Closed),
            }
        };
        match ping_result {
            Ok(()) => {
                if self.shared.terminal.load(Ordering::Acquire) {
                    close_sink_slot(
                        &mut sink_slot,
                        RealtimeClose::normal("MiniMax Bidi T2A session closed"),
                    )
                    .await;
                    Err(MiniMaxBidiTtsError::Closed)
                } else {
                    Ok(())
                }
            }
            Err(RealtimeError::InvalidInput { .. }) => {
                if self.shared.terminal.load(Ordering::Acquire) {
                    close_sink_slot(
                        &mut sink_slot,
                        RealtimeClose::normal("MiniMax Bidi T2A session closed"),
                    )
                    .await;
                    Err(MiniMaxBidiTtsError::Closed)
                } else {
                    Err(MiniMaxBidiTtsError::PingUnsupported)
                }
            }
            Err(RealtimeError::Closed) if self.shared.terminal.load(Ordering::Acquire) => {
                close_sink_slot(
                    &mut sink_slot,
                    RealtimeClose::normal("MiniMax Bidi T2A session closed"),
                )
                .await;
                self.shared.state.lock().await.status = SessionStatus::Failed;
                Err(MiniMaxBidiTtsError::Closed)
            }
            Err(error) => {
                self.shared.state.lock().await.status = SessionStatus::Failed;
                self.shared.request_terminal();
                close_sink_slot(
                    &mut sink_slot,
                    RealtimeClose::normal("MiniMax Bidi T2A ping failed"),
                )
                .await;
                Err(MiniMaxBidiTtsError::Realtime(error))
            }
        }
    }

    /// Locally close the connection. This does not claim that the task finished.
    pub async fn abort(&mut self) -> Result<(), MiniMaxBidiTtsError> {
        {
            let mut state = self.shared.state.lock().await;
            if matches!(
                state.status,
                SessionStatus::Finished | SessionStatus::Aborted
            ) {
                return Ok(());
            }
            state.status = SessionStatus::Aborted;
            state.finish_send_result.take();
        }
        self.shared
            .close_sink(RealtimeClose::normal("MiniMax Bidi T2A session aborted"))
            .await
    }

    async fn send_control(
        &mut self,
        event: &'static str,
        operation: &'static str,
        control: ControlSend,
    ) -> Result<(), MiniMaxBidiTtsError> {
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(MiniMaxBidiTtsError::Closed);
        }
        let frame = json_frame(&json!({"event":event}), self.limits.max_frame_bytes)?;
        {
            let mut state = self.shared.state.lock().await;
            if self.shared.terminal.load(Ordering::Acquire) {
                return Err(MiniMaxBidiTtsError::Closed);
            }
            match (control, state.status) {
                (ControlSend::Flush, SessionStatus::Ready) => {
                    state.status = SessionStatus::SendingFlush;
                }
                (ControlSend::Cancel, SessionStatus::Ready | SessionStatus::FlushPending) => {
                    state.status = SessionStatus::SendingCancel;
                }
                (
                    ControlSend::Cancel,
                    SessionStatus::SendingFlush | SessionStatus::FlushAckedWhileSending,
                ) => {
                    return Err(MiniMaxBidiTtsError::ControlPending);
                }
                (_, SessionStatus::Failed | SessionStatus::Aborted | SessionStatus::Finished) => {
                    return Err(MiniMaxBidiTtsError::Closed);
                }
                _ => return Err(MiniMaxBidiTtsError::ControlPending),
            }
        }
        let mut pending_write = PendingWriteGuard::new(self.shared.clone());
        let send_result = self.send_frame(frame).await;
        let mut state = self.shared.state.lock().await;
        match send_result {
            Ok(()) => {
                if state.status == SessionStatus::Failed
                    || self.shared.terminal.load(Ordering::Acquire)
                {
                    state.status = SessionStatus::Failed;
                    pending_write.complete();
                    return Err(MiniMaxBidiTtsError::Closed);
                }
                state.status = match (control, state.status) {
                    (ControlSend::Flush, SessionStatus::SendingFlush) => {
                        SessionStatus::FlushPending
                    }
                    (ControlSend::Flush, SessionStatus::FlushAckedWhileSending) => {
                        SessionStatus::Ready
                    }
                    (ControlSend::Cancel, SessionStatus::SendingCancel) => {
                        SessionStatus::CancelPending
                    }
                    (ControlSend::Cancel, SessionStatus::CancelAckedWhileSending) => {
                        SessionStatus::Ready
                    }
                    (_, SessionStatus::Failed | SessionStatus::Aborted) => state.status,
                    (_, status) => status,
                };
                if state.status == SessionStatus::Ready {
                    state.finish_send_result.take();
                }
                pending_write.complete();
                Ok(())
            }
            Err(source) => {
                state.status = SessionStatus::Failed;
                self.shared.request_terminal();
                pending_write.complete();
                Err(MiniMaxBidiTtsError::OutcomeUnknown {
                    operation,
                    reason: source.to_string(),
                })
            }
        }
    }

    async fn send_frame(&self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(RealtimeError::Closed);
        }
        let _writer_active = WriterActivityGuard::new(self.shared.clone());
        let mut sink_slot = self.shared.sink.lock().await;
        let send_result = {
            let sink = sink_slot.as_mut().ok_or(RealtimeError::Closed)?;
            let mut close_receiver = self.shared.close_receiver.lock().await;
            match select(Box::pin(sink.send(frame)), Box::pin(close_receiver.next())).await {
                Either::Left((result, _)) => result,
                Either::Right((_, _)) => Err(RealtimeError::Closed),
            }
        };
        if self.shared.terminal.load(Ordering::Acquire) {
            close_sink_slot(
                &mut sink_slot,
                RealtimeClose::normal("MiniMax Bidi T2A session closed"),
            )
            .await;
            Err(RealtimeError::Closed)
        } else if send_result.is_err() {
            close_sink_slot(
                &mut sink_slot,
                RealtimeClose::normal("MiniMax Bidi T2A send failed"),
            )
            .await;
            send_result
        } else {
            send_result
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ControlSend {
    Flush,
    Cancel,
}

/// MiniMax Bidi T2A service bound to one explicit regional route and account.
pub struct MiniMaxBidiTtsService {
    transport: Arc<dyn RealtimeTransport>,
    scope: MiniMaxBidiTtsScope,
    endpoint: String,
    connect_timeout: Duration,
}

impl fmt::Debug for MiniMaxBidiTtsService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsService")
            .field("scope", &self.scope)
            .field("endpoint", &"<fixed regional WSS route>")
            .field("connect_timeout", &self.connect_timeout)
            .finish_non_exhaustive()
    }
}

impl MiniMaxBidiTtsService {
    pub fn new(
        transport: Arc<dyn RealtimeTransport>,
        config: MiniMaxBidiTtsConfig,
    ) -> Result<Self, MiniMaxBidiTtsError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.connect_timeout.is_zero() {
            return Err(invalid("connect_timeout must be positive"));
        }
        let endpoint = bidi_endpoint(config.region).to_owned();
        let scope = MiniMaxBidiTtsScope {
            provider_id: ProviderId::new("minimax"),
            profile_name: config.profile_name,
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&endpoint),
            account_scope: config.account_scope,
            region: config.region,
        };
        Ok(Self {
            transport,
            scope,
            endpoint,
            connect_timeout: config.connect_timeout,
        })
    }

    pub fn scope(&self) -> &MiniMaxBidiTtsScope {
        &self.scope
    }

    /// Connect to the Bidi task using the voice ID from the typed request.
    pub async fn connect(
        &self,
        credentials: &MiniMaxBidiTtsCredentials,
        request: &MiniMaxBidiTtsRequest,
        limits: MiniMaxBidiTtsLimits,
    ) -> Result<MiniMaxBidiTtsSession, MiniMaxBidiTtsError> {
        validate_credential(&credentials.api_key)?;
        limits.validate()?;
        // Shared task_start schema and structural validation are independent
        // from ordinary WSS endpoint/region capability checks.
        let mut start_event = request
            .parameters
            .start_event()
            .map_err(|error| invalid(error.to_string()))?;
        if let Some(session_id) = &request.session_id {
            start_event["session_id"] = Value::String(session_id.clone());
        }
        let start_frame = json_frame(&start_event, limits.max_frame_bytes)?;
        let deadline = Deadline::after(Some(self.connect_timeout));
        let connect_request = RealtimeConnectRequest {
            endpoint: self.endpoint.clone(),
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", credentials.api_key.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let mut connection = deadline
            .run(self.transport.connect(connect_request))
            .await
            .map_err(|_| MiniMaxBidiTtsError::Timeout {
                operation: "WebSocket connect",
            })??;

        let connected_native = receive_setup_json(
            &mut connection,
            deadline,
            limits.max_frame_bytes,
            "connected_success",
        )
        .await?;
        validate_setup_status(&connected_native)?;
        if event_name(&connected_native) != Some("connected_success") {
            return Err(protocol(
                "the first server event was not connected_success",
                Some(connected_native),
            ));
        }

        let send_start = deadline.run(connection.outbound.send(start_frame)).await;
        match send_start {
            Err(_) | Ok(Err(_)) => {
                return Err(MiniMaxBidiTtsError::OutcomeUnknown {
                    operation: "task_start",
                    reason: "task_start send timed out or failed".into(),
                })
            }
            Ok(Ok(())) => {}
        }

        let mut setup_events = Vec::new();
        let task_started_native = loop {
            let native = receive_setup_json(
                &mut connection,
                deadline,
                limits.max_frame_bytes,
                "task_started",
            )
            .await
            .map_err(|error| MiniMaxBidiTtsError::OutcomeUnknown {
                operation: "task_start",
                reason: error.to_string(),
            })?;
            validate_setup_status(&native).map_err(|error| {
                if let MiniMaxBidiTtsError::Provider { .. } = error {
                    error
                } else {
                    MiniMaxBidiTtsError::OutcomeUnknown {
                        operation: "task_start",
                        reason: error.to_string(),
                    }
                }
            })?;
            match event_name(&native) {
                Some("task_started") => break native,
                Some("task_failed") => return Err(provider_error(&native)),
                _ => {
                    if setup_events.len() >= limits.max_setup_events {
                        return Err(MiniMaxBidiTtsError::OutcomeUnknown {
                            operation: "task_start",
                            reason: "MiniMax did not acknowledge task_started within the setup event limit".into(),
                        });
                    }
                    setup_events.push(native);
                }
            }
        };

        let session_id = string_field(&task_started_native, "session_id")
            .or_else(|| string_field(&connected_native, "session_id"))
            .or_else(|| request.session_id.clone());
        let trace_id = string_field(&task_started_native, "trace_id")
            .or_else(|| string_field(&connected_native, "trace_id"));
        let connect_id = string_field(&task_started_native, "connect_id")
            .or_else(|| string_field(&connected_native, "connect_id"));
        let metadata = Arc::new(MiniMaxBidiTtsMetadata {
            scope: self.scope.clone(),
            connected_native,
            task_started_native,
            setup_events,
            session_id: session_id.clone(),
            trace_id: trace_id.clone(),
            connect_id: connect_id.clone(),
        });
        let (close_signal, close_receiver) = mpsc::unbounded();
        let (reader_signal, terminal_receiver) = mpsc::unbounded();
        let shared = Arc::new(SharedSession {
            state: AsyncMutex::new(SharedState {
                status: SessionStatus::Ready,
                audio_bytes: 0,
                finish_send_result: None,
            }),
            sink: AsyncMutex::new(Some(connection.outbound)),
            terminal: AtomicBool::new(false),
            writer_active: AtomicBool::new(false),
            close_signal,
            close_receiver: AsyncMutex::new(close_receiver),
            reader_signal,
        });
        Ok(MiniMaxBidiTtsSession {
            input: MiniMaxBidiTtsInput {
                shared: shared.clone(),
                limits,
                metadata: metadata.clone(),
            },
            events: MiniMaxBidiTtsEvents {
                inbound: connection.inbound,
                shared,
                limits,
                metadata,
                last_session_id: session_id,
                last_trace_id: trace_id,
                last_connect_id: connect_id,
                task_finished_seen: false,
                task_failed_seen: false,
                eof_seen: false,
                terminal_receiver,
            },
        })
    }

    /// Connect after validating a voice reference from the matching MiniMax
    /// HTTP `/v1` scope. All checks happen before the WSS handshake.
    pub async fn connect_with_voice(
        &self,
        credentials: &MiniMaxBidiTtsCredentials,
        request: &MiniMaxBidiTtsRequest,
        voice: &MiniMaxVoiceRef,
        limits: MiniMaxBidiTtsLimits,
    ) -> Result<MiniMaxBidiTtsSession, MiniMaxBidiTtsError> {
        self.validate_voice_reference(voice)?;
        if request
            .parameters
            .timbre_weights
            .as_ref()
            .is_some_and(|weights| !weights.is_empty())
        {
            return Err(invalid(
                "connect_with_voice cannot override voice_setting.voice_id when timbre_weights are used",
            ));
        }
        let mut request = request.clone();
        request.parameters.voice_setting.voice_id = voice.voice_id().to_owned();
        self.connect(credentials, &request, limits).await
    }

    fn validate_voice_reference(&self, voice: &MiniMaxVoiceRef) -> Result<(), MiniMaxBidiTtsError> {
        let reference_scope = voice.scope();
        let expected_http_root = match self.scope.region {
            MiniMaxBidiTtsRegion::International => "https://api.minimax.io/v1",
            MiniMaxBidiTtsRegion::ChinaMainland => "https://api.minimax.cn/v1",
        };
        if MiniMaxVoiceRef::new(
            reference_scope.clone(),
            voice.kind(),
            voice.voice_id().to_owned(),
        )
        .is_err()
            || reference_scope.provider_id != self.scope.provider_id
            || reference_scope.profile_name != self.scope.profile_name
            || reference_scope.account_scope != self.scope.account_scope
            || reference_scope.region != self.scope.region
            || reference_scope.endpoint_fingerprint
                != provider_file_endpoint_fingerprint(expected_http_root)
        {
            return Err(invalid(
                "MiniMax voice reference does not match this provider, profile, account, region, or verified HTTP /v1 endpoint",
            ));
        }
        Ok(())
    }
}

/// Errors from a MiniMax Bidi T2A connection or task.
#[derive(Debug, Error)]
pub enum MiniMaxBidiTtsError {
    #[error("invalid MiniMax Bidi T2A request: {0}")]
    InvalidRequest(String),
    #[error("MiniMax API credential is empty, malformed, or too large")]
    InvalidCredential,
    #[error("MiniMax Bidi T2A timed out while waiting for {operation}")]
    Timeout { operation: &'static str },
    #[error("MiniMax Bidi T2A session is closed")]
    Closed,
    #[error("MiniMax Bidi T2A is waiting for a task control acknowledgement")]
    ControlPending,
    #[error("the active realtime transport does not support explicit WebSocket Ping frames")]
    PingUnsupported,
    #[error("the outcome of MiniMax {operation} is unknown: {reason}")]
    OutcomeUnknown {
        operation: &'static str,
        reason: String,
    },
    #[error("MiniMax Bidi T2A connection was interrupted (session {session_id:?}, trace {trace_id:?}, connect {connect_id:?}): {reason}")]
    Interrupted {
        session_id: Option<String>,
        trace_id: Option<String>,
        connect_id: Option<String>,
        reason: String,
    },
    #[error("MiniMax Bidi T2A protocol error: {message}")]
    Protocol {
        message: String,
        native: Option<Box<Value>>,
        raw_frame: Option<Bytes>,
    },
    #[error("MiniMax Bidi T2A provider error {status_code:?}: {status_message:?}")]
    Provider {
        status_code: Option<i64>,
        status_message: Option<String>,
        native: Value,
    },
    #[error("MiniMax Bidi T2A output exceeded the configured {max}-byte session limit")]
    AudioLimitExceeded { max: usize },
    #[error(transparent)]
    Realtime(#[from] RealtimeError),
}

fn validate_credential(credential: &Secret<String>) -> Result<(), MiniMaxBidiTtsError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || value.len() > MAX_CREDENTIAL_BYTES || value.contains(['\r', '\n'])
    {
        return Err(MiniMaxBidiTtsError::InvalidCredential);
    }
    Ok(())
}

fn bidi_endpoint(region: MiniMaxBidiTtsRegion) -> &'static str {
    match region {
        MiniMaxBidiTtsRegion::International => MINIMAX_BIDI_TTS_INTERNATIONAL_ENDPOINT,
        MiniMaxBidiTtsRegion::ChinaMainland => MINIMAX_BIDI_TTS_CHINA_ENDPOINT,
    }
}

async fn receive_setup_json(
    connection: &mut RealtimeConnection,
    deadline: Deadline,
    max_frame_bytes: usize,
    operation: &'static str,
) -> Result<Value, MiniMaxBidiTtsError> {
    let incoming = deadline
        .run(connection.inbound.next())
        .await
        .map_err(|_| MiniMaxBidiTtsError::Timeout { operation })?;
    let frame = incoming.ok_or(RealtimeError::UnexpectedRemoteClose)??;
    if frame.len() > max_frame_bytes {
        return Err(RealtimeError::FrameTooLarge {
            actual: frame.len(),
            max: max_frame_bytes,
        }
        .into());
    }
    let RealtimeFrame::Text(raw) = frame else {
        return Err(MiniMaxBidiTtsError::Protocol {
            message: "MiniMax setup events must be JSON text frames".into(),
            native: None,
            raw_frame: None,
        });
    };
    serde_json::from_slice(&raw).map_err(|_| MiniMaxBidiTtsError::Protocol {
        message: "MiniMax sent invalid JSON during setup".into(),
        native: None,
        raw_frame: Some(raw),
    })
}

fn validate_setup_status(native: &Value) -> Result<(), MiniMaxBidiTtsError> {
    validate_base_response_shape(native)
        .map_err(|message| protocol(message, Some(native.clone())))?;
    if provider_status(native).0.is_some_and(|status| status != 0) {
        Err(provider_error(native))
    } else {
        Ok(())
    }
}

fn validate_base_response_shape(native: &Value) -> Result<(), String> {
    let Some(base_resp) = native.get("base_resp") else {
        return Ok(());
    };
    let Some(base_resp) = base_resp.as_object() else {
        return Err("base_resp must be a JSON object when present".into());
    };
    if base_resp
        .get("status_code")
        .is_some_and(|value| !value.is_i64())
    {
        return Err("base_resp.status_code must be an integer when present".into());
    }
    if base_resp
        .get("status_msg")
        .is_some_and(|value| !value.is_string())
    {
        return Err("base_resp.status_msg must be a string when present".into());
    }
    Ok(())
}

fn provider_error(native: &Value) -> MiniMaxBidiTtsError {
    let (status_code, status_message) = provider_status(native);
    MiniMaxBidiTtsError::Provider {
        status_code,
        status_message,
        native: native.clone(),
    }
}

fn provider_status(native: &Value) -> (Option<i64>, Option<String>) {
    let response = native.get("base_resp");
    (
        response
            .and_then(|value| value.get("status_code"))
            .and_then(Value::as_i64),
        response
            .and_then(|value| value.get("status_msg"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    )
}

fn event_name(native: &Value) -> Option<&str> {
    native.get("event").and_then(Value::as_str)
}

fn string_field(native: &Value, field: &str) -> Option<String> {
    native.get(field).and_then(Value::as_str).map(str::to_owned)
}

struct DecodedAudioDelta {
    audio: Vec<u8>,
    is_final: bool,
    extra_info: Option<Value>,
}

fn decode_audio_delta(native: &Value) -> Result<Option<DecodedAudioDelta>, String> {
    let Some(data) = native.get("data") else {
        return Ok(None);
    };
    let Some(data) = data.as_object() else {
        return Err("task_continued data must be an object when present".into());
    };
    let Some(audio) = data.get("audio") else {
        return Ok(None);
    };
    let Some(audio) = audio.as_str() else {
        return Err("data.audio must be a hexadecimal string".into());
    };
    let is_final = native
        .get("is_final")
        .and_then(Value::as_bool)
        .ok_or_else(|| "audio event is missing boolean is_final".to_owned())?;
    let audio = decode_hex(audio)?;
    Ok(Some(DecodedAudioDelta {
        audio,
        is_final,
        extra_info: native.get("extra_info").cloned(),
    }))
}

fn has_audio_payload(native: &Value) -> bool {
    native
        .get("data")
        .and_then(Value::as_object)
        .is_some_and(|data| data.contains_key("audio"))
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("data.audio contains an odd number of hexadecimal characters".into());
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high =
            hex_nibble(pair[0]).ok_or_else(|| "data.audio is not valid hexadecimal".to_owned())?;
        let low =
            hex_nibble(pair[1]).ok_or_else(|| "data.audio is not valid hexadecimal".to_owned())?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn json_frame(value: &Value, max_frame_bytes: usize) -> Result<RealtimeFrame, MiniMaxBidiTtsError> {
    let bytes =
        serde_json::to_vec(value).map_err(|_| invalid("could not encode MiniMax Bidi event"))?;
    if bytes.len() > max_frame_bytes {
        return Err(RealtimeError::FrameTooLarge {
            actual: bytes.len(),
            max: max_frame_bytes,
        }
        .into());
    }
    Ok(RealtimeFrame::text(Bytes::from(bytes)))
}

fn ensure_ready(status: SessionStatus) -> Result<(), MiniMaxBidiTtsError> {
    match status {
        SessionStatus::Ready => Ok(()),
        SessionStatus::Failed | SessionStatus::Aborted | SessionStatus::Finished => {
            Err(MiniMaxBidiTtsError::Closed)
        }
        _ => Err(MiniMaxBidiTtsError::ControlPending),
    }
}

fn ensure_input_ready(
    shared: &SharedSession,
    status: SessionStatus,
) -> Result<(), MiniMaxBidiTtsError> {
    if shared.terminal.load(Ordering::Acquire) {
        return Err(MiniMaxBidiTtsError::Closed);
    }
    ensure_ready(status)
}

fn invalid(message: impl Into<String>) -> MiniMaxBidiTtsError {
    MiniMaxBidiTtsError::InvalidRequest(message.into())
}

fn protocol(message: impl Into<String>, native: Option<Value>) -> MiniMaxBidiTtsError {
    MiniMaxBidiTtsError::Protocol {
        message: message.into(),
        native: native.map(Box::new),
        raw_frame: None,
    }
}

/// Incremental events from the provider's WebSocket. Successful completion
/// requires both a valid `task_finished` event and clean EOF; `task_failed`
/// is surfaced once before the terminal stream returns `None`.
pub struct MiniMaxBidiTtsEvents {
    inbound: futures::stream::BoxStream<'static, Result<RealtimeFrame, RealtimeError>>,
    shared: Arc<SharedSession>,
    limits: MiniMaxBidiTtsLimits,
    metadata: Arc<MiniMaxBidiTtsMetadata>,
    last_session_id: Option<String>,
    last_trace_id: Option<String>,
    last_connect_id: Option<String>,
    task_finished_seen: bool,
    task_failed_seen: bool,
    eof_seen: bool,
    terminal_receiver: mpsc::UnboundedReceiver<()>,
}

impl fmt::Debug for MiniMaxBidiTtsEvents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxBidiTtsEvents")
            .field("metadata", &self.metadata)
            .field("limits", &self.limits)
            .field("task_finished_seen", &self.task_finished_seen)
            .field("eof_seen", &self.eof_seen)
            .finish_non_exhaustive()
    }
}

impl MiniMaxBidiTtsEvents {
    pub fn metadata(&self) -> &MiniMaxBidiTtsMetadata {
        &self.metadata
    }

    pub async fn next(&mut self) -> Result<Option<MiniMaxBidiTtsEvent>, MiniMaxBidiTtsError> {
        if self.eof_seen {
            return Ok(None);
        }
        if self.task_failed_seen {
            self.eof_seen = true;
            self.shared
                .close_sink_if_idle(RealtimeClose::normal("MiniMax Bidi T2A task failed"))
                .await;
            return Ok(None);
        }
        if self.shared.terminal.load(Ordering::Acquire) {
            self.eof_seen = true;
            self.shared
                .close_sink_if_idle(RealtimeClose::normal(
                    "MiniMax Bidi T2A session outcome is unknown",
                ))
                .await;
            let mut state = self.shared.state.lock().await;
            if !matches!(
                state.status,
                SessionStatus::Aborted | SessionStatus::Finished
            ) {
                state.status = SessionStatus::Failed;
            }
            state.finish_send_result.take();
            drop(state);
            self.inbound = futures::stream::empty().boxed();
            return Err(self.interrupted(
                "an outbound operation was canceled or failed before its result was confirmed",
            ));
        }
        let frame = match select(
            Box::pin(self.inbound.next()),
            Box::pin(self.terminal_receiver.next()),
        )
        .await
        {
            Either::Left((Some(frame), _)) => frame,
            Either::Left((None, _)) => {
                self.inbound = futures::stream::empty().boxed();
                return self.finish_at_eof().await;
            }
            Either::Right((_, _)) => return self.finish_after_terminal_signal().await,
        };
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                {
                    let mut state = self.shared.state.lock().await;
                    state.status = SessionStatus::Failed;
                    state.finish_send_result.take();
                }
                self.shared.request_terminal();
                self.shared
                    .close_sink_if_idle(RealtimeClose::normal("MiniMax Bidi T2A transport failed"))
                    .await;
                self.inbound = futures::stream::empty().boxed();
                return Err(self.interrupted(error.to_string()));
            }
        };
        if frame.len() > self.limits.max_frame_bytes {
            {
                let mut state = self.shared.state.lock().await;
                state.status = SessionStatus::Failed;
                state.finish_send_result.take();
            }
            self.shared.request_terminal();
            self.shared
                .close_sink_if_idle(RealtimeClose::normal(
                    "MiniMax Bidi T2A frame limit exceeded",
                ))
                .await;
            self.inbound = futures::stream::empty().boxed();
            return Err(RealtimeError::FrameTooLarge {
                actual: frame.len(),
                max: self.limits.max_frame_bytes,
            }
            .into());
        }
        let raw = match frame {
            RealtimeFrame::Text(raw) => raw,
            RealtimeFrame::Binary(raw) => {
                return Ok(Some(self.malformed_frame(
                    raw,
                    None,
                    "MiniMax Bidi T2A events must be JSON text frames",
                )))
            }
        };
        let native: Value = match serde_json::from_slice(&raw) {
            Ok(native) => native,
            Err(_) => {
                return Ok(Some(self.malformed_frame(
                    raw,
                    None,
                    "MiniMax sent invalid JSON",
                )))
            }
        };
        if !native.is_object() {
            return Ok(Some(self.event(
                MiniMaxBidiTtsEventKind::Malformed {
                    event: None,
                    reason: "provider event must be a JSON object".into(),
                },
                Some(native),
                None,
            )));
        }

        let event_name = event_name(&native).map(str::to_owned);
        if native.get("event").is_some_and(|event| !event.is_string()) {
            return Ok(Some(self.event(
                MiniMaxBidiTtsEventKind::Malformed {
                    event: None,
                    reason: "event must be a string when present".into(),
                },
                Some(native),
                None,
            )));
        }
        self.update_ids(&native);
        if let Err(reason) = validate_base_response_shape(&native) {
            return Ok(Some(self.event(
                MiniMaxBidiTtsEventKind::Malformed {
                    event: event_name,
                    reason,
                },
                Some(native),
                None,
            )));
        }
        let (status_code, status_message) = provider_status(&native);
        if let Some(code) = status_code.filter(|code| *code != 0) {
            if matches!(code, 2204 | 2205) {
                return Ok(Some(self.event(
                    MiniMaxBidiTtsEventKind::SoftError {
                        status_code: code,
                        status_message,
                    },
                    Some(native),
                    None,
                )));
            }
            self.fail_provider_task().await;
            return Ok(Some(self.event(
                MiniMaxBidiTtsEventKind::TaskFailed {
                    status_code: Some(code),
                    status_message,
                },
                Some(native),
                None,
            )));
        }

        let kind = match event_name.as_deref() {
            Some("task_failed") => {
                self.fail_provider_task().await;
                MiniMaxBidiTtsEventKind::TaskFailed {
                    status_code,
                    status_message,
                }
            }
            Some("sentence_start") => MiniMaxBidiTtsEventKind::SentenceStart,
            Some("sentence_end") => MiniMaxBidiTtsEventKind::SentenceEnd,
            Some("task_canceled") => {
                let mut state = self.shared.state.lock().await;
                match state.status {
                    SessionStatus::CancelPending => state.status = SessionStatus::Ready,
                    SessionStatus::SendingCancel => {
                        state.status = SessionStatus::CancelAckedWhileSending
                    }
                    _ => {
                        drop(state);
                        return Ok(Some(self.event(
                            MiniMaxBidiTtsEventKind::Malformed {
                                event: event_name,
                                reason:
                                    "task_canceled arrived without a pending task_cancel".into(),
                            },
                            Some(native),
                            None,
                        )));
                    }
                }
                MiniMaxBidiTtsEventKind::TaskCanceled
            }
            Some("task_flushed") => {
                let mut state = self.shared.state.lock().await;
                match state.status {
                    SessionStatus::FlushPending => state.status = SessionStatus::Ready,
                    SessionStatus::SendingFlush => {
                        state.status = SessionStatus::FlushAckedWhileSending
                    }
                    // A cancel can supersede a pending flush. Preserve a late
                    // acknowledgement but do not let it release the cancel gate.
                    SessionStatus::SendingCancel | SessionStatus::CancelPending => {}
                    _ => {
                        drop(state);
                        return Ok(Some(self.event(
                            MiniMaxBidiTtsEventKind::Malformed {
                                event: event_name,
                                reason:
                                    "task_flushed arrived without a matching pending flush".into(),
                            },
                            Some(native),
                            None,
                        )));
                    }
                }
                MiniMaxBidiTtsEventKind::TaskFlushed
            }
            Some("task_finished") => {
                let mut state = self.shared.state.lock().await;
                match state.status {
                    SessionStatus::FinishPending => state.status = SessionStatus::FinishAcked,
                    SessionStatus::SendingFinish => {
                        state.status = SessionStatus::FinishAckedWhileSending
                    }
                    _ => {
                        drop(state);
                        return Ok(Some(self.event(
                            MiniMaxBidiTtsEventKind::Malformed {
                                event: event_name,
                                reason:
                                    "task_finished arrived without a pending task_finish".into(),
                            },
                            Some(native),
                            None,
                        )));
                    }
                }
                self.task_finished_seen = true;
                MiniMaxBidiTtsEventKind::TaskFinished
            }
            event
                if event == Some("task_continued")
                    || (native.get("event").is_none() && has_audio_payload(&native)) =>
            {
                match decode_audio_delta(&native) {
                    Ok(Some(delta)) => {
                        let mut state = self.shared.state.lock().await;
                        let next_total = state.audio_bytes.saturating_add(delta.audio.len());
                        if next_total > self.limits.max_audio_bytes_per_session {
                            state.status = SessionStatus::Failed;
                            state.finish_send_result.take();
                            drop(state);
                            self.shared.request_terminal();
                            self.shared
                                .close_sink_if_idle(RealtimeClose::normal(
                                    "MiniMax Bidi T2A audio limit exceeded",
                                ))
                                .await;
                            self.inbound = futures::stream::empty().boxed();
                            return Err(MiniMaxBidiTtsError::AudioLimitExceeded {
                                max: self.limits.max_audio_bytes_per_session,
                            });
                        }
                        state.audio_bytes = next_total;
                        MiniMaxBidiTtsEventKind::AudioDelta {
                            audio: Bytes::from(delta.audio),
                            is_final: delta.is_final,
                            extra_info: delta.extra_info,
                        }
                    }
                    Ok(None) => MiniMaxBidiTtsEventKind::Native { event: event_name },
                    Err(reason) => {
                        return Ok(Some(self.event(
                            MiniMaxBidiTtsEventKind::Malformed {
                                event: event_name,
                                reason,
                            },
                            Some(native),
                            None,
                        )))
                    }
                }
            }
            _ => MiniMaxBidiTtsEventKind::Native { event: event_name },
        };
        Ok(Some(self.event(kind, Some(native), None)))
    }

    async fn finish_after_terminal_signal(
        &mut self,
    ) -> Result<Option<MiniMaxBidiTtsEvent>, MiniMaxBidiTtsError> {
        self.eof_seen = true;
        self.inbound = futures::stream::empty().boxed();
        self.shared
            .close_sink_if_idle(RealtimeClose::normal(
                "MiniMax Bidi T2A session became terminal",
            ))
            .await;
        {
            let mut state = self.shared.state.lock().await;
            if !matches!(
                state.status,
                SessionStatus::Aborted | SessionStatus::Finished
            ) {
                state.status = SessionStatus::Failed;
            }
            state.finish_send_result.take();
        }
        Err(self.interrupted(
            "the MiniMax Bidi T2A session became terminal while waiting for a provider event",
        ))
    }

    async fn fail_provider_task(&mut self) {
        {
            let mut state = self.shared.state.lock().await;
            state.status = SessionStatus::Failed;
            state.finish_send_result.take();
        }
        self.task_failed_seen = true;
        self.shared.request_terminal();
        self.shared
            .close_sink_if_idle(RealtimeClose::normal("MiniMax Bidi T2A task failed"))
            .await;
        self.inbound = futures::stream::empty().boxed();
    }

    async fn finish_at_eof(&mut self) -> Result<Option<MiniMaxBidiTtsEvent>, MiniMaxBidiTtsError> {
        loop {
            let (finish_send_result, completed) = {
                let mut state = self.shared.state.lock().await;
                match state.status {
                    SessionStatus::FinishAckedWhileSending => {
                        (state.finish_send_result.take(), false)
                    }
                    SessionStatus::FinishAcked => {
                        state.status = SessionStatus::Finished;
                        (None, true)
                    }
                    SessionStatus::Failed if self.task_failed_seen => (None, true),
                    SessionStatus::Finished => (None, true),
                    _ => {
                        state.status = SessionStatus::Failed;
                        (None, false)
                    }
                }
            };
            if completed {
                self.eof_seen = true;
                self.shared
                    .close_sink_if_idle(RealtimeClose::normal("MiniMax Bidi T2A task reached EOF"))
                    .await;
                return Ok(None);
            }
            if let Some(receiver) = finish_send_result {
                match receiver.await {
                    Ok(true) => continue,
                    Ok(false) | Err(_) => {
                        self.shared.state.lock().await.status = SessionStatus::Failed;
                        self.shared.request_terminal();
                        self.eof_seen = true;
                        self.shared
                            .close_sink_if_idle(RealtimeClose::normal(
                                "MiniMax Bidi T2A task_finish outcome is unknown",
                            ))
                            .await;
                        return Err(MiniMaxBidiTtsError::OutcomeUnknown {
                            operation: "task_finish",
                            reason: "task_finished arrived before task_finish send was confirmed"
                                .into(),
                        });
                    }
                }
            }
            self.shared.request_terminal();
            self.eof_seen = true;
            self.shared
                .close_sink_if_idle(RealtimeClose::normal(
                    "MiniMax Bidi T2A closed before confirmed completion",
                ))
                .await;
            return Err(self.interrupted(if self.task_finished_seen {
                "WebSocket closed while task_finish send was unresolved"
            } else {
                "WebSocket closed before a valid task_finished event"
            }));
        }
    }

    fn event(
        &self,
        kind: MiniMaxBidiTtsEventKind,
        native: Option<Value>,
        raw_frame: Option<Bytes>,
    ) -> MiniMaxBidiTtsEvent {
        MiniMaxBidiTtsEvent {
            kind,
            session_id: self.last_session_id.clone(),
            trace_id: self.last_trace_id.clone(),
            connect_id: self.last_connect_id.clone(),
            native,
            raw_frame,
        }
    }

    fn malformed_frame(
        &self,
        raw_frame: Bytes,
        event: Option<String>,
        reason: &str,
    ) -> MiniMaxBidiTtsEvent {
        self.event(
            MiniMaxBidiTtsEventKind::Malformed {
                event,
                reason: reason.into(),
            },
            None,
            Some(raw_frame),
        )
    }

    fn update_ids(&mut self, native: &Value) {
        if let Some(value) = string_field(native, "session_id") {
            self.last_session_id = Some(value);
        }
        if let Some(value) = string_field(native, "trace_id") {
            self.last_trace_id = Some(value);
        }
        if let Some(value) = string_field(native, "connect_id") {
            self.last_connect_id = Some(value);
        }
    }

    fn interrupted(&self, reason: impl Into<String>) -> MiniMaxBidiTtsError {
        MiniMaxBidiTtsError::Interrupted {
            session_id: self.last_session_id.clone(),
            trace_id: self.last_trace_id.clone(),
            connect_id: self.last_connect_id.clone(),
            reason: reason.into(),
        }
    }
}
