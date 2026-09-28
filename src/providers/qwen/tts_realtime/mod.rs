//! Standalone Qwen-TTS Realtime: incremental text input and audio output.
//!
//! This is neither Qwen-Omni nor Qwen-Audio-TTS/CosyVoice. The host polls the
//! input and event halves concurrently; there is no background driver,
//! automatic replay, reconnect, device playback or local speech model.

use crate::{
    client::RequestOptions,
    files::provider_file_endpoint_fingerprint,
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
    runtime::Deadline,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::Bytes;
use futures::{
    channel::mpsc,
    future::{select, Either},
    lock::Mutex as AsyncMutex,
    stream::BoxStream,
    StreamExt,
};
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::Duration,
};

pub use crate::providers::qwen::tts::{QwenTtsLanguage, QwenTtsRegion as QwenTtsRealtimeRegion};

pub const QWEN_TTS_REALTIME_BEIJING_ENDPOINT: &str =
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime";
pub const QWEN_TTS_REALTIME_SINGAPORE_ENDPOINT: &str =
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime";
const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// Regional route and non-secret identity. Each connect call supplies fresh
/// credentials through `RequestOptions`; workspace is sent in the documented
/// `X-DashScope-WorkSpace` handshake header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsRealtimeConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: QwenTtsRealtimeRegion,
    pub workspace_id: String,
    pub connect_timeout: Duration,
}

impl QwenTtsRealtimeConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenTtsRealtimeRegion,
        workspace_id: impl Into<String>,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
            connect_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsRealtimeScope {
    pub profile_name: String,
    pub account_scope: String,
    pub region: QwenTtsRealtimeRegion,
    pub workspace_id: String,
    pub endpoint_fingerprint: String,
}

/// Voice origin is explicit: custom cloning/design voices cannot be silently
/// used with system-voice models. Account ownership and voice availability
/// remain provider-authoritative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenTtsRealtimeVoice {
    System(String),
    Cloned(String),
    Designed(String),
}

impl QwenTtsRealtimeVoice {
    pub fn id(&self) -> &str {
        match self {
            Self::System(id) | Self::Cloned(id) | Self::Designed(id) => id,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QwenTtsRealtimeMode {
    #[default]
    ServerCommit,
    Commit,
}

impl QwenTtsRealtimeMode {
    fn wire(self) -> &'static str {
        match self {
            Self::ServerCommit => "server_commit",
            Self::Commit => "commit",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QwenTtsRealtimeFormat {
    #[default]
    Pcm,
    Wav,
    Mp3,
    Opus,
}

impl QwenTtsRealtimeFormat {
    fn wire(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
        }
    }
}

/// Model names are validated against the documented regional catalog. All
/// current stable and dated Qwen-TTS Realtime IDs are supported. Instructions
/// are only available on the Instruct family; the provider enforces its 1600
/// token limit and Chinese/English instruction-language requirement.
#[derive(Clone, PartialEq)]
pub struct QwenTtsRealtimeRequest {
    pub model: String,
    pub voice: QwenTtsRealtimeVoice,
    pub mode: QwenTtsRealtimeMode,
    pub language_type: QwenTtsLanguage,
    pub format: QwenTtsRealtimeFormat,
    pub sample_rate: u32,
    pub speech_rate: Option<f64>,
    pub volume: Option<u8>,
    pub pitch_rate: Option<f64>,
    pub bit_rate: Option<u16>,
    pub instructions: Option<String>,
    pub optimize_instructions: Option<bool>,
}

impl QwenTtsRealtimeRequest {
    pub fn new(model: impl Into<String>, voice: QwenTtsRealtimeVoice) -> Self {
        Self {
            model: model.into(),
            voice,
            mode: Default::default(),
            language_type: QwenTtsLanguage::Auto,
            format: Default::default(),
            sample_rate: 24000,
            speech_rate: None,
            volume: None,
            pitch_rate: None,
            bit_rate: None,
            instructions: None,
            optimize_instructions: None,
        }
    }
}

impl fmt::Debug for QwenTtsRealtimeRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsRealtimeRequest")
            .field("model", &self.model)
            .field("voice", &self.voice)
            .field("mode", &self.mode)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .field(
                "instructions",
                &self.instructions.as_ref().map(|_| "<redacted>"),
            )
            .finish_non_exhaustive()
    }
}

/// Local resource limits, not provider quotas. Sends have no queue: an input
/// method awaits the transport, applying backpressure. Events are decoded one
/// at a time, without accumulating an audio artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenTtsRealtimeLimits {
    pub max_frame_bytes: usize,
    pub max_audio_bytes: usize,
    pub max_text_bytes: usize,
    pub max_setup_events: usize,
    pub max_active_responses: usize,
    pub operation_timeout: Duration,
    pub finish_timeout: Duration,
}

impl Default for QwenTtsRealtimeLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 4 * 1024 * 1024,
            max_audio_bytes: 512 * 1024 * 1024,
            max_text_bytes: 64 * 1024,
            max_setup_events: 16,
            max_active_responses: 64,
            operation_timeout: Duration::from_secs(30),
            finish_timeout: Duration::from_secs(60),
        }
    }
}

impl QwenTtsRealtimeLimits {
    fn validate(self) -> Result<(), QwenTtsRealtimeError> {
        if self.max_frame_bytes == 0
            || self.max_frame_bytes > 4 * 1024 * 1024
            || self.max_audio_bytes == 0
            || self.max_audio_bytes > 512 * 1024 * 1024
            || self.max_text_bytes == 0
            || self.max_text_bytes > 256 * 1024
            || self.max_setup_events == 0
            || self.max_setup_events > 32
            || self.max_active_responses == 0
            || self.max_active_responses > 128
            || self.operation_timeout.is_zero()
            || self.finish_timeout.is_zero()
        {
            return Err(invalid("invalid Qwen TTS Realtime resource limits"));
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub struct QwenTtsRealtimeMetadata {
    pub scope: QwenTtsRealtimeScope,
    pub session_id: String,
    pub model: String,
    pub format: QwenTtsRealtimeFormat,
    pub sample_rate: u32,
    pub created_native: Value,
    pub updated_native: Value,
    pub setup_events: Vec<Value>,
}

impl fmt::Debug for QwenTtsRealtimeMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsRealtimeMetadata")
            .field("scope", &self.scope)
            .field("session_id", &self.session_id)
            .field("model", &self.model)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, PartialEq)]
pub struct QwenTtsRealtimeEvent {
    pub kind: QwenTtsRealtimeEventKind,
    /// Exact event, including response status, usage and provider extensions.
    pub native: Value,
}

impl fmt::Debug for QwenTtsRealtimeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsRealtimeEvent")
            .field("kind", &self.kind)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum QwenTtsRealtimeEventKind {
    AudioDelta {
        data: Bytes,
        response_id: String,
        item_id: String,
    },
    AudioDone {
        response_id: String,
        item_id: String,
    },
    ResponseDone {
        response_id: String,
        status: String,
    },
    BufferCommitted {
        item_id: String,
    },
    BufferCleared,
    /// Emitted only after confirmed `session.finish`, `session.finished`, and
    /// clean WebSocket EOF. Receiving the native event alone is insufficient.
    SessionFinished,
    Native,
}

impl fmt::Debug for QwenTtsRealtimeEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioDelta {
                data,
                response_id,
                item_id,
            } => f
                .debug_struct("AudioDelta")
                .field("audio_bytes", &data.len())
                .field("response_id", response_id)
                .field("item_id", item_id)
                .finish(),
            Self::AudioDone {
                response_id,
                item_id,
            } => f
                .debug_struct("AudioDone")
                .field("response_id", response_id)
                .field("item_id", item_id)
                .finish(),
            Self::ResponseDone {
                response_id,
                status,
            } => f
                .debug_struct("ResponseDone")
                .field("response_id", response_id)
                .field("status", status)
                .finish(),
            Self::BufferCommitted { item_id } => f
                .debug_struct("BufferCommitted")
                .field("item_id", item_id)
                .finish(),
            Self::BufferCleared => f.write_str("BufferCleared"),
            Self::SessionFinished => f.write_str("SessionFinished"),
            Self::Native => f.write_str("Native"),
        }
    }
}

#[derive(thiserror::Error)]
pub enum QwenTtsRealtimeError {
    #[error("invalid Qwen TTS Realtime input: {0}")]
    InvalidInput(String),
    #[error("Qwen TTS Realtime account scope does not match")]
    ScopeMismatch,
    #[error("Qwen TTS Realtime session is closed")]
    Closed,
    #[error("a Qwen TTS Realtime control is awaiting acknowledgement")]
    ControlPending,
    #[error("Qwen TTS Realtime {operation} outcome is unknown: {reason}")]
    OutcomeUnknown {
        operation: &'static str,
        reason: String,
    },
    #[error("Qwen TTS Realtime connection interrupted: {reason}")]
    Interrupted { reason: String },
    #[error("Qwen TTS Realtime provider error: {message}")]
    Provider { message: String, native: Box<Value> },
    #[error("invalid Qwen TTS Realtime event: {message}")]
    Protocol {
        message: String,
        native: Option<Box<Value>>,
        raw_frame: Option<Bytes>,
    },
    #[error("WebSocket Ping is unsupported by the injected transport")]
    PingUnsupported,
    #[error(transparent)]
    Realtime(#[from] RealtimeError),
}

impl fmt::Debug for QwenTtsRealtimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Raw payloads can contain audio, user text and server diagnostics.
        f.debug_tuple("QwenTtsRealtimeError")
            .field(&self.to_string())
            .finish()
    }
}

#[derive(Clone)]
pub struct QwenTtsRealtimeService {
    transport: Arc<dyn RealtimeTransport>,
    scope: QwenTtsRealtimeScope,
    endpoint: String,
    connect_timeout: Duration,
}

impl QwenTtsRealtimeService {
    pub fn new(
        transport: Arc<dyn RealtimeTransport>,
        config: QwenTtsRealtimeConfig,
    ) -> Result<Self, QwenTtsRealtimeError> {
        if config.profile_name.trim().is_empty()
            || config.account_scope.trim().is_empty()
            || config.connect_timeout.is_zero()
            || !valid_workspace(&config.workspace_id)
        {
            return Err(invalid(
                "profile, account, DNS-label workspace and positive connect timeout are required",
            ));
        }
        let endpoint = match config.region {
            QwenTtsRealtimeRegion::Beijing => QWEN_TTS_REALTIME_BEIJING_ENDPOINT,
            QwenTtsRealtimeRegion::Singapore => QWEN_TTS_REALTIME_SINGAPORE_ENDPOINT,
        }
        .to_owned();
        Ok(Self {
            transport,
            scope: QwenTtsRealtimeScope {
                profile_name: config.profile_name,
                account_scope: config.account_scope,
                region: config.region,
                workspace_id: config.workspace_id,
                endpoint_fingerprint: provider_file_endpoint_fingerprint(&endpoint),
            },
            endpoint,
            connect_timeout: config.connect_timeout,
        })
    }

    pub fn scope(&self) -> &QwenTtsRealtimeScope {
        &self.scope
    }

    /// Wait for `session.created` and the acknowledgement of `session.update`
    /// before exposing a writable session. Setup and its unknown events remain
    /// available in session metadata. Credentials are never stored.
    pub async fn connect(
        &self,
        options: &RequestOptions,
        request: &QwenTtsRealtimeRequest,
        limits: QwenTtsRealtimeLimits,
    ) -> Result<QwenTtsRealtimeSession, QwenTtsRealtimeError> {
        limits.validate()?;
        validate_request(request, self.scope.region, limits)?;
        if options
            .account_scope
            .as_deref()
            .is_some_and(|scope| scope != self.scope.account_scope)
        {
            return Err(QwenTtsRealtimeError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| invalid("a regional API key is required"))?;
        let key = credential.expose_secret();
        if key.trim().is_empty()
            || key.len() > 16 * 1024
            || key.bytes().any(|byte| byte.is_ascii_control())
        {
            return Err(invalid(
                "API key is empty, too large, or contains control characters",
            ));
        }
        let setup = session_update(request)?;
        let frame = json_frame(&setup, limits.max_frame_bytes)?;
        let connect_request = RealtimeConnectRequest {
            endpoint: format!("{}?model={}", self.endpoint, request.model),
            headers: vec![
                ("Authorization".into(), format!("Bearer {key}")),
                (
                    "X-DashScope-WorkSpace".into(),
                    self.scope.workspace_id.clone(),
                ),
            ],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let deadline = Deadline::after(Some(self.connect_timeout));
        let mut connection = deadline
            .run(self.transport.connect(connect_request))
            .await
            .map_err(|_| interrupted("WebSocket handshake timed out"))??;
        let created_native =
            receive_setup(&mut connection, deadline, limits.max_frame_bytes).await?;
        if event_name(&created_native)? != "session.created" {
            return Err(protocol(
                "first server event must be session.created",
                Some(created_native),
            ));
        }
        let session_id = required_string(&created_native, "/session/id")?.to_owned();
        deadline
            .run(connection.outbound.send(frame))
            .await
            .map_err(|_| unknown("session.update", "setup send timed out"))?
            .map_err(|error| unknown("session.update", &error.to_string()))?;
        let mut setup_events = Vec::new();
        let updated_native = loop {
            let native = receive_setup(&mut connection, deadline, limits.max_frame_bytes).await?;
            if event_name(&native)? == "session.updated" {
                break native;
            }
            if setup_events.len() >= limits.max_setup_events {
                return Err(protocol(
                    "setup acknowledgement event limit exceeded",
                    Some(native),
                ));
            }
            setup_events.push(native);
        };
        validate_updated(&updated_native, request, &session_id)?;
        let metadata = Arc::new(QwenTtsRealtimeMetadata {
            scope: self.scope.clone(),
            session_id,
            model: request.model.clone(),
            format: request.format,
            sample_rate: request.sample_rate,
            created_native,
            updated_native,
            setup_events,
        });
        let (reader_signal, notices) = mpsc::unbounded();
        let (writer_signal, writer_notices) = mpsc::unbounded();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                status: Status::Ready,
                pending: None,
                finish_deadline: Deadline::default(),
                may_have_buffered_text: false,
            }),
            terminal: AtomicBool::new(false),
            sink: AsyncMutex::new(Some(connection.outbound)),
            reader_signal,
            writer_signal,
            writer_notices: AsyncMutex::new(writer_notices),
        });
        Ok(QwenTtsRealtimeSession {
            input: QwenTtsRealtimeInput {
                shared: shared.clone(),
                metadata: metadata.clone(),
                limits,
                mode: request.mode,
            },
            events: QwenTtsRealtimeEvents {
                shared,
                metadata,
                limits,
                mode: request.mode,
                inbound: connection.inbound,
                notices,
                active_responses: BTreeSet::new(),
                audio_bytes: 0,
                finished_native: None,
                eof_received: false,
                pending_terminal: None,
                done: false,
            },
        })
    }
}

pub struct QwenTtsRealtimeSession {
    input: QwenTtsRealtimeInput,
    events: QwenTtsRealtimeEvents,
}
impl QwenTtsRealtimeSession {
    pub fn metadata(&self) -> &QwenTtsRealtimeMetadata {
        &self.input.metadata
    }
    pub fn into_parts(self) -> (QwenTtsRealtimeInput, QwenTtsRealtimeEvents) {
        (self.input, self.events)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Ready,
    Finishing,
    Finished,
    Failed,
    Aborted,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Append,
    Commit,
    Clear,
    Finish,
    Ping,
}
impl Operation {
    fn wire(self) -> &'static str {
        match self {
            Self::Append => "input_text_buffer.append",
            Self::Commit => "input_text_buffer.commit",
            Self::Clear => "input_text_buffer.clear",
            Self::Finish => "session.finish",
            Self::Ping => "WebSocket Ping",
        }
    }
}
struct Pending {
    operation: Operation,
    sent: bool,
    acked: bool,
}
struct State {
    status: Status,
    pending: Option<Pending>,
    finish_deadline: Deadline,
    may_have_buffered_text: bool,
}
struct Shared {
    state: Mutex<State>,
    terminal: AtomicBool,
    sink: AsyncMutex<Option<Box<dyn RealtimeSink>>>,
    reader_signal: mpsc::UnboundedSender<()>,
    writer_signal: mpsc::UnboundedSender<()>,
    writer_notices: AsyncMutex<mpsc::UnboundedReceiver<()>>,
}
impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn stop(&self, status: Status) {
        {
            let mut state = self.state();
            if !matches!(state.status, Status::Finished | Status::Aborted) {
                state.status = status;
            }
        }
        if !self.terminal.swap(true, Ordering::AcqRel) {
            let _ = self.reader_signal.unbounded_send(());
            let _ = self.writer_signal.unbounded_send(());
        }
    }
    fn release_if_idle(&self) {
        if let Some(mut sink) = self.sink.try_lock() {
            sink.take();
        }
    }
    async fn close_if_idle(&self) {
        if let Some(mut sink) = self.sink.try_lock() {
            close_sink(&mut sink).await;
        }
    }
}
struct WriteGuard {
    shared: Arc<Shared>,
    armed: bool,
}
impl Drop for WriteGuard {
    fn drop(&mut self) {
        if self.armed {
            self.shared.stop(Status::Failed);
        }
    }
}

/// Single writer. Poll this concurrently with the event half. Canceling an
/// in-flight send fails the session closed and wakes a blocked event reader.
pub struct QwenTtsRealtimeInput {
    shared: Arc<Shared>,
    metadata: Arc<QwenTtsRealtimeMetadata>,
    limits: QwenTtsRealtimeLimits,
    mode: QwenTtsRealtimeMode,
}
impl QwenTtsRealtimeInput {
    pub fn metadata(&self) -> &QwenTtsRealtimeMetadata {
        &self.metadata
    }
    pub async fn append_text(&mut self, text: &str) -> Result<(), QwenTtsRealtimeError> {
        if text.is_empty() || text.len() > self.limits.max_text_bytes {
            return Err(invalid("text must be nonempty and within max_text_bytes"));
        }
        self.send(Operation::Append, Some(text), None).await
    }
    /// Explicit commit is valid in both server_commit and commit modes.
    pub async fn commit(&mut self) -> Result<(), QwenTtsRealtimeError> {
        {
            let state = self.shared.state();
            if state.status != Status::Ready || self.shared.terminal.load(Ordering::Acquire) {
                return Err(QwenTtsRealtimeError::Closed);
            }
            if !state.may_have_buffered_text {
                return Err(invalid("cannot commit a known-empty text buffer"));
            }
        }
        self.send(Operation::Commit, None, None).await
    }
    /// Clear only uncommitted text. Already created audio responses continue.
    pub async fn clear_buffer(&mut self) -> Result<(), QwenTtsRealtimeError> {
        self.send(Operation::Clear, None, None).await
    }
    /// Stop accepting text and flush the tail. Success is confirmed by the
    /// event half only after session.finished and clean EOF.
    pub async fn finish(&mut self) -> Result<(), QwenTtsRealtimeError> {
        self.send(Operation::Finish, None, None).await
    }
    /// Transport-level Ping only. No provider JSON ping or timer is invented.
    pub async fn ping(&mut self, payload: Bytes) -> Result<(), QwenTtsRealtimeError> {
        if payload.len() > 125 {
            return Err(invalid("WebSocket Ping payload exceeds 125 bytes"));
        }
        self.send(Operation::Ping, None, Some(payload)).await
    }
    pub async fn abort(&mut self) -> Result<(), QwenTtsRealtimeError> {
        self.shared.stop(Status::Aborted);
        self.shared.close_if_idle().await;
        Ok(())
    }

    async fn send(
        &mut self,
        operation: Operation,
        text: Option<&str>,
        ping: Option<Bytes>,
    ) -> Result<(), QwenTtsRealtimeError> {
        {
            let state = self.shared.state();
            if state.status != Status::Ready || self.shared.terminal.load(Ordering::Acquire) {
                return Err(QwenTtsRealtimeError::Closed);
            }
            if state.pending.is_some() {
                return Err(QwenTtsRealtimeError::ControlPending);
            }
        }
        let id = event_id()?;
        let mut native = json!({"event_id":id, "type": operation.wire()});
        if let Some(text) = text {
            native["text"] = json!(text);
        }
        let frame = json_frame(&native, self.limits.max_frame_bytes)?;
        {
            let mut state = self.shared.state();
            if state.status != Status::Ready || self.shared.terminal.load(Ordering::Acquire) {
                return Err(QwenTtsRealtimeError::Closed);
            }
            state.pending = Some(Pending {
                operation,
                sent: false,
                acked: false,
            });
            if operation == Operation::Finish {
                state.status = Status::Finishing;
                state.finish_deadline = Deadline::after(Some(self.limits.finish_timeout));
                // Finish starts/finishes once; terminal notification also occurs once.
                let _ = self.shared.reader_signal.unbounded_send(());
            }
        }
        let mut guard = WriteGuard {
            shared: self.shared.clone(),
            armed: true,
        };
        let result = self.write(frame, ping).await;
        if operation == Operation::Ping
            && matches!(result, Err(RealtimeError::InvalidInput { .. }))
            && !self.shared.terminal.load(Ordering::Acquire)
        {
            self.shared.state().pending = None;
            guard.armed = false;
            return Err(QwenTtsRealtimeError::PingUnsupported);
        }
        match result {
            Ok(()) => {
                if self.shared.terminal.load(Ordering::Acquire) {
                    guard.armed = false;
                    return Err(QwenTtsRealtimeError::Closed);
                }
                {
                    let mut state = self.shared.state();
                    if let Some(pending) = &mut state.pending {
                        pending.sent = true;
                        if operation != Operation::Finish
                            && (pending.acked
                                || matches!(operation, Operation::Append | Operation::Ping))
                        {
                            state.pending = None;
                        }
                    }
                    if operation == Operation::Append {
                        state.may_have_buffered_text = true;
                    }
                    if operation == Operation::Clear
                        || (operation == Operation::Commit
                            && self.mode == QwenTtsRealtimeMode::Commit)
                    {
                        state.may_have_buffered_text = false;
                    }
                    // Automatic server commits have no correlation to local
                    // appends. Preserve the unknown/nonempty state until an
                    // explicit clear is acknowledged.
                }
                if operation == Operation::Finish {
                    let _ = self.shared.reader_signal.unbounded_send(());
                }
                guard.armed = false;
                Ok(())
            }
            Err(error) => {
                self.shared.stop(Status::Failed);
                self.shared.close_if_idle().await;
                guard.armed = false;
                Err(unknown(operation.wire(), &error.to_string()))
            }
        }
    }

    async fn write(&self, frame: RealtimeFrame, ping: Option<Bytes>) -> Result<(), RealtimeError> {
        let mut slot = self.shared.sink.lock().await;
        if self.shared.terminal.load(Ordering::Acquire) {
            close_sink(&mut slot).await;
            return Err(RealtimeError::Closed);
        }
        let sink = slot.as_mut().ok_or(RealtimeError::Closed)?;
        let mut notices = self.shared.writer_notices.lock().await;
        let operation = async {
            if let Some(payload) = ping {
                sink.ping(payload).await
            } else {
                sink.send(frame).await
            }
        };
        let result = match Deadline::after(Some(self.limits.operation_timeout))
            .run(select(Box::pin(operation), Box::pin(notices.next())))
            .await
        {
            Ok(Either::Left((result, _))) => result,
            Ok(Either::Right((_, _))) => Err(RealtimeError::Closed),
            Err(_) => Err(RealtimeError::Transport {
                message: "Qwen TTS Realtime write timed out".into(),
            }),
        };
        if self.shared.terminal.load(Ordering::Acquire) {
            close_sink(&mut slot).await;
            return Err(RealtimeError::Closed);
        }
        result
    }
}
impl Drop for QwenTtsRealtimeInput {
    fn drop(&mut self) {
        self.shared.stop(Status::Aborted);
        self.shared.release_if_idle();
    }
}

pub struct QwenTtsRealtimeEvents {
    shared: Arc<Shared>,
    metadata: Arc<QwenTtsRealtimeMetadata>,
    limits: QwenTtsRealtimeLimits,
    mode: QwenTtsRealtimeMode,
    inbound: BoxStream<'static, Result<RealtimeFrame, RealtimeError>>,
    notices: mpsc::UnboundedReceiver<()>,
    active_responses: BTreeSet<String>,
    audio_bytes: usize,
    finished_native: Option<Value>,
    eof_received: bool,
    pending_terminal: Option<Result<Option<QwenTtsRealtimeEvent>, QwenTtsRealtimeError>>,
    done: bool,
}
impl QwenTtsRealtimeEvents {
    pub fn metadata(&self) -> &QwenTtsRealtimeMetadata {
        &self.metadata
    }
    pub fn audio_bytes_received(&self) -> usize {
        self.audio_bytes
    }
    pub async fn next(&mut self) -> Result<Option<QwenTtsRealtimeEvent>, QwenTtsRealtimeError> {
        if self.done && self.pending_terminal.is_none() {
            return Ok(None);
        }
        if self.pending_terminal.is_none() {
            let result = self.next_inner().await;
            let terminal = match &result {
                Err(_) => true,
                Ok(Some(QwenTtsRealtimeEvent {
                    kind: QwenTtsRealtimeEventKind::SessionFinished,
                    ..
                })) => true,
                Ok(Some(QwenTtsRealtimeEvent {
                    kind: QwenTtsRealtimeEventKind::ResponseDone { status, .. },
                    ..
                })) => status != "completed",
                _ => false,
            };
            if !terminal {
                return result;
            }
            // Store before any cleanup await: canceling next() must not lose
            // its terminal event or error. A retry drains this exact result.
            self.pending_terminal = Some(result);
            self.done = true;
            self.inbound = futures::stream::empty().boxed();
            self.shared.stop(Status::Failed);
        }
        self.shared.close_if_idle().await;
        self.pending_terminal
            .take()
            .expect("terminal result retained")
    }

    async fn next_inner(&mut self) -> Result<Option<QwenTtsRealtimeEvent>, QwenTtsRealtimeError> {
        loop {
            if self.shared.terminal.load(Ordering::Acquire) {
                return Err(interrupted(
                    "session aborted or an outbound send was canceled",
                ));
            }
            if self.eof_received {
                return self.finish_at_eof().await;
            }
            let deadline = self.shared.state().finish_deadline;
            let next = deadline
                .run(select(
                    Box::pin(self.inbound.next()),
                    Box::pin(self.notices.next()),
                ))
                .await
                .map_err(|_| interrupted("timed out awaiting session.finished and clean EOF"))?;
            let frame = match next {
                Either::Right((_, _)) => continue,
                Either::Left((None, _)) => {
                    self.eof_received = true;
                    return self.finish_at_eof().await;
                }
                Either::Left((Some(frame), _)) => {
                    frame.map_err(|error| interrupted(&error.to_string()))?
                }
            };
            let native = parse_frame(frame, self.limits.max_frame_bytes)?;
            provider_error(&native)?;
            if self.finished_native.is_some() {
                return Err(protocol(
                    "data received after session.finished",
                    Some(native),
                ));
            }
            let event_type = event_name(&native)?;
            let kind = match event_type {
                "input_text_buffer.committed" => {
                    self.acknowledge(Operation::Commit, &native)?;
                    QwenTtsRealtimeEventKind::BufferCommitted {
                        item_id: native
                            .get("item_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                protocol("committed event is missing item_id", Some(native.clone()))
                            })?
                            .to_owned(),
                    }
                }
                "input_text_buffer.cleared" => {
                    self.acknowledge(Operation::Clear, &native)?;
                    QwenTtsRealtimeEventKind::BufferCleared
                }
                "response.created" => {
                    let id = required_string(&native, "/response/id")?.to_owned();
                    if self.active_responses.len() >= self.limits.max_active_responses
                        || !self.active_responses.insert(id)
                    {
                        return Err(protocol(
                            "duplicate response ID or active response limit exceeded",
                            Some(native),
                        ));
                    }
                    QwenTtsRealtimeEventKind::Native
                }
                "response.audio.delta" | "response.audio.done" => {
                    let response_id = required_string(&native, "/response_id")?.to_owned();
                    let item_id = required_string(&native, "/item_id")?.to_owned();
                    if !self.active_responses.contains(&response_id) {
                        return Err(protocol(
                            "audio event has no active response.created",
                            Some(native),
                        ));
                    }
                    if event_type == "response.audio.done" {
                        QwenTtsRealtimeEventKind::AudioDone {
                            response_id,
                            item_id,
                        }
                    } else {
                        let data = STANDARD
                            .decode(required_string(&native, "/delta")?)
                            .map_err(|_| {
                                protocol("audio delta is not valid Base64", Some(native.clone()))
                            })?;
                        if data.len() > self.limits.max_audio_bytes.saturating_sub(self.audio_bytes)
                        {
                            return Err(protocol(
                                "session audio byte limit exceeded",
                                Some(native),
                            ));
                        }
                        self.audio_bytes += data.len();
                        QwenTtsRealtimeEventKind::AudioDelta {
                            data: Bytes::from(data),
                            response_id,
                            item_id,
                        }
                    }
                }
                "response.done" => {
                    let response_id = required_string(&native, "/response/id")?.to_owned();
                    let status = required_string(&native, "/response/status")?.to_owned();
                    if !self.active_responses.remove(&response_id) {
                        return Err(protocol(
                            "response.done has no active response.created",
                            Some(native),
                        ));
                    }
                    if status != "completed" {
                        self.shared.stop(Status::Failed);
                    }
                    QwenTtsRealtimeEventKind::ResponseDone {
                        response_id,
                        status,
                    }
                }
                "session.finished" => {
                    if !self.active_responses.is_empty() {
                        return Err(protocol(
                            "session.finished precedes response.done",
                            Some(native),
                        ));
                    }
                    self.acknowledge(Operation::Finish, &native)?;
                    self.finished_native = Some(native);
                    continue;
                }
                "session.created" | "session.updated" => {
                    return Err(protocol(
                        "unexpected session reinitialization",
                        Some(native),
                    ))
                }
                _ => QwenTtsRealtimeEventKind::Native,
            };
            return Ok(Some(QwenTtsRealtimeEvent { kind, native }));
        }
    }

    fn acknowledge(
        &self,
        operation: Operation,
        native: &Value,
    ) -> Result<(), QwenTtsRealtimeError> {
        let mut state = self.shared.state();
        if let Some(pending) = &mut state.pending {
            if pending.operation == operation {
                if pending.acked {
                    return Err(protocol(
                        "duplicate control acknowledgement",
                        Some(native.clone()),
                    ));
                }
                pending.acked = true;
                if pending.sent && operation != Operation::Finish {
                    state.pending = None;
                }
                return Ok(());
            }
        }
        if operation == Operation::Commit
            && (self.mode == QwenTtsRealtimeMode::ServerCommit || state.status == Status::Finishing)
        {
            return Ok(());
        }
        Err(protocol(
            "control acknowledgement has no matching pending operation",
            Some(native.clone()),
        ))
    }

    async fn finish_at_eof(
        &mut self,
    ) -> Result<Option<QwenTtsRealtimeEvent>, QwenTtsRealtimeError> {
        loop {
            let (confirmed, sending, deadline) = {
                let state = self.shared.state();
                let finishing = state.status == Status::Finishing;
                let confirmed = finishing
                    && state.pending.as_ref().is_some_and(|pending| {
                        pending.operation == Operation::Finish && pending.sent && pending.acked
                    });
                let sending = finishing
                    && state.pending.as_ref().is_some_and(|pending| {
                        pending.operation == Operation::Finish && !pending.sent
                    });
                (confirmed, sending, state.finish_deadline)
            };
            if confirmed
                && self.finished_native.is_some()
                && !self.shared.terminal.load(Ordering::Acquire)
            {
                self.shared.stop(Status::Finished);
                return Ok(Some(QwenTtsRealtimeEvent {
                    kind: QwenTtsRealtimeEventKind::SessionFinished,
                    native: self.finished_native.take().expect("checked"),
                }));
            }
            if self.finished_native.is_some()
                && sending
                && !self.shared.terminal.load(Ordering::Acquire)
            {
                // Keep the notification receiver in self. Canceling next()
                // cannot consume and lose the pending send confirmation.
                if matches!(deadline.run(self.notices.next()).await, Ok(Some(()))) {
                    continue;
                }
            }
            return Err(interrupted(
                "WebSocket EOF without confirmed finish send and session.finished",
            ));
        }
    }
}
impl Drop for QwenTtsRealtimeEvents {
    fn drop(&mut self) {
        self.shared.stop(Status::Aborted);
        self.shared.release_if_idle();
    }
}

async fn close_sink(slot: &mut Option<Box<dyn RealtimeSink>>) {
    if let Some(mut sink) = slot.take() {
        let _ = Deadline::after(Some(CLOSE_TIMEOUT))
            .run(sink.close(RealtimeClose::normal("Qwen TTS Realtime closed locally")))
            .await;
    }
}

#[derive(PartialEq, Eq)]
enum Family {
    Flash,
    Instruct,
    Clone,
    Design,
    Legacy,
}
fn model_family(model: &str) -> Result<Family, QwenTtsRealtimeError> {
    Ok(match model {
        "qwen3-tts-flash-realtime"
        | "qwen3-tts-flash-realtime-2025-11-27"
        | "qwen3-tts-flash-realtime-2025-09-18" => Family::Flash,
        "qwen3-tts-instruct-flash-realtime" | "qwen3-tts-instruct-flash-realtime-2026-01-22" => {
            Family::Instruct
        }
        "qwen3-tts-vc-realtime-2026-01-15" | "qwen3-tts-vc-realtime-2025-11-27" => Family::Clone,
        "qwen3-tts-vd-realtime-2026-01-15" | "qwen3-tts-vd-realtime-2025-12-16" => Family::Design,
        "qwen-tts-realtime" | "qwen-tts-realtime-latest" | "qwen-tts-realtime-2025-07-15" => {
            Family::Legacy
        }
        _ => {
            return Err(invalid(
                "model is not a documented standalone Qwen-TTS Realtime model",
            ))
        }
    })
}

fn validate_request(
    request: &QwenTtsRealtimeRequest,
    region: QwenTtsRealtimeRegion,
    limits: QwenTtsRealtimeLimits,
) -> Result<(), QwenTtsRealtimeError> {
    let family = model_family(&request.model)?;
    if family == Family::Legacy && region != QwenTtsRealtimeRegion::Beijing {
        return Err(invalid(
            "legacy Qwen-TTS-Realtime is documented only in Beijing",
        ));
    }
    let voice_matches = matches!(
        (&family, &request.voice),
        (
            Family::Flash | Family::Instruct | Family::Legacy,
            QwenTtsRealtimeVoice::System(_)
        ) | (Family::Clone, QwenTtsRealtimeVoice::Cloned(_))
            | (Family::Design, QwenTtsRealtimeVoice::Designed(_))
    );
    if !voice_matches || request.voice.id().trim().is_empty() || request.voice.id().len() > 1024 {
        return Err(invalid(
            "voice ID is empty, too long, or its origin does not match the model family",
        ));
    }
    if !matches!(request.sample_rate, 8000 | 16000 | 24000 | 48000) {
        return Err(invalid("sample_rate must be 8000, 16000, 24000, or 48000"));
    }
    if family == Family::Legacy
        && (request.format != QwenTtsRealtimeFormat::Pcm
            || request.sample_rate != 24000
            || request.speech_rate.is_some()
            || request.volume.is_some()
            || request.pitch_rate.is_some()
            || request.bit_rate.is_some())
    {
        return Err(invalid("legacy Qwen-TTS-Realtime accepts PCM 24000 only, without speed, volume, pitch or bitrate controls"));
    }
    if request
        .speech_rate
        .is_some_and(|v| !v.is_finite() || !(0.5..=2.0).contains(&v))
        || request
            .pitch_rate
            .is_some_and(|v| !v.is_finite() || !(0.5..=2.0).contains(&v))
    {
        return Err(invalid(
            "speech_rate and pitch_rate must be finite values from 0.5 through 2.0",
        ));
    }
    if request.volume.is_some_and(|v| v > 100) {
        return Err(invalid("volume must be 0 through 100"));
    }
    if request.bit_rate.is_some_and(|v| !(6..=510).contains(&v))
        || (request.bit_rate.is_some() && request.format != QwenTtsRealtimeFormat::Opus)
    {
        return Err(invalid(
            "bit_rate requires Opus and must be 6 through 510 kbps",
        ));
    }
    if family != Family::Instruct
        && (request.instructions.is_some() || request.optimize_instructions.is_some())
    {
        return Err(invalid(
            "instructions and optimize_instructions require Qwen3-TTS-Instruct-Flash-Realtime",
        ));
    }
    if request
        .instructions
        .as_ref()
        .is_some_and(|s| s.len() > limits.max_text_bytes)
    {
        return Err(invalid("instructions exceed the local text-byte bound"));
    }
    Ok(())
}

fn session_update(request: &QwenTtsRealtimeRequest) -> Result<Value, QwenTtsRealtimeError> {
    let mut session = json!({
        "voice":request.voice.id(), "mode":request.mode.wire(),
        "language_type":language_wire(request.language_type),
        "response_format":request.format.wire(), "sample_rate":request.sample_rate,
    });
    for (key, value) in [
        ("speech_rate", request.speech_rate.map(|v| json!(v))),
        ("volume", request.volume.map(|v| json!(v))),
        ("pitch_rate", request.pitch_rate.map(|v| json!(v))),
        ("bit_rate", request.bit_rate.map(|v| json!(v))),
        (
            "instructions",
            request.instructions.as_ref().map(|v| json!(v)),
        ),
        (
            "optimize_instructions",
            request.optimize_instructions.map(|v| json!(v)),
        ),
    ] {
        if let Some(value) = value {
            session[key] = value;
        }
    }
    Ok(json!({"event_id":event_id()?, "type":"session.update", "session":session}))
}

fn validate_updated(
    native: &Value,
    request: &QwenTtsRealtimeRequest,
    session_id: &str,
) -> Result<(), QwenTtsRealtimeError> {
    if let Some(language) = native.pointer("/session/language_type") {
        if language.as_str() != Some(language_wire(request.language_type)) {
            return Err(protocol(
                "session.updated language differs from requested language",
                Some(native.clone()),
            ));
        }
    }
    let reported_model = required_string(native, "/session/model")?;
    if canonical_model(reported_model) != canonical_model(&request.model) {
        return Err(protocol(
            "session.updated model differs from requested model",
            Some(native.clone()),
        ));
    }
    for (path, expected) in [
        ("/session/id", session_id),
        ("/session/voice", request.voice.id()),
        ("/session/mode", request.mode.wire()),
        ("/session/response_format", request.format.wire()),
    ] {
        if native.pointer(path).and_then(Value::as_str) != Some(expected) {
            return Err(protocol(
                "session.updated did not acknowledge requested session configuration",
                Some(native.clone()),
            ));
        }
    }
    if native
        .pointer("/session/sample_rate")
        .and_then(Value::as_u64)
        != Some(request.sample_rate as u64)
    {
        return Err(protocol(
            "session.updated sample rate differs from request",
            Some(native.clone()),
        ));
    }
    Ok(())
}

// Documented aliases are permitted to echo their equivalent dated model ID.
fn canonical_model(model: &str) -> &str {
    match model {
        "qwen3-tts-flash-realtime" => "qwen3-tts-flash-realtime-2025-11-27",
        "qwen3-tts-instruct-flash-realtime" => "qwen3-tts-instruct-flash-realtime-2026-01-22",
        "qwen-tts-realtime" | "qwen-tts-realtime-latest" => "qwen-tts-realtime-2025-07-15",
        other => other,
    }
}

async fn receive_setup(
    connection: &mut RealtimeConnection,
    deadline: Deadline,
    max: usize,
) -> Result<Value, QwenTtsRealtimeError> {
    let frame = deadline
        .run(connection.inbound.next())
        .await
        .map_err(|_| interrupted("session setup timed out"))?
        .ok_or_else(|| interrupted("WebSocket ended during session setup"))??;
    let native = parse_frame(frame, max)?;
    provider_error(&native)?;
    Ok(native)
}
fn parse_frame(frame: RealtimeFrame, max: usize) -> Result<Value, QwenTtsRealtimeError> {
    if frame.len() > max {
        return Err(RealtimeError::FrameTooLarge {
            actual: frame.len(),
            max,
        }
        .into());
    }
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(protocol(
            "standalone Qwen TTS requires JSON text frames",
            None,
        ));
    };
    let native: Value =
        serde_json::from_slice(&bytes).map_err(|_| QwenTtsRealtimeError::Protocol {
            message: "invalid JSON frame".into(),
            native: None,
            raw_frame: Some(bytes.clone()),
        })?;
    event_name(&native)?;
    Ok(native)
}
fn event_name(native: &Value) -> Result<&str, QwenTtsRealtimeError> {
    required_string(native, "/type")
}
fn required_string<'a>(native: &'a Value, path: &str) -> Result<&'a str, QwenTtsRealtimeError> {
    native
        .pointer(path)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| protocol(&format!("missing or empty {path}"), Some(native.clone())))
}
fn provider_error(native: &Value) -> Result<(), QwenTtsRealtimeError> {
    if native.get("type").and_then(Value::as_str) == Some("error") {
        return Err(QwenTtsRealtimeError::Provider {
            message: native
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("provider rejected the event")
                .to_owned(),
            native: Box::new(native.clone()),
        });
    }
    Ok(())
}
fn json_frame(native: &Value, max: usize) -> Result<RealtimeFrame, QwenTtsRealtimeError> {
    let bytes =
        serde_json::to_vec(native).map_err(|_| invalid("could not encode provider event"))?;
    if bytes.len() > max {
        return Err(RealtimeError::FrameTooLarge {
            actual: bytes.len(),
            max,
        }
        .into());
    }
    Ok(RealtimeFrame::Text(Bytes::from(bytes)))
}
fn valid_workspace(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 63
        && bytes[0].is_ascii_alphanumeric()
        && bytes[bytes.len() - 1].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'-')
}
fn language_wire(language: QwenTtsLanguage) -> &'static str {
    match language {
        QwenTtsLanguage::Auto => "Auto",
        QwenTtsLanguage::Chinese => "Chinese",
        QwenTtsLanguage::English => "English",
        QwenTtsLanguage::German => "German",
        QwenTtsLanguage::Italian => "Italian",
        QwenTtsLanguage::Portuguese => "Portuguese",
        QwenTtsLanguage::Spanish => "Spanish",
        QwenTtsLanguage::Japanese => "Japanese",
        QwenTtsLanguage::Korean => "Korean",
        QwenTtsLanguage::French => "French",
        QwenTtsLanguage::Russian => "Russian",
    }
}
fn invalid(message: &str) -> QwenTtsRealtimeError {
    QwenTtsRealtimeError::InvalidInput(message.into())
}
fn interrupted(reason: &str) -> QwenTtsRealtimeError {
    QwenTtsRealtimeError::Interrupted {
        reason: reason.into(),
    }
}
fn unknown(operation: &'static str, reason: &str) -> QwenTtsRealtimeError {
    QwenTtsRealtimeError::OutcomeUnknown {
        operation,
        reason: reason.into(),
    }
}
fn protocol(message: &str, native: Option<Value>) -> QwenTtsRealtimeError {
    QwenTtsRealtimeError::Protocol {
        message: message.into(),
        native: native.map(Box::new),
        raw_frame: None,
    }
}

fn event_id() -> Result<String, QwenTtsRealtimeError> {
    let mut bytes = [0_u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| invalid("could not generate a unique session event ID"))?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let mut id = String::with_capacity(36);
    for (index, byte) in bytes.iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            id.push('-');
        }
        use std::fmt::Write;
        write!(&mut id, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(id)
}
