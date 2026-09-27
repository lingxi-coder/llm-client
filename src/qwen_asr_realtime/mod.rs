//! Standalone Qwen3-ASR Realtime over WebSocket.
//!
//! The host polls the input and event halves concurrently. This protocol
//! recognizes audio; it does not generate Omni responses or execute tools.

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
    future::{select, BoxFuture, Either},
    lock::Mutex as AsyncMutex,
    stream::BoxStream,
    FutureExt, StreamExt,
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

pub use crate::qwen_asr::{QwenAsrLanguage, QwenAsrRegion as QwenAsrRealtimeRegion};
const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAsrRealtimeConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: QwenAsrRealtimeRegion,
    pub workspace_id: String,
    pub connect_timeout: Duration,
    /// Emit `X-DashScope-DataInspection: enable` only when explicitly selected.
    pub data_inspection: bool,
}
impl QwenAsrRealtimeConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenAsrRealtimeRegion,
        workspace_id: impl Into<String>,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
            connect_timeout: Duration::from_secs(30),
            data_inspection: false,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAsrRealtimeScope {
    pub profile_name: String,
    pub account_scope: String,
    pub region: QwenAsrRealtimeRegion,
    pub workspace_id: String,
    pub endpoint_fingerprint: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QwenAsrRealtimeAudioFormat {
    #[default]
    Pcm,
    Opus,
}
impl QwenAsrRealtimeAudioFormat {
    fn wire(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Opus => "opus",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QwenAsrRealtimeTurnDetection {
    Manual,
    ServerVad {
        threshold: f64,
        silence_duration_ms: u32,
    },
}
impl Default for QwenAsrRealtimeTurnDetection {
    fn default() -> Self {
        Self::ServerVad {
            threshold: 0.2,
            silence_duration_ms: 800,
        }
    }
}

/// One ASR session. Corpus text is passed intact; the provider enforces its
/// 10,000-token limit. Client byte limits do not pretend to count tokens.
#[derive(Clone, PartialEq)]
pub struct QwenAsrRealtimeRequest {
    pub model: String,
    pub format: QwenAsrRealtimeAudioFormat,
    pub sample_rate: u32,
    pub language: Option<QwenAsrLanguage>,
    pub corpus_text: Option<String>,
    pub turn_detection: QwenAsrRealtimeTurnDetection,
}
impl QwenAsrRealtimeRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            format: Default::default(),
            sample_rate: 16000,
            language: None,
            corpus_text: None,
            turn_detection: Default::default(),
        }
    }
}
impl fmt::Debug for QwenAsrRealtimeRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAsrRealtimeRequest")
            .field("model", &self.model)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .field("language", &self.language)
            .field("turn_detection", &self.turn_detection)
            .field(
                "corpus_text",
                &self.corpus_text.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Local bounds. Appends await transport backpressure rather than filling an
/// unbounded queue. Transcript accounting includes repeated partial previews.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenAsrRealtimeLimits {
    pub max_frame_bytes: usize,
    pub max_audio_bytes: usize,
    pub max_audio_chunk_bytes: usize,
    pub max_text_bytes: usize,
    pub max_transcript_bytes: usize,
    pub max_setup_events: usize,
    pub max_active_items: usize,
    pub operation_timeout: Duration,
    pub finish_timeout: Duration,
}
impl Default for QwenAsrRealtimeLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 4 * 1024 * 1024,
            max_audio_bytes: 512 * 1024 * 1024,
            max_audio_chunk_bytes: 1024 * 1024,
            max_text_bytes: 64 * 1024,
            max_transcript_bytes: 16 * 1024 * 1024,
            max_setup_events: 16,
            max_active_items: 64,
            operation_timeout: Duration::from_secs(30),
            finish_timeout: Duration::from_secs(60),
        }
    }
}
impl QwenAsrRealtimeLimits {
    fn validate(self) -> Result<(), QwenAsrRealtimeError> {
        if self.max_frame_bytes == 0
            || self.max_frame_bytes > 16 * 1024 * 1024
            || self.max_audio_bytes == 0
            || self.max_audio_bytes > 512 * 1024 * 1024
            || self.max_audio_chunk_bytes == 0
            || self.max_audio_chunk_bytes > 10 * 1024 * 1024
            || self.max_text_bytes == 0
            || self.max_text_bytes > 1024 * 1024
            || self.max_transcript_bytes == 0
            || self.max_transcript_bytes > 64 * 1024 * 1024
            || self.max_setup_events == 0
            || self.max_setup_events > 32
            || self.max_active_items == 0
            || self.max_active_items > 128
            || self.operation_timeout.is_zero()
            || self.finish_timeout.is_zero()
        {
            return Err(invalid("invalid ASR Realtime resource limits"));
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub struct QwenAsrRealtimeMetadata {
    pub scope: QwenAsrRealtimeScope,
    pub session_id: String,
    pub model: String,
    pub format: QwenAsrRealtimeAudioFormat,
    pub sample_rate: u32,
    pub created_native: Value,
    pub updated_native: Value,
    pub setup_events: Vec<Value>,
}
impl fmt::Debug for QwenAsrRealtimeMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAsrRealtimeMetadata")
            .field("scope", &self.scope)
            .field("session_id", &self.session_id)
            .field("model", &self.model)
            .field("format", &self.format)
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}
#[derive(Clone, PartialEq)]
pub struct QwenAsrRealtimeEvent {
    pub kind: QwenAsrRealtimeEventKind,
    pub native: Value,
}
impl fmt::Debug for QwenAsrRealtimeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAsrRealtimeEvent")
            .field("kind", &self.kind)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}
#[derive(Clone, PartialEq, Eq)]
pub enum QwenAsrRealtimeEventKind {
    SpeechStarted {
        item_id: String,
        audio_start_ms: u64,
    },
    SpeechStopped {
        item_id: String,
        audio_end_ms: u64,
    },
    BufferCommitted {
        item_id: String,
        previous_item_id: Option<String>,
    },
    ItemCreated {
        item_id: String,
    },
    /// `text` is the current confirmed prefix and `stash` its revisable suffix.
    /// Replace the item's preview with `text + stash`; do not append previews.
    PartialTranscript {
        item_id: String,
        content_index: u32,
        text: String,
        stash: String,
        language: Option<String>,
        emotion: Option<String>,
    },
    TranscriptCompleted {
        item_id: String,
        content_index: u32,
        transcript: String,
        language: Option<String>,
        emotion: Option<String>,
    },
    /// This item failed; other items and subsequent audio remain usable. No
    /// failed audio is resent automatically.
    TranscriptFailed {
        item_id: String,
        content_index: u32,
        code: Option<String>,
        message: Option<String>,
    },
    /// Finish send confirmed, provider session.finished, local close confirmed,
    /// and clean EOF. This means the session drained, not that every item was
    /// transcribed successfully; callers must retain item failure outcomes.
    SessionFinished,
    Native,
}
impl fmt::Debug for QwenAsrRealtimeEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpeechStarted {
                item_id,
                audio_start_ms,
            } => f
                .debug_struct("SpeechStarted")
                .field("item_id", item_id)
                .field("audio_start_ms", audio_start_ms)
                .finish(),
            Self::SpeechStopped {
                item_id,
                audio_end_ms,
            } => f
                .debug_struct("SpeechStopped")
                .field("item_id", item_id)
                .field("audio_end_ms", audio_end_ms)
                .finish(),
            Self::BufferCommitted {
                item_id,
                previous_item_id,
            } => f
                .debug_struct("BufferCommitted")
                .field("item_id", item_id)
                .field("previous_item_id", previous_item_id)
                .finish(),
            Self::ItemCreated { item_id } => f
                .debug_struct("ItemCreated")
                .field("item_id", item_id)
                .finish(),
            Self::PartialTranscript {
                item_id,
                content_index,
                ..
            } => f
                .debug_struct("PartialTranscript")
                .field("item_id", item_id)
                .field("content_index", content_index)
                .finish_non_exhaustive(),
            Self::TranscriptCompleted {
                item_id,
                content_index,
                ..
            } => f
                .debug_struct("TranscriptCompleted")
                .field("item_id", item_id)
                .field("content_index", content_index)
                .finish_non_exhaustive(),
            Self::TranscriptFailed {
                item_id,
                content_index,
                code,
                ..
            } => f
                .debug_struct("TranscriptFailed")
                .field("item_id", item_id)
                .field("content_index", content_index)
                .field("code", code)
                .finish_non_exhaustive(),
            Self::SessionFinished => f.write_str("SessionFinished"),
            Self::Native => f.write_str("Native"),
        }
    }
}

#[derive(thiserror::Error)]
pub enum QwenAsrRealtimeError {
    #[error("invalid Qwen ASR Realtime input: {0}")]
    InvalidInput(String),
    #[error("Qwen ASR Realtime account scope does not match")]
    ScopeMismatch,
    #[error("unsupported Qwen ASR Realtime operation: {0}")]
    UnsupportedOperation(String),
    #[error("Qwen ASR Realtime session is closed")]
    Closed,
    #[error("a Qwen ASR Realtime control is awaiting acknowledgement")]
    ControlPending,
    #[error("Qwen ASR Realtime {operation} outcome is unknown: {reason}")]
    OutcomeUnknown {
        operation: &'static str,
        reason: String,
    },
    #[error("Qwen ASR Realtime connection interrupted: {reason}")]
    Interrupted { reason: String },
    #[error("Qwen ASR Realtime provider error: {message}")]
    Provider { message: String, native: Box<Value> },
    #[error("invalid Qwen ASR Realtime event: {message}")]
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

impl fmt::Debug for QwenAsrRealtimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Raw payloads can contain audio, user text and server diagnostics.
        f.debug_tuple("QwenAsrRealtimeError")
            .field(&self.to_string())
            .finish()
    }
}

#[derive(Clone)]
pub struct QwenAsrRealtimeService {
    transport: Arc<dyn RealtimeTransport>,
    scope: QwenAsrRealtimeScope,
    endpoint: String,
    connect_timeout: Duration,
    data_inspection: bool,
}
impl QwenAsrRealtimeService {
    pub fn new(
        transport: Arc<dyn RealtimeTransport>,
        config: QwenAsrRealtimeConfig,
    ) -> Result<Self, QwenAsrRealtimeError> {
        if config.profile_name.trim().is_empty()
            || config.account_scope.trim().is_empty()
            || !valid_workspace(&config.workspace_id)
            || config.connect_timeout.is_zero()
        {
            return Err(invalid(
                "profile, account, DNS-label workspace and positive connect timeout are required",
            ));
        }
        let region = match config.region {
            QwenAsrRealtimeRegion::Beijing => "cn-beijing",
            QwenAsrRealtimeRegion::Singapore => "ap-southeast-1",
        };
        let endpoint = format!(
            "wss://{}.{region}.maas.aliyuncs.com/api-ws/v1/realtime",
            config.workspace_id
        );
        Ok(Self {
            transport,
            scope: QwenAsrRealtimeScope {
                profile_name: config.profile_name,
                account_scope: config.account_scope,
                region: config.region,
                workspace_id: config.workspace_id,
                endpoint_fingerprint: provider_file_endpoint_fingerprint(&endpoint),
            },
            endpoint,
            connect_timeout: config.connect_timeout,
            data_inspection: config.data_inspection,
        })
    }
    pub fn scope(&self) -> &QwenAsrRealtimeScope {
        &self.scope
    }
    pub async fn connect(
        &self,
        options: &RequestOptions,
        request: &QwenAsrRealtimeRequest,
        limits: QwenAsrRealtimeLimits,
    ) -> Result<QwenAsrRealtimeSession, QwenAsrRealtimeError> {
        limits.validate()?;
        validate_request(request, limits)?;
        if options
            .account_scope
            .as_deref()
            .is_some_and(|v| v != self.scope.account_scope)
        {
            return Err(QwenAsrRealtimeError::ScopeMismatch);
        }
        let key = options
            .credential
            .as_ref()
            .ok_or_else(|| invalid("regional API key is required"))?
            .expose_secret();
        if key.trim().is_empty()
            || key.len() > 16 * 1024
            || key.bytes().any(|b| b.is_ascii_control())
        {
            return Err(invalid(
                "API key is empty, too large, or contains control characters",
            ));
        }
        let setup = json_frame(&session_update(request)?, limits.max_frame_bytes)?;
        let mut headers = vec![
            ("Authorization".into(), format!("Bearer {key}")),
            (
                "X-DashScope-WorkSpace".into(),
                self.scope.workspace_id.clone(),
            ),
        ];
        if self.data_inspection {
            headers.push(("X-DashScope-DataInspection".into(), "enable".into()));
        }
        let deadline = Deadline::after(Some(self.connect_timeout)).cap(options.total_timeout);
        let mut connection = deadline
            .run(self.transport.connect(RealtimeConnectRequest {
                endpoint: format!("{}?model={}", self.endpoint, request.model),
                headers,
                max_frame_bytes: limits.max_frame_bytes,
            }))
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
            .run(connection.outbound.send(setup))
            .await
            .map_err(|_| unknown("session.update", "setup send timed out"))?
            .map_err(|e| unknown("session.update", &e.to_string()))?;
        let mut setup_events = Vec::new();
        let updated_native = loop {
            let native = receive_setup(&mut connection, deadline, limits.max_frame_bytes).await?;
            if event_name(&native)? == "session.updated" {
                break native;
            }
            if setup_events.len() >= limits.max_setup_events {
                return Err(protocol("setup event limit exceeded", Some(native)));
            }
            setup_events.push(native);
        };
        validate_updated(&updated_native, request, &session_id)?;
        let metadata = Arc::new(QwenAsrRealtimeMetadata {
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
                awaiting_commit: false,
                buffered_audio: 0,
                finish_sent: false,
                finish_acked: false,
                finish_deadline: Deadline::default(),
            }),
            terminal: AtomicBool::new(false),
            sink: AsyncMutex::new(Some(connection.outbound)),
            reader_signal,
            writer_signal,
            writer_notices: AsyncMutex::new(writer_notices),
        });
        Ok(QwenAsrRealtimeSession {
            input: QwenAsrRealtimeInput {
                shared: shared.clone(),
                metadata: metadata.clone(),
                limits,
                turn_detection: request.turn_detection,
                audio_bytes: 0,
            },
            events: QwenAsrRealtimeEvents {
                shared,
                metadata,
                limits,
                turn_detection: request.turn_detection,
                inbound: connection.inbound,
                notices,
                active_items: BTreeSet::new(),
                transcript_bytes: 0,
                finished_native: None,
                eof_received: false,
                pending_terminal: None,
                closing: None,
                close_confirmed: false,
                done: false,
            },
        })
    }
}
pub struct QwenAsrRealtimeSession {
    input: QwenAsrRealtimeInput,
    events: QwenAsrRealtimeEvents,
}
impl QwenAsrRealtimeSession {
    pub fn metadata(&self) -> &QwenAsrRealtimeMetadata {
        &self.input.metadata
    }
    pub fn into_parts(self) -> (QwenAsrRealtimeInput, QwenAsrRealtimeEvents) {
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
    Finish,
    Ping,
}
impl Operation {
    fn wire(self) -> &'static str {
        match self {
            Self::Append => "input_audio_buffer.append",
            Self::Commit => "input_audio_buffer.commit",
            Self::Finish => "session.finish",
            Self::Ping => "WebSocket Ping",
        }
    }
}
struct State {
    status: Status,
    awaiting_commit: bool,
    buffered_audio: usize,
    finish_sent: bool,
    finish_acked: bool,
    finish_deadline: Deadline,
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

/// Poll writes concurrently with `QwenAsrRealtimeEvents::next`. Canceling a
/// write makes its outcome unknown and wakes the event reader without replay.
pub struct QwenAsrRealtimeInput {
    shared: Arc<Shared>,
    metadata: Arc<QwenAsrRealtimeMetadata>,
    limits: QwenAsrRealtimeLimits,
    turn_detection: QwenAsrRealtimeTurnDetection,
    audio_bytes: usize,
}
impl QwenAsrRealtimeInput {
    pub fn metadata(&self) -> &QwenAsrRealtimeMetadata {
        &self.metadata
    }
    pub fn audio_bytes_sent(&self) -> usize {
        self.audio_bytes
    }
    pub async fn append_audio(&mut self, audio: &[u8]) -> Result<(), QwenAsrRealtimeError> {
        if audio.is_empty()
            || audio.len() > self.limits.max_audio_chunk_bytes
            || audio.len() > self.limits.max_audio_bytes.saturating_sub(self.audio_bytes)
        {
            return Err(invalid(
                "audio chunk is empty or exceeds configured chunk/session limits",
            ));
        }
        if self.metadata.format == QwenAsrRealtimeAudioFormat::Pcm && !audio.len().is_multiple_of(2)
        {
            return Err(invalid("PCM16 audio must contain complete 16-bit samples"));
        }
        self.send(Operation::Append, Some(audio), None).await?;
        self.audio_bytes += audio.len();
        Ok(())
    }
    pub async fn commit(&mut self) -> Result<(), QwenAsrRealtimeError> {
        if self.turn_detection != QwenAsrRealtimeTurnDetection::Manual {
            return Err(QwenAsrRealtimeError::UnsupportedOperation(
                "input_audio_buffer.commit is disabled in VAD mode".into(),
            ));
        }
        if self.shared.state().buffered_audio == 0 {
            return Err(invalid("cannot commit an empty audio buffer"));
        }
        self.send(Operation::Commit, None, None).await
    }
    /// Manual audio must first be committed. Finish may follow a successfully
    /// sent commit before its acknowledgement has been consumed.
    pub async fn finish(&mut self) -> Result<(), QwenAsrRealtimeError> {
        if self.turn_detection == QwenAsrRealtimeTurnDetection::Manual
            && self.shared.state().buffered_audio != 0
        {
            return Err(invalid(
                "commit buffered audio before finishing a manual ASR session",
            ));
        }
        self.send(Operation::Finish, None, None).await
    }
    pub async fn abort(&mut self) -> Result<(), QwenAsrRealtimeError> {
        self.shared.stop(Status::Aborted);
        self.shared.close_if_idle().await;
        Ok(())
    }
    pub async fn ping(&mut self, payload: Bytes) -> Result<(), QwenAsrRealtimeError> {
        if payload.len() > 125 {
            return Err(invalid("WebSocket Ping payload exceeds 125 bytes"));
        }
        self.send(Operation::Ping, None, Some(payload)).await
    }
    async fn send(
        &mut self,
        operation: Operation,
        audio: Option<&[u8]>,
        ping: Option<Bytes>,
    ) -> Result<(), QwenAsrRealtimeError> {
        {
            let state = self.shared.state();
            if state.status != Status::Ready || self.shared.terminal.load(Ordering::Acquire) {
                return Err(QwenAsrRealtimeError::Closed);
            }
            if state.awaiting_commit && operation != Operation::Finish {
                return Err(QwenAsrRealtimeError::ControlPending);
            }
        }
        let mut native = json!({"event_id":event_id()?,"type":operation.wire()});
        if let Some(audio) = audio {
            native["audio"] = json!(STANDARD.encode(audio));
        }
        let frame = json_frame(&native, self.limits.max_frame_bytes)?;
        {
            let mut state = self.shared.state();
            if state.status != Status::Ready || self.shared.terminal.load(Ordering::Acquire) {
                return Err(QwenAsrRealtimeError::Closed);
            }
            if operation == Operation::Commit {
                state.awaiting_commit = true;
            }
            if operation == Operation::Finish {
                state.status = Status::Finishing;
                state.finish_deadline = Deadline::after(Some(self.limits.finish_timeout));
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
            guard.armed = false;
            return Err(QwenAsrRealtimeError::PingUnsupported);
        }
        match result {
            Ok(()) => {
                if self.shared.terminal.load(Ordering::Acquire) {
                    guard.armed = false;
                    return Err(QwenAsrRealtimeError::Closed);
                }
                {
                    let mut state = self.shared.state();
                    match operation {
                        Operation::Append => {
                            state.buffered_audio = state
                                .buffered_audio
                                .saturating_add(audio.map_or(0, <[u8]>::len))
                        }
                        Operation::Commit => state.buffered_audio = 0,
                        Operation::Finish => state.finish_sent = true,
                        Operation::Ping => {}
                    }
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
                message: "ASR Realtime write timed out".into(),
            }),
        };
        if self.shared.terminal.load(Ordering::Acquire) {
            close_sink(&mut slot).await;
            return Err(RealtimeError::Closed);
        }
        result
    }
}
impl Drop for QwenAsrRealtimeInput {
    fn drop(&mut self) {
        self.shared.stop(Status::Aborted);
        self.shared.release_if_idle();
    }
}

pub struct QwenAsrRealtimeEvents {
    shared: Arc<Shared>,
    metadata: Arc<QwenAsrRealtimeMetadata>,
    limits: QwenAsrRealtimeLimits,
    turn_detection: QwenAsrRealtimeTurnDetection,
    inbound: BoxStream<'static, Result<RealtimeFrame, RealtimeError>>,
    notices: mpsc::UnboundedReceiver<()>,
    active_items: BTreeSet<String>,
    transcript_bytes: usize,
    finished_native: Option<Value>,
    eof_received: bool,
    pending_terminal: Option<Result<Option<QwenAsrRealtimeEvent>, QwenAsrRealtimeError>>,
    closing: Option<BoxFuture<'static, Result<(), QwenAsrRealtimeError>>>,
    close_confirmed: bool,
    done: bool,
}
enum ReadStep {
    Frame(Option<Result<RealtimeFrame, RealtimeError>>),
    Notice,
    Close(Result<(), QwenAsrRealtimeError>),
}
impl QwenAsrRealtimeEvents {
    pub fn metadata(&self) -> &QwenAsrRealtimeMetadata {
        &self.metadata
    }
    pub fn transcript_bytes_received(&self) -> usize {
        self.transcript_bytes
    }
    pub async fn next(&mut self) -> Result<Option<QwenAsrRealtimeEvent>, QwenAsrRealtimeError> {
        if self.done && self.pending_terminal.is_none() {
            return Ok(None);
        }
        if self.pending_terminal.is_none() {
            let result = self.next_inner().await;
            let terminal = matches!(
                &result,
                Err(_)
                    | Ok(Some(QwenAsrRealtimeEvent {
                        kind: QwenAsrRealtimeEventKind::SessionFinished,
                        ..
                    }))
            );
            if !terminal {
                return result;
            }
            self.pending_terminal = Some(result);
            self.done = true;
            self.inbound = futures::stream::empty().boxed();
            self.closing = None;
            self.shared.stop(Status::Failed);
        }
        self.shared.close_if_idle().await;
        self.pending_terminal
            .take()
            .expect("terminal result retained")
    }
    async fn next_inner(&mut self) -> Result<Option<QwenAsrRealtimeEvent>, QwenAsrRealtimeError> {
        loop {
            if self.shared.terminal.load(Ordering::Acquire) {
                return Err(interrupted(
                    "session aborted or an outbound send was canceled",
                ));
            }
            if self.eof_received {
                return self.finish_at_eof().await;
            }
            self.begin_graceful_close().await?;
            let deadline = self.shared.state().finish_deadline;
            let step = deadline
                .run(async {
                    let read = select(Box::pin(self.inbound.next()), Box::pin(self.notices.next()));
                    if let Some(close) = self.closing.as_mut() {
                        match select(Box::pin(read), close.as_mut()).await {
                            Either::Left((Either::Left((frame, _)), _)) => ReadStep::Frame(frame),
                            Either::Left((Either::Right((_, _)), _)) => ReadStep::Notice,
                            Either::Right((result, _)) => ReadStep::Close(result),
                        }
                    } else {
                        match read.await {
                            Either::Left((frame, _)) => ReadStep::Frame(frame),
                            Either::Right((_, _)) => ReadStep::Notice,
                        }
                    }
                })
                .await
                .map_err(|_| interrupted("timed out awaiting final recognition and clean close"))?;
            let frame = match step {
                ReadStep::Notice => continue,
                ReadStep::Close(result) => {
                    self.closing = None;
                    result?;
                    self.close_confirmed = true;
                    continue;
                }
                ReadStep::Frame(None) => {
                    self.eof_received = true;
                    continue;
                }
                ReadStep::Frame(Some(frame)) => {
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
            let kind = self.decode_event(&native)?;
            if let Some(kind) = kind {
                return Ok(Some(QwenAsrRealtimeEvent { kind, native }));
            }
            self.finished_native = Some(native);
        }
    }

    fn decode_event(
        &mut self,
        native: &Value,
    ) -> Result<Option<QwenAsrRealtimeEventKind>, QwenAsrRealtimeError> {
        let kind = match event_name(native)? {
            "input_audio_buffer.speech_started" | "input_audio_buffer.speech_stopped" => {
                if self.turn_detection == QwenAsrRealtimeTurnDetection::Manual {
                    return Err(protocol(
                        "VAD speech event in manual mode",
                        Some(native.clone()),
                    ));
                }
                let item_id = required_string(native, "/item_id")?.to_owned();
                self.track_item(&item_id, native)?;
                if event_name(native)? == "input_audio_buffer.speech_started" {
                    QwenAsrRealtimeEventKind::SpeechStarted {
                        item_id,
                        audio_start_ms: required_u64(native, "/audio_start_ms")?,
                    }
                } else {
                    QwenAsrRealtimeEventKind::SpeechStopped {
                        item_id,
                        audio_end_ms: required_u64(native, "/audio_end_ms")?,
                    }
                }
            }
            "input_audio_buffer.committed" => {
                let item_id = required_string(native, "/item_id")?.to_owned();
                if self.turn_detection == QwenAsrRealtimeTurnDetection::Manual {
                    let mut state = self.shared.state();
                    if !state.awaiting_commit {
                        return Err(protocol(
                            "manual commit acknowledgement has no pending commit",
                            Some(native.clone()),
                        ));
                    }
                    state.awaiting_commit = false;
                }
                self.track_item(&item_id, native)?;
                QwenAsrRealtimeEventKind::BufferCommitted {
                    item_id,
                    previous_item_id: optional_string(native, "/previous_item_id")?,
                }
            }
            "conversation.item.created" => {
                let item_id = required_string(native, "/item/id")?.to_owned();
                self.track_item(&item_id, native)?;
                QwenAsrRealtimeEventKind::ItemCreated { item_id }
            }
            "conversation.item.input_audio_transcription.text" => {
                let item_id = required_string(native, "/item_id")?.to_owned();
                let text = string_allow_empty(native, "/text")?.to_owned();
                let stash = string_allow_empty(native, "/stash")?.to_owned();
                self.add_text_bytes(text.len().saturating_add(stash.len()), native)?;
                self.track_item(&item_id, native)?;
                QwenAsrRealtimeEventKind::PartialTranscript {
                    item_id,
                    content_index: content_index(native)?,
                    text,
                    stash,
                    language: optional_string(native, "/language")?,
                    emotion: optional_string(native, "/emotion")?,
                }
            }
            "conversation.item.input_audio_transcription.completed" => {
                let item_id = required_string(native, "/item_id")?.to_owned();
                let transcript = string_allow_empty(native, "/transcript")?.to_owned();
                self.add_text_bytes(transcript.len(), native)?;
                self.active_items.remove(&item_id);
                QwenAsrRealtimeEventKind::TranscriptCompleted {
                    item_id,
                    content_index: content_index(native)?,
                    transcript,
                    language: optional_string(native, "/language")?,
                    emotion: optional_string(native, "/emotion")?,
                }
            }
            "conversation.item.input_audio_transcription.failed" => {
                let item_id = required_string(native, "/item_id")?.to_owned();
                self.active_items.remove(&item_id);
                QwenAsrRealtimeEventKind::TranscriptFailed {
                    item_id,
                    content_index: content_index(native)?,
                    code: optional_string(native, "/error/code")?,
                    message: optional_string(native, "/error/message")?,
                }
            }
            "session.finished" => {
                if !self.active_items.is_empty() {
                    return Err(protocol(
                        "session.finished precedes final transcription for active items",
                        Some(native.clone()),
                    ));
                }
                let mut state = self.shared.state();
                if state.status != Status::Finishing || state.finish_acked || state.awaiting_commit
                {
                    return Err(protocol("session.finished without a pending finish or before commit acknowledgement",Some(native.clone())));
                }
                state.finish_acked = true;
                return Ok(None);
            }
            "session.created" | "session.updated" => {
                return Err(protocol(
                    "unexpected session reinitialization",
                    Some(native.clone()),
                ))
            }
            _ => QwenAsrRealtimeEventKind::Native,
        };
        Ok(Some(kind))
    }
    fn track_item(&mut self, id: &str, native: &Value) -> Result<(), QwenAsrRealtimeError> {
        if !self.active_items.contains(id)
            && self.active_items.len() >= self.limits.max_active_items
        {
            return Err(protocol(
                "active ASR item limit exceeded",
                Some(native.clone()),
            ));
        }
        self.active_items.insert(id.to_owned());
        Ok(())
    }
    fn add_text_bytes(&mut self, len: usize, native: &Value) -> Result<(), QwenAsrRealtimeError> {
        if len > self.limits.max_text_bytes
            || len
                > self
                    .limits
                    .max_transcript_bytes
                    .saturating_sub(self.transcript_bytes)
        {
            return Err(protocol(
                "transcription text exceeds configured event/session limits",
                Some(native.clone()),
            ));
        }
        self.transcript_bytes += len;
        Ok(())
    }

    /// ASR requires the client to close after session.finished. Move the sink
    /// into a stored future so cancellation of next() neither drops nor repeats
    /// the close operation. No sink lock is held while reading inbound events.
    async fn begin_graceful_close(&mut self) -> Result<(), QwenAsrRealtimeError> {
        let ready = {
            let state = self.shared.state();
            state.status == Status::Finishing && state.finish_sent && state.finish_acked
        };
        if self.finished_native.is_none()
            || !ready
            || self.closing.is_some()
            || self.close_confirmed
        {
            return Ok(());
        }
        let mut slot = self.shared.sink.lock().await;
        if self.shared.terminal.load(Ordering::Acquire) {
            return Err(interrupted("session was aborted before graceful close"));
        }
        let mut sink = slot
            .take()
            .ok_or_else(|| interrupted("WebSocket sink unavailable for graceful close"))?;
        let deadline = self.shared.state().finish_deadline;
        self.closing = Some(
            async move {
                deadline
                    .run(sink.close(RealtimeClose::normal("Qwen ASR recognition finished")))
                    .await
                    .map_err(|_| interrupted("graceful close timed out"))?
                    .map_err(|error| interrupted(&error.to_string()))
            }
            .boxed(),
        );
        Ok(())
    }
    async fn finish_at_eof(
        &mut self,
    ) -> Result<Option<QwenAsrRealtimeEvent>, QwenAsrRealtimeError> {
        loop {
            if self.shared.terminal.load(Ordering::Acquire) {
                return Err(interrupted("session aborted during final close"));
            }
            let (sent, acked, finishing, deadline) = {
                let state = self.shared.state();
                (
                    state.finish_sent,
                    state.finish_acked,
                    state.status == Status::Finishing,
                    state.finish_deadline,
                )
            };
            if self.finished_native.is_none() || !finishing || !acked {
                return Err(interrupted(
                    "WebSocket EOF before confirmed session.finished",
                ));
            }
            if !sent {
                if matches!(deadline.run(self.notices.next()).await, Ok(Some(()))) {
                    continue;
                }
                return Err(interrupted("finish send was not confirmed before EOF"));
            }
            self.begin_graceful_close().await?;
            if self.close_confirmed {
                self.shared.stop(Status::Finished);
                return Ok(Some(QwenAsrRealtimeEvent {
                    kind: QwenAsrRealtimeEventKind::SessionFinished,
                    native: self.finished_native.take().expect("checked"),
                }));
            }
            if let Some(close) = self.closing.as_mut() {
                match deadline
                    .run(select(close.as_mut(), Box::pin(self.notices.next())))
                    .await
                {
                    Ok(Either::Left((result, _))) => {
                        self.closing = None;
                        result?;
                        self.close_confirmed = true;
                    }
                    Ok(Either::Right((_, _))) => continue,
                    Err(_) => return Err(interrupted("timed out confirming graceful close")),
                }
            } else {
                return Err(interrupted("graceful close was not initiated"));
            }
        }
    }
}
impl Drop for QwenAsrRealtimeEvents {
    fn drop(&mut self) {
        self.shared.stop(Status::Aborted);
        self.shared.release_if_idle();
    }
}
async fn close_sink(slot: &mut Option<Box<dyn RealtimeSink>>) {
    if let Some(mut sink) = slot.take() {
        let _ = Deadline::after(Some(CLOSE_TIMEOUT))
            .run(sink.close(RealtimeClose::normal("Qwen ASR Realtime closed locally")))
            .await;
    }
}

fn validate_request(
    request: &QwenAsrRealtimeRequest,
    limits: QwenAsrRealtimeLimits,
) -> Result<(), QwenAsrRealtimeError> {
    if !matches!(
        request.model.as_str(),
        "qwen3-asr-flash-realtime"
            | "qwen3-asr-flash-realtime-2026-02-10"
            | "qwen3-asr-flash-realtime-2025-10-27"
    ) {
        return Err(invalid(
            "model is not a documented standalone Qwen3-ASR-Realtime model",
        ));
    }
    if !matches!(request.sample_rate, 8000 | 16000) {
        return Err(invalid("ASR sample rate must be 8000 or 16000 Hz"));
    }
    if let QwenAsrRealtimeTurnDetection::ServerVad {
        threshold,
        silence_duration_ms,
    } = request.turn_detection
    {
        if !threshold.is_finite()
            || !(-1.0..=1.0).contains(&threshold)
            || !(200..=6000).contains(&silence_duration_ms)
        {
            return Err(invalid(
                "VAD threshold must be finite in [-1,1]; silence duration must be 200..6000 ms",
            ));
        }
    }
    if request
        .corpus_text
        .as_ref()
        .is_some_and(|text| text.len() > limits.max_text_bytes)
    {
        return Err(invalid("corpus text exceeds the local byte bound"));
    }
    Ok(())
}
fn session_update(request: &QwenAsrRealtimeRequest) -> Result<Value, QwenAsrRealtimeError> {
    let mut session = json!({"input_audio_format":request.format.wire(),"sample_rate":request.sample_rate,"turn_detection":turn_detection_wire(request.turn_detection)});
    let mut transcription = serde_json::Map::new();
    if let Some(language) = request.language {
        transcription.insert("language".into(), json!(language));
    }
    if let Some(text) = &request.corpus_text {
        transcription.insert("corpus".into(), json!({"text":text}));
    }
    if !transcription.is_empty() {
        session["input_audio_transcription"] = Value::Object(transcription);
    }
    Ok(json!({"event_id":event_id()?,"type":"session.update","session":session}))
}
fn turn_detection_wire(turn: QwenAsrRealtimeTurnDetection) -> Value {
    match turn {
        QwenAsrRealtimeTurnDetection::Manual => Value::Null,
        QwenAsrRealtimeTurnDetection::ServerVad {
            threshold,
            silence_duration_ms,
        } => {
            json!({"type":"server_vad","threshold":threshold,"silence_duration_ms":silence_duration_ms})
        }
    }
}
fn validate_updated(
    native: &Value,
    request: &QwenAsrRealtimeRequest,
    session_id: &str,
) -> Result<(), QwenAsrRealtimeError> {
    if required_string(native, "/session/id")? != session_id
        || canonical_model(required_string(native, "/session/model")?)
            != canonical_model(&request.model)
    {
        return Err(protocol(
            "session.updated identity differs from requested session",
            Some(native.clone()),
        ));
    }
    let format = required_string(native, "/session/input_audio_format")?;
    if format != request.format.wire()
        && !(request.format == QwenAsrRealtimeAudioFormat::Pcm && format == "pcm16")
    {
        return Err(protocol(
            "session.updated audio format differs from request",
            Some(native.clone()),
        ));
    }
    if native
        .pointer("/session/sample_rate")
        .is_some_and(|v| v.as_u64() != Some(request.sample_rate as u64))
    {
        return Err(protocol(
            "session.updated sample rate differs from request",
            Some(native.clone()),
        ));
    }
    if let Some(modalities) = native.pointer("/session/modalities") {
        if modalities != &json!(["text"]) {
            return Err(protocol(
                "ASR session must have text-only output",
                Some(native.clone()),
            ));
        }
    }
    if let Some(turn) = native.pointer("/session/turn_detection") {
        match request.turn_detection {
            QwenAsrRealtimeTurnDetection::Manual if !turn.is_null() => {
                return Err(protocol(
                    "manual session acknowledged VAD",
                    Some(native.clone()),
                ))
            }
            QwenAsrRealtimeTurnDetection::ServerVad {
                threshold,
                silence_duration_ms,
            } => {
                if turn.get("type").and_then(Value::as_str) != Some("server_vad")
                    || turn
                        .get("threshold")
                        .is_some_and(|v| v.as_f64() != Some(threshold))
                    || turn
                        .get("silence_duration_ms")
                        .is_some_and(|v| v.as_u64() != Some(silence_duration_ms as u64))
                {
                    return Err(protocol(
                        "session.updated VAD configuration differs from request",
                        Some(native.clone()),
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some(language) = native.pointer("/session/input_audio_transcription/language") {
        if request
            .language
            .is_some_and(|expected| language != &json!(expected))
        {
            return Err(protocol(
                "session.updated language differs from request",
                Some(native.clone()),
            ));
        }
    }
    if let Some(corpus) = native.pointer("/session/input_audio_transcription/corpus/text") {
        if request
            .corpus_text
            .as_ref()
            .is_some_and(|expected| corpus.as_str() != Some(expected))
        {
            return Err(protocol(
                "session.updated corpus differs from request",
                Some(native.clone()),
            ));
        }
    }
    Ok(())
}
fn canonical_model(model: &str) -> &str {
    if model == "qwen3-asr-flash-realtime" {
        "qwen3-asr-flash-realtime-2025-10-27"
    } else {
        model
    }
}
fn required_u64(native: &Value, path: &str) -> Result<u64, QwenAsrRealtimeError> {
    native
        .pointer(path)
        .and_then(Value::as_u64)
        .ok_or_else(|| protocol(&format!("missing or invalid {path}"), Some(native.clone())))
}
fn content_index(native: &Value) -> Result<u32, QwenAsrRealtimeError> {
    u32::try_from(required_u64(native, "/content_index")?)
        .map_err(|_| protocol("content_index overflows", Some(native.clone())))
}
fn string_allow_empty<'a>(native: &'a Value, path: &str) -> Result<&'a str, QwenAsrRealtimeError> {
    native
        .pointer(path)
        .and_then(Value::as_str)
        .ok_or_else(|| protocol(&format!("missing or invalid {path}"), Some(native.clone())))
}
fn optional_string(native: &Value, path: &str) -> Result<Option<String>, QwenAsrRealtimeError> {
    match native.pointer(path) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(v)) => Ok(Some(v.clone())),
        _ => Err(protocol(
            &format!("invalid optional string {path}"),
            Some(native.clone()),
        )),
    }
}

async fn receive_setup(
    connection: &mut RealtimeConnection,
    deadline: Deadline,
    max: usize,
) -> Result<Value, QwenAsrRealtimeError> {
    let frame = deadline
        .run(connection.inbound.next())
        .await
        .map_err(|_| interrupted("session setup timed out"))?
        .ok_or_else(|| interrupted("WebSocket ended during session setup"))??;
    let native = parse_frame(frame, max)?;
    provider_error(&native)?;
    Ok(native)
}
fn parse_frame(frame: RealtimeFrame, max: usize) -> Result<Value, QwenAsrRealtimeError> {
    if frame.len() > max {
        return Err(RealtimeError::FrameTooLarge {
            actual: frame.len(),
            max,
        }
        .into());
    }
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(protocol(
            "standalone Qwen ASR requires JSON text frames",
            None,
        ));
    };
    let native: Value =
        serde_json::from_slice(&bytes).map_err(|_| QwenAsrRealtimeError::Protocol {
            message: "invalid JSON frame".into(),
            native: None,
            raw_frame: Some(bytes.clone()),
        })?;
    event_name(&native)?;
    Ok(native)
}
fn event_name(native: &Value) -> Result<&str, QwenAsrRealtimeError> {
    required_string(native, "/type")
}
fn required_string<'a>(native: &'a Value, path: &str) -> Result<&'a str, QwenAsrRealtimeError> {
    native
        .pointer(path)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| protocol(&format!("missing or empty {path}"), Some(native.clone())))
}
fn provider_error(native: &Value) -> Result<(), QwenAsrRealtimeError> {
    if native.get("type").and_then(Value::as_str) == Some("error") {
        return Err(QwenAsrRealtimeError::Provider {
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
fn json_frame(native: &Value, max: usize) -> Result<RealtimeFrame, QwenAsrRealtimeError> {
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

fn invalid(message: &str) -> QwenAsrRealtimeError {
    QwenAsrRealtimeError::InvalidInput(message.into())
}
fn interrupted(reason: &str) -> QwenAsrRealtimeError {
    QwenAsrRealtimeError::Interrupted {
        reason: reason.into(),
    }
}
fn unknown(operation: &'static str, reason: &str) -> QwenAsrRealtimeError {
    QwenAsrRealtimeError::OutcomeUnknown {
        operation,
        reason: reason.into(),
    }
}
fn protocol(message: &str, native: Option<Value>) -> QwenAsrRealtimeError {
    QwenAsrRealtimeError::Protocol {
        message: message.into(),
        native: native.map(Box::new),
        raw_frame: None,
    }
}

fn event_id() -> Result<String, QwenAsrRealtimeError> {
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
