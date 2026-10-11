//! OpenAI GPT-Live primary WebSocket adapter.
//!
//! GPT-Live uses `/v1/live/sessions` and a `session.start` handshake. It is a
//! separate protocol from the Realtime API's `/v1/realtime` endpoint.

use crate::realtime::{
    validate_frame, validate_limits, RealtimeClose, RealtimeConnectRequest, RealtimeConnection,
    RealtimeError, RealtimeFrame, RealtimeLimits, RealtimeTransport,
};
use crate::{files::provider_file_endpoint_fingerprint, protocol::Secret};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{
    channel::{mpsc, oneshot},
    future::poll_fn,
    stream, FutureExt, Sink, SinkExt, StreamExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    fmt,
    pin::Pin,
    sync::{Arc, Mutex},
    task::Poll,
};
use url::Url;

const MAX_SETUP_PREFACE_FRAMES: usize = 16;
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const MAX_SAFETY_IDENTIFIER_BYTES: usize = 256;
const MAX_LIVE_TOOL_COUNT: usize = 128;
const MAX_LIVE_HISTORY_MESSAGES: usize = 128;

/// Query-free WebSocket endpoint for the GPT-Live primary connection.
pub const OPENAI_LIVE_WEBSOCKET_ENDPOINT: &str = "wss://api.openai.com/v1/live/sessions";

/// The model ID currently documented for GPT-Live primary sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpenAiLiveModel {
    #[default]
    #[serde(rename = "gpt-live-1")]
    GptLive1,
}

impl OpenAiLiveModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GptLive1 => "gpt-live-1",
        }
    }
}

/// Primary endpoint route. The provider endpoint has no model query parameter;
/// the model is sent inside `session.start`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiLiveRoute {
    endpoint: String,
}

impl OpenAiLiveRoute {
    pub fn new() -> Self {
        Self {
            endpoint: OPENAI_LIVE_WEBSOCKET_ENDPOINT.into(),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        let expected = Url::parse(OPENAI_LIVE_WEBSOCKET_ENDPOINT).map_err(|_| {
            RealtimeError::InvalidConfig {
                message: "OpenAI Live endpoint constant is invalid".into(),
            }
        })?;
        let actual = Url::parse(&self.endpoint).map_err(|_| RealtimeError::InvalidConfig {
            message: "OpenAI Live endpoint is invalid".into(),
        })?;
        if actual != expected || actual.query().is_some() || actual.fragment().is_some() {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Live route must use the query-free primary session endpoint"
                    .into(),
            });
        }
        Ok(())
    }

    fn endpoint_fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(&self.endpoint)
    }
}

impl Default for OpenAiLiveRoute {
    fn default() -> Self {
        Self::new()
    }
}

/// Non-secret profile and account identity bound to the fixed Live endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiLiveScope {
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
}

impl OpenAiLiveScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        route: &OpenAiLiveRoute,
    ) -> Result<Self, RealtimeError> {
        route.validate()?;
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            endpoint_fingerprint: route.endpoint_fingerprint(),
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

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        if self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || self.endpoint_fingerprint.trim().is_empty()
        {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Live scope requires profile, account, and endpoint identity"
                    .into(),
            });
        }
        Ok(())
    }

    fn validate_route(&self, route: &OpenAiLiveRoute) -> Result<(), RealtimeError> {
        route.validate()?;
        self.validate()?;
        if self.endpoint_fingerprint != route.endpoint_fingerprint() {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Live scope belongs to a different endpoint".into(),
            });
        }
        Ok(())
    }
}

/// Audio format shared by GPT-Live input and output for the whole session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OpenAiLiveAudioFormat {
    #[default]
    Pcm16Mono24Khz,
    Pcm16Mono16Khz,
    G711MuLaw8Khz,
    G711ALaw8Khz,
}

impl OpenAiLiveAudioFormat {
    fn wire_value(self) -> Value {
        match self {
            Self::Pcm16Mono24Khz => json!({"type":"audio/pcm","rate":24000}),
            Self::Pcm16Mono16Khz => json!({"type":"audio/pcm","rate":16000}),
            Self::G711MuLaw8Khz => json!({"type":"audio/pcmu","rate":8000}),
            Self::G711ALaw8Khz => json!({"type":"audio/pcma","rate":8000}),
        }
    }

    fn validate_data(self, data: &Bytes) -> Result<(), RealtimeError> {
        if data.is_empty() {
            return Err(RealtimeError::InvalidInput {
                message: "OpenAI Live audio chunks must not be empty".into(),
            });
        }
        if matches!(self, Self::Pcm16Mono24Khz | Self::Pcm16Mono16Khz)
            && !data.len().is_multiple_of(2)
        {
            return Err(RealtimeError::InvalidInput {
                message: "OpenAI Live PCM16 chunks must contain complete samples".into(),
            });
        }
        Ok(())
    }
}

/// Responses backend function/web-search tool accepted by the documented Live
/// delegation subset. This adapter advertises tools but never executes them.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiLiveResponsesTool {
    Function {
        name: String,
        description: Option<String>,
        parameters: Value,
    },
    WebSearch,
}

impl OpenAiLiveResponsesTool {
    pub fn function(name: impl Into<String>, parameters: Value) -> Self {
        Self::Function {
            name: name.into(),
            description: None,
            parameters,
        }
    }

    pub fn with_description(
        mut self,
        description: impl Into<String>,
    ) -> Result<Self, RealtimeError> {
        if let Self::Function {
            description: field, ..
        } = &mut self
        {
            *field = Some(description.into());
            Ok(self)
        } else {
            Err(RealtimeError::InvalidInput {
                message: "a description can only be set on an OpenAI Live function tool".into(),
            })
        }
    }

    fn name(&self) -> Option<&str> {
        match self {
            Self::Function { name, .. } => Some(name),
            Self::WebSearch => None,
        }
    }

    fn to_wire(&self) -> Value {
        match self {
            Self::Function {
                name,
                description,
                parameters,
            } => {
                let mut wire = json!({
                    "type": "function",
                    "name": name,
                    "parameters": parameters,
                });
                if let Some(description) = description {
                    wire["description"] = Value::String(description.clone());
                }
                wire
            }
            Self::WebSearch => json!({"type":"web_search"}),
        }
    }
}

/// Tool-choice modes documented for Responses delegation in GPT-Live.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OpenAiLiveToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

impl OpenAiLiveToolChoice {
    fn to_wire(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".into()),
            Self::None => Value::String("none".into()),
            Self::Required => Value::String("required".into()),
            Self::Function(name) => json!({"type":"function","name":name}),
        }
    }
}

/// Full Responses-backend setup configured at `session.start`.
#[derive(Clone, PartialEq)]
pub struct OpenAiLiveResponsesConfig {
    pub model: String,
    pub instructions: Option<String>,
    pub tools: Vec<OpenAiLiveResponsesTool>,
    pub tool_choice: Option<OpenAiLiveToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub max_output_tokens: Option<u32>,
}

impl fmt::Debug for OpenAiLiveResponsesConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiLiveResponsesConfig")
            .field("model", &self.model)
            .field("instructions", &self.instructions.as_ref().map(|_| "<set>"))
            .field("tool_count", &self.tools.len())
            .field("tool_choice", &self.tool_choice)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("max_output_tokens", &self.max_output_tokens)
            .finish()
    }
}

impl OpenAiLiveResponsesConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instructions: None,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Sparse, documented Responses delegation settings accepted by `session.update`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpenAiLiveResponsesUpdate {
    pub model: Option<String>,
    pub instructions: Option<String>,
    pub tools: Option<Vec<OpenAiLiveResponsesTool>>,
    pub tool_choice: Option<OpenAiLiveToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub max_output_tokens: Option<u32>,
}

/// Backend delegation mode. This adapter does not own any delegated workflow.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiLiveDelegation {
    Client,
    Responses(OpenAiLiveResponsesConfig),
}

/// Text role for one documented GPT-Live startup history message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiLiveHistoryRole {
    Developer,
    User,
    Assistant,
}

/// One text-only message seeded into the GPT-Live session at startup.
///
/// Live accepts one text part per message. Developer and user messages are
/// encoded as `input_text`; assistant messages are encoded as `output_text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiLiveHistoryMessage {
    pub role: OpenAiLiveHistoryRole,
    pub text: String,
}

impl OpenAiLiveHistoryMessage {
    pub fn new(role: OpenAiLiveHistoryRole, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
        }
    }

    fn to_wire(&self) -> Value {
        let (role, content_type) = match self.role {
            OpenAiLiveHistoryRole::Developer => ("developer", "input_text"),
            OpenAiLiveHistoryRole::User => ("user", "input_text"),
            OpenAiLiveHistoryRole::Assistant => ("assistant", "output_text"),
        };
        json!({
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": self.text}],
        })
    }
}

/// Startup configuration for a primary GPT-Live WebSocket session.
#[derive(Clone)]
pub struct OpenAiLiveConfig {
    pub model: OpenAiLiveModel,
    pub instructions: Option<String>,
    /// Prior text messages. Live documents at most 128 messages and 8,192
    /// combined tokens; the provider enforces the token bound.
    pub input: Vec<OpenAiLiveHistoryMessage>,
    pub audio_format: OpenAiLiveAudioFormat,
    pub voice: Option<String>,
    pub delegation: OpenAiLiveDelegation,
    /// Opt in to provider-side storage for later forking. Defaults to false.
    pub store: bool,
}

impl Default for OpenAiLiveConfig {
    fn default() -> Self {
        Self {
            model: OpenAiLiveModel::GptLive1,
            instructions: None,
            input: Vec::new(),
            audio_format: OpenAiLiveAudioFormat::Pcm16Mono24Khz,
            voice: None,
            delegation: OpenAiLiveDelegation::Client,
            store: false,
        }
    }
}

impl fmt::Debug for OpenAiLiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiLiveConfig")
            .field("model", &self.model)
            .field("instructions", &self.instructions.as_ref().map(|_| "<set>"))
            .field("input_message_count", &self.input.len())
            .field("audio_format", &self.audio_format)
            .field("voice", &self.voice)
            .field("delegation", &self.delegation)
            .field("store", &self.store)
            .finish()
    }
}

/// Explicit commands for a primary GPT-Live session. Tool execution and
/// application-specific delegation remain with the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiLiveCommand {
    AppendAudio {
        data: Bytes,
        event_id: Option<String>,
    },
    /// Add a user message to a Responses backend. It does not start a response.
    AppendResponsesText {
        text: String,
        event_id: Option<String>,
    },
    /// Submit a caller-executed function result to the Responses backend.
    FunctionOutput {
        call_id: String,
        output: String,
        event_id: Option<String>,
    },
    ContinueResponses {
        event_id: Option<String>,
    },
    UpdateResponses {
        update: OpenAiLiveResponsesUpdate,
        event_id: Option<String>,
    },
    AppendInstructions {
        content: String,
        delegation_id: Option<String>,
        event_id: Option<String>,
    },
    AppendThinking {
        content: String,
        delegation_id: Option<String>,
        event_id: Option<String>,
    },
    AppendCommentary {
        content: String,
        delegation_id: Option<String>,
        event_id: Option<String>,
    },
    MuteAudio {
        event_id: Option<String>,
    },
    UnmuteAudio {
        event_id: Option<String>,
    },
    RequestClose {
        event_id: Option<String>,
    },
}

/// Provider-specific events, including terminal usage and untouched nested
/// Responses backend events.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiLiveEvent {
    SessionStarted {
        session: Value,
        client_event_id: Option<String>,
        native: Value,
    },
    SessionUpdated {
        session: Value,
        client_event_id: Option<String>,
        native: Value,
    },
    InputAudioMuted {
        client_event_id: Option<String>,
        native: Value,
    },
    InputAudioUnmuted {
        client_event_id: Option<String>,
        native: Value,
    },
    InstructionsAppended {
        start_ms: Option<f64>,
        end_ms: Option<f64>,
        client_event_id: Option<String>,
        native: Value,
    },
    ThinkingAppended {
        start_ms: Option<f64>,
        end_ms: Option<f64>,
        client_event_id: Option<String>,
        native: Value,
    },
    CommentaryAppended {
        start_ms: Option<f64>,
        end_ms: Option<f64>,
        client_event_id: Option<String>,
        native: Value,
    },
    InputTranscriptDelta {
        delta: String,
        start_ms: Option<f64>,
        end_ms: Option<f64>,
        native: Value,
    },
    OutputTranscriptDelta {
        delta: String,
        start_ms: Option<f64>,
        end_ms: Option<f64>,
        native: Value,
    },
    OutputAudioDelta {
        data: Bytes,
        format: OpenAiLiveAudioFormat,
        native: Value,
    },
    DelegationCreated {
        delegation_id: Option<String>,
        target: Option<String>,
        response_id: Option<String>,
        offset_ms: Option<f64>,
        native: Value,
    },
    ResponseEvent {
        delegation_id: Option<String>,
        event: Value,
        native: Value,
    },
    UsageUpdated {
        seconds: Option<f64>,
        context_window: Option<Value>,
        native: Value,
    },
    SessionClosed {
        reason: Option<String>,
        session: Value,
        usage: Value,
        native: Value,
    },
    ProviderError {
        error_type: Option<String>,
        code: Option<String>,
        message: String,
        param: Option<String>,
        client_event_id: Option<String>,
        native: Value,
    },
    ConnectionInterrupted {
        message: String,
    },
    TransportClosed {
        code: u16,
        reason: String,
    },
    /// A caller command was accepted before the terminal event but could not
    /// be sent because the server had already closed the Live session.
    CommandDroppedAfterSessionClosed {
        event_id: Option<String>,
    },
    Native {
        event_type: String,
        native: Value,
    },
}

/// Connected primary GPT-Live session. The host runs the returned driver and
/// reads events until finalization before closing the transport.
pub struct OpenAiLiveSession {
    control: OpenAiLiveControl,
    events: OpenAiLiveEvents,
    scope: OpenAiLiveScope,
}

impl OpenAiLiveSession {
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        route: OpenAiLiveRoute,
        scope: OpenAiLiveScope,
        credential: Secret<String>,
        safety_identifier: Option<String>,
        config: OpenAiLiveConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, OpenAiLiveDriver), RealtimeError> {
        validate_limits(limits)?;
        route.validate()?;
        scope.validate_route(&route)?;
        validate_credential(&credential)?;
        validate_safety_identifier(safety_identifier.as_deref())?;
        validate_config(&config)?;

        let setup_frame = encode_json_frame(build_session_start(&config))?;
        validate_frame(&setup_frame, limits.max_frame_bytes)?;
        let mut headers = vec![(
            "Authorization".into(),
            format!("Bearer {}", credential.expose_secret()),
        )];
        if let Some(identifier) = safety_identifier {
            headers.push(("OpenAI-Safety-Identifier".into(), identifier));
        }
        let request = RealtimeConnectRequest {
            endpoint: route.endpoint.clone(),
            headers,
            max_frame_bytes: limits.max_frame_bytes,
        };
        let setup_transport = OpenAiLiveSetupTransport {
            inner: transport,
            setup_frame,
            max_frame_bytes: limits.max_frame_bytes,
            expected_model: config.model,
        };
        let connection = setup_transport.connect(request).await?;
        let (command_sender, command_receiver) = mpsc::channel(limits.outbound_capacity - 1);
        let (event_sender, event_receiver) = mpsc::channel(limits.event_capacity - 1);
        let state = Arc::new(Mutex::new(OpenAiLiveState::default()));
        let (abort_tx, abort_rx) = oneshot::channel();
        let driver = OpenAiLiveDriver {
            abort: Some(abort_rx),
            commands: command_receiver,
            events: event_sender,
            connection,
            max_frame_bytes: limits.max_frame_bytes,
            audio_format: config.audio_format,
            state: state.clone(),
        };
        let control = OpenAiLiveControl {
            abort: Arc::new(Mutex::new(Some(abort_tx))),
            commands: Arc::new(Mutex::new(command_sender)),
            state: state.clone(),
            max_frame_bytes: limits.max_frame_bytes,
            audio_format: config.audio_format,
            delegation: config.delegation.clone(),
        };
        Ok((
            Self {
                control,
                events: OpenAiLiveEvents {
                    receiver: event_receiver,
                    state,
                },
                scope,
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (OpenAiLiveControl, OpenAiLiveEvents) {
        (self.control, self.events)
    }

    pub fn scope(&self) -> &OpenAiLiveScope {
        &self.scope
    }
}

#[derive(Default)]
struct OpenAiLiveState {
    close_requested: bool,
    session_closed: bool,
    transport_closed: bool,
}

/// Bounded command handle. It never runs tools or starts delegated Responses
/// work implicitly; callers must submit outputs and continue explicitly.
#[derive(Clone)]
pub struct OpenAiLiveControl {
    abort: Arc<
        Mutex<Option<oneshot::Sender<(RealtimeClose, oneshot::Sender<Result<(), RealtimeError>>)>>>,
    >,
    commands: Arc<Mutex<mpsc::Sender<DriverCommand>>>,
    state: Arc<Mutex<OpenAiLiveState>>,
    max_frame_bytes: usize,
    audio_format: OpenAiLiveAudioFormat,
    delegation: OpenAiLiveDelegation,
}

impl OpenAiLiveControl {
    pub fn send(&self, command: OpenAiLiveCommand) -> Result<(), RealtimeError> {
        let mut state = self.state.lock().unwrap();
        if state.close_requested {
            return Err(RealtimeError::InvalidInput {
                message: "OpenAI Live does not accept commands after session.close".into(),
            });
        }
        let closing = matches!(&command, OpenAiLiveCommand::RequestClose { .. });
        let frames = encode_command(&command, self.audio_format, &self.delegation)?;
        if frames.is_empty() {
            return Err(RealtimeError::InvalidInput {
                message: "OpenAI Live command encoded to no frames".into(),
            });
        }
        let mut total = 0usize;
        for frame in &frames {
            validate_frame(frame, self.max_frame_bytes)?;
            total = total.saturating_add(frame.len());
        }
        if total > self.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: total,
                max: self.max_frame_bytes,
            });
        }
        self.commands
            .lock()
            .unwrap()
            .try_send(DriverCommand::SendBatch(frames))
            .map_err(|error| {
                if error.is_full() {
                    RealtimeError::QueueFull
                } else {
                    RealtimeError::Closed
                }
            })?;
        if closing {
            state.close_requested = true;
        }
        Ok(())
    }

    pub fn append_audio(&self, data: Bytes, event_id: Option<String>) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::AppendAudio { data, event_id })
    }

    pub fn append_responses_text(
        &self,
        text: impl Into<String>,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::AppendResponsesText {
            text: text.into(),
            event_id,
        })
    }

    pub fn function_output(
        &self,
        call_id: impl Into<String>,
        output: impl Into<String>,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::FunctionOutput {
            call_id: call_id.into(),
            output: output.into(),
            event_id,
        })
    }

    pub fn continue_responses(&self, event_id: Option<String>) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::ContinueResponses { event_id })
    }

    pub fn update_responses(
        &self,
        update: OpenAiLiveResponsesUpdate,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::UpdateResponses { update, event_id })
    }

    pub fn append_instructions(
        &self,
        content: impl Into<String>,
        delegation_id: Option<String>,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::AppendInstructions {
            content: content.into(),
            delegation_id,
            event_id,
        })
    }

    pub fn append_commentary(
        &self,
        content: impl Into<String>,
        delegation_id: Option<String>,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::AppendCommentary {
            content: content.into(),
            delegation_id,
            event_id,
        })
    }

    pub fn append_thinking(
        &self,
        content: impl Into<String>,
        delegation_id: Option<String>,
        event_id: Option<String>,
    ) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::AppendThinking {
            content: content.into(),
            delegation_id,
            event_id,
        })
    }

    pub fn mute_audio(&self, event_id: Option<String>) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::MuteAudio { event_id })
    }

    pub fn unmute_audio(&self, event_id: Option<String>) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::UnmuteAudio { event_id })
    }

    /// Request protocol-level closure; keep reading through `SessionClosed`.
    pub fn request_close(&self, event_id: Option<String>) -> Result<(), RealtimeError> {
        self.send(OpenAiLiveCommand::RequestClose { event_id })
    }

    /// Close the WebSocket after `session.closed` confirmed final usage.
    pub async fn close_after_session_closed(
        &self,
        close: RealtimeClose,
    ) -> Result<(), RealtimeError> {
        {
            let state = self.state.lock().unwrap();
            if !state.session_closed {
                return Err(RealtimeError::InvalidInput {
                    message: "OpenAI Live transport close requires the session.closed event".into(),
                });
            }
            if state.transport_closed {
                return Ok(());
            }
        }
        self.close_transport(close).await
    }

    /// Explicitly abort the WebSocket when startup, finalization, or transport
    /// handling fails. An abort does not confirm final usage.
    pub async fn abort(&self, close: RealtimeClose) -> Result<(), RealtimeError> {
        let close = RealtimeClose::new(close.code, close.reason)?;
        let (ack_tx, ack_rx) = oneshot::channel();
        {
            let mut state = self.state.lock().unwrap();
            if state.transport_closed {
                return Ok(());
            }
            state.close_requested = true;
            let mut commands = self.commands.lock().unwrap();
            let abort = self
                .abort
                .lock()
                .unwrap()
                .take()
                .ok_or(RealtimeError::Closed)?;
            commands.close_channel();
            abort
                .send((close, ack_tx))
                .map_err(|_| RealtimeError::Closed)?;
        }
        ack_rx.await.map_err(|_| RealtimeError::Closed)?
    }

    async fn close_transport(&self, close: RealtimeClose) -> Result<(), RealtimeError> {
        let close = RealtimeClose::new(close.code, close.reason)?;
        {
            let mut state = self.state.lock().unwrap();
            if state.transport_closed {
                return Ok(());
            }
            state.close_requested = true;
        }
        let (ack_tx, ack_rx) = oneshot::channel();
        let commands = self.commands.clone();
        let mut command = Some(DriverCommand::CloseTransport(close, ack_tx));
        poll_fn(move |cx| {
            let mut sender = commands.lock().unwrap();
            match Pin::new(&mut *sender).poll_ready(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(_)) => Poll::Ready(Err(RealtimeError::Closed)),
                Poll::Ready(Ok(())) => {
                    let command = command.take().expect("close transport queues once");
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

/// Event receiver. Observing `SessionClosed` enables the graceful explicit
/// transport close helper on the control handle.
pub struct OpenAiLiveEvents {
    receiver: mpsc::Receiver<OpenAiLiveEvent>,
    state: Arc<Mutex<OpenAiLiveState>>,
}

impl OpenAiLiveEvents {
    pub async fn next(&mut self) -> Option<OpenAiLiveEvent> {
        let event = self.receiver.next().await?;
        if matches!(event, OpenAiLiveEvent::SessionClosed { .. }) {
            let mut state = self.state.lock().unwrap();
            state.session_closed = true;
            state.close_requested = true;
        }
        Some(event)
    }
}

/// Primary GPT-Live event pump. Unlike generic realtime codecs, Live has a
/// provider-specific handshake and terminal session lifecycle.
pub struct OpenAiLiveDriver {
    abort: Option<oneshot::Receiver<(RealtimeClose, oneshot::Sender<Result<(), RealtimeError>>)>>,
    commands: mpsc::Receiver<DriverCommand>,
    events: mpsc::Sender<OpenAiLiveEvent>,
    connection: RealtimeConnection,
    max_frame_bytes: usize,
    audio_format: OpenAiLiveAudioFormat,
    state: Arc<Mutex<OpenAiLiveState>>,
}

impl OpenAiLiveDriver {
    pub async fn run(mut self) -> Result<(), RealtimeError> {
        let abort = self.abort.take().expect("Live driver runs once");
        let (close, ack) = {
            let pump = self.run_loop().fuse();
            let abort = async move {
                match abort.await {
                    Ok(request) => request,
                    Err(_) => futures::future::pending().await,
                }
            }
            .fuse();
            futures::pin_mut!(pump, abort);
            futures::select_biased! { request = abort => request, result = pump => return result }
        };
        self.commands.close();
        while self.commands.try_recv().is_ok() {}
        self.connection.outbound.abort();
        let _ = ack.send(Ok(()));
        let _ = self.events.try_send(OpenAiLiveEvent::TransportClosed {
            code: close.code,
            reason: close.reason,
        });
        Ok(())
    }
    async fn run_loop(&mut self) -> Result<(), RealtimeError> {
        loop {
            let command = self.commands.next().fuse();
            let incoming = self.connection.inbound.next().fuse();
            futures::pin_mut!(command, incoming);
            match futures::future::select(command, incoming).await {
                futures::future::Either::Left((Some(DriverCommand::SendBatch(frames)), _)) => {
                    for frame in frames {
                        if self.state.lock().unwrap().session_closed {
                            self.emit(OpenAiLiveEvent::CommandDroppedAfterSessionClosed {
                                event_id: frame_event_id(&frame),
                            })
                            .await?;
                            continue;
                        }
                        validate_frame(&frame, self.max_frame_bytes)?;
                        if let Err(error) = self.connection.outbound.send(frame).await {
                            let _ = self
                                .emit(OpenAiLiveEvent::ConnectionInterrupted {
                                    message: error.to_string(),
                                })
                                .await;
                            return Err(error);
                        }
                    }
                }
                futures::future::Either::Left((
                    Some(DriverCommand::CloseTransport(close, ack)),
                    _,
                )) => {
                    let result = self.connection.outbound.close(close.clone()).await;
                    let _ = ack.send(result.clone());
                    if result.is_ok() {
                        self.emit(OpenAiLiveEvent::TransportClosed {
                            code: close.code,
                            reason: close.reason,
                        })
                        .await?;
                    }
                    return result;
                }
                futures::future::Either::Left((None, _)) => {
                    let close = RealtimeClose::normal("GPT-Live controls dropped");
                    self.connection.outbound.close(close.clone()).await?;
                    self.emit(OpenAiLiveEvent::TransportClosed {
                        code: close.code,
                        reason: close.reason,
                    })
                    .await?;
                    return Ok(());
                }
                futures::future::Either::Right((Some(Ok(frame)), _)) => {
                    validate_frame(&frame, self.max_frame_bytes)?;
                    let event = match decode_event(frame, self.audio_format) {
                        Ok(event) => event,
                        Err(error) => {
                            let _ = self
                                .emit(OpenAiLiveEvent::ConnectionInterrupted {
                                    message: error.to_string(),
                                })
                                .await;
                            return Err(error);
                        }
                    };
                    if matches!(&event, OpenAiLiveEvent::SessionClosed { .. }) {
                        let mut state = self.state.lock().unwrap();
                        state.session_closed = true;
                        state.close_requested = true;
                    }
                    self.emit(event).await?;
                }
                futures::future::Either::Right((Some(Err(error)), _)) => {
                    let _ = self
                        .emit(OpenAiLiveEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
                futures::future::Either::Right((None, _)) => {
                    if self.state.lock().unwrap().session_closed {
                        return Ok(());
                    }
                    let error = RealtimeError::UnexpectedRemoteClose;
                    let _ = self
                        .emit(OpenAiLiveEvent::ConnectionInterrupted {
                            message: error.to_string(),
                        })
                        .await;
                    return Err(error);
                }
            }
        }
    }

    async fn emit(&mut self, event: OpenAiLiveEvent) -> Result<(), RealtimeError> {
        self.events
            .send(event)
            .await
            .map_err(|_| RealtimeError::EventReceiverDropped)
    }
}

impl Drop for OpenAiLiveDriver {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.transport_closed = true;
        state.close_requested = true;
    }
}

#[async_trait]
impl RealtimeTransport for OpenAiLiveSetupTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let mut connection = self.inner.connect(request).await?;
        connection.outbound.send(self.setup_frame.clone()).await?;
        let mut preface = Vec::new();
        let mut started = false;
        for _ in 0..MAX_SETUP_PREFACE_FRAMES {
            let frame = connection
                .inbound
                .next()
                .await
                .ok_or(RealtimeError::UnexpectedRemoteClose)??;
            validate_frame(&frame, self.max_frame_bytes)?;
            let native = parse_json_frame(&frame)?;
            let event_type = get_event_type(&native)?;
            if event_type == "error" {
                return Err(RealtimeError::Transport {
                    message: setup_error_message(&native),
                });
            }
            preface.push(Ok(frame));
            if event_type == "session.started" {
                let Some(server_model) = native
                    .get("session")
                    .and_then(|session| session.get("model"))
                    .and_then(Value::as_str)
                else {
                    return Err(RealtimeError::Transport {
                        message: "OpenAI Live session.started omitted its required model".into(),
                    });
                };
                if server_model != self.expected_model.as_str() {
                    return Err(RealtimeError::Transport {
                        message: "OpenAI Live session started with a different model".into(),
                    });
                }
                started = true;
                break;
            }
        }
        if !started {
            return Err(RealtimeError::Transport {
                message: "OpenAI Live did not return session.started within the setup limit".into(),
            });
        }
        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct OpenAiLiveSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
    expected_model: OpenAiLiveModel,
}

enum DriverCommand {
    SendBatch(Vec<RealtimeFrame>),
    CloseTransport(RealtimeClose, oneshot::Sender<Result<(), RealtimeError>>),
}

fn validate_credential(credential: &Secret<String>) -> Result<(), RealtimeError> {
    let value = credential.expose_secret();
    if value.trim().is_empty()
        || value.len() > MAX_CREDENTIAL_BYTES
        || value
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live bearer credential is empty or invalid".into(),
        });
    }
    Ok(())
}

fn validate_safety_identifier(identifier: Option<&str>) -> Result<(), RealtimeError> {
    if identifier.is_some_and(|value| {
        value.trim().is_empty()
            || value.len() > MAX_SAFETY_IDENTIFIER_BYTES
            || value
                .bytes()
                .any(|byte| byte.is_ascii_control() || byte == b' ')
    }) {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live safety identifier must be non-empty and a single header value"
                .into(),
        });
    }
    Ok(())
}

fn validate_config(config: &OpenAiLiveConfig) -> Result<(), RealtimeError> {
    if config
        .instructions
        .as_ref()
        .is_some_and(|text| text.contains('\0') || text.len() > 64 * 1024)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live instructions must be NUL-free and at most 64 KiB locally".into(),
        });
    }
    if config.input.len() > MAX_LIVE_HISTORY_MESSAGES {
        return Err(RealtimeError::InvalidConfig {
            message: format!(
                "OpenAI Live startup history exceeds the documented {MAX_LIVE_HISTORY_MESSAGES}-message limit"
            ),
        });
    }
    if config
        .input
        .iter()
        .any(|message| message.text.contains('\0'))
    {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live startup history text must be NUL-free".into(),
        });
    }
    validate_voice(config.voice.as_deref())?;
    if let OpenAiLiveDelegation::Responses(responses) = &config.delegation {
        validate_responses_config(responses)?;
    }
    Ok(())
}

fn validate_voice(voice: Option<&str>) -> Result<(), RealtimeError> {
    if voice
        .is_some_and(|voice| voice.trim().is_empty() || voice.contains('\0') || voice.len() > 256)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live voice must be non-empty, NUL-free, and at most 256 bytes".into(),
        });
    }
    Ok(())
}

fn validate_responses_config(config: &OpenAiLiveResponsesConfig) -> Result<(), RealtimeError> {
    validate_nonempty_field(&config.model, "Responses backend model", 128)?;
    if config
        .instructions
        .as_ref()
        .is_some_and(|instructions| instructions.contains('\0') || instructions.len() > 64 * 1024)
    {
        return Err(RealtimeError::InvalidConfig {
            message:
                "OpenAI Live Responses instructions must be NUL-free and at most 64 KiB locally"
                    .into(),
        });
    }
    validate_responses_tools(&config.tools)?;
    if config.max_output_tokens.is_some_and(|tokens| tokens < 16) {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live Responses max_output_tokens must be at least 16".into(),
        });
    }
    if let Some(choice) = &config.tool_choice {
        validate_tool_choice(choice, &config.tools, true)?;
    }
    Ok(())
}

fn validate_responses_update(update: &OpenAiLiveResponsesUpdate) -> Result<(), RealtimeError> {
    if update.model.is_none()
        && update.instructions.is_none()
        && update.tools.is_none()
        && update.tool_choice.is_none()
        && update.parallel_tool_calls.is_none()
        && update.max_output_tokens.is_none()
    {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Live Responses update must change at least one documented field"
                .into(),
        });
    }
    if let Some(model) = &update.model {
        validate_nonempty_field(model, "Responses backend model", 128)?;
    }
    if update
        .instructions
        .as_ref()
        .is_some_and(|text| text.contains('\0') || text.len() > 64 * 1024)
    {
        return Err(RealtimeError::InvalidInput {
            message:
                "OpenAI Live Responses instructions must be NUL-free and at most 64 KiB locally"
                    .into(),
        });
    }
    if let Some(tools) = &update.tools {
        validate_responses_tools(tools)?;
    }
    if update.max_output_tokens.is_some_and(|tokens| tokens < 16) {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Live Responses max_output_tokens must be at least 16".into(),
        });
    }
    if let Some(choice) = &update.tool_choice {
        let tools = update.tools.as_deref().unwrap_or(&[]);
        validate_tool_choice(choice, tools, update.tools.is_some())?;
    }
    Ok(())
}

fn validate_responses_tools(tools: &[OpenAiLiveResponsesTool]) -> Result<(), RealtimeError> {
    if tools.len() > MAX_LIVE_TOOL_COUNT {
        return Err(RealtimeError::InvalidConfig {
            message: format!("OpenAI Live tool list exceeds local bound {MAX_LIVE_TOOL_COUNT}"),
        });
    }
    let mut names = HashSet::with_capacity(tools.len());
    for tool in tools {
        match tool {
            OpenAiLiveResponsesTool::Function {
                name,
                description,
                parameters,
            } => {
                validate_tool_name(name)?;
                if !names.insert(name.as_str()) {
                    return Err(RealtimeError::InvalidConfig {
                        message: "OpenAI Live Responses function tool names must be unique".into(),
                    });
                }
                if !parameters.is_object() {
                    return Err(RealtimeError::InvalidConfig {
                        message: "OpenAI Live function parameters must be a JSON Schema object"
                            .into(),
                    });
                }
                if description
                    .as_ref()
                    .is_some_and(|description| description.contains('\0'))
                {
                    return Err(RealtimeError::InvalidConfig {
                        message: "OpenAI Live function descriptions must be NUL-free".into(),
                    });
                }
            }
            OpenAiLiveResponsesTool::WebSearch => {
                // The provider owns validation of repeated server-tool entries.
            }
        }
    }
    Ok(())
}

fn validate_tool_choice(
    choice: &OpenAiLiveToolChoice,
    tools: &[OpenAiLiveResponsesTool],
    tools_explicit: bool,
) -> Result<(), RealtimeError> {
    match choice {
        OpenAiLiveToolChoice::Required if tools_explicit && tools.is_empty() => {
            Err(RealtimeError::InvalidConfig {
                message: "OpenAI Live required tool choice needs a non-empty tool list".into(),
            })
        }
        OpenAiLiveToolChoice::Function(name) => {
            validate_tool_name(name)?;
            if tools_explicit && !tools.iter().any(|tool| tool.name() == Some(name.as_str())) {
                return Err(RealtimeError::InvalidConfig {
                    message:
                        "OpenAI Live named tool choice must reference a function in the same config"
                            .into(),
                });
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_tool_name(name: &str) -> Result<(), RealtimeError> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if !valid {
        return Err(RealtimeError::InvalidConfig {
            message: "OpenAI Live function tool names must use 1–64 letters, digits, `_`, or `-`"
                .into(),
        });
    }
    Ok(())
}

fn validate_nonempty_field(value: &str, label: &str, max_len: usize) -> Result<(), RealtimeError> {
    if value.trim().is_empty() || value.contains('\0') || value.len() > max_len {
        return Err(RealtimeError::InvalidConfig {
            message: format!("{label} must be non-empty, NUL-free, and at most {max_len} bytes"),
        });
    }
    Ok(())
}

fn build_session_start(config: &OpenAiLiveConfig) -> Value {
    let mut session = json!({
        "model": config.model.as_str(),
        "audio": {
            "format": config.audio_format.wire_value(),
        },
        "delegation": match &config.delegation {
            OpenAiLiveDelegation::Client => json!({"type":"client"}),
            OpenAiLiveDelegation::Responses(responses) => build_responses_config(responses),
        },
    });
    if let Some(instructions) = &config.instructions {
        session["instructions"] = Value::String(instructions.clone());
    }
    if !config.input.is_empty() {
        session["input"] = Value::Array(
            config
                .input
                .iter()
                .map(|message| message.to_wire())
                .collect(),
        );
    }
    if let Some(voice) = &config.voice {
        session["audio"]["output"] = json!({"voice":voice});
    }
    if config.store {
        session["store"] = Value::Bool(true);
    }
    json!({"type":"session.start","session":session})
}

fn build_responses_config(config: &OpenAiLiveResponsesConfig) -> Value {
    let mut responses = json!({"model":config.model});
    if let Some(instructions) = &config.instructions {
        responses["instructions"] = Value::String(instructions.clone());
    }
    if !config.tools.is_empty() {
        responses["tools"] = Value::Array(
            config
                .tools
                .iter()
                .map(OpenAiLiveResponsesTool::to_wire)
                .collect(),
        );
    }
    if let Some(choice) = &config.tool_choice {
        responses["tool_choice"] = choice.to_wire();
    }
    if let Some(parallel) = config.parallel_tool_calls {
        responses["parallel_tool_calls"] = Value::Bool(parallel);
    }
    if let Some(max_output_tokens) = config.max_output_tokens {
        responses["max_output_tokens"] = json!(max_output_tokens);
    }
    json!({"type":"responses","responses":responses})
}

fn encode_command(
    command: &OpenAiLiveCommand,
    audio_format: OpenAiLiveAudioFormat,
    delegation: &OpenAiLiveDelegation,
) -> Result<Vec<RealtimeFrame>, RealtimeError> {
    let with_event_id = |mut value: Value, event_id: &Option<String>| {
        if let Some(event_id) = event_id {
            value["event_id"] = Value::String(event_id.clone());
        }
        value
    };
    let response_delegation = matches!(delegation, OpenAiLiveDelegation::Responses(_));
    let value = match command {
        OpenAiLiveCommand::AppendAudio { data, event_id } => {
            validate_event_id(event_id.as_deref())?;
            audio_format.validate_data(data)?;
            with_event_id(
                json!({"type":"session.input_audio.append","audio":STANDARD.encode(data)}),
                event_id,
            )
        }
        OpenAiLiveCommand::AppendResponsesText { text, event_id } => {
            require_responses(response_delegation, "Responses input item")?;
            validate_event_id(event_id.as_deref())?;
            validate_nonempty_input(text, "Responses text item")?;
            with_event_id(
                json!({
                    "type":"response.item.create",
                    "item": {
                        "type":"message",
                        "role":"user",
                        "content":[{"type":"input_text","text":text}]
                    }
                }),
                event_id,
            )
        }
        OpenAiLiveCommand::FunctionOutput {
            call_id,
            output,
            event_id,
        } => {
            require_responses(response_delegation, "Responses function output")?;
            validate_event_id(event_id.as_deref())?;
            validate_nonempty_input(call_id, "function call_id")?;
            if output.contains('\0') {
                return Err(RealtimeError::InvalidInput {
                    message: "OpenAI Live function output must be NUL-free".into(),
                });
            }
            with_event_id(
                json!({
                    "type":"response.item.create",
                    "item":{"type":"function_call_output","call_id":call_id,"output":output}
                }),
                event_id,
            )
        }
        OpenAiLiveCommand::ContinueResponses { event_id } => {
            require_responses(response_delegation, "Responses continuation")?;
            validate_event_id(event_id.as_deref())?;
            with_event_id(json!({"type":"response.create"}), event_id)
        }
        OpenAiLiveCommand::UpdateResponses { update, event_id } => {
            require_responses(response_delegation, "Responses settings update")?;
            validate_event_id(event_id.as_deref())?;
            validate_responses_update(update)?;
            let mut responses = MapLike::new();
            if let Some(model) = &update.model {
                responses.insert("model", json!(model));
            }
            if let Some(instructions) = &update.instructions {
                responses.insert("instructions", json!(instructions));
            }
            if let Some(tools) = &update.tools {
                responses.insert(
                    "tools",
                    Value::Array(tools.iter().map(OpenAiLiveResponsesTool::to_wire).collect()),
                );
            }
            if let Some(choice) = &update.tool_choice {
                responses.insert("tool_choice", choice.to_wire());
            }
            if let Some(parallel) = update.parallel_tool_calls {
                responses.insert("parallel_tool_calls", json!(parallel));
            }
            if let Some(max_output_tokens) = update.max_output_tokens {
                responses.insert("max_output_tokens", json!(max_output_tokens));
            }
            with_event_id(
                json!({"type":"session.update","session":{"delegation":{"type":"responses","responses":responses.value()}}}),
                event_id,
            )
        }
        OpenAiLiveCommand::AppendInstructions {
            content,
            delegation_id,
            event_id,
        } => {
            validate_event_id(event_id.as_deref())?;
            validate_context_append(content, delegation_id.as_deref(), response_delegation)?;
            with_event_id(
                json!({
                    "type":"session.instructions.append",
                    "content":content,
                    "delegation_id":delegation_id,
                }),
                event_id,
            )
        }
        OpenAiLiveCommand::AppendCommentary {
            content,
            delegation_id,
            event_id,
        } => {
            validate_event_id(event_id.as_deref())?;
            validate_context_append(content, delegation_id.as_deref(), response_delegation)?;
            with_event_id(
                json!({
                    "type":"session.commentary.append",
                    "content":content,
                    "delegation_id":delegation_id,
                }),
                event_id,
            )
        }
        OpenAiLiveCommand::AppendThinking {
            content,
            delegation_id,
            event_id,
        } => {
            validate_event_id(event_id.as_deref())?;
            validate_context_append(content, delegation_id.as_deref(), response_delegation)?;
            with_event_id(
                json!({
                    "type":"session.thinking.append",
                    "content":content,
                    "delegation_id":delegation_id,
                }),
                event_id,
            )
        }
        OpenAiLiveCommand::MuteAudio { event_id } => {
            validate_event_id(event_id.as_deref())?;
            with_event_id(json!({"type":"session.input_audio.mute"}), event_id)
        }
        OpenAiLiveCommand::UnmuteAudio { event_id } => {
            validate_event_id(event_id.as_deref())?;
            with_event_id(json!({"type":"session.input_audio.unmute"}), event_id)
        }
        OpenAiLiveCommand::RequestClose { event_id } => {
            validate_event_id(event_id.as_deref())?;
            with_event_id(json!({"type":"session.close"}), event_id)
        }
    };
    Ok(vec![encode_json_frame(value)?])
}

fn require_responses(enabled: bool, feature: &str) -> Result<(), RealtimeError> {
    if enabled {
        Ok(())
    } else {
        Err(RealtimeError::InvalidInput {
            message: format!("{feature} requires Responses delegation"),
        })
    }
}

fn validate_event_id(event_id: Option<&str>) -> Result<(), RealtimeError> {
    if event_id.is_some_and(|id| id.is_empty() || id.len() > 512 || id.contains('\0')) {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Live event_id must be non-empty, NUL-free, and at most 512 bytes"
                .into(),
        });
    }
    Ok(())
}

fn validate_nonempty_input(value: &str, label: &str) -> Result<(), RealtimeError> {
    if value.trim().is_empty() || value.contains('\0') {
        return Err(RealtimeError::InvalidInput {
            message: format!("OpenAI Live {label} must be non-empty and NUL-free"),
        });
    }
    Ok(())
}

fn validate_context_append(
    content: &str,
    delegation_id: Option<&str>,
    responses_delegation: bool,
) -> Result<(), RealtimeError> {
    validate_nonempty_input(content, "context append")?;
    if delegation_id.is_some_and(|id| id.is_empty() || id.contains('\0')) {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Live delegation_id must be non-empty and NUL-free when present".into(),
        });
    }
    if responses_delegation && delegation_id.is_some() {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Live Responses delegation context must use delegation_id: null".into(),
        });
    }
    Ok(())
}

fn encode_json_frame(value: Value) -> Result<RealtimeFrame, RealtimeError> {
    serde_json::to_vec(&value)
        .map(Bytes::from)
        .map(RealtimeFrame::Text)
        .map_err(|_| RealtimeError::Codec {
            message: "could not encode OpenAI Live JSON event".into(),
        })
}

fn frame_event_id(frame: &RealtimeFrame) -> Option<String> {
    parse_json_frame(frame)
        .ok()?
        .get("event_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

struct MapLike(serde_json::Map<String, Value>);

impl MapLike {
    fn new() -> Self {
        Self(serde_json::Map::new())
    }
    fn insert(&mut self, key: &str, value: Value) {
        self.0.insert(key.to_owned(), value);
    }
    fn value(self) -> Value {
        Value::Object(self.0)
    }
}

fn parse_json_frame(frame: &RealtimeFrame) -> Result<Value, RealtimeError> {
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(RealtimeError::Codec {
            message: "OpenAI Live events must be JSON text frames".into(),
        });
    };
    serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
        message: "OpenAI Live event is not valid JSON".into(),
    })
}

fn get_event_type(native: &Value) -> Result<&str, RealtimeError> {
    native
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| RealtimeError::Codec {
            message: "OpenAI Live event is missing string `type`".into(),
        })
}

fn setup_error_message(native: &Value) -> String {
    let error = native.get("error").unwrap_or(&Value::Null);
    match error.get("code").and_then(Value::as_str) {
        Some(code) if code.len() <= 128 => {
            format!("OpenAI Live rejected session.start (code: {code})")
        }
        _ => "OpenAI Live rejected session.start".into(),
    }
}

fn decode_event(
    frame: RealtimeFrame,
    format: OpenAiLiveAudioFormat,
) -> Result<OpenAiLiveEvent, RealtimeError> {
    let native = parse_json_frame(&frame)?;
    let event_type = get_event_type(&native)?;
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let number = |key: &str| native.get(key).and_then(Value::as_f64);
    let client_event_id = string("client_event_id");
    Ok(match event_type {
        "session.started" => OpenAiLiveEvent::SessionStarted {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            client_event_id,
            native,
        },
        "session.updated" => OpenAiLiveEvent::SessionUpdated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            client_event_id,
            native,
        },
        "session.input_audio.muted" => OpenAiLiveEvent::InputAudioMuted {
            client_event_id,
            native,
        },
        "session.input_audio.unmuted" => OpenAiLiveEvent::InputAudioUnmuted {
            client_event_id,
            native,
        },
        "session.instructions.appended" => OpenAiLiveEvent::InstructionsAppended {
            start_ms: number("start_ms"),
            end_ms: number("end_ms"),
            client_event_id,
            native,
        },
        "session.thinking.appended" => OpenAiLiveEvent::ThinkingAppended {
            start_ms: number("start_ms"),
            end_ms: number("end_ms"),
            client_event_id,
            native,
        },
        "session.commentary.appended" => OpenAiLiveEvent::CommentaryAppended {
            start_ms: number("start_ms"),
            end_ms: number("end_ms"),
            client_event_id,
            native,
        },
        "session.input_transcript.delta" => OpenAiLiveEvent::InputTranscriptDelta {
            delta: string("delta").unwrap_or_default(),
            start_ms: number("start_ms"),
            end_ms: number("end_ms"),
            native,
        },
        "session.output_transcript.delta" => OpenAiLiveEvent::OutputTranscriptDelta {
            delta: string("delta").unwrap_or_default(),
            start_ms: number("start_ms"),
            end_ms: number("end_ms"),
            native,
        },
        "session.output_audio.delta" => {
            let encoded = string("delta").ok_or_else(|| RealtimeError::Codec {
                message: "OpenAI Live output audio is missing base64 `delta`".into(),
            })?;
            let data = STANDARD.decode(encoded).map_err(|_| RealtimeError::Codec {
                message: "OpenAI Live output audio `delta` is not valid base64".into(),
            })?;
            OpenAiLiveEvent::OutputAudioDelta {
                data: Bytes::from(data),
                format,
                native,
            }
        }
        "session.delegation.created" => {
            let delegation = native.get("delegation").unwrap_or(&Value::Null);
            OpenAiLiveEvent::DelegationCreated {
                delegation_id: delegation
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                target: delegation
                    .get("target")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                response_id: delegation
                    .get("response_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                offset_ms: number("offset_ms"),
                native,
            }
        }
        "response.event" => OpenAiLiveEvent::ResponseEvent {
            delegation_id: string("delegation_id"),
            event: native.get("event").cloned().unwrap_or(Value::Null),
            native,
        },
        "session.usage.updated" => OpenAiLiveEvent::UsageUpdated {
            seconds: native
                .get("usage")
                .and_then(|usage| usage.get("seconds"))
                .and_then(Value::as_f64),
            context_window: native.get("context_window").cloned(),
            native,
        },
        "session.closed" => OpenAiLiveEvent::SessionClosed {
            reason: string("reason"),
            session: native.get("session").cloned().unwrap_or(Value::Null),
            usage: native.get("usage").cloned().unwrap_or(Value::Null),
            native,
        },
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            OpenAiLiveEvent::ProviderError {
                error_type: error.get("type").and_then(Value::as_str).map(str::to_owned),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("OpenAI Live provider error")
                    .to_owned(),
                param: error
                    .get("param")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                client_event_id: error
                    .get("client_event_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or(client_event_id),
                native,
            }
        }
        _ => OpenAiLiveEvent::Native {
            event_type: event_type.to_owned(),
            native,
        },
    })
}
