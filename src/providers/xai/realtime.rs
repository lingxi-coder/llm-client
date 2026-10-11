//! xAI Speech-to-Speech Realtime WebSocket session adapter.

use crate::realtime::{
    validate_frame, validate_limits, RealtimeAudioFormat, RealtimeCodec, RealtimeConnectRequest,
    RealtimeConnection, RealtimeControl, RealtimeDriver, RealtimeError, RealtimeEvent,
    RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeToolResult,
    RealtimeTransport, MAX_REALTIME_TOOL_RESULTS,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fmt, sync::Arc};

const XAI_PCM_DEFAULT_RATE: u32 = 24_000;
const MAX_SETUP_PREFACE_FRAMES: usize = 16;

/// xAI's automatic turn detector or explicit client-committed turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XaiTurnDetection {
    /// Server-side VAD detects speech and starts responses automatically.
    ServerVad,
    /// The host sends `CommitAudio` after appending one utterance.
    Manual,
}

/// The reasoning mode documented for xAI Voice sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XaiRealtimeReasoningEffort {
    High,
    None,
}

impl XaiRealtimeReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::None => "none",
        }
    }
}

/// Optional xAI server VAD controls. The provider documents bounds for the
/// threshold, silence duration, and prefix padding; it publishes no numeric
/// range for the idle timeout.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct XaiRealtimeVadOptions {
    pub threshold: Option<f64>,
    pub silence_duration_ms: Option<u64>,
    pub prefix_padding_ms: Option<u64>,
    pub idle_timeout_ms: Option<u64>,
}

/// Select the documented JSON/base64 or raw binary WebSocket audio path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum XaiRealtimeAudioTransport {
    #[default]
    Json,
    Binary,
}

/// A client-executed function advertised in the xAI Voice session.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiRealtimeFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl XaiRealtimeFunctionTool {
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A persisted xAI conversation ID bound to its model, endpoint route, and
/// caller-chosen credential scope. Use [`Self::import`] when restoring a
/// conversation ID stored outside this process.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct XaiRealtimeResumeRef {
    conversation_id: String,
    model: String,
    endpoint_scope: String,
    credential_scope: String,
}

impl XaiRealtimeResumeRef {
    /// Import a conversation ID with the exact non-secret identity fields that
    /// will be checked again before a resumed connection is opened.
    pub fn import(
        conversation_id: impl Into<String>,
        model: impl Into<String>,
        endpoint: &str,
        credential_scope: impl Into<String>,
    ) -> Result<Self, RealtimeError> {
        let conversation_id = conversation_id.into();
        let model = model.into();
        let credential_scope = credential_scope.into();
        if model.trim().is_empty() || model.trim() != model.as_str() {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime resume references require a model matching the endpoint model value".into(),
            });
        }
        let endpoint_url = url::Url::parse(endpoint).map_err(|_| RealtimeError::InvalidConfig {
            message: "xAI Realtime endpoint must be an absolute wss:// URL".into(),
        })?;
        let endpoint_models = endpoint_url
            .query_pairs()
            .filter(|(name, _)| name == "model")
            .map(|(_, value)| value.into_owned())
            .collect::<Vec<_>>();
        let endpoint_model_matches = match endpoint_models.as_slice() {
            [] => true,
            [endpoint_model] => endpoint_model == &model,
            _ => false,
        };
        if !endpoint_model_matches {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime resume reference model must match its endpoint model query"
                    .into(),
            });
        }
        let endpoint_scope = endpoint_scope(endpoint)?;
        if conversation_id.trim().is_empty()
            || model.trim().is_empty()
            || credential_scope.trim().is_empty()
        {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime resume references require non-empty conversation, model, and credential scope values".into(),
            });
        }
        Ok(Self {
            conversation_id,
            model,
            endpoint_scope,
            credential_scope,
        })
    }

    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn credential_scope(&self) -> &str {
        &self.credential_scope
    }
}

impl fmt::Debug for XaiRealtimeResumeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiRealtimeResumeRef")
            .field("conversation_id", &"<redacted>")
            .field("model", &self.model)
            .field("endpoint_scope", &"<scoped>")
            .field("credential_scope", &"<redacted>")
            .finish()
    }
}

/// Session configuration sent in xAI's `session.update` event.
///
/// The model is selected by the `model` query parameter in the connection
/// endpoint. Audio uses JSON/base64 transport by default; input and output can
/// independently opt into the documented binary transport.
#[derive(Clone)]
pub struct XaiRealtimeConfig {
    /// Model sent in the Realtime endpoint's `model` query parameter.
    pub model: String,
    /// Stable, non-secret account identity chosen by the host. Required when
    /// resumption is enabled so saved references cannot cross accounts.
    pub credential_scope: String,
    pub voice: Option<String>,
    pub instructions: Option<String>,
    pub turn_detection: XaiTurnDetection,
    pub vad: XaiRealtimeVadOptions,
    pub reasoning_effort: Option<XaiRealtimeReasoningEffort>,
    /// Enable xAI's cumulative input transcription events (`grok-transcribe`).
    pub input_transcription: bool,
    pub input_transcription_language_hint: Option<String>,
    pub input_transcription_keyterms: Vec<String>,
    pub input_audio_format: RealtimeAudioFormat,
    pub output_audio_format: RealtimeAudioFormat,
    pub input_audio_transport: XaiRealtimeAudioTransport,
    pub output_audio_transport: XaiRealtimeAudioTransport,
    pub output_audio_speed: Option<f64>,
    pub replace: BTreeMap<String, String>,
    pub tools: Vec<XaiRealtimeFunctionTool>,
    pub resumption_enabled: bool,
    /// Scoped conversation reference from `conversation.created` or an
    /// explicit [`XaiRealtimeResumeRef::import`].
    pub resume_ref: Option<XaiRealtimeResumeRef>,
}

impl Default for XaiRealtimeConfig {
    fn default() -> Self {
        Self {
            model: "grok-voice-latest".into(),
            credential_scope: String::new(),
            voice: Some("eve".into()),
            instructions: None,
            turn_detection: XaiTurnDetection::ServerVad,
            vad: XaiRealtimeVadOptions::default(),
            reasoning_effort: None,
            input_transcription: false,
            input_transcription_language_hint: None,
            input_transcription_keyterms: Vec::new(),
            input_audio_format: RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: XAI_PCM_DEFAULT_RATE,
            },
            output_audio_format: RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: XAI_PCM_DEFAULT_RATE,
            },
            input_audio_transport: XaiRealtimeAudioTransport::Json,
            output_audio_transport: XaiRealtimeAudioTransport::Json,
            output_audio_speed: None,
            replace: BTreeMap::new(),
            tools: Vec::new(),
            resumption_enabled: false,
            resume_ref: None,
        }
    }
}

impl fmt::Debug for XaiRealtimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiRealtimeConfig")
            .field("voice", &self.voice)
            .field("model", &self.model)
            .field("credential_scope", &"<redacted>")
            .field("instructions", &self.instructions.as_ref().map(|_| "<set>"))
            .field("turn_detection", &self.turn_detection)
            .field("vad", &self.vad)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("input_transcription", &self.input_transcription)
            .field(
                "input_transcription_language_hint",
                &self
                    .input_transcription_language_hint
                    .as_ref()
                    .map(|_| "<set>"),
            )
            .field(
                "input_transcription_keyterms",
                &self.input_transcription_keyterms.len(),
            )
            .field("input_audio_format", &self.input_audio_format)
            .field("output_audio_format", &self.output_audio_format)
            .field("input_audio_transport", &self.input_audio_transport)
            .field("output_audio_transport", &self.output_audio_transport)
            .field("output_audio_speed", &self.output_audio_speed)
            .field("replace_entries", &self.replace.len())
            .field("tools", &self.tools.len())
            .field("resumption_enabled", &self.resumption_enabled)
            .field("resume_ref", &self.resume_ref)
            .finish()
    }
}

/// Typed xAI Realtime server events. Unrecognized event types retain their
/// original JSON so newer provider events remain inspectable.
#[derive(Debug, Clone, PartialEq)]
pub enum XaiRealtimeEvent {
    Realtime(RealtimeEvent),
    SessionCreated {
        session: Value,
        native: Value,
    },
    ConversationCreated {
        conversation: Value,
        native: Value,
    },
    SessionUpdated {
        session: Value,
        native: Value,
    },
    InputAudioBufferCleared {
        native: Value,
    },
    InputAudioBufferTimeoutTriggered {
        native: Value,
    },
    ConversationItemAdded {
        item_id: Option<String>,
        native: Value,
    },
    ConversationItemDeleted {
        item_id: Option<String>,
        native: Value,
    },
    ConversationItemTruncated {
        item_id: Option<String>,
        native: Value,
    },
    ResponseOutputItemAdded {
        native: Value,
    },
    ResponseOutputItemDone {
        native: Value,
    },
    ResponseContentPartAdded {
        native: Value,
    },
    ResponseContentPartDone {
        native: Value,
    },
    McpEvent {
        event_type: String,
        native: Value,
    },
    SpeechStarted {
        item_id: Option<String>,
        audio_start_ms: Option<u64>,
        native: Value,
    },
    SpeechStopped {
        item_id: Option<String>,
        audio_end_ms: Option<u64>,
        native: Value,
    },
    InputAudioCommitted {
        item_id: Option<String>,
        previous_item_id: Option<String>,
        native: Value,
    },
    InputTranscriptionUpdated {
        item_id: Option<String>,
        transcript: String,
        native: Value,
    },
    InputTranscriptionCompleted {
        item_id: Option<String>,
        transcript: String,
        native: Value,
    },
    ResponseCreated {
        response_id: Option<String>,
        native: Value,
    },
    AudioDelta {
        response_id: Option<String>,
        item_id: Option<String>,
        output_index: Option<u64>,
        content_index: Option<u64>,
        data: Bytes,
        format: RealtimeAudioFormat,
        native: Value,
    },
    AudioDone {
        response_id: Option<String>,
        item_id: Option<String>,
        native: Value,
    },
    /// Audio delivered in a binary WebSocket frame. Binary frames do not
    /// carry the JSON response/item identifiers available on JSON deltas.
    AudioBinaryDelta {
        data: Bytes,
        format: RealtimeAudioFormat,
    },
    AudioTranscriptDelta {
        response_id: Option<String>,
        item_id: Option<String>,
        delta: String,
        native: Value,
    },
    AudioTranscriptDone {
        response_id: Option<String>,
        item_id: Option<String>,
        transcript: String,
        native: Value,
    },
    TextDelta {
        response_id: Option<String>,
        item_id: Option<String>,
        delta: String,
        native: Value,
    },
    TextDone {
        response_id: Option<String>,
        item_id: Option<String>,
        text: String,
        native: Value,
    },
    FunctionCallArgumentsDelta {
        response_id: Option<String>,
        item_id: Option<String>,
        call_id: Option<String>,
        delta: String,
        native: Value,
    },
    FunctionCallArgumentsDone {
        response_id: Option<String>,
        item_id: Option<String>,
        call_id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
        native: Value,
    },
    ResponseDone {
        response_id: Option<String>,
        response: Value,
        native: Value,
    },
    ProviderError {
        code: Option<String>,
        message: String,
        native: Value,
    },
    Native {
        event_type: String,
        native: Value,
    },
}

/// A connected xAI Speech-to-Speech session. `connect` waits for
/// `session.updated`, so its returned control can immediately send input.
pub struct XaiRealtimeSession {
    control: XaiRealtimeControl,
    events: XaiRealtimeEvents,
}

impl XaiRealtimeSession {
    /// Connect to an xAI Realtime endpoint, wait for `session.created`, send
    /// `session.update`, then wait for its `session.updated` acknowledgement.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        mut request: RealtimeConnectRequest,
        config: XaiRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        validate_endpoint(&request.endpoint)?;
        apply_session_route(&mut request, &config)?;
        let setup = build_session_update(&config)?;
        let setup_bytes = serde_json::to_vec(&setup).map_err(|_| RealtimeError::Codec {
            message: "could not encode xAI Realtime session update".into(),
        })?;
        if setup_bytes.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: setup_bytes.len(),
                max: limits.max_frame_bytes,
            });
        }

        let setup_transport = XaiSetupTransport {
            inner: transport,
            setup_frame: RealtimeFrame::Text(Bytes::from(setup_bytes)),
            max_frame_bytes: limits.max_frame_bytes,
        };
        let codec = Arc::new(XaiRealtimeCodec {
            input_audio_format: config.input_audio_format,
            output_audio_format: config.output_audio_format.clone(),
            input_audio_transport: config.input_audio_transport,
            output_audio_transport: config.output_audio_transport,
            turn_detection: config.turn_detection,
        });
        let (session, driver) =
            RealtimeSession::connect(&setup_transport, request, codec, limits).await?;
        let (control, events) = session.into_parts();
        Ok((
            Self {
                control: XaiRealtimeControl { inner: control },
                events: XaiRealtimeEvents {
                    inner: events,
                    output_audio_format: config.output_audio_format,
                    output_audio_transport: config.output_audio_transport,
                },
            },
            driver,
        ))
    }

    pub fn into_realtime_parts(self) -> (RealtimeControl, crate::realtime::RealtimeEvents) {
        (self.control.inner, self.events.inner)
    }

    pub fn into_parts(self) -> (XaiRealtimeControl, XaiRealtimeEvents) {
        (self.control, self.events)
    }
}

/// xAI-specific command surface layered over the bounded generic control
/// handle. Provider commands are JSON-encoded and obey the same queue/frame
/// limits as generic realtime inputs.
#[derive(Clone)]
pub struct XaiRealtimeControl {
    inner: RealtimeControl,
}

impl XaiRealtimeControl {
    pub fn send(&self, input: RealtimeInput) -> Result<(), RealtimeError> {
        self.inner.send(input)
    }

    pub fn capabilities(&self) -> crate::realtime::RealtimeCapabilities {
        self.inner.capabilities()
    }
    pub async fn abort(&self, close: crate::realtime::RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.abort(close).await
    }

    pub fn interrupt(&self) -> Result<(), RealtimeError> {
        self.inner.interrupt()
    }

    pub async fn close(&self, close: crate::realtime::RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.close(close).await
    }

    pub fn send_command(&self, command: XaiRealtimeCommand) -> Result<(), RealtimeError> {
        match command {
            XaiRealtimeCommand::ClearAudioBuffer => self.inner.send(RealtimeInput::ClearAudio),
            XaiRealtimeCommand::ForceMessage {
                text,
                interruptible,
            } => {
                if text.trim().is_empty() {
                    return Err(RealtimeError::InvalidInput {
                        message: "xAI force_message requires non-empty text".into(),
                    });
                }
                let mut item = json!({
                    "type": "force_message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text }]
                });
                if let Some(interruptible) = interruptible {
                    item["interruptible"] = Value::Bool(interruptible);
                }
                self.send_json(json!({ "type": "conversation.item.create", "item": item }))
            }
            XaiRealtimeCommand::CreateResponse { instructions } => {
                let mut event = json!({ "type": "response.create" });
                if let Some(instructions) = instructions {
                    event["response"] = json!({ "instructions": instructions });
                }
                self.send_json(event)
            }
        }
    }

    fn send_json(&self, value: Value) -> Result<(), RealtimeError> {
        let bytes = serde_json::to_vec(&value).map_err(|_| RealtimeError::Codec {
            message: "could not encode xAI Realtime command".into(),
        })?;
        let frame = RealtimeFrame::Text(Bytes::from(bytes));
        self.inner.send_provider_frame(frame)
    }
}

/// Explicit client-to-server commands documented for xAI Voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XaiRealtimeCommand {
    /// Discard audio appended to the current, uncommitted input buffer.
    ClearAudioBuffer,
    /// Insert a scripted assistant utterance without requesting a model turn.
    ForceMessage {
        text: String,
        /// `None` preserves xAI's documented default (`true`).
        interruptible: Option<bool>,
    },
    /// Request a response, optionally overriding session instructions for
    /// this response only.
    CreateResponse { instructions: Option<String> },
}

/// Receiver that converts the xAI JSON event union into typed provider events.
pub struct XaiRealtimeEvents {
    output_audio_transport: XaiRealtimeAudioTransport,
    inner: crate::realtime::RealtimeEvents,
    output_audio_format: RealtimeAudioFormat,
}

impl XaiRealtimeEvents {
    pub async fn next(&mut self) -> Option<XaiRealtimeEvent> {
        loop {
            match self.inner.next().await? {
                RealtimeEvent::ProviderEvent { name, native } => {
                    return Some(parse_native_event(
                        &name,
                        &native,
                        &self.output_audio_format,
                    ))
                }
                RealtimeEvent::AudioDelta { data, format, .. }
                    if self.output_audio_transport == XaiRealtimeAudioTransport::Binary =>
                {
                    return Some(XaiRealtimeEvent::AudioBinaryDelta { data, format })
                }
                event @ (RealtimeEvent::Closed { .. }
                | RealtimeEvent::ConnectionInterrupted { .. }) => {
                    return Some(XaiRealtimeEvent::Realtime(event))
                }
                event @ RealtimeEvent::ProviderError { code: Some(_), .. } if matches!(&event, RealtimeEvent::ProviderError { code:Some(code),.. } if matches!(code.as_str(), "invalid_provider_frame" | "frame_too_large")) => {
                    return Some(XaiRealtimeEvent::Realtime(event))
                }
                _ => {}
            }
        }
    }
}

struct XaiSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
}

#[async_trait]
impl RealtimeTransport for XaiSetupTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let mut connection = self.inner.connect(request).await?;
        let mut preface = Vec::new();
        let mut update_sent = false;
        let mut update_acknowledged = false;

        for _ in 0..MAX_SETUP_PREFACE_FRAMES {
            let frame = connection
                .inbound
                .next()
                .await
                .ok_or(RealtimeError::UnexpectedRemoteClose)??;
            validate_frame(&frame, self.max_frame_bytes)?;
            let RealtimeFrame::Text(bytes) = &frame else {
                return Err(RealtimeError::Codec {
                    message: "xAI Realtime session setup requires JSON text events".into(),
                });
            };
            let message: Value =
                serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
                    message: "xAI Realtime setup event is not valid JSON".into(),
                })?;
            let event_type = message.get("type").and_then(Value::as_str).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "xAI Realtime setup event is missing string `type`".into(),
                }
            })?;
            if event_type == "error" {
                return Err(RealtimeError::Transport {
                    message: "xAI rejected the Realtime session configuration".into(),
                });
            }
            preface.push(Ok(frame));
            if event_type == "session.created" && !update_sent {
                connection.outbound.send(self.setup_frame.clone()).await?;
                update_sent = true;
            }
            if event_type == "session.updated" && update_sent {
                update_acknowledged = true;
                break;
            }
        }
        if !update_acknowledged {
            return Err(RealtimeError::Transport {
                message:
                    "xAI Realtime did not acknowledge session.update within the setup preface limit"
                        .into(),
            });
        }

        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct XaiRealtimeCodec {
    input_audio_format: RealtimeAudioFormat,
    output_audio_format: RealtimeAudioFormat,
    input_audio_transport: XaiRealtimeAudioTransport,
    output_audio_transport: XaiRealtimeAudioTransport,
    turn_detection: XaiTurnDetection,
}

impl RealtimeCodec for XaiRealtimeCodec {
    fn capabilities(&self) -> crate::realtime::RealtimeCapabilities {
        crate::realtime::RealtimeCapabilities {
            tools: true,
            input_transcription: true,
            output_transcription: true,
            usage: true,
            interruption: self.turn_detection == XaiTurnDetection::Manual,
            session_resumption: true,
            ..Default::default()
        }
    }
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let messages = match input {
            RealtimeInput::ImportHistory { .. } => return Err(RealtimeError::InvalidInput { message: "history import is not implemented by this adapter".into() }),
            RealtimeInput::FinishSession => {
                return Err(RealtimeError::InvalidInput {
                    message: "explicit session finishing is not implemented by this adapter".into(),
                });
            }
            RealtimeInput::Text(text) => vec![
                json!({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }]
                    }
                }),
                json!({ "type": "response.create" }),
            ],
            RealtimeInput::Audio { data, format } => {
                validate_audio_format(format)?;
                if format != &self.input_audio_format {
                    return Err(RealtimeError::InvalidInput {
                        message: format!(
                            "audio format {format:?} does not match the xAI session input format {:?}",
                            self.input_audio_format
                        ),
                    });
                }
                match self.input_audio_transport {
                    XaiRealtimeAudioTransport::Json => vec![json!({
                        "type": "input_audio_buffer.append",
                        "audio": STANDARD.encode(data)
                    })],
                    XaiRealtimeAudioTransport::Binary => {
                        return Ok(vec![RealtimeFrame::Binary(data.clone())]);
                    }
                }
            }
            RealtimeInput::Image { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI Realtime image input is not supported by this codec".into(),
                });
            }
            RealtimeInput::CommitAudio if self.turn_detection == XaiTurnDetection::Manual => {
                vec![json!({ "type": "input_audio_buffer.commit" })]
            }
            RealtimeInput::CommitAudio => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI input audio commit requires manual turn detection".into(),
                })
            }
            RealtimeInput::Interrupt if self.turn_detection == XaiTurnDetection::Manual => {
                vec![json!({ "type": "response.cancel" })]
            }
            RealtimeInput::Interrupt => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI Realtime uses server VAD for turn interruption; manual cancel requires manual turn detection".into(),
                })
            }
            RealtimeInput::ClearAudio => vec![json!({ "type": "input_audio_buffer.clear" })],
            RealtimeInput::ToolResult { call_id, output , .. } => {
                vec![function_call_output(call_id, output)?]
            }
            RealtimeInput::ToolResults { results } => function_call_outputs(results)?,
            RealtimeInput::ContinueResponse => vec![json!({ "type": "response.create" })],
            RealtimeInput::RetrieveItem { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI Realtime documents conversation.item.retrieve as unsupported".into(),
                });
            }
            RealtimeInput::DeleteItem { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI Realtime documents conversation.item.delete but does not publish its event payload schema".into(),
                });
            }
            RealtimeInput::TruncateAudio { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "xAI Realtime documents conversation.item.truncate but does not publish its event payload schema".into(),
                });
            }
        };
        messages
            .into_iter()
            .map(|message| {
                serde_json::to_vec(&message)
                    .map(|bytes| RealtimeFrame::Text(Bytes::from(bytes)))
                    .map_err(|_| RealtimeError::Codec {
                        message: "could not encode xAI Realtime client event".into(),
                    })
            })
            .collect()
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let bytes = match frame {
            RealtimeFrame::Text(bytes) => bytes,
            RealtimeFrame::Binary(data) => {
                return match self.output_audio_transport {
                    XaiRealtimeAudioTransport::Binary => Ok(vec![RealtimeEvent::AudioDelta {
                        data,
                        format: self.output_audio_format.clone(),
                        item_id: None,
                    }]),
                    XaiRealtimeAudioTransport::Json => Err(RealtimeError::Codec {
                        message: "xAI sent a binary frame while JSON audio output was configured"
                            .into(),
                    }),
                };
            }
        };
        let native: Value = serde_json::from_slice(&bytes).map_err(|_| RealtimeError::Codec {
            message: "xAI Realtime server event is not valid JSON".into(),
        })?;
        let name =
            native
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| RealtimeError::Codec {
                    message: "xAI Realtime server event is missing string `type`".into(),
                })?;
        if matches!(name, "response.output_audio.delta" | "response.audio.delta") {
            if self.output_audio_transport == XaiRealtimeAudioTransport::Binary {
                return Err(RealtimeError::Codec {
                    message: "xAI sent a JSON audio delta while binary audio output was configured"
                        .into(),
                });
            }
            let audio = native.get("delta").and_then(Value::as_str).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "xAI Realtime audio delta is missing base64 `delta`".into(),
                }
            })?;
            STANDARD.decode(audio).map_err(|_| RealtimeError::Codec {
                message: "xAI Realtime audio delta is not valid base64".into(),
            })?;
        }
        crate::realtime::normalize_json_events(native, self.output_audio_format.clone(), true)
    }
}

fn function_call_outputs(results: &[RealtimeToolResult]) -> Result<Vec<Value>, RealtimeError> {
    if results.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "xAI Realtime tool result batch must not be empty".into(),
        });
    }
    if results.len() > MAX_REALTIME_TOOL_RESULTS {
        return Err(RealtimeError::InvalidInput {
            message: format!(
                "xAI Realtime tool result batch exceeds the local limit of {MAX_REALTIME_TOOL_RESULTS} results"
            ),
        });
    }
    let mut call_ids = std::collections::HashSet::with_capacity(results.len());
    for result in results {
        validate_call_id(&result.call_id)?;
        if !call_ids.insert(result.call_id.as_str()) {
            return Err(RealtimeError::InvalidInput {
                message: "xAI Realtime tool result batch contains a duplicate call_id".into(),
            });
        }
    }
    results
        .iter()
        .map(|result| function_call_output(&result.call_id, &result.output))
        .collect()
}

fn function_call_output(call_id: &str, result: &Value) -> Result<Value, RealtimeError> {
    validate_call_id(call_id)?;
    let output = serde_json::to_string(result).map_err(|_| RealtimeError::Codec {
        message: "could not encode xAI Realtime function output".into(),
    })?;
    Ok(json!({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }
    }))
}

fn validate_call_id(call_id: &str) -> Result<(), RealtimeError> {
    if call_id.trim().is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "xAI Realtime function result requires a non-empty call_id".into(),
        });
    }
    Ok(())
}

fn build_session_update(config: &XaiRealtimeConfig) -> Result<Value, RealtimeError> {
    let input_format = wire_audio_format(&config.input_audio_format)?;
    let output_format = wire_audio_format(&config.output_audio_format)?;
    validate_config(config)?;
    let mut session = serde_json::Map::new();
    if let Some(voice) = &config.voice {
        if voice.trim().is_empty() {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime voice must not be empty".into(),
            });
        }
        session.insert("voice".into(), Value::String(voice.clone()));
    }
    if let Some(instructions) = &config.instructions {
        session.insert("instructions".into(), Value::String(instructions.clone()));
    }
    if let Some(reasoning_effort) = config.reasoning_effort {
        session.insert(
            "reasoning".into(),
            json!({ "effort": reasoning_effort.as_str() }),
        );
    }
    let mut turn_detection = match config.turn_detection {
        XaiTurnDetection::ServerVad => json!({ "type": "server_vad" }),
        XaiTurnDetection::Manual => json!({ "type": null }),
    };
    if config.turn_detection == XaiTurnDetection::ServerVad {
        if let Some(threshold) = config.vad.threshold {
            turn_detection["threshold"] = json!(threshold);
        }
        if let Some(silence_duration_ms) = config.vad.silence_duration_ms {
            turn_detection["silence_duration_ms"] = json!(silence_duration_ms);
        }
        if let Some(prefix_padding_ms) = config.vad.prefix_padding_ms {
            turn_detection["prefix_padding_ms"] = json!(prefix_padding_ms);
        }
        if let Some(idle_timeout_ms) = config.vad.idle_timeout_ms {
            turn_detection["idle_timeout_ms"] = json!(idle_timeout_ms);
        }
    }
    session.insert("turn_detection".into(), turn_detection);
    let mut audio = json!({
        "input": { "format": input_format, "transport": audio_transport_name(config.input_audio_transport) },
        "output": { "format": output_format, "transport": audio_transport_name(config.output_audio_transport) }
    });
    if config.input_transcription {
        audio["input"]["transcription"] = json!({ "model": "grok-transcribe" });
    }
    if let Some(language_hint) = &config.input_transcription_language_hint {
        audio["input"]["transcription"]["language_hint"] = Value::String(language_hint.clone());
    }
    if !config.input_transcription_keyterms.is_empty() {
        audio["input"]["transcription"]["keyterms"] = json!(config.input_transcription_keyterms);
    }
    if let Some(speed) = config.output_audio_speed {
        audio["output"]["speed"] = json!(speed);
    }
    session.insert("audio".into(), audio);
    let tools = encode_function_tools(&config.tools)?;
    if !tools.is_empty() {
        session.insert("tools".into(), Value::Array(tools));
    }
    if !config.replace.is_empty() {
        session.insert(
            "replace".into(),
            serde_json::to_value(&config.replace).map_err(|_| RealtimeError::Codec {
                message: "could not encode xAI Realtime replacements".into(),
            })?,
        );
    }
    if config.resumption_enabled {
        session.insert("resumption".into(), json!({ "enabled": true }));
    }
    Ok(json!({ "type": "session.update", "session": session }))
}

fn wire_audio_format(format: &RealtimeAudioFormat) -> Result<Value, RealtimeError> {
    match format {
        RealtimeAudioFormat::Pcm16 { sample_rate_hz }
            if matches!(
                *sample_rate_hz,
                8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000
            ) =>
        {
            Ok(json!({ "type": "audio/pcm", "rate": sample_rate_hz }))
        }
        RealtimeAudioFormat::G711MuLaw => Ok(json!({ "type": "audio/pcmu" })),
        RealtimeAudioFormat::G711ALaw => Ok(json!({ "type": "audio/pcma" })),
        RealtimeAudioFormat::Encoded { mime_type } if mime_type == "audio/opus" => {
            Ok(json!({ "type": "audio/opus" }))
        }
        _ => Err(RealtimeError::InvalidConfig {
            message: format!("unsupported xAI Realtime audio format: {format:?}"),
        }),
    }
}

fn validate_audio_format(format: &RealtimeAudioFormat) -> Result<(), RealtimeError> {
    match format {
        RealtimeAudioFormat::Pcm16 { sample_rate_hz }
            if matches!(
                *sample_rate_hz,
                8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000
            ) =>
        {
            Ok(())
        }
        RealtimeAudioFormat::G711MuLaw | RealtimeAudioFormat::G711ALaw => Ok(()),
        RealtimeAudioFormat::Encoded { mime_type } if mime_type == "audio/opus" => Ok(()),
        _ => Err(RealtimeError::InvalidInput {
            message: format!("unsupported xAI Realtime audio format: {format:?}"),
        }),
    }
}

fn audio_transport_name(transport: XaiRealtimeAudioTransport) -> &'static str {
    match transport {
        XaiRealtimeAudioTransport::Json => "json",
        XaiRealtimeAudioTransport::Binary => "binary",
    }
}

fn validate_config(config: &XaiRealtimeConfig) -> Result<(), RealtimeError> {
    if config.model.trim().is_empty() || config.model.trim() != config.model.as_str() {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime model must not be empty".into(),
        });
    }
    if let Some(threshold) = config.vad.threshold {
        if !threshold.is_finite() || !(0.1..=0.9).contains(&threshold) {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime VAD threshold must be between 0.1 and 0.9".into(),
            });
        }
    }
    for (name, value) in [
        ("silence_duration_ms", config.vad.silence_duration_ms),
        ("prefix_padding_ms", config.vad.prefix_padding_ms),
    ] {
        if value.is_some_and(|value| value > 10_000) {
            return Err(RealtimeError::InvalidConfig {
                message: format!("xAI Realtime VAD {name} must not exceed 10000"),
            });
        }
    }
    if config.turn_detection == XaiTurnDetection::Manual
        && (config.vad.threshold.is_some()
            || config.vad.silence_duration_ms.is_some()
            || config.vad.prefix_padding_ms.is_some()
            || config.vad.idle_timeout_ms.is_some())
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime VAD options require server_vad turn detection".into(),
        });
    }
    if !config.input_transcription
        && (config.input_transcription_language_hint.is_some()
            || !config.input_transcription_keyterms.is_empty())
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime transcription hints require input_transcription".into(),
        });
    }
    if config
        .input_transcription_language_hint
        .as_ref()
        .is_some_and(|hint| hint.trim().is_empty())
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime transcription language hint must not be empty".into(),
        });
    }
    if config.input_transcription_keyterms.len() > 100
        || config
            .input_transcription_keyterms
            .iter()
            .any(|term| term.chars().count() > 50)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime transcription supports at most 100 keyterms of at most 50 characters each".into(),
        });
    }
    if config
        .output_audio_speed
        .is_some_and(|speed| !speed.is_finite() || !(0.7..=1.5).contains(&speed))
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime output audio speed must be between 0.7 and 1.5".into(),
        });
    }
    if config.resumption_enabled && config.credential_scope.trim().is_empty() {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime resumption requires a non-empty host credential scope".into(),
        });
    }
    if config.resume_ref.is_some() && !config.resumption_enabled {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime resume reference requires resumption_enabled".into(),
        });
    }
    if config.replace.keys().any(|phrase| phrase.trim().is_empty()) {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime replacement phrases must not be empty".into(),
        });
    }
    let _ = encode_function_tools(&config.tools)?;
    Ok(())
}

fn encode_function_tools(tools: &[XaiRealtimeFunctionTool]) -> Result<Vec<Value>, RealtimeError> {
    let mut names = std::collections::HashSet::with_capacity(tools.len());
    let mut encoded = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool.name.trim().is_empty() || !tool.parameters.is_object() {
            return Err(RealtimeError::InvalidConfig {
                message:
                    "xAI Realtime function tools require a non-empty name and object JSON schema"
                        .into(),
            });
        }
        if !names.insert(tool.name.as_str()) {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime function tool names must be unique".into(),
            });
        }
        let mut value = json!({
            "type": "function",
            "name": tool.name,
            "parameters": tool.parameters
        });
        if let Some(description) = &tool.description {
            value["description"] = Value::String(description.clone());
        }
        encoded.push(value);
    }
    Ok(encoded)
}

fn apply_session_route(
    request: &mut RealtimeConnectRequest,
    config: &XaiRealtimeConfig,
) -> Result<(), RealtimeError> {
    let mut endpoint =
        url::Url::parse(&request.endpoint).map_err(|_| RealtimeError::InvalidConfig {
            message: "xAI Realtime endpoint must be an absolute wss:// URL".into(),
        })?;
    if let Some(resume_ref) = &config.resume_ref {
        let endpoint_scope = endpoint_scope(&request.endpoint)?;
        if resume_ref.conversation_id.trim().is_empty()
            || resume_ref.model != config.model
            || resume_ref.endpoint_scope != endpoint_scope
            || resume_ref.credential_scope != config.credential_scope
        {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime resume reference belongs to a different model, endpoint, or credential scope".into(),
            });
        }
    }

    let existing_models = endpoint
        .query_pairs()
        .filter(|(name, _)| name == "model")
        .map(|(_, value)| value.into_owned())
        .collect::<Vec<_>>();
    match existing_models.as_slice() {
        [model] if model == &config.model => {}
        [] => {
            endpoint
                .query_pairs_mut()
                .append_pair("model", &config.model);
        }
        _ => {
            return Err(RealtimeError::InvalidConfig {
                message:
                    "xAI Realtime endpoint model conflicts with or duplicates the configured model"
                        .into(),
            });
        }
    }

    let existing_conversation_ids = endpoint
        .query_pairs()
        .filter(|(name, _)| name == "conversation_id")
        .map(|(_, value)| value.into_owned())
        .collect::<Vec<_>>();
    match (&config.resume_ref, existing_conversation_ids.as_slice()) {
        (Some(resume_ref), [conversation_id]) if conversation_id == &resume_ref.conversation_id => {
        }
        (Some(resume_ref), []) => {
            endpoint
                .query_pairs_mut()
                .append_pair("conversation_id", &resume_ref.conversation_id);
        }
        (None, []) => {}
        _ => {
            return Err(RealtimeError::InvalidConfig {
                message: "xAI Realtime conversation_id query must match an explicit scoped resume reference".into(),
            });
        }
    }
    request.endpoint = endpoint.into();
    Ok(())
}

fn endpoint_scope(endpoint: &str) -> Result<String, RealtimeError> {
    let parsed = url::Url::parse(endpoint).map_err(|_| RealtimeError::InvalidConfig {
        message: "xAI Realtime endpoint must be an absolute wss:// URL".into(),
    })?;
    if parsed.scheme() != "wss"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime endpoint must be a wss:// URL without user-info".into(),
        });
    }
    let mut query = Vec::new();
    for (name, value) in parsed.query_pairs() {
        match name.as_ref() {
            "model" | "conversation_id" => {}
            "reasoning.effort" => query.push((name.into_owned(), value.into_owned())),
            name if is_credential_query_name(name) => {}
            _ => {
                return Err(RealtimeError::InvalidConfig {
                    message:
                        "xAI Realtime resume references require documented route query parameters"
                            .into(),
                });
            }
        }
    }
    query.sort();
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (name, value) in query {
        serializer.append_pair(&name, &value);
    }
    let query = serializer.finish();
    let base = format!("{}{}", parsed.origin().ascii_serialization(), parsed.path());
    if query.is_empty() {
        Ok(base)
    } else {
        Ok(format!("{base}?{query}"))
    }
}

fn is_credential_query_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "key" | "api_key" | "apikey" | "access_token" | "token" | "client_secret"
    )
}

fn validate_endpoint(endpoint: &str) -> Result<(), RealtimeError> {
    let parsed = url::Url::parse(endpoint).map_err(|_| RealtimeError::InvalidConfig {
        message: "xAI Realtime endpoint must be an absolute wss:// URL".into(),
    })?;
    if parsed.scheme() != "wss"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(RealtimeError::InvalidConfig {
            message: "xAI Realtime endpoint must be a wss:// URL without user-info".into(),
        });
    }
    Ok(())
}

fn parse_native_event(
    event_type: &str,
    native: &Value,
    output_audio_format: &RealtimeAudioFormat,
) -> XaiRealtimeEvent {
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let number = |key: &str| native.get(key).and_then(Value::as_u64);
    let nested_string = |outer: &str, key: &str| {
        native
            .get(outer)
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    match event_type {
        "session.created" => XaiRealtimeEvent::SessionCreated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "conversation.created" => XaiRealtimeEvent::ConversationCreated {
            conversation: native.get("conversation").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "session.updated" => XaiRealtimeEvent::SessionUpdated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "input_audio_buffer.cleared" => XaiRealtimeEvent::InputAudioBufferCleared {
            native: native.clone(),
        },
        "input_audio_buffer.timeout_triggered" => {
            XaiRealtimeEvent::InputAudioBufferTimeoutTriggered {
                native: native.clone(),
            }
        }
        "conversation.item.added" | "conversation.item.created" => {
            XaiRealtimeEvent::ConversationItemAdded {
                item_id: string("item_id").or_else(|| nested_string("item", "id")),
                native: native.clone(),
            }
        }
        "conversation.item.deleted" => XaiRealtimeEvent::ConversationItemDeleted {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "conversation.item.truncated" => XaiRealtimeEvent::ConversationItemTruncated {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "response.output_item.added" => XaiRealtimeEvent::ResponseOutputItemAdded {
            native: native.clone(),
        },
        "response.output_item.done" => XaiRealtimeEvent::ResponseOutputItemDone {
            native: native.clone(),
        },
        "response.content_part.added" => XaiRealtimeEvent::ResponseContentPartAdded {
            native: native.clone(),
        },
        "response.content_part.done" => XaiRealtimeEvent::ResponseContentPartDone {
            native: native.clone(),
        },
        "input_audio_buffer.speech_started" => XaiRealtimeEvent::SpeechStarted {
            item_id: string("item_id"),
            audio_start_ms: number("audio_start_ms"),
            native: native.clone(),
        },
        "input_audio_buffer.speech_stopped" => XaiRealtimeEvent::SpeechStopped {
            item_id: string("item_id"),
            audio_end_ms: number("audio_end_ms"),
            native: native.clone(),
        },
        "input_audio_buffer.committed" => XaiRealtimeEvent::InputAudioCommitted {
            item_id: string("item_id"),
            previous_item_id: string("previous_item_id"),
            native: native.clone(),
        },
        "conversation.item.input_audio_transcription.updated" => {
            XaiRealtimeEvent::InputTranscriptionUpdated {
                item_id: string("item_id"),
                transcript: string("transcript").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.completed" => {
            XaiRealtimeEvent::InputTranscriptionCompleted {
                item_id: string("item_id"),
                transcript: string("transcript").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "response.created" => XaiRealtimeEvent::ResponseCreated {
            response_id: nested_string("response", "id"),
            native: native.clone(),
        },
        "response.output_audio.delta" | "response.audio.delta" => {
            let data = string("delta")
                .and_then(|delta| STANDARD.decode(delta).ok())
                .map(Bytes::from)
                .unwrap_or_default();
            XaiRealtimeEvent::AudioDelta {
                response_id: string("response_id"),
                item_id: string("item_id"),
                output_index: number("output_index"),
                content_index: number("content_index"),
                data,
                format: output_audio_format.clone(),
                native: native.clone(),
            }
        }
        "response.output_audio.done" | "response.audio.done" => XaiRealtimeEvent::AudioDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            native: native.clone(),
        },
        "response.output_audio_transcript.delta" => XaiRealtimeEvent::AudioTranscriptDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.output_audio_transcript.done" => XaiRealtimeEvent::AudioTranscriptDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            transcript: string("transcript").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.delta" | "response.output_text.delta" => XaiRealtimeEvent::TextDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.done" | "response.output_text.done" => XaiRealtimeEvent::TextDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            text: string("text").unwrap_or_default(),
            native: native.clone(),
        },
        "response.function_call_arguments.delta" => XaiRealtimeEvent::FunctionCallArgumentsDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            call_id: string("call_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.function_call_arguments.done" => XaiRealtimeEvent::FunctionCallArgumentsDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            call_id: string("call_id"),
            name: string("name"),
            arguments: string("arguments"),
            native: native.clone(),
        },
        "response.done" => {
            let response = native.get("response").cloned().unwrap_or(Value::Null);
            let response_id = response
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            XaiRealtimeEvent::ResponseDone {
                response_id,
                response,
                native: native.clone(),
            }
        }
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            XaiRealtimeEvent::ProviderError {
                code: error
                    .get("code")
                    .and_then(value_to_string)
                    .or_else(|| string("code")),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| native.get("message").and_then(Value::as_str))
                    .unwrap_or("xAI Realtime provider error")
                    .to_owned(),
                native: native.clone(),
            }
        }
        "mcp_list_tools.in_progress"
        | "mcp_list_tools.completed"
        | "mcp_list_tools.failed"
        | "response.mcp_call_arguments.delta"
        | "response.mcp_call_arguments.done"
        | "response.mcp_call.in_progress"
        | "response.mcp_call.completed"
        | "response.mcp_call.failed" => XaiRealtimeEvent::McpEvent {
            event_type: event_type.to_owned(),
            native: native.clone(),
        },
        _ => XaiRealtimeEvent::Native {
            event_type: event_type.to_owned(),
            native: native.clone(),
        },
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_i64().map(|value| value.to_string()))
}
