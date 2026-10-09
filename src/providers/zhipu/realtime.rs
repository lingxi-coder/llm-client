//! GLM Realtime WebSocket adapter for Zhipu's mainland Voice API.
//!
//! This adapter follows the first-party MetaGLM protocol guide. It deliberately
//! exposes only the mainland endpoint currently documented by that guide.

use crate::realtime::{
    validate_frame, validate_limits, RealtimeAudioFormat, RealtimeCodec, RealtimeConnectRequest,
    RealtimeConnection, RealtimeControl, RealtimeDriver, RealtimeError, RealtimeEvent,
    RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeToolResult,
    RealtimeTransport, MAX_REALTIME_TOOL_RESULTS,
};
use crate::{files::provider_file_endpoint_fingerprint, protocol::Secret};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashSet, fmt, sync::Arc};

const MAX_SETUP_PREFACE_FRAMES: usize = 16;
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

/// Mainland GLM Realtime endpoint in the official MetaGLM protocol guide.
pub const GLM_REALTIME_MAINLAND_ENDPOINT: &str = "wss://open.bigmodel.cn/api/paas/v4/realtime";

/// Region of the documented GLM Realtime route. The international realtime
/// route is not included because no first-party endpoint contract was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmRealtimeRegion {
    MainlandChina,
}

/// A GLM Realtime route. It is fixed to the currently documented mainland
/// endpoint so region-specific credentials cannot be sent to an inferred host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRealtimeRoute {
    region: GlmRealtimeRegion,
    endpoint: String,
}

impl GlmRealtimeRoute {
    pub fn mainland_china() -> Self {
        Self {
            region: GlmRealtimeRegion::MainlandChina,
            endpoint: GLM_REALTIME_MAINLAND_ENDPOINT.into(),
        }
    }

    pub fn region(&self) -> GlmRealtimeRegion {
        self.region
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        if self.region != GlmRealtimeRegion::MainlandChina
            || self.endpoint != GLM_REALTIME_MAINLAND_ENDPOINT
        {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime currently supports only the documented mainland endpoint"
                    .into(),
            });
        }
        Ok(())
    }

    fn endpoint_scope(&self) -> String {
        provider_file_endpoint_fingerprint(&self.endpoint)
    }
}

/// Non-secret account identity bound to one GLM Realtime region and endpoint.
/// Credentials are never stored in this scope; pass one credential to each
/// [`GlmRealtimeSession::connect`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRealtimeScope {
    profile_name: String,
    account_scope: String,
    region: GlmRealtimeRegion,
    endpoint_fingerprint: String,
}

impl GlmRealtimeScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        route: &GlmRealtimeRoute,
    ) -> Result<Self, RealtimeError> {
        route.validate()?;
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region: route.region,
            endpoint_fingerprint: route.endpoint_scope(),
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

    pub fn region(&self) -> GlmRealtimeRegion {
        self.region
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
                message: "GLM Realtime scope requires a profile, account, region, and endpoint"
                    .into(),
            });
        }
        Ok(())
    }

    fn validate_route(&self, route: &GlmRealtimeRoute) -> Result<(), RealtimeError> {
        route.validate()?;
        self.validate()?;
        if self.region != route.region || self.endpoint_fingerprint != route.endpoint_scope() {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime account scope belongs to a different region or endpoint"
                    .into(),
            });
        }
        Ok(())
    }
}

/// Client- or server-side voice activity detection documented by GLM Realtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmRealtimeTurnDetection {
    ServerVad,
    ClientVad,
}

/// GLM's documented audio output encoding. The protocol guide does not specify
/// a PCM sample rate, so the adapter does not attach one to returned bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmRealtimeOutputAudioFormat {
    Pcm,
    Mp3,
}

/// One function tool advertised through GLM's flattened Realtime session
/// schema. `parameters` is the JSON Schema object sent to the provider.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmRealtimeFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl GlmRealtimeFunctionTool {
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

/// GLM Realtime's documented session-level function selection values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmRealtimeToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

impl GlmRealtimeToolChoice {
    fn as_wire_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".into()),
            Self::None => Value::String("none".into()),
            Self::Required => Value::String("required".into()),
            Self::Function(name) => json!({ "type": "function", "function": name }),
        }
    }
}

/// Session configuration sent in GLM's native `session.update` event.
#[derive(Clone)]
pub struct GlmRealtimeConfig {
    pub instructions: Option<String>,
    pub turn_detection: GlmRealtimeTurnDetection,
    pub output_audio_format: GlmRealtimeOutputAudioFormat,
    pub tools: Vec<GlmRealtimeFunctionTool>,
    pub tool_choice: Option<GlmRealtimeToolChoice>,
}

impl Default for GlmRealtimeConfig {
    fn default() -> Self {
        Self {
            instructions: None,
            turn_detection: GlmRealtimeTurnDetection::ServerVad,
            output_audio_format: GlmRealtimeOutputAudioFormat::Pcm,
            tools: Vec::new(),
            tool_choice: None,
        }
    }
}

impl fmt::Debug for GlmRealtimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmRealtimeConfig")
            .field("instructions", &self.instructions.as_ref().map(|_| "<set>"))
            .field("turn_detection", &self.turn_detection)
            .field("output_audio_format", &self.output_audio_format)
            .field("tool_count", &self.tools.len())
            .field("has_tool_choice", &self.tool_choice.is_some())
            .finish()
    }
}

/// Typed events defined by the GLM Realtime protocol. Unrecognized events are
/// preserved as native JSON. Text deltas are the provider's audio transcript
/// events; GLM does not document a separate response-text delta event.
#[derive(Debug, Clone, PartialEq)]
pub enum GlmRealtimeEvent {
    Realtime(RealtimeEvent),
    SessionCreated {
        session: Value,
        native: Value,
    },
    SessionUpdated {
        session: Value,
        native: Value,
    },
    ConversationCreated {
        conversation: Value,
        native: Value,
    },
    InputAudioCommitted {
        item_id: Option<String>,
        native: Value,
    },
    SpeechStarted {
        item_id: Option<String>,
        native: Value,
    },
    SpeechStopped {
        item_id: Option<String>,
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
        format: GlmRealtimeOutputAudioFormat,
        native: Value,
    },
    AudioDone {
        response_id: Option<String>,
        item_id: Option<String>,
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
        transcript: String,
        native: Value,
    },
    FunctionCallArgumentsDone {
        response_id: Option<String>,
        item_id: Option<String>,
        output_index: Option<u64>,
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
        error_type: Option<String>,
        code: Option<String>,
        message: String,
        native: Value,
    },
    Native {
        event_type: String,
        native: Value,
    },
}

/// A connected GLM Realtime session. `connect` returns after it sees
/// `session.created`, sends `session.update`, and receives `session.updated`.
pub struct GlmRealtimeSession {
    control: RealtimeControl,
    events: GlmRealtimeEvents,
    scope: GlmRealtimeScope,
}

impl GlmRealtimeSession {
    /// Connect once with an explicit route/account scope and an ephemeral
    /// bearer credential. The credential may be an API key or a host-issued
    /// JWT; it is only placed in the handshake header for this connection.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        route: GlmRealtimeRoute,
        scope: GlmRealtimeScope,
        credential: Secret<String>,
        config: GlmRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        route.validate()?;
        scope.validate_route(&route)?;
        validate_credential(&credential)?;
        let setup = build_session_update(&config)?;
        let setup_bytes = serde_json::to_vec(&setup).map_err(|_| RealtimeError::Codec {
            message: "could not encode GLM Realtime session update".into(),
        })?;
        if setup_bytes.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: setup_bytes.len(),
                max: limits.max_frame_bytes,
            });
        }

        let request = RealtimeConnectRequest {
            endpoint: route.endpoint.clone(),
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let setup_transport = GlmSetupTransport {
            inner: transport,
            setup_frame: RealtimeFrame::Text(Bytes::from(setup_bytes)),
            max_frame_bytes: limits.max_frame_bytes,
        };
        let codec = Arc::new(GlmRealtimeCodec {
            turn_detection: config.turn_detection,
        });
        let (session, driver) =
            RealtimeSession::connect(&setup_transport, request, codec, limits).await?;
        let (control, events) = session.into_parts();
        Ok((
            Self {
                control,
                events: GlmRealtimeEvents {
                    inner: events,
                    output_audio_format: config.output_audio_format,
                },
                scope,
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (RealtimeControl, GlmRealtimeEvents) {
        (self.control, self.events)
    }

    pub fn scope(&self) -> &GlmRealtimeScope {
        &self.scope
    }
}

/// Receiver that converts GLM's native JSON event union into typed events.
pub struct GlmRealtimeEvents {
    inner: crate::realtime::RealtimeEvents,
    output_audio_format: GlmRealtimeOutputAudioFormat,
}

impl GlmRealtimeEvents {
    pub async fn next(&mut self) -> Option<GlmRealtimeEvent> {
        let event = self.inner.next().await?;
        let RealtimeEvent::ProviderEvent { name, native } = &event else {
            return Some(GlmRealtimeEvent::Realtime(event));
        };
        Some(parse_native_event(name, native, self.output_audio_format))
    }
}

struct GlmSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
}

#[async_trait]
impl RealtimeTransport for GlmSetupTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let mut connection = self.inner.connect(request).await?;
        let mut preface = Vec::new();
        let mut session_created = false;
        let mut update_sent = false;
        let mut update_acknowledged = false;

        for _ in 0..MAX_SETUP_PREFACE_FRAMES {
            let frame = connection
                .inbound
                .next()
                .await
                .ok_or(RealtimeError::UnexpectedRemoteClose)??;
            validate_frame(&frame, self.max_frame_bytes)?;
            let message = parse_native_frame(&frame)?;
            let event_type = event_type(&message)?;
            if event_type == "error" {
                return Err(RealtimeError::Transport {
                    message: setup_error_message(&message),
                });
            }
            preface.push(Ok(frame));
            match event_type {
                "session.created" if !update_sent => {
                    session_created = true;
                    connection.outbound.send(self.setup_frame.clone()).await?;
                    update_sent = true;
                }
                "session.updated" if update_sent => {
                    update_acknowledged = true;
                    break;
                }
                _ => {}
            }
        }
        if !session_created || !update_acknowledged {
            return Err(RealtimeError::Transport {
                message: "GLM Realtime did not acknowledge session.update within the setup limit"
                    .into(),
            });
        }

        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct GlmRealtimeCodec {
    turn_detection: GlmRealtimeTurnDetection,
}

impl RealtimeCodec for GlmRealtimeCodec {
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let messages = match input {
            RealtimeInput::RetrieveItem { .. }
            | RealtimeInput::DeleteItem { .. }
            | RealtimeInput::TruncateAudio { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "conversation item retrieval, deletion and truncation are not supported by this adapter".into(),
                });
            }
            RealtimeInput::ClearAudio => {
                vec![json!({ "type": "input_audio_buffer.clear" })]
            }
            RealtimeInput::FinishSession => {
                return Err(RealtimeError::InvalidInput {
                    message: "explicit session finishing is not implemented by this adapter".into(),
                });
            }
            RealtimeInput::Text(text) => {
                validate_text(text)?;
                vec![
                    json!({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{ "type": "input_text", "text": text }]
                        }
                    }),
                    json!({ "type": "response.create" }),
                ]
            }
            RealtimeInput::Audio { data, format } => {
                validate_input_audio(data, format)?;
                vec![json!({
                    "type": "input_audio_buffer.append",
                    "audio": STANDARD.encode(data)
                })]
            }
            RealtimeInput::Image { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "GLM Realtime image input is not supported by this audio/text adapter"
                        .into(),
                });
            }
            RealtimeInput::CommitAudio
                if self.turn_detection == GlmRealtimeTurnDetection::ClientVad =>
            {
                vec![json!({ "type": "input_audio_buffer.commit" })]
            }
            RealtimeInput::CommitAudio => {
                return Err(RealtimeError::InvalidInput {
                    message: "GLM Realtime audio commit is only valid with client_vad".into(),
                })
            }
            // `server_vad` starts responses automatically for audio turns, but
            // the documented function-output flow explicitly uses
            // `response.create` after the host inserts function_call_output.
            // Keep that continuation under caller control in either VAD mode.
            RealtimeInput::ContinueResponse => vec![json!({ "type": "response.create" })],
            RealtimeInput::Interrupt => vec![json!({ "type": "response.cancel" })],
            RealtimeInput::ToolResult { call_id, output , .. } => {
                vec![encode_function_call_output(call_id, output)?]
            }
            RealtimeInput::ToolResults { results } => encode_function_call_outputs(results)?,
        };
        messages
            .into_iter()
            .map(|message| {
                serde_json::to_vec(&message)
                    .map(|bytes| RealtimeFrame::Text(Bytes::from(bytes)))
                    .map_err(|_| RealtimeError::Codec {
                        message: "could not encode GLM Realtime client event".into(),
                    })
            })
            .collect()
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let native = parse_native_frame(&frame)?;
        let name = event_type(&native)?;
        if name == "response.audio.delta" {
            let audio = native.get("delta").and_then(Value::as_str).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "GLM Realtime audio delta is missing base64 `delta`".into(),
                }
            })?;
            STANDARD.decode(audio).map_err(|_| RealtimeError::Codec {
                message: "GLM Realtime audio delta is not valid base64".into(),
            })?;
        }
        Ok(vec![RealtimeEvent::ProviderEvent {
            name: name.to_owned(),
            native,
        }])
    }
}

fn build_session_update(config: &GlmRealtimeConfig) -> Result<Value, RealtimeError> {
    if config
        .instructions
        .as_ref()
        .is_some_and(|text| text.contains('\0') || text.len() > 64 * 1024)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "GLM Realtime instructions must be NUL-free and at most 64 KiB".into(),
        });
    }
    let mut session = serde_json::Map::new();
    session.insert("input_audio_format".into(), Value::String("wav".into()));
    session.insert(
        "output_audio_format".into(),
        Value::String(
            match config.output_audio_format {
                GlmRealtimeOutputAudioFormat::Pcm => "pcm",
                GlmRealtimeOutputAudioFormat::Mp3 => "mp3",
            }
            .into(),
        ),
    );
    if let Some(instructions) = &config.instructions {
        session.insert("instructions".into(), Value::String(instructions.clone()));
    }
    let tools = encode_session_tools(config)?;
    if !tools.is_empty() {
        session.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = &config.tool_choice {
        session.insert("tool_choice".into(), tool_choice.as_wire_value());
    }
    let detection = match config.turn_detection {
        GlmRealtimeTurnDetection::ServerVad => "server_vad",
        GlmRealtimeTurnDetection::ClientVad => "client_vad",
    };
    session.insert("turn_detection".into(), json!({ "type": detection }));
    session.insert(
        "beta_fields".into(),
        json!({
            "chat_mode": "audio",
            "tts_source": "e2e",
            "auto_search": false
        }),
    );
    Ok(json!({ "type": "session.update", "session": session }))
}

fn encode_session_tools(config: &GlmRealtimeConfig) -> Result<Vec<Value>, RealtimeError> {
    let mut names = HashSet::with_capacity(config.tools.len());
    let mut tools = Vec::with_capacity(config.tools.len());
    for tool in &config.tools {
        if tool.name.trim().is_empty() || tool.name.contains('\0') {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime function tool names must be non-empty and NUL-free".into(),
            });
        }
        if !names.insert(tool.name.as_str()) {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime function tool names must be unique".into(),
            });
        }
        if !tool.parameters.is_object() {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime function tool parameters must be a JSON Schema object"
                    .into(),
            });
        }
        if tool
            .description
            .as_deref()
            .is_some_and(|description| description.contains('\0'))
        {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime function tool descriptions must be NUL-free".into(),
            });
        }
        let mut wire = json!({
            "type": "function",
            "name": tool.name,
            "parameters": tool.parameters,
        });
        if let Some(description) = &tool.description {
            wire["description"] = Value::String(description.clone());
        }
        tools.push(wire);
    }

    match &config.tool_choice {
        None | Some(GlmRealtimeToolChoice::Auto) | Some(GlmRealtimeToolChoice::None) => {}
        Some(GlmRealtimeToolChoice::Required) => {}
        Some(GlmRealtimeToolChoice::Function(name))
            if name.trim().is_empty() || !names.contains(name.as_str()) =>
        {
            return Err(RealtimeError::InvalidConfig {
                message: "GLM Realtime function tool choice must name a configured function".into(),
            });
        }
        Some(GlmRealtimeToolChoice::Function(_)) => {}
    }
    Ok(tools)
}

fn encode_function_call_outputs(
    results: &[RealtimeToolResult],
) -> Result<Vec<Value>, RealtimeError> {
    if results.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "GLM Realtime tool result batch must not be empty".into(),
        });
    }
    if results.len() > MAX_REALTIME_TOOL_RESULTS {
        return Err(RealtimeError::InvalidInput {
            message: format!(
                "GLM Realtime tool result batch exceeds the local limit of {MAX_REALTIME_TOOL_RESULTS} results"
            ),
        });
    }
    let mut call_ids = HashSet::with_capacity(results.len());
    for result in results {
        validate_call_id(&result.call_id)?;
        if !call_ids.insert(result.call_id.as_str()) {
            return Err(RealtimeError::InvalidInput {
                message: "GLM Realtime tool result batch contains a duplicate call_id".into(),
            });
        }
    }
    results
        .iter()
        .map(|result| encode_function_call_output(&result.call_id, &result.output))
        .collect()
}

fn encode_function_call_output(call_id: &str, output: &Value) -> Result<Value, RealtimeError> {
    validate_call_id(call_id)?;
    let output = serde_json::to_string(output).map_err(|_| RealtimeError::Codec {
        message: "could not encode GLM Realtime function output".into(),
    })?;
    Ok(json!({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
    }))
}

fn validate_call_id(call_id: &str) -> Result<(), RealtimeError> {
    if call_id.trim().is_empty() || call_id.contains('\0') {
        return Err(RealtimeError::InvalidInput {
            message: "GLM Realtime function result requires a non-empty, NUL-free call_id".into(),
        });
    }
    Ok(())
}

fn validate_input_audio(data: &Bytes, format: &RealtimeAudioFormat) -> Result<(), RealtimeError> {
    if data.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "GLM Realtime audio input must not be empty".into(),
        });
    }
    if format
        != &(RealtimeAudioFormat::Encoded {
            mime_type: "audio/wav".into(),
        })
    {
        return Err(RealtimeError::InvalidInput {
            message: "GLM Realtime accepts WAV-encoded input audio only".into(),
        });
    }
    Ok(())
}

fn validate_text(text: &str) -> Result<(), RealtimeError> {
    if text.trim().is_empty() || text.contains('\0') {
        return Err(RealtimeError::InvalidInput {
            message: "GLM Realtime text input must be non-empty and NUL-free".into(),
        });
    }
    Ok(())
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
            message: "GLM Realtime bearer credential is empty or invalid".into(),
        });
    }
    Ok(())
}

fn parse_native_frame(frame: &RealtimeFrame) -> Result<Value, RealtimeError> {
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(RealtimeError::Codec {
            message: "GLM Realtime JSON/base64 protocol cannot decode binary frames".into(),
        });
    };
    serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
        message: "GLM Realtime event is not valid JSON".into(),
    })
}

fn event_type(native: &Value) -> Result<&str, RealtimeError> {
    native
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| RealtimeError::Codec {
            message: "GLM Realtime event is missing string `type`".into(),
        })
}

fn setup_error_message(native: &Value) -> String {
    let error = native.get("error").unwrap_or(&Value::Null);
    let code = error.get("code").and_then(Value::as_str);
    match code {
        Some(code) if code.len() <= 128 => {
            format!("GLM Realtime rejected session setup (code: {code})")
        }
        _ => "GLM Realtime rejected session setup".into(),
    }
}

fn parse_native_event(
    event_type: &str,
    native: &Value,
    output_audio_format: GlmRealtimeOutputAudioFormat,
) -> GlmRealtimeEvent {
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let number = |key: &str| native.get(key).and_then(Value::as_u64);
    let nested = |outer: &str, key: &str| {
        native
            .get(outer)
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    match event_type {
        "session.created" => GlmRealtimeEvent::SessionCreated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "session.updated" => GlmRealtimeEvent::SessionUpdated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "conversation.created" => GlmRealtimeEvent::ConversationCreated {
            conversation: native.get("conversation").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "input_audio_buffer.committed" => GlmRealtimeEvent::InputAudioCommitted {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "input_audio_buffer.speech_started" => GlmRealtimeEvent::SpeechStarted {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "input_audio_buffer.speech_stopped" => GlmRealtimeEvent::SpeechStopped {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "conversation.item.input_audio_transcription.completed" => {
            GlmRealtimeEvent::InputTranscriptionCompleted {
                item_id: string("item_id"),
                transcript: string("transcript").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "response.created" => GlmRealtimeEvent::ResponseCreated {
            response_id: nested("response", "id"),
            native: native.clone(),
        },
        "response.audio.delta" => {
            let data = string("delta")
                .and_then(|delta| STANDARD.decode(delta).ok())
                .map(Bytes::from)
                .unwrap_or_default();
            GlmRealtimeEvent::AudioDelta {
                response_id: string("response_id"),
                item_id: string("item_id"),
                output_index: number("output_index"),
                content_index: number("content_index"),
                data,
                format: output_audio_format,
                native: native.clone(),
            }
        }
        "response.audio.done" => GlmRealtimeEvent::AudioDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            native: native.clone(),
        },
        "response.audio_transcript.delta" => GlmRealtimeEvent::TextDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.audio_transcript.done" => GlmRealtimeEvent::TextDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            transcript: string("transcript").unwrap_or_default(),
            native: native.clone(),
        },
        "response.function_call_arguments.done" => {
            let call_id = string("call_id").filter(|value| !value.is_empty());
            GlmRealtimeEvent::FunctionCallArgumentsDone {
                response_id: string("response_id"),
                item_id: string("item_id").filter(|value| !value.is_empty()),
                output_index: number("output_index"),
                call_id,
                name: string("name").filter(|value| !value.is_empty()),
                arguments: string("arguments").filter(|value| !value.is_empty()),
                native: native.clone(),
            }
        }
        "response.done" => {
            let response = native.get("response").cloned().unwrap_or(Value::Null);
            let response_id = response
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            GlmRealtimeEvent::ResponseDone {
                response_id,
                response,
                native: native.clone(),
            }
        }
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            GlmRealtimeEvent::ProviderError {
                error_type: error.get("type").and_then(Value::as_str).map(str::to_owned),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("GLM Realtime provider error")
                    .to_owned(),
                native: native.clone(),
            }
        }
        _ => GlmRealtimeEvent::Native {
            event_type: event_type.to_owned(),
            native: native.clone(),
        },
    }
}
