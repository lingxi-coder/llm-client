//! Gemini Live (`BidiGenerateContent`) session setup and event adapter.

use crate::realtime::{
    validate_frame, validate_limits, RealtimeAudioFormat, RealtimeCodec, RealtimeConnectRequest,
    RealtimeConnection, RealtimeControl, RealtimeDriver, RealtimeError, RealtimeEvent,
    RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeTransport,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, Mutex},
};

const GEMINI_PCM_SAMPLE_RATE_HZ: u32 = 16_000;
const MAX_SETUP_PREFACE_FRAMES: usize = 8;
const MAX_TRACKED_TOOL_CALLS: usize = 1_024;
const MAX_TRACKED_TOOL_NAME_BYTES: usize = 256 * 1024;

/// Gemini Live setup fields and optional, explicitly scoped resumption state.
///
/// `model` accepts either a short model name or a `models/...` resource name.
/// The default generation configuration requests audio output. Set
/// `realtime_input_config` to disable automatic activity detection when the
/// host will send manual activity boundaries.
#[derive(Clone)]
pub struct GeminiLiveConfig {
    pub model: String,
    pub generation_config: Value,
    pub system_instruction: Option<String>,
    pub tools: Vec<Value>,
    pub realtime_input_config: Option<Value>,
    /// Additional documented top-level setup fields, such as
    /// `contextWindowCompression` or `inputAudioTranscription`.
    pub additional_setup_fields: Map<String, Value>,
    /// A stable, non-secret account identity chosen by the host. It is used to
    /// prevent a resume handle from crossing provider accounts.
    pub credential_scope: String,
    /// Ask Gemini to emit `sessionResumptionUpdate` messages, even when this
    /// connection starts a fresh session and has no previous handle.
    pub enable_session_resumption: bool,
    pub resume_handle: Option<GeminiLiveResumeHandle>,
}

impl GeminiLiveConfig {
    pub fn new(model: impl Into<String>, credential_scope: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            generation_config: json!({ "responseModalities": ["AUDIO"] }),
            system_instruction: None,
            tools: Vec::new(),
            realtime_input_config: None,
            additional_setup_fields: Map::new(),
            credential_scope: credential_scope.into(),
            enable_session_resumption: false,
            resume_handle: None,
        }
    }
}

impl fmt::Debug for GeminiLiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiLiveConfig")
            .field("model", &self.model)
            .field("generation_config", &self.generation_config)
            .field(
                "system_instruction",
                &self.system_instruction.as_ref().map(|_| "<set>"),
            )
            .field("tools_count", &self.tools.len())
            .field("realtime_input_config", &self.realtime_input_config)
            .field(
                "additional_setup_fields",
                &self.additional_setup_fields.keys(),
            )
            .field("credential_scope", &"<redacted>")
            .field("resume_handle", &self.resume_handle)
            .finish()
    }
}

/// A Gemini Live resumption handle bound to its model, endpoint, and host
/// credential scope. The opaque provider token is omitted from `Debug`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiLiveResumeHandle {
    handle: String,
    model: String,
    endpoint_scope: String,
    credential_scope: String,
}

impl GeminiLiveResumeHandle {
    pub fn handle(&self) -> &str {
        &self.handle
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn credential_scope(&self) -> &str {
        &self.credential_scope
    }
}

impl fmt::Debug for GeminiLiveResumeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiLiveResumeHandle")
            .field("handle", &"<redacted>")
            .field("model", &self.model)
            .field("endpoint_scope", &self.endpoint_scope)
            .field("credential_scope", &"<redacted>")
            .finish()
    }
}

/// One server update about whether the current Gemini Live state can resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiLiveResumeUpdate {
    pub resumable: bool,
    pub handle: Option<GeminiLiveResumeHandle>,
}

/// Events from Gemini Live. Events represented by the generic realtime API are
/// wrapped in `Realtime`; resumption updates expose a scoped typed handle.
#[derive(Debug, Clone, PartialEq)]
pub enum GeminiLiveEvent {
    Realtime(RealtimeEvent),
    SessionResumptionUpdate(GeminiLiveResumeUpdate),
}

/// A connected Gemini Live session. The connection is ready before this value
/// is returned, so callers can enqueue inputs without racing `setupComplete`.
pub struct GeminiLiveSession {
    control: RealtimeControl,
    events: GeminiLiveEvents,
}

impl GeminiLiveSession {
    /// Connect, send the one-time `setup` message, and wait for
    /// `setupComplete` before returning control to the host.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        request: RealtimeConnectRequest,
        config: GeminiLiveConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        let model = model_resource_name(&config.model)?;
        if config.credential_scope.trim().is_empty() {
            return Err(RealtimeError::InvalidConfig {
                message: "Gemini Live requires a non-empty host credential scope".into(),
            });
        }
        let endpoint_scope = endpoint_scope(&request.endpoint)?;
        if let Some(resume_handle) = &config.resume_handle {
            if resume_handle.handle.is_empty()
                || resume_handle.model != model
                || resume_handle.endpoint_scope != endpoint_scope
                || resume_handle.credential_scope != config.credential_scope
            {
                return Err(RealtimeError::InvalidConfig {
                    message: "Gemini Live resume handle belongs to a different model, endpoint, or credential scope".into(),
                });
            }
        }

        let (setup, manual_activity_detection) = build_setup(&config, &model)?;
        let setup =
            serde_json::to_vec(&json!({ "setup": setup })).map_err(|_| RealtimeError::Codec {
                message: "could not encode Gemini Live setup JSON".into(),
            })?;
        if setup.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: setup.len(),
                max: limits.max_frame_bytes,
            });
        }

        let scope = GeminiResumeScope {
            model,
            endpoint_scope,
            credential_scope: config.credential_scope,
        };
        let codec = Arc::new(GeminiLiveCodec {
            manual_activity_detection,
            tool_names: Mutex::new(ToolNameTable::default()),
        });
        let setup_transport = GeminiSetupTransport {
            inner: transport,
            setup_frame: RealtimeFrame::Text(Bytes::from(setup)),
            max_frame_bytes: limits.max_frame_bytes,
        };
        let (session, driver) =
            RealtimeSession::connect(&setup_transport, request, codec, limits).await?;
        let (control, events) = session.into_parts();
        Ok((
            Self {
                control,
                events: GeminiLiveEvents {
                    inner: events,
                    scope,
                    resume_enabled: config.enable_session_resumption
                        || config.resume_handle.is_some(),
                },
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (RealtimeControl, GeminiLiveEvents) {
        (self.control, self.events)
    }
}

/// Bounded event receiver with a typed projection for Gemini's scoped
/// `sessionResumptionUpdate` server message.
pub struct GeminiLiveEvents {
    inner: crate::realtime::RealtimeEvents,
    scope: GeminiResumeScope,
    resume_enabled: bool,
}

impl GeminiLiveEvents {
    pub async fn next(&mut self) -> Option<GeminiLiveEvent> {
        let event = self.inner.next().await?;
        let RealtimeEvent::ProviderEvent { name, native } = &event else {
            return Some(GeminiLiveEvent::Realtime(event));
        };
        if name != "sessionResumptionUpdate" || !self.resume_enabled {
            return Some(GeminiLiveEvent::Realtime(event));
        }
        let update = native.get("sessionResumptionUpdate").unwrap_or(native);
        let resumable = update
            .get("resumable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let handle = resumable
            .then(|| update.get("newHandle").and_then(Value::as_str))
            .flatten()
            .filter(|value| !value.is_empty())
            .map(|handle| self.scope.resume_handle(handle.to_owned()));
        Some(GeminiLiveEvent::SessionResumptionUpdate(
            GeminiLiveResumeUpdate { resumable, handle },
        ))
    }
}

#[derive(Clone)]
struct GeminiResumeScope {
    model: String,
    endpoint_scope: String,
    credential_scope: String,
}

impl GeminiResumeScope {
    fn resume_handle(&self, handle: String) -> GeminiLiveResumeHandle {
        GeminiLiveResumeHandle {
            handle,
            model: self.model.clone(),
            endpoint_scope: self.endpoint_scope.clone(),
            credential_scope: self.credential_scope.clone(),
        }
    }
}

struct GeminiSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
}

#[async_trait]
impl RealtimeTransport for GeminiSetupTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let mut connection = self.inner.connect(request).await?;
        connection.outbound.send(self.setup_frame.clone()).await?;

        let mut preface = Vec::new();
        let mut setup_complete = false;
        for _ in 0..MAX_SETUP_PREFACE_FRAMES {
            let frame = connection
                .inbound
                .next()
                .await
                .ok_or(RealtimeError::UnexpectedRemoteClose)??;
            validate_frame(&frame, self.max_frame_bytes)?;
            let RealtimeFrame::Text(bytes) = &frame else {
                return Err(RealtimeError::Codec {
                    message: "Gemini Live setup responses must be JSON text frames".into(),
                });
            };
            let message: Value =
                serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
                    message: "Gemini Live setup response is not valid JSON".into(),
                })?;
            if message.get("error").is_some() {
                return Err(RealtimeError::Codec {
                    message: "Gemini Live rejected the session setup".into(),
                });
            }
            setup_complete = message.get("setupComplete").is_some();
            preface.push(Ok(frame));
            if setup_complete {
                break;
            }
        }
        if !setup_complete {
            return Err(RealtimeError::Transport {
                message: "Gemini Live did not send setupComplete within the setup preface limit"
                    .into(),
            });
        }

        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct GeminiLiveCodec {
    manual_activity_detection: bool,
    tool_names: Mutex<ToolNameTable>,
}

#[derive(Clone, Default)]
struct ToolNameTable {
    entries: HashMap<String, String>,
    retained_bytes: usize,
}

impl ToolNameTable {
    fn insert(&mut self, call_id: String, name: String) -> Result<(), RealtimeError> {
        let previous_name_bytes = self.entries.get(&call_id).map_or(0, String::len);
        let previous_entry_bytes = self
            .entries
            .get(&call_id)
            .map_or(0, |name| call_id.len() + name.len());
        if previous_name_bytes == 0
            && !self.entries.contains_key(&call_id)
            && self.entries.len() >= MAX_TRACKED_TOOL_CALLS
        {
            return Err(RealtimeError::Codec {
                message: "Gemini Live outstanding tool-call tracking limit reached".into(),
            });
        }
        let new_bytes = call_id.len().saturating_add(name.len());
        let next_bytes = self
            .retained_bytes
            .saturating_sub(previous_entry_bytes)
            .saturating_add(new_bytes);
        if next_bytes > MAX_TRACKED_TOOL_NAME_BYTES {
            return Err(RealtimeError::Codec {
                message: "Gemini Live tool-call name storage limit reached".into(),
            });
        }
        self.retained_bytes = next_bytes;
        self.entries.insert(call_id, name);
        Ok(())
    }

    fn remove(&mut self, call_id: &str) {
        if let Some(name) = self.entries.remove(call_id) {
            self.retained_bytes = self
                .retained_bytes
                .saturating_sub(call_id.len() + name.len());
        }
    }
}

impl RealtimeCodec for GeminiLiveCodec {
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let messages = match input {
            RealtimeInput::RetrieveItem { .. }
            | RealtimeInput::DeleteItem { .. }
            | RealtimeInput::TruncateAudio { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "conversation item retrieval, deletion and truncation are not supported by this adapter".into(),
                });
            }
            RealtimeInput::ClearAudio | RealtimeInput::FinishSession => {
                return Err(RealtimeError::InvalidInput {
                    message: "explicit audio clearing and session finishing are not implemented by this adapter".into(),
                });
            }
            RealtimeInput::Text(text) => {
                let text = json!({ "realtimeInput": { "text": text } });
                if self.manual_activity_detection {
                    vec![
                        json!({ "realtimeInput": { "activityStart": {} } }),
                        text,
                        json!({ "realtimeInput": { "activityEnd": {} } }),
                    ]
                } else {
                    vec![text]
                }
            }
            RealtimeInput::Audio { data, format } => {
                let expected_format = RealtimeAudioFormat::Pcm16 {
                    sample_rate_hz: GEMINI_PCM_SAMPLE_RATE_HZ,
                };
                if format != &expected_format {
                    return Err(RealtimeError::InvalidInput {
                        message: "Gemini Live input audio must be mono PCM16 at 16 kHz".into(),
                    });
                }
                vec![json!({
                    "realtimeInput": {
                        "audio": {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": STANDARD.encode(data)
                        }
                    }
                })]
            }
            RealtimeInput::CommitAudio if self.manual_activity_detection => {
                vec![json!({ "realtimeInput": { "activityEnd": {} } })]
            }
            RealtimeInput::CommitAudio => {
                vec![json!({ "realtimeInput": { "audioStreamEnd": true } })]
            }
            RealtimeInput::Interrupt if self.manual_activity_detection => {
                vec![json!({ "realtimeInput": { "activityStart": {} } })]
            }
            RealtimeInput::Interrupt => return Err(RealtimeError::InvalidInput {
                message:
                    "Gemini Live activityStart requires automatic activity detection to be disabled"
                        .into(),
            }),
            RealtimeInput::ToolResult { call_id, output } => {
                let name = self
                    .tool_names
                    .lock()
                    .unwrap()
                    .entries
                    .get(call_id)
                    .cloned()
                    .ok_or_else(|| RealtimeError::InvalidInput {
                        message: "Gemini Live tool result has no matching function call".into(),
                    })?;
                if !output.is_object() {
                    return Err(RealtimeError::InvalidInput {
                        message: "Gemini Live function responses must be JSON objects".into(),
                    });
                }
                let message = json!({
                    "toolResponse": {
                        "functionResponses": [{
                            "id": call_id,
                            "name": name,
                            "response": output
                        }]
                    }
                });
                vec![message]
            }
            RealtimeInput::Image { data, mime_type } => {
                if data.is_empty() || !matches!(mime_type.as_str(), "image/jpeg" | "image/png") {
                    return Err(RealtimeError::InvalidInput {
                        message: "Gemini Live video frames require non-empty JPEG or PNG data"
                            .into(),
                    });
                }
                vec![json!({ "realtimeInput": { "video": {
                    "mimeType": mime_type,
                    "data": STANDARD.encode(data)
                } } })]
            }
            RealtimeInput::ToolResults { results } => {
                if results.is_empty() || results.len() > crate::realtime::MAX_REALTIME_TOOL_RESULTS
                {
                    return Err(RealtimeError::InvalidInput {
                        message:
                            "Gemini Live tool response batch is empty or exceeds the local limit"
                                .into(),
                    });
                }
                let names = self.tool_names.lock().unwrap();
                let mut seen = HashSet::new();
                let mut responses = Vec::with_capacity(results.len());
                for result in results {
                    if !seen.insert(&result.call_id) || !result.output.is_object() {
                        return Err(RealtimeError::InvalidInput {
                            message: "Gemini Live tool responses require unique call IDs and JSON objects".into(),
                        });
                    }
                    let name = names.entries.get(&result.call_id).ok_or_else(|| {
                        RealtimeError::InvalidInput {
                            message: "Gemini Live tool result has no matching function call".into(),
                        }
                    })?;
                    responses.push(json!({
                        "id": result.call_id,
                        "name": name,
                        "response": result.output
                    }));
                }
                vec![json!({ "toolResponse": { "functionResponses": responses } })]
            }
            RealtimeInput::ContinueResponse => {
                return Err(RealtimeError::InvalidInput {
                    message: "Gemini Live does not use explicit response continuation".into(),
                });
            }
        };
        messages
            .into_iter()
            .map(|message| {
                serde_json::to_vec(&message)
                    .map(|bytes| RealtimeFrame::Text(Bytes::from(bytes)))
                    .map_err(|_| RealtimeError::Codec {
                        message: "could not encode Gemini Live client message".into(),
                    })
            })
            .collect()
    }

    fn input_queued(&self, input: &RealtimeInput) {
        match input {
            RealtimeInput::ToolResult { call_id, .. } => {
                self.tool_names.lock().unwrap().remove(call_id);
            }
            RealtimeInput::ToolResults { results } => {
                let mut names = self.tool_names.lock().unwrap();
                for result in results {
                    names.remove(&result.call_id);
                }
            }
            _ => {}
        }
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let RealtimeFrame::Text(bytes) = frame else {
            return Err(RealtimeError::Codec {
                message: "Gemini Live server messages must be JSON text frames".into(),
            });
        };
        let native: Value = serde_json::from_slice(&bytes).map_err(|_| RealtimeError::Codec {
            message: "Gemini Live server message is not valid JSON".into(),
        })?;
        if let Some(error) = native.get("error") {
            return Ok(vec![RealtimeEvent::ProviderError {
                code: error.get("code").and_then(value_to_string),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Gemini Live provider error")
                    .to_owned(),
            }]);
        }
        if native.get("setupComplete").is_some() {
            return Ok(vec![RealtimeEvent::SessionReady]);
        }
        if let Some(server_content) = native.get("serverContent") {
            return self.decode_server_content(server_content);
        }
        if let Some(tool_call) = native.get("toolCall") {
            return self.decode_tool_call(tool_call, &native);
        }
        if let Some(cancellation) = native.get("toolCallCancellation") {
            let ids = cancellation
                .get("ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            let mut names = self.tool_names.lock().unwrap();
            for id in ids {
                names.remove(id);
            }
            return Ok(vec![RealtimeEvent::ProviderEvent {
                name: "toolCallCancellation".into(),
                native,
            }]);
        }
        let name = ["sessionResumptionUpdate", "goAway"]
            .into_iter()
            .find(|field| native.get(*field).is_some())
            .unwrap_or("unknown");
        Ok(vec![RealtimeEvent::ProviderEvent {
            name: name.into(),
            native,
        }])
    }
}

impl GeminiLiveCodec {
    fn decode_server_content(
        &self,
        server_content: &Value,
    ) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let mut events = Vec::new();
        if let Some(parts) = server_content
            .get("modelTurn")
            .and_then(|turn| turn.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    events.push(RealtimeEvent::TextDelta {
                        text: text.to_owned(),
                        item_id: None,
                        final_chunk: false,
                    });
                }
                if let Some(inline_data) = part.get("inlineData") {
                    if let (Some(encoded), Some(mime_type)) = (
                        inline_data.get("data").and_then(Value::as_str),
                        inline_data.get("mimeType").and_then(Value::as_str),
                    ) {
                        let data = STANDARD.decode(encoded).map_err(|_| RealtimeError::Codec {
                            message: "Gemini Live inline media data is not valid base64".into(),
                        })?;
                        events.push(RealtimeEvent::AudioDelta {
                            data: Bytes::from(data),
                            format: gemini_audio_format(mime_type),
                            item_id: None,
                        });
                    } else {
                        events.push(RealtimeEvent::ProviderEvent {
                            name: "modelTurnPart".into(),
                            native: part.clone(),
                        });
                    }
                }
            }
        }
        if let Some(transcription) = server_content
            .get("outputTranscription")
            .and_then(|value| value.get("text"))
            .and_then(Value::as_str)
        {
            events.push(RealtimeEvent::TextDelta {
                text: transcription.to_owned(),
                item_id: None,
                final_chunk: false,
            });
        }
        for (name, key) in [
            ("inputTranscription", "inputTranscription"),
            ("interimInputTranscription", "interimInputTranscription"),
        ] {
            if let Some(transcription) = server_content.get(key) {
                events.push(RealtimeEvent::ProviderEvent {
                    name: name.into(),
                    native: transcription.clone(),
                });
            }
        }
        let interrupted = server_content
            .get("interrupted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if interrupted {
            events.push(RealtimeEvent::Interrupted);
        }
        if server_content
            .get("turnComplete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            if let Some(RealtimeEvent::TextDelta { final_chunk, .. }) = events
                .iter_mut()
                .rev()
                .find(|event| matches!(event, RealtimeEvent::TextDelta { .. }))
            {
                *final_chunk = true;
            }
            events.push(RealtimeEvent::TurnCompleted {
                turn_id: None,
                status: Some(
                    if interrupted {
                        "interrupted"
                    } else {
                        "completed"
                    }
                    .into(),
                ),
            });
        }
        if server_content
            .get("generationComplete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            events.push(RealtimeEvent::ProviderEvent {
                name: "generationComplete".into(),
                native: Value::Bool(true),
            });
        }
        if events.is_empty() {
            events.push(RealtimeEvent::ProviderEvent {
                name: "serverContent".into(),
                native: server_content.clone(),
            });
        }
        Ok(events)
    }

    fn decode_tool_call(
        &self,
        tool_call: &Value,
        full_message: &Value,
    ) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let Some(calls) = tool_call.get("functionCalls").and_then(Value::as_array) else {
            return Ok(vec![RealtimeEvent::ProviderEvent {
                name: "toolCall".into(),
                native: full_message.clone(),
            }]);
        };
        let mut names = self.tool_names.lock().unwrap();
        let mut staged = names.clone();
        let mut seen = HashSet::new();
        let mut events = Vec::with_capacity(calls.len());
        for call in calls {
            let (Some(call_id), Some(name)) = (
                call.get("id").and_then(Value::as_str),
                call.get("name").and_then(Value::as_str),
            ) else {
                return Ok(vec![RealtimeEvent::ProviderEvent {
                    name: "toolCall".into(),
                    native: full_message.clone(),
                }]);
            };
            let arguments = call
                .get("args")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            if call_id.trim().is_empty()
                || name.trim().is_empty()
                || !seen.insert(call_id)
                || !arguments.is_object()
            {
                return Ok(vec![RealtimeEvent::ProviderEvent {
                    name: "toolCall".into(),
                    native: full_message.clone(),
                }]);
            }
            staged.insert(call_id.to_owned(), name.to_owned())?;
            events.push(RealtimeEvent::ToolCall {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments,
            });
        }
        // A malformed or oversized later call must not register earlier calls
        // that were never delivered to the host.
        *names = staged;
        if events.is_empty() {
            events.push(RealtimeEvent::ProviderEvent {
                name: "toolCall".into(),
                native: full_message.clone(),
            });
        }
        Ok(events)
    }
}

fn build_setup(
    config: &GeminiLiveConfig,
    model: &str,
) -> Result<(Map<String, Value>, bool), RealtimeError> {
    if !config.generation_config.is_object() {
        return Err(RealtimeError::InvalidConfig {
            message: "Gemini Live generation_config must be a JSON object".into(),
        });
    }
    if config
        .realtime_input_config
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Gemini Live realtime_input_config must be a JSON object".into(),
        });
    }
    let mut setup = Map::new();
    setup.insert("model".into(), Value::String(model.to_owned()));
    setup.insert("generationConfig".into(), config.generation_config.clone());
    if let Some(instruction) = &config.system_instruction {
        setup.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": instruction }] }),
        );
    }
    if !config.tools.is_empty() {
        setup.insert("tools".into(), Value::Array(config.tools.clone()));
    }
    if let Some(input_config) = &config.realtime_input_config {
        setup.insert("realtimeInputConfig".into(), input_config.clone());
    }
    if config.enable_session_resumption || config.resume_handle.is_some() {
        let session_resumption = config
            .resume_handle
            .as_ref()
            .map(|resume_handle| json!({ "handle": resume_handle.handle }))
            .unwrap_or_else(|| json!({}));
        setup.insert("sessionResumption".into(), session_resumption);
    }
    for (field, value) in &config.additional_setup_fields {
        if matches!(
            field.as_str(),
            "model"
                | "generationConfig"
                | "systemInstruction"
                | "tools"
                | "realtimeInputConfig"
                | "sessionResumption"
        ) {
            return Err(RealtimeError::InvalidConfig {
                message: "Gemini Live additional_setup_fields contains a reserved field".into(),
            });
        }
        setup.insert(field.clone(), value.clone());
    }
    let manual_activity_detection = config
        .realtime_input_config
        .as_ref()
        .and_then(|value| value.get("automaticActivityDetection"))
        .and_then(|value| value.get("disabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok((setup, manual_activity_detection))
}

fn model_resource_name(model: &str) -> Result<String, RealtimeError> {
    let model = model.trim();
    if model.is_empty() {
        return Err(RealtimeError::InvalidConfig {
            message: "Gemini Live model must not be empty".into(),
        });
    }
    if model.starts_with("models/") {
        Ok(model.to_owned())
    } else {
        Ok(format!("models/{model}"))
    }
}

fn endpoint_scope(endpoint: &str) -> Result<String, RealtimeError> {
    let parsed = url::Url::parse(endpoint).map_err(|_| RealtimeError::InvalidConfig {
        message: "Gemini Live endpoint must be an absolute wss:// URL".into(),
    })?;
    if parsed.scheme() != "wss"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Gemini Live endpoint must be a wss:// URL without user-info".into(),
        });
    }
    Ok(format!(
        "{}{}",
        parsed.origin().ascii_serialization(),
        parsed.path()
    ))
}

fn gemini_audio_format(mime_type: &str) -> RealtimeAudioFormat {
    let lower = mime_type.to_ascii_lowercase();
    if lower.starts_with("audio/pcm") {
        if let Some(sample_rate_hz) = lower
            .split(';')
            .skip(1)
            .find_map(|part| part.trim().strip_prefix("rate="))
            .and_then(|rate| rate.parse().ok())
        {
            return RealtimeAudioFormat::Pcm16 { sample_rate_hz };
        }
    }
    RealtimeAudioFormat::Encoded {
        mime_type: mime_type.to_owned(),
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_i64().map(|value| value.to_string()))
}
