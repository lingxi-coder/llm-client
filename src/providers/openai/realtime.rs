//! OpenAI Realtime WebSocket JSON event codec.

use crate::realtime::{
    RealtimeAudioFormat, RealtimeCapabilities, RealtimeCodec, RealtimeError, RealtimeEvent,
    RealtimeFrame, RealtimeHistoryItem, RealtimeInput, RealtimeRole, RealtimeToolResult,
    MAX_REALTIME_TOOL_RESULTS,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use serde_json::{json, Value};

/// Audio encodings understood by the OpenAI Realtime session configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiAudioFormat {
    Pcm16,
    G711MuLaw,
    G711ALaw,
}

impl OpenAiAudioFormat {
    fn wire_format(self) -> Value {
        match self {
            Self::Pcm16 => json!({"type": "audio/pcm", "rate": 24000}),
            Self::G711MuLaw => json!({"type": "audio/pcmu"}),
            Self::G711ALaw => json!({"type": "audio/pcma"}),
        }
    }

    fn core_format(self) -> RealtimeAudioFormat {
        match self {
            Self::Pcm16 => RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 24_000,
            },
            Self::G711MuLaw => RealtimeAudioFormat::G711MuLaw,
            Self::G711ALaw => RealtimeAudioFormat::G711ALaw,
        }
    }
}

/// Requested response type. OpenAI currently accepts either text or audio
/// output for Realtime, with audio output also carrying transcript events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiOutputMode {
    Text,
    Audio,
}

/// Voice selected for OpenAI Realtime audio output. Named built-in voices are
/// sent as strings; custom voices are sent as an object containing their ID.
/// Names are intentionally open-ended so newer provider voices remain usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiRealtimeVoice {
    BuiltIn(String),
    Custom { id: String },
}

impl OpenAiRealtimeVoice {
    fn as_wire_value(&self) -> Value {
        match self {
            Self::BuiltIn(name) => Value::String(name.clone()),
            Self::Custom { id } => json!({ "id": id }),
        }
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        let value = match self {
            Self::BuiltIn(name) => name,
            Self::Custom { id } => id,
        };
        if value.trim().is_empty() || value.contains('\0') {
            return Err(RealtimeError::InvalidConfig {
                message:
                    "OpenAI Realtime voice names and custom IDs must be non-empty and NUL-free"
                        .into(),
            });
        }
        Ok(())
    }
}

/// One first-party function tool advertised to an OpenAI Realtime session.
/// Only fields documented for Realtime function tools are exposed here.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiRealtimeFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl OpenAiRealtimeFunctionTool {
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

/// OpenAI Realtime's documented function tool selection modes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OpenAiRealtimeToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

impl OpenAiRealtimeToolChoice {
    fn as_wire_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".into()),
            Self::None => Value::String("none".into()),
            Self::Required => Value::String("required".into()),
            Self::Function(name) => json!({ "type": "function", "name": name }),
        }
    }
}

impl OpenAiOutputMode {
    fn wire_name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Audio => "audio",
        }
    }
}

/// Settings encoded into the session's initial `session.update` event. The
/// model is selected by the caller's WebSocket endpoint or provider route.
#[derive(Debug, Clone)]
pub struct OpenAiRealtimeConfig {
    pub instructions: Option<String>,
    /// Enable input transcripts separately from native audio understanding.
    pub input_transcription_model: Option<String>,
    /// Disable server VAD; the host commits input and explicitly continues.
    pub manual_turns: bool,
    pub output_mode: OpenAiOutputMode,
    pub input_audio_format: OpenAiAudioFormat,
    pub output_audio_format: OpenAiAudioFormat,
    /// Optional named built-in or project-approved custom output voice.
    pub voice: Option<OpenAiRealtimeVoice>,
    pub tools: Vec<OpenAiRealtimeFunctionTool>,
    pub tool_choice: OpenAiRealtimeToolChoice,
}

impl Default for OpenAiRealtimeConfig {
    fn default() -> Self {
        Self {
            instructions: None,
            input_transcription_model: Some("gpt-4o-mini-transcribe".into()),
            manual_turns: false,
            output_mode: OpenAiOutputMode::Audio,
            input_audio_format: OpenAiAudioFormat::Pcm16,
            output_audio_format: OpenAiAudioFormat::Pcm16,
            voice: None,
            tools: Vec::new(),
            tool_choice: OpenAiRealtimeToolChoice::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiRealtimeCodec {
    config: OpenAiRealtimeConfig,
}

impl OpenAiRealtimeCodec {
    pub fn new(config: OpenAiRealtimeConfig) -> Self {
        Self { config }
    }

    fn json_frame(&self, value: Value) -> Result<RealtimeFrame, RealtimeError> {
        serde_json::to_vec(&value)
            .map(Bytes::from)
            .map(RealtimeFrame::Text)
            .map_err(|error| RealtimeError::Codec {
                message: format!("could not encode JSON event: {error}"),
            })
    }
}

impl Default for OpenAiRealtimeCodec {
    fn default() -> Self {
        Self::new(OpenAiRealtimeConfig::default())
    }
}

impl RealtimeCodec for OpenAiRealtimeCodec {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            history_import: true,
            tools: true,
            input_transcription: self.config.input_transcription_model.is_some(),
            output_transcription: true,
            usage: true,
            interruption: true,
            audio_truncation: true,
            session_resumption: false,
        }
    }
    fn initial_frames(&self) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        if let Some(voice) = &self.config.voice {
            voice.validate()?;
        }
        let mut session = json!({
            "type": "realtime",
            "output_modalities": [self.config.output_mode.wire_name()],
            "audio": {
                "input": { "format": self.config.input_audio_format.wire_format() },
                "output": { "format": self.config.output_audio_format.wire_format() }
            }
        });
        if let Some(model) = &self.config.input_transcription_model {
            if model.trim().is_empty() || model.contains('\0') {
                return Err(RealtimeError::InvalidConfig {
                    message: "OpenAI input transcription model must be nonempty and NUL-free"
                        .into(),
                });
            }
            session["audio"]["input"]["transcription"] = json!({"model":model});
        }
        if self.config.manual_turns {
            session["audio"]["input"]["turn_detection"] = Value::Null;
        }
        if let Some(instructions) = &self.config.instructions {
            session["instructions"] = Value::String(instructions.clone());
        }
        if let Some(voice) = &self.config.voice {
            session["audio"]["output"]["voice"] = voice.as_wire_value();
        }
        let tools = encode_session_tools(&self.config)?;
        if !tools.is_empty() {
            session["tools"] = Value::Array(tools);
        }
        if !self.config.tools.is_empty()
            || !matches!(&self.config.tool_choice, OpenAiRealtimeToolChoice::Auto)
        {
            session["tool_choice"] = self.config.tool_choice.as_wire_value();
        }
        Ok(vec![self.json_frame(json!({
            "type": "session.update",
            "session": session
        }))?])
    }

    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let values = match input {
            RealtimeInput::ImportHistory { items } => encode_history(items)?,
            RealtimeInput::ClearAudio => vec![json!({
                "type": "input_audio_buffer.clear"
            })],
            RealtimeInput::FinishSession => {
                return Err(RealtimeError::InvalidInput {
                    message: "session finishing is not an OpenAI Realtime client event; close the transport after the host finishes the session".into(),
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
                if format != &self.config.input_audio_format.core_format() {
                    return Err(RealtimeError::InvalidInput {
                        message: format!(
                            "audio format {format:?} does not match the OpenAI session format {:?}",
                            self.config.input_audio_format.core_format()
                        ),
                    });
                }
                vec![json!({
                    "type": "input_audio_buffer.append",
                    "audio": STANDARD.encode(data)
                })]
            }
            RealtimeInput::Image { data, mime_type } => {
                validate_image(data, mime_type)?;
                let data_url = format!("data:{mime_type};base64,{}", STANDARD.encode(data));
                vec![json!({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_image", "image_url": data_url }]
                    }
                })]
            }
            RealtimeInput::RetrieveItem { item_id } => {
                validate_item_id(item_id)?;
                vec![json!({
                    "type": "conversation.item.retrieve",
                    "item_id": item_id,
                })]
            }
            RealtimeInput::DeleteItem { item_id } => {
                validate_item_id(item_id)?;
                vec![json!({
                    "type": "conversation.item.delete",
                    "item_id": item_id,
                })]
            }
            RealtimeInput::TruncateAudio {
                item_id,
                content_index,
                audio_end_ms,
            } => {
                validate_item_id(item_id)?;
                if *content_index != 0 {
                    return Err(RealtimeError::InvalidInput {
                        message:
                            "OpenAI Realtime conversation.item.truncate requires content_index 0"
                                .into(),
                    });
                }
                vec![json!({
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                })]
            }
            RealtimeInput::CommitAudio => vec![json!({
                "type": "input_audio_buffer.commit"
            })],
            RealtimeInput::ToolResult {
                call_id, output, ..
            } => {
                vec![encode_function_call_output(call_id, output)?]
            }
            RealtimeInput::ToolResults { results } => encode_function_call_outputs(results)?,
            RealtimeInput::ContinueResponse => vec![json!({ "type": "response.create" })],
            RealtimeInput::Interrupt => vec![json!({ "type": "response.cancel" })],
        };
        values
            .into_iter()
            .map(|value| self.json_frame(value))
            .collect()
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        crate::realtime::decode_json_events(frame, self.config.output_audio_format.core_format())
    }
}

fn encode_session_tools(config: &OpenAiRealtimeConfig) -> Result<Vec<Value>, RealtimeError> {
    let mut names = std::collections::HashSet::with_capacity(config.tools.len());
    let mut tools = Vec::with_capacity(config.tools.len());
    for tool in &config.tools {
        if tool.name.trim().is_empty() || tool.name.contains('\0') {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime function tool names must be non-empty and NUL-free"
                    .into(),
            });
        }
        if !names.insert(tool.name.as_str()) {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime function tool names must be unique".into(),
            });
        }
        if !tool.parameters.is_object() {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime function tool parameters must be a JSON Schema object"
                    .into(),
            });
        }
        if tool
            .description
            .as_deref()
            .is_some_and(|description| description.contains('\0'))
        {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime function tool descriptions must be NUL-free".into(),
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
        OpenAiRealtimeToolChoice::Auto | OpenAiRealtimeToolChoice::None => {}
        OpenAiRealtimeToolChoice::Required if config.tools.is_empty() => {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime required tool choice needs at least one function tool"
                    .into(),
            });
        }
        OpenAiRealtimeToolChoice::Required => {}
        OpenAiRealtimeToolChoice::Function(name)
            if name.trim().is_empty() || !names.contains(name.as_str()) =>
        {
            return Err(RealtimeError::InvalidConfig {
                message: "OpenAI Realtime function tool choice must name a configured function"
                    .into(),
            });
        }
        OpenAiRealtimeToolChoice::Function(_) => {}
    }
    Ok(tools)
}

fn validate_image(data: &Bytes, mime_type: &str) -> Result<(), RealtimeError> {
    let valid_image_mime = mime_type.split_once('/').is_some_and(|(kind, subtype)| {
        kind.eq_ignore_ascii_case("image")
            && !subtype.is_empty()
            && subtype.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'.' | b'+' | b'-'
                    )
            })
    });
    if data.is_empty() || !valid_image_mime {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Realtime image input requires non-empty bytes and an image MIME type"
                .into(),
        });
    }
    Ok(())
}

fn validate_item_id(item_id: &str) -> Result<(), RealtimeError> {
    if item_id.trim().is_empty() || item_id.contains('\0') {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Realtime conversation item IDs must be non-empty and NUL-free".into(),
        });
    }
    Ok(())
}

fn encode_function_call_outputs(
    results: &[RealtimeToolResult],
) -> Result<Vec<Value>, RealtimeError> {
    if results.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "OpenAI Realtime tool result batch must not be empty".into(),
        });
    }
    if results.len() > MAX_REALTIME_TOOL_RESULTS {
        return Err(RealtimeError::InvalidInput {
            message: format!(
                "OpenAI Realtime tool result batch exceeds the local limit of {MAX_REALTIME_TOOL_RESULTS} results"
            ),
        });
    }
    let mut call_ids = std::collections::HashSet::with_capacity(results.len());
    for result in results {
        validate_call_id(&result.call_id)?;
        if !call_ids.insert(result.call_id.as_str()) {
            return Err(RealtimeError::InvalidInput {
                message: "OpenAI Realtime tool result batch contains a duplicate call_id".into(),
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
    let output = serde_json::to_string(output).map_err(|error| RealtimeError::Codec {
        message: format!("could not encode tool result: {error}"),
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
            message: "OpenAI Realtime function result requires a non-empty, NUL-free call_id"
                .into(),
        });
    }
    Ok(())
}

/// Encode host history without a response.create. Validate the entire batch
/// before dispatch; no partial history is sent when a later item is invalid.
pub(crate) fn encode_history(items: &[RealtimeHistoryItem]) -> Result<Vec<Value>, RealtimeError> {
    if items.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "history must contain at least one item".into(),
        });
    }
    let mut ids = std::collections::HashSet::new();
    items.iter().map(|entry| {
        let (id, mut item) = match entry {
            RealtimeHistoryItem::Message { item_id, role, text } => {
                if text.contains('\0') { return Err(RealtimeError::InvalidInput { message: "history text must be NUL-free".into() }); }
                let (role, kind) = match role { RealtimeRole::User => ("user", "input_text"), RealtimeRole::System => ("system", "input_text"), RealtimeRole::Assistant => ("assistant", "output_text") };
                (item_id, json!({ "type":"message", "role":role, "content":[{"type":kind,"text":text}] }))
            }
            RealtimeHistoryItem::ToolCall { item_id, call_id, name, arguments } => {
                validate_call_id(call_id)?;
                if name.trim().is_empty() || name.contains('\0') || !arguments.is_object() { return Err(RealtimeError::InvalidInput { message: "history tool calls require a name and JSON object arguments".into() }); }
                (item_id, json!({"type":"function_call","call_id":call_id,"name":name,"arguments":arguments.to_string()}))
            }
            RealtimeHistoryItem::ToolResult { item_id, call_id, output, .. } => {
                validate_call_id(call_id)?;
                (item_id, json!({"type":"function_call_output","call_id":call_id,"output":output.to_string()}))
            }
        };
        if let Some(id) = id {
            validate_item_id(id)?;
            if !ids.insert(id) { return Err(RealtimeError::InvalidInput { message: "history item IDs must be unique".into() }); }
            item["id"] = json!(id);
        }
        Ok(json!({"type":"conversation.item.create","item":item}))
    }).collect()
}
