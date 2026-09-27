//! Native streaming calls to a published Model Studio Knowledge Q&A service.
//!
//! The service is stateless: callers supply their complete conversation
//! history for each request. This module preserves every SSE data payload and
//! never executes the hosted agent's tool calls in the local process.

use super::{
    invalid, valid_resource_id, QwenKnowledgeDispatch, QwenKnowledgeError, QwenKnowledgeFileRef,
    QwenKnowledgeScope, QwenKnowledgeService, DEFAULT_TIMEOUT,
};
use crate::{
    framing::sse::SseFrameSplitter,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, StreamResponse},
};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde_json::{json, Map, Value};
use std::{collections::VecDeque, fmt};
use thiserror::Error;
use url::Url;

const KNOWLEDGE_CHAT_PATH: &str = "/api/v2/apps/knowledge/chat";
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_STREAM_BYTES: usize = 64 * 1024 * 1024;
const SSE_PARSE_CHUNK_BYTES: usize = 64 * 1024;
const MAX_SESSION_FILES: usize = 10;

/// Opaque reference to a published Knowledge Q&A service in one workspace.
///
/// The reference binds the caller-supplied application ID to the exact
/// provider profile, account, region, workspace, and endpoint that supplied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeChatRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    agent_id: String,
}

impl QwenKnowledgeChatRef {
    /// Bind an ID obtained from an already-published Knowledge Q&A service.
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        agent_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let agent_id = agent_id.into();
        if !valid_resource_id(&agent_id) {
            return Err(invalid("Knowledge Chat agent_id is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            agent_id,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

impl QwenKnowledgeScope {
    /// Bind the ID of a Knowledge Q&A service already published in this scope.
    pub fn knowledge_chat_ref(
        &self,
        agent_id: impl Into<String>,
    ) -> Result<QwenKnowledgeChatRef, QwenKnowledgeError> {
        QwenKnowledgeChatRef::from_scope(self, agent_id)
    }
}

/// A JSON content part supported by the Knowledge Chat message format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenKnowledgeChatContentPart {
    Text(String),
    ImageUrl(String),
}

impl QwenKnowledgeChatContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Supply a caller-accessible HTTP(S) image URL. The client does not fetch it.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl(url.into())
    }

    fn validate(&self) -> Result<(), QwenKnowledgeChatError> {
        if let Self::ImageUrl(value) = self {
            validate_image_url(value)?;
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        match self {
            Self::Text(text) => json!({"type": "text", "text": text}),
            Self::ImageUrl(url) => {
                json!({"type": "image_url", "image_url": {"url": url}})
            }
        }
    }
}

/// Caller-owned message content. It can be plain text or a multimodal array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenKnowledgeChatContent {
    Text(String),
    Parts(Vec<QwenKnowledgeChatContentPart>),
}

impl QwenKnowledgeChatContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn parts(parts: impl IntoIterator<Item = QwenKnowledgeChatContentPart>) -> Self {
        Self::Parts(parts.into_iter().collect())
    }

    fn validate(&self) -> Result<(), QwenKnowledgeChatError> {
        if let Self::Parts(parts) = self {
            if parts.is_empty() {
                return Err(QwenKnowledgeChatError::InvalidInput(
                    "Knowledge Chat content-part arrays must not be empty".into(),
                ));
            }
            for part in parts {
                part.validate()?;
            }
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        match self {
            Self::Text(text) => json!(text),
            Self::Parts(parts) => Value::Array(parts.iter().map(|part| part.to_value()).collect()),
        }
    }
}

/// Message role accepted by the published Knowledge Chat service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenKnowledgeChatRole {
    User,
    Assistant,
    Tool,
}

impl QwenKnowledgeChatRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One hosted function-call record carried as ordinary conversation history.
///
/// Function arguments remain the original JSON string. This type does not
/// deserialize arguments or make the hosted tool executable on the client.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenKnowledgeChatToolCall {
    id: String,
    name: String,
    arguments: String,
    index: Option<u32>,
}

impl QwenKnowledgeChatToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            index: None,
        }
    }

    /// Retain the provider's call index when one was present in the SSE frame.
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = Some(index);
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &str {
        &self.arguments
    }

    fn validate(&self) -> Result<(), QwenKnowledgeChatError> {
        if self.id.is_empty() || self.name.is_empty() || self.arguments.is_empty() {
            return Err(QwenKnowledgeChatError::InvalidInput(
                "Knowledge Chat tool-call ID, function name, and arguments must be nonempty".into(),
            ));
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        let mut value = json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        });
        if let Some(index) = self.index {
            value["index"] = json!(index);
        }
        value
    }
}

impl fmt::Debug for QwenKnowledgeChatToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeChatToolCall")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("arguments", &"<redacted provider arguments>")
            .field("index", &self.index)
            .finish()
    }
}

/// One item in the caller-owned stateless message history.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenKnowledgeChatMessage {
    role: QwenKnowledgeChatRole,
    content: QwenKnowledgeChatContent,
    tool_calls: Vec<QwenKnowledgeChatToolCall>,
    tool_call_id: Option<String>,
}

impl QwenKnowledgeChatMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::new(
            QwenKnowledgeChatRole::User,
            QwenKnowledgeChatContent::text(text),
        )
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::new(
            QwenKnowledgeChatRole::Assistant,
            QwenKnowledgeChatContent::text(text),
        )
    }

    pub fn user_parts(parts: impl IntoIterator<Item = QwenKnowledgeChatContentPart>) -> Self {
        Self::new(
            QwenKnowledgeChatRole::User,
            QwenKnowledgeChatContent::parts(parts),
        )
    }

    pub fn assistant_parts(parts: impl IntoIterator<Item = QwenKnowledgeChatContentPart>) -> Self {
        Self::new(
            QwenKnowledgeChatRole::Assistant,
            QwenKnowledgeChatContent::parts(parts),
        )
    }

    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: impl IntoIterator<Item = QwenKnowledgeChatToolCall>,
    ) -> Self {
        Self {
            role: QwenKnowledgeChatRole::Assistant,
            content: QwenKnowledgeChatContent::text(content),
            tool_calls: tool_calls.into_iter().collect(),
            tool_call_id: None,
        }
    }

    pub fn tool_return(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: QwenKnowledgeChatRole::Tool,
            content: QwenKnowledgeChatContent::text(content),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    pub fn tool_return_content(
        tool_call_id: impl Into<String>,
        content: QwenKnowledgeChatContent,
    ) -> Self {
        Self {
            role: QwenKnowledgeChatRole::Tool,
            content,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    fn new(role: QwenKnowledgeChatRole, content: QwenKnowledgeChatContent) -> Self {
        Self {
            role,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn role(&self) -> QwenKnowledgeChatRole {
        self.role
    }

    fn validate(&self) -> Result<(), QwenKnowledgeChatError> {
        self.content.validate()?;
        match self.role {
            QwenKnowledgeChatRole::User => {
                if !self.tool_calls.is_empty() || self.tool_call_id.is_some() {
                    return Err(QwenKnowledgeChatError::InvalidInput(
                        "user messages cannot carry tool-call fields".into(),
                    ));
                }
            }
            QwenKnowledgeChatRole::Assistant => {
                if self.tool_call_id.is_some() {
                    return Err(QwenKnowledgeChatError::InvalidInput(
                        "assistant messages cannot carry tool_call_id".into(),
                    ));
                }
                for tool_call in &self.tool_calls {
                    tool_call.validate()?;
                }
            }
            QwenKnowledgeChatRole::Tool => {
                if !self.tool_calls.is_empty()
                    || self.tool_call_id.as_deref().is_none_or(str::is_empty)
                {
                    return Err(QwenKnowledgeChatError::InvalidInput(
                        "tool messages require tool_call_id and cannot carry tool_calls".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        let mut value = json!({
            "role": self.role.as_str(),
            "content": self.content.to_value(),
        });
        if !self.tool_calls.is_empty() {
            value["tool_calls"] = Value::Array(
                self.tool_calls
                    .iter()
                    .map(QwenKnowledgeChatToolCall::to_value)
                    .collect(),
            );
        }
        if let Some(id) = &self.tool_call_id {
            value["tool_call_id"] = json!(id);
        }
        value
    }
}

impl fmt::Debug for QwenKnowledgeChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeChatMessage")
            .field("role", &self.role)
            .field("content", &"<redacted conversation content>")
            .field("tool_calls", &self.tool_calls)
            .field("tool_call_id", &self.tool_call_id)
            .finish()
    }
}

/// One single-turn request to an already-published Knowledge Q&A service.
///
/// Message history is caller-managed. The request never creates or persists a
/// server conversation, and the endpoint does not return a session ID.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenKnowledgeChatRequest {
    chat: QwenKnowledgeChatRef,
    messages: Vec<QwenKnowledgeChatMessage>,
    session_files: Vec<QwenKnowledgeFileRef>,
    request_id: Option<String>,
    enable_cache_control: Option<bool>,
}

impl QwenKnowledgeChatRequest {
    pub fn new(
        chat: QwenKnowledgeChatRef,
        messages: impl IntoIterator<Item = QwenKnowledgeChatMessage>,
    ) -> Self {
        Self {
            chat,
            messages: messages.into_iter().collect(),
            session_files: Vec::new(),
            request_id: None,
            enable_cache_control: None,
        }
    }

    pub fn with_session_files(
        mut self,
        files: impl IntoIterator<Item = QwenKnowledgeFileRef>,
    ) -> Self {
        self.session_files = files.into_iter().collect();
        self
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_cache_control(mut self, enabled: bool) -> Self {
        self.enable_cache_control = Some(enabled);
        self
    }

    pub fn chat_ref(&self) -> &QwenKnowledgeChatRef {
        &self.chat
    }

    pub fn messages(&self) -> &[QwenKnowledgeChatMessage] {
        &self.messages
    }

    pub fn session_files(&self) -> &[QwenKnowledgeFileRef] {
        &self.session_files
    }

    fn validate(&self) -> Result<(), QwenKnowledgeChatError> {
        if self.messages.is_empty() {
            return Err(QwenKnowledgeChatError::InvalidInput(
                "Knowledge Chat requires at least one history message".into(),
            ));
        }
        for message in &self.messages {
            message.validate()?;
        }
        if self.session_files.len() > MAX_SESSION_FILES {
            return Err(QwenKnowledgeChatError::InvalidInput(format!(
                "Knowledge Chat accepts at most {MAX_SESSION_FILES} session file IDs"
            )));
        }
        Ok(())
    }

    fn to_body(&self) -> Result<Bytes, QwenKnowledgeChatError> {
        let mut input = Map::new();
        input.insert(
            "messages".into(),
            Value::Array(
                self.messages
                    .iter()
                    .map(QwenKnowledgeChatMessage::to_value)
                    .collect(),
            ),
        );
        if let Some(request_id) = &self.request_id {
            input.insert("request_id".into(), json!(request_id));
        }

        let mut agent_options = Map::new();
        agent_options.insert("agent_id".into(), json!(self.chat.agent_id));
        if !self.session_files.is_empty() {
            agent_options.insert(
                "session_files".into(),
                json!(self
                    .session_files
                    .iter()
                    .map(QwenKnowledgeFileRef::file_id)
                    .collect::<Vec<_>>()),
            );
        }
        if let Some(enabled) = self.enable_cache_control {
            agent_options.insert("enable_cache_control".into(), json!(enabled));
        }
        let body = json!({
            "input": Value::Object(input),
            "parameters": {"agent_options": Value::Object(agent_options)},
            "stream": true,
        });
        let bytes = serde_json::to_vec(&body).map_err(|_| {
            QwenKnowledgeChatError::InvalidInput(
                "Knowledge Chat request cannot be encoded as JSON".into(),
            )
        })?;
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(QwenKnowledgeChatError::InvalidInput(format!(
                "Knowledge Chat request exceeds the local {} MiB body limit",
                MAX_REQUEST_BYTES / 1024 / 1024
            )));
        }
        Ok(Bytes::from(bytes))
    }
}

impl fmt::Debug for QwenKnowledgeChatRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeChatRequest")
            .field("chat", &self.chat)
            .field("messages", &"<redacted conversation history>")
            .field("message_count", &self.messages.len())
            .field("session_file_count", &self.session_files.len())
            .field("request_id", &self.request_id)
            .field("enable_cache_control", &self.enable_cache_control)
            .finish()
    }
}

/// Provider diagnostic returned in HTTP or SSE error data.
#[derive(Clone, PartialEq)]
pub struct QwenKnowledgeChatProviderError {
    pub status: Option<u16>,
    pub code: Option<String>,
    pub message: String,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Display for QwenKnowledgeChatProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = &self.code {
            write!(f, "{code}: ")?;
        }
        f.write_str(&self.message)
    }
}

impl fmt::Debug for QwenKnowledgeChatProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeChatProviderError")
            .field("status", &self.status)
            .field("code", &self.code)
            .field("message", &self.message)
            .field("request_id", &self.request_id)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum QwenKnowledgeChatError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error(transparent)]
    Knowledge(#[from] QwenKnowledgeError),
    #[error("invalid Qwen Knowledge Chat input: {0}")]
    InvalidInput(String),
    #[error("Qwen Knowledge Chat call outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen Knowledge Chat provider returned HTTP {status}: {error}")]
    Provider {
        status: u16,
        error: Box<QwenKnowledgeChatProviderError>,
        dispatch: QwenKnowledgeDispatch,
    },
    #[error("Qwen Knowledge Chat accepted the call but returned an invalid response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: QwenKnowledgeDispatch,
    },
    #[error("Qwen Knowledge Chat stream was interrupted: {source}")]
    StreamInterrupted {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
}

impl QwenKnowledgeChatError {
    pub fn dispatch(&self) -> QwenKnowledgeDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) => QwenKnowledgeDispatch::NotSent,
            Self::Knowledge(error) => error.dispatch(),
            Self::OutcomeUnknown { .. } | Self::StreamInterrupted { .. } => {
                QwenKnowledgeDispatch::Unknown
            }
            Self::Provider { dispatch, .. } | Self::InvalidResponse { dispatch, .. } => *dispatch,
        }
    }
}

/// One unmodified native Knowledge Chat SSE data payload.
#[derive(Clone, PartialEq)]
pub struct QwenKnowledgeChatEvent {
    native: Value,
    complete: bool,
}

impl QwenKnowledgeChatEvent {
    pub fn native(&self) -> &Value {
        &self.native
    }

    /// True only for the stop frame emitted after a clean SSE EOF.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn step(&self) -> Option<&str> {
        self.native
            .pointer("/output/choices/0/message/extra/step")
            .and_then(Value::as_str)
    }

    pub fn step_change(&self) -> Option<&str> {
        self.native
            .pointer("/output/choices/0/message/extra/step_change")
            .and_then(Value::as_str)
    }
}

impl fmt::Debug for QwenKnowledgeChatEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeChatEvent")
            .field("step", &self.step())
            .field("step_change", &self.step_change())
            .field("complete", &self.complete)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}

/// One HTTP-accepted, native SSE chat call. Dropping this stream cancels the
/// response read. The final stop frame is withheld until the body ends cleanly.
pub struct QwenKnowledgeChatStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    chat: QwenKnowledgeChatRef,
    request_id: Option<String>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_bytes: Bytes,
    pending_error: Option<LlmError>,
    pending_completion: Option<QwenKnowledgeChatEvent>,
    wire_bytes: usize,
    ended: bool,
    done: bool,
}

impl QwenKnowledgeChatStream {
    fn new(response: StreamResponse, chat: QwenKnowledgeChatRef) -> Self {
        let request_id = response.header("x-request-id").map(str::to_owned);
        Self {
            body: response.body,
            chat,
            request_id,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_bytes: Bytes::new(),
            pending_error: None,
            pending_completion: None,
            wire_bytes: 0,
            ended: false,
            done: false,
        }
    }

    pub fn chat_ref(&self) -> &QwenKnowledgeChatRef {
        &self.chat
    }

    /// The provider request ID when it has appeared in a response header or frame.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Read one raw provider frame. Success is confirmed only by an event with
    /// `output.choices[0].finish_reason=stop` followed by clean HTTP EOF.
    pub async fn next_event(
        &mut self,
    ) -> Result<Option<QwenKnowledgeChatEvent>, QwenKnowledgeChatError> {
        if self.done {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                match self.decode_event(&frame) {
                    Ok(Some(event)) => return Ok(Some(event)),
                    Ok(None) => continue,
                    Err(error) => {
                        self.terminate();
                        return Err(error);
                    }
                }
            }
            if let Some(source) = self.pending_error.take() {
                let error = QwenKnowledgeChatError::StreamInterrupted {
                    source,
                    request_id: self.request_id.clone(),
                };
                self.terminate();
                return Err(error);
            }
            if !self.pending_bytes.is_empty() {
                let len = self.pending_bytes.len().min(SSE_PARSE_CHUNK_BYTES);
                let bytes = self.pending_bytes.split_to(len);
                let (frames, error) = self.splitter.push_batch(&bytes);
                self.ready.extend(frames);
                self.pending_error = error;
                continue;
            }
            if self.ended {
                if let Some(mut completion) = self.pending_completion.take() {
                    if completion
                        .native
                        .get("request_id")
                        .and_then(Value::as_str)
                        .is_some()
                    {
                        self.request_id = completion
                            .native
                            .get("request_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                    }
                    completion.complete = true;
                    self.terminate();
                    return Ok(Some(completion));
                }
                let error = QwenKnowledgeChatError::StreamInterrupted {
                    source: LlmError::StreamInterrupted {
                        message: "Knowledge Chat SSE ended before finish_reason was stop".into(),
                    },
                    request_id: self.request_id.clone(),
                };
                self.terminate();
                return Err(error);
            }
            match self.body.next().await {
                Some(Ok(bytes)) => {
                    if bytes.len() > MAX_STREAM_BYTES.saturating_sub(self.wire_bytes) {
                        self.pending_error = Some(LlmError::StreamInterrupted {
                            message: "Knowledge Chat SSE response exceeds the 64 MiB wire limit"
                                .into(),
                        });
                    } else {
                        self.wire_bytes += bytes.len();
                        self.pending_bytes = bytes;
                    }
                }
                Some(Err(error)) => self.pending_error = Some(error),
                None => {
                    self.ended = true;
                    match self.splitter.finish() {
                        Ok(Some(frame)) => self.ready.push_back(frame),
                        Ok(None) => {}
                        Err(error) => self.pending_error = Some(error),
                    }
                }
            }
        }
    }

    fn decode_event(
        &mut self,
        frame: &[u8],
    ) -> Result<Option<QwenKnowledgeChatEvent>, QwenKnowledgeChatError> {
        let native: Value = serde_json::from_slice(frame).map_err(|_| {
            self.invalid_event(
                "SSE data is not a JSON object",
                Value::String(String::from_utf8_lossy(frame).into_owned()),
            )
        })?;
        if !native.is_object() {
            return Err(self.invalid_event("SSE data must be a JSON object", native));
        }
        if let Some(id) = native.get("request_id").and_then(Value::as_str) {
            self.request_id = Some(id.to_owned());
        }

        let code = match native.get("code") {
            None => None,
            Some(Value::String(code)) if code.is_empty() || code == "200" => {
                (!code.is_empty()).then(|| code.clone())
            }
            Some(Value::String(code)) => Some(code.clone()),
            Some(_) => {
                return Err(self.invalid_event("SSE code must be a string when present", native));
            }
        };
        let status_code = match native.get("status_code") {
            None => None,
            Some(Value::Number(status)) => status
                .as_u64()
                .and_then(|status| u16::try_from(status).ok())
                .map(Some)
                .ok_or_else(|| {
                    self.invalid_event(
                        "SSE status_code must be an unsigned 16-bit integer",
                        native.clone(),
                    )
                })?,
            Some(_) => {
                return Err(self.invalid_event(
                    "SSE status_code must be an unsigned integer when present",
                    native,
                ));
            }
        };
        if code.as_deref().is_some_and(|code| code != "200")
            || status_code.is_some_and(|status| !(200..300).contains(&status))
        {
            let message = native
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Model Studio reported a Knowledge Chat error")
                .to_owned();
            let request_id = self.request_id.clone();
            return Err(QwenKnowledgeChatError::Provider {
                status: status_code.unwrap_or(200),
                error: Box::new(QwenKnowledgeChatProviderError {
                    status: status_code,
                    code,
                    message,
                    request_id,
                    native,
                }),
                dispatch: QwenKnowledgeDispatch::Accepted,
            });
        }

        if self.pending_completion.is_some() {
            return Err(self.invalid_event("SSE data arrived after the final stop payload", native));
        }

        let finish_reason = native
            .pointer("/output/choices/0/finish_reason")
            .and_then(Value::as_str);
        if finish_reason == Some("stop") {
            self.pending_completion = Some(QwenKnowledgeChatEvent {
                native,
                complete: false,
            });
            return Ok(None);
        }
        if finish_reason.is_some_and(|reason| !reason.is_empty()) {
            return Err(self.invalid_event(
                "unsupported output.choices[0].finish_reason in SSE payload",
                native,
            ));
        }

        Ok(Some(QwenKnowledgeChatEvent {
            native,
            complete: false,
        }))
    }

    fn invalid_event(&self, message: &str, native: Value) -> QwenKnowledgeChatError {
        QwenKnowledgeChatError::InvalidResponse {
            message: message.into(),
            request_id: self.request_id.clone(),
            native: Box::new(native),
            dispatch: QwenKnowledgeDispatch::Accepted,
        }
    }

    fn terminate(&mut self) {
        self.done = true;
        self.body = futures::stream::empty().boxed();
        self.pending_bytes = Bytes::new();
        self.ready.clear();
        self.pending_completion = None;
        self.splitter = SseFrameSplitter::new();
    }
}

impl<'a> QwenKnowledgeService<'a> {
    /// Start one streamed request to a service that the caller has published.
    /// The method sends once and never retries an ambiguous or billable call.
    pub async fn knowledge_chat(
        &self,
        request: &QwenKnowledgeChatRequest,
    ) -> Result<QwenKnowledgeChatStream, QwenKnowledgeChatError> {
        self.ensure_supported_region()?;
        if request.chat.scope != self.scope
            || request.chat.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || !valid_resource_id(&request.chat.agent_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "Knowledge Chat reference belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        request.validate()?;
        for file in &request.session_files {
            self.validate_file_ref(file)?;
        }
        let body = request.to_body()?;
        let url = self.operation_url(KNOWLEDGE_CHAT_PATH, &[])?;
        let http_request = HttpRequest {
            method: "POST".into(),
            url: url.to_string(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", self.credential.expose_secret()),
                ),
                ("content-type".into(), "application/json".into()),
                ("accept".into(), "text/event-stream".into()),
            ],
            body,
            timeout: Some(DEFAULT_TIMEOUT),
        };
        let response = HttpExecutor::new(self.http)
            .send(http_request)
            .await
            .map_err(|source| QwenKnowledgeChatError::OutcomeUnknown {
                source,
                request_id: None,
            })?;
        let request_id = response.header("x-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            let status = response.status;
            let response = HttpExecutor::collect_response(response, None)
                .await
                .map_err(|source| QwenKnowledgeChatError::OutcomeUnknown {
                    source,
                    request_id: request_id.clone(),
                })?;
            let provider_error = decode_provider_error(status, &response.body, request_id.clone());
            let dispatch = if (400..500).contains(&status) && status != 408 {
                QwenKnowledgeDispatch::Rejected
            } else {
                QwenKnowledgeDispatch::Unknown
            };
            return Err(QwenKnowledgeChatError::Provider {
                status,
                error: Box::new(provider_error),
                dispatch,
            });
        }
        let content_type = response.header("content-type");
        if content_type.is_some_and(|value| {
            !value
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        }) {
            let response = HttpExecutor::collect_response(response, Some(MAX_REQUEST_BYTES))
                .await
                .map_err(|source| QwenKnowledgeChatError::OutcomeUnknown {
                    source,
                    request_id: request_id.clone(),
                })?;
            let native = serde_json::from_slice(&response.body)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into()));
            let request_id = native
                .get("request_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or(request_id);
            return Err(QwenKnowledgeChatError::InvalidResponse {
                message: "successful Knowledge Chat response did not use text/event-stream".into(),
                request_id,
                native: Box::new(native),
                dispatch: QwenKnowledgeDispatch::Accepted,
            });
        }
        Ok(QwenKnowledgeChatStream::new(response, request.chat.clone()))
    }
}

fn validate_image_url(value: &str) -> Result<(), QwenKnowledgeChatError> {
    let url = Url::parse(value).map_err(|_| {
        QwenKnowledgeChatError::InvalidInput(
            "Knowledge Chat image_url must be a valid HTTP(S) URL".into(),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(QwenKnowledgeChatError::InvalidInput(
            "Knowledge Chat image_url must be an HTTP(S) URL with a host and no embedded credentials".into(),
        ));
    }
    Ok(())
}

fn decode_provider_error(
    status: u16,
    bytes: &[u8],
    header_request_id: Option<String>,
) -> QwenKnowledgeChatProviderError {
    let native = serde_json::from_slice(bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()));
    let code = native
        .get("code")
        .or_else(|| native.get("Code"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let message = native
        .get("message")
        .or_else(|| native.get("Message"))
        .and_then(Value::as_str)
        .unwrap_or("Model Studio returned an HTTP error")
        .to_owned();
    let request_id = native
        .get("request_id")
        .or_else(|| native.get("requestId"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(header_request_id);
    QwenKnowledgeChatProviderError {
        status: Some(status),
        code,
        message,
        request_id,
        native,
    }
}
