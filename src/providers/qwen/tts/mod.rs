//! HTTP synthesis through Alibaba Cloud Model Studio's Qwen3-TTS-Flash API.
//!
//! Each call submits complete text once, returning either a temporary audio
//! URL or native SSE audio deltas followed by that URL. The service does not
//! download audio, retry, or route through a Chat endpoint.

use crate::{
    client::RequestOptions,
    framing::sse::SseFrameSplitter,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, StreamResponse, Transport},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde_json::{json, Value};
use std::{collections::VecDeque, fmt};
use thiserror::Error;
use url::Url;

/// Beijing DashScope endpoint for Qwen3-TTS-Flash non-streaming HTTP calls.
pub const QWEN_TTS_BEIJING_HTTP_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";
/// Singapore DashScope endpoint for Qwen3-TTS-Flash non-streaming HTTP calls.
pub const QWEN_TTS_SINGAPORE_HTTP_ENDPOINT: &str =
    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";

const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TEXT_CHARACTERS: usize = 600;
const MAX_STREAM_BYTES: usize = 64 * 1024 * 1024;
const SSE_PARSE_CHUNK_BYTES: usize = 64 * 1024;

/// Region of the Model Studio account. Beijing and Singapore use different
/// API keys. The Qwen-TTS HTTP reference documents the DashScope generation
/// route rather than a workspace-specific route; the workspace remains part
/// of the caller-owned scope and is never inferred from another service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QwenTtsRegion {
    Beijing,
    Singapore,
}

impl QwenTtsRegion {
    fn endpoint(self) -> &'static str {
        match self {
            Self::Beijing => QWEN_TTS_BEIJING_HTTP_ENDPOINT,
            Self::Singapore => QWEN_TTS_SINGAPORE_HTTP_ENDPOINT,
        }
    }
}

/// Account and workspace identity bound to this Qwen TTS client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsScope {
    profile_name: String,
    account_scope: String,
    region: QwenTtsRegion,
    workspace_id: String,
}

impl QwenTtsScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenTtsRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, QwenTtsError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
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

    pub fn region(&self) -> QwenTtsRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    fn validate(&self) -> Result<(), QwenTtsError> {
        if self.profile_name.trim().is_empty() || self.account_scope.trim().is_empty() {
            return Err(QwenTtsError::InvalidInput(
                "profile name and account scope must be non-empty".into(),
            ));
        }
        let id = self.workspace_id.as_bytes();
        if id.is_empty()
            || id.len() > 63
            || !id[0].is_ascii_alphanumeric()
            || !id[id.len() - 1].is_ascii_alphanumeric()
            || !id
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        {
            return Err(QwenTtsError::InvalidInput(
                "workspace ID must be one DNS label containing only letters, digits, and internal hyphens".into(),
            ));
        }
        Ok(())
    }
}

/// Languages accepted by Qwen3-TTS-Flash's `language_type` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTtsLanguage {
    Auto,
    Chinese,
    English,
    German,
    Italian,
    Portuguese,
    Spanish,
    Japanese,
    Korean,
    French,
    Russian,
}

impl QwenTtsLanguage {
    fn wire_value(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Chinese => "Chinese",
            Self::English => "English",
            Self::German => "German",
            Self::Italian => "Italian",
            Self::Portuguese => "Portuguese",
            Self::Spanish => "Spanish",
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::French => "French",
            Self::Russian => "Russian",
        }
    }
}

/// One complete-text Qwen3-TTS-Flash request, with buffered or SSE output.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenTtsRequest {
    text: String,
    voice: String,
    language_type: Option<QwenTtsLanguage>,
}

impl QwenTtsRequest {
    pub fn new(text: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: voice.into(),
            language_type: None,
        }
    }

    pub fn with_language_type(mut self, language: QwenTtsLanguage) -> Self {
        self.language_type = Some(language);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }

    pub fn language_type(&self) -> Option<QwenTtsLanguage> {
        self.language_type
    }

    fn validate(&self) -> Result<(), QwenTtsError> {
        if self.text.trim().is_empty() {
            return Err(QwenTtsError::InvalidInput(
                "text to synthesize must be non-empty".into(),
            ));
        }
        if self.text.chars().count() > MAX_TEXT_CHARACTERS {
            return Err(QwenTtsError::InvalidInput(format!(
                "Qwen3-TTS-Flash accepts at most {MAX_TEXT_CHARACTERS} input characters"
            )));
        }
        if self.voice.trim().is_empty() {
            return Err(QwenTtsError::InvalidInput("voice must be non-empty".into()));
        }
        Ok(())
    }
}

impl fmt::Debug for QwenTtsRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsRequest")
            .field("text", &"<redacted spoken text>")
            .field("voice", &self.voice)
            .field("language_type", &self.language_type)
            .finish()
    }
}

/// One non-streaming audio result. The URL is signed, expires after 24 hours,
/// and is redacted from `Debug`; the service does not download it.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenTtsSynthesis {
    scope: QwenTtsScope,
    pub request_id: Option<String>,
    audio_url: String,
    pub audio_id: Option<String>,
    pub expires_at_unix_seconds: Option<u64>,
    pub finish_reason: Option<String>,
    pub characters: Option<u64>,
}

impl QwenTtsSynthesis {
    pub fn scope(&self) -> &QwenTtsScope {
        &self.scope
    }

    pub fn audio_url(&self) -> &str {
        &self.audio_url
    }
}

impl fmt::Debug for QwenTtsSynthesis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsSynthesis")
            .field("scope", &self.scope)
            .field("request_id", &self.request_id)
            .field("audio_url", &"<redacted short-lived URL>")
            .field("audio_id", &self.audio_id)
            .field("expires_at_unix_seconds", &self.expires_at_unix_seconds)
            .field("finish_reason", &self.finish_reason)
            .field("characters", &self.characters)
            .finish()
    }
}

/// How far a synthesis call may have progressed at Model Studio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTtsDispatchOutcome {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

/// An error payload received after HTTP acceptance. The complete native JSON
/// retains provider diagnostics, usage and extensions without printing them.
#[derive(Clone, PartialEq)]
pub struct QwenTtsProviderError {
    pub code: Option<String>,
    pub message: String,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Display for QwenTtsProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = &self.code {
            write!(f, "{code}: ")?;
        }
        f.write_str(&self.message)
    }
}

impl fmt::Debug for QwenTtsProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsProviderError")
            .field("code", &self.code)
            .field("message", &self.message)
            .field("request_id", &self.request_id)
            .field("native", &"<redacted provider error payload>")
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum QwenTtsError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Qwen TTS input: {0}")]
    InvalidInput(String),
    #[error("Qwen TTS account scope does not match the configured service")]
    ScopeMismatch,
    #[error("Qwen TTS request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen TTS request was rejected with HTTP {status}: {message}")]
    Rejected {
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen TTS accepted the request but returned an invalid response: {message}")]
    AcceptedInvalidResponse {
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen TTS accepted the request but its audio stream was interrupted: {source}")]
    StreamInterrupted {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen TTS stream returned a provider error: {error}")]
    StreamProvider { error: Box<QwenTtsProviderError> },
}

impl QwenTtsError {
    /// Classifies a failed synthesis call or stream. No request is retried by
    /// this service; `Unknown` and `Accepted` may already be billable.
    pub fn dispatch_outcome(&self) -> QwenTtsDispatchOutcome {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::ScopeMismatch => {
                QwenTtsDispatchOutcome::NotSent
            }
            Self::OutcomeUnknown { .. } => QwenTtsDispatchOutcome::Unknown,
            Self::Rejected { .. } => QwenTtsDispatchOutcome::Rejected,
            Self::AcceptedInvalidResponse { .. }
            | Self::StreamInterrupted { .. }
            | Self::StreamProvider { .. } => QwenTtsDispatchOutcome::Accepted,
        }
    }
}

/// One native SSE payload and its typed audio projection. `native` preserves
/// request/audio IDs, character usage and provider extensions. Debug redacts
/// audio, signed URLs and the native payload.
#[derive(Clone, PartialEq)]
pub struct QwenTtsStreamEvent {
    pub kind: QwenTtsStreamEventKind,
    pub native: Value,
}

impl fmt::Debug for QwenTtsStreamEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTtsStreamEvent")
            .field("kind", &self.kind)
            .field("native", &"<redacted provider payload>")
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum QwenTtsStreamEventKind {
    /// A PCM16 little-endian, 24 kHz, mono audio segment. Concatenate bytes in
    /// arrival order; the service does not play or convert audio.
    AudioDelta { data: Bytes },
    /// An intermediate payload with no audio bytes and no completion marker.
    Metadata,
    /// A valid native `finish_reason: "stop"` payload, emitted only after clean
    /// HTTP EOF. The native payload retains the full usage object.
    Completed { synthesis: QwenTtsSynthesis },
}

impl fmt::Debug for QwenTtsStreamEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioDelta { data } => f
                .debug_struct("AudioDelta")
                .field("audio_bytes", &data.len())
                .finish(),
            Self::Metadata => f.write_str("Metadata"),
            Self::Completed { synthesis } => f
                .debug_struct("Completed")
                .field("synthesis", synthesis)
                .finish(),
        }
    }
}

/// Native SSE output from one complete-text request. Dropping the stream
/// cancels the HTTP read. Read through `Completed` to confirm success: a final
/// URL alone, an audio delta, or an ordinary early EOF is not completion.
///
/// SSE events are bounded to 8 MiB, the complete wire response to 64 MiB, and
/// parsing proceeds in 64 KiB slices to bound queued events. No complete audio
/// artifact is accumulated. `RequestOptions::total_timeout` covers body reads.
pub struct QwenTtsStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    scope: QwenTtsScope,
    request_id: Option<String>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_bytes: Bytes,
    pending_error: Option<LlmError>,
    pending_completion: Option<QwenTtsStreamEvent>,
    wire_bytes: usize,
    ended: bool,
    done: bool,
}

impl QwenTtsStream {
    fn new(response: StreamResponse, scope: QwenTtsScope) -> Self {
        let request_id = response.header("x-request-id").map(str::to_owned);
        Self {
            body: response.body,
            scope,
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

    pub fn scope(&self) -> &QwenTtsScope {
        &self.scope
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Return an audio/metadata event, or the final result after clean EOF.
    /// Every error terminates the stream and releases the response body.
    pub async fn next_event(&mut self) -> Result<Option<QwenTtsStreamEvent>, QwenTtsError> {
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
                let error = QwenTtsError::StreamInterrupted {
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
                if let Some(completion) = self.pending_completion.take() {
                    self.terminate();
                    return Ok(Some(completion));
                }
                let error = QwenTtsError::StreamInterrupted {
                    source: LlmError::StreamInterrupted {
                        message: "Qwen TTS stream ended before output.finish_reason was stop"
                            .into(),
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
                            message: "Qwen TTS SSE response exceeds the 64 MiB wire limit".into(),
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

    fn decode_event(&mut self, frame: &[u8]) -> Result<Option<QwenTtsStreamEvent>, QwenTtsError> {
        let native: Value = serde_json::from_slice(frame)
            .map_err(|_| self.invalid_event("SSE data is not a JSON object"))?;
        if let Some(id) = native.get("request_id").and_then(Value::as_str) {
            self.request_id = Some(id.to_owned());
        }
        let code = native
            .get("code")
            .and_then(Value::as_str)
            .filter(|code| !code.is_empty())
            .map(str::to_owned);
        if code.is_some()
            || native
                .get("status_code")
                .and_then(Value::as_u64)
                .is_some_and(|status| !(200..300).contains(&status))
        {
            let message = native
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Model Studio reported a synthesis error")
                .to_owned();
            return Err(QwenTtsError::StreamProvider {
                error: Box::new(QwenTtsProviderError {
                    code,
                    message,
                    request_id: self.request_id.clone(),
                    native,
                }),
            });
        }
        if self.pending_completion.is_some() {
            return Err(self.invalid_event("SSE data arrived after the final stop payload"));
        }
        let audio = native
            .pointer("/output/audio")
            .and_then(Value::as_object)
            .ok_or_else(|| self.invalid_event("missing output.audio object in SSE payload"))?;
        let data = audio
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| self.invalid_event("missing output.audio.data string in SSE payload"))?;
        let finish_reason = native.pointer("/output/finish_reason");
        if finish_reason.and_then(Value::as_str) == Some("stop") {
            if !data.is_empty() {
                return Err(self.invalid_event("final stop payload must have empty audio.data"));
            }
            let mut synthesis = decode_synthesis(native.clone(), &self.scope)?;
            if synthesis.request_id.is_none() {
                synthesis.request_id = self.request_id.clone();
            }
            self.pending_completion = Some(QwenTtsStreamEvent {
                kind: QwenTtsStreamEventKind::Completed { synthesis },
                native,
            });
            return Ok(None);
        }
        if finish_reason.is_some_and(|reason| !reason.is_null()) {
            return Err(self.invalid_event("unsupported output.finish_reason in SSE payload"));
        }
        let decoded = STANDARD
            .decode(data)
            .map_err(|_| self.invalid_event("output.audio.data is not valid Base64"))?;
        Ok(Some(QwenTtsStreamEvent {
            kind: if decoded.is_empty() {
                QwenTtsStreamEventKind::Metadata
            } else {
                QwenTtsStreamEventKind::AudioDelta {
                    data: Bytes::from(decoded),
                }
            },
            native,
        }))
    }

    fn invalid_event(&self, message: &str) -> QwenTtsError {
        QwenTtsError::AcceptedInvalidResponse {
            message: message.into(),
            request_id: self.request_id.clone(),
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

/// Qwen3-TTS-Flash's complete-text Model Studio HTTP service.
#[derive(Clone)]
pub struct QwenTtsService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    scope: QwenTtsScope,
}

impl<'a> QwenTtsService<'a> {
    fn pin(&self) -> Result<Self, crate::protocol::LlmError> {
        let mut pinned = self.clone();
        pinned.binding = self
            .binding
            .as_ref()
            .map(crate::providers::binding::ProviderBinding::pinned)
            .transpose()?;
        Ok(pinned)
    }

    pub(crate) fn with_binding(
        mut self,
        binding: &crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding.clone());
        self
    }

    pub fn new(transport: &'a dyn Transport, scope: QwenTtsScope) -> Result<Self, QwenTtsError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            transport,
            scope,
        })
    }

    pub fn scope(&self) -> &QwenTtsScope {
        &self.scope
    }

    /// Synthesize one complete text input and return Model Studio's temporary
    /// result URL. The method sends one request and never downloads audio or
    /// retries an ambiguous result.
    pub async fn synthesize(
        &self,
        request: &QwenTtsRequest,
        options: &RequestOptions,
    ) -> Result<QwenTtsSynthesis, QwenTtsError> {
        let pinned_service = self.pin()?;
        let response = HttpExecutor::new(pinned_service.transport)
            .execute_bounded(
                pinned_service.http_request(request, options, false)?,
                MAX_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| QwenTtsError::OutcomeUnknown {
                source,
                request_id: None,
            })?;

        if !(200..300).contains(&response.status) {
            let (code, message, request_id) = error_details(&response.body);
            let request_id =
                request_id.or_else(|| response.header("x-request-id").map(str::to_owned));
            if (400..500).contains(&response.status) && response.status != 408 {
                return Err(QwenTtsError::Rejected {
                    status: response.status,
                    code,
                    message,
                    request_id,
                });
            }
            return Err(QwenTtsError::OutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "Model Studio returned HTTP {} after TTS submission: {}",
                        response.status, message
                    ),
                },
                request_id,
            });
        }

        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            QwenTtsError::AcceptedInvalidResponse {
                message: format!("response JSON could not be decoded: {error}"),
                request_id: None,
            }
        })?;
        decode_synthesis(value, &pinned_service.scope)
    }

    /// Stream native PCM audio from one complete-text HTTP request. The same
    /// regional route and request body as `synthesize` are used, with
    /// `X-DashScope-SSE: enable`. This is not bidirectional/realtime text input.
    pub async fn synthesize_stream(
        &self,
        request: &QwenTtsRequest,
        options: &RequestOptions,
    ) -> Result<QwenTtsStream, QwenTtsError> {
        let pinned_service = self.pin()?;
        let response = HttpExecutor::new(pinned_service.transport)
            .send(pinned_service.http_request(request, options, true)?)
            .await
            .map_err(|source| QwenTtsError::OutcomeUnknown {
                source,
                request_id: None,
            })?;
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, None)
                .await
                .map_err(|source| QwenTtsError::OutcomeUnknown {
                    source,
                    request_id: None,
                })?;
            let (code, message, request_id) = error_details(&response.body);
            let request_id =
                request_id.or_else(|| response.header("x-request-id").map(str::to_owned));
            if (400..500).contains(&response.status) && response.status != 408 {
                return Err(QwenTtsError::Rejected {
                    status: response.status,
                    code,
                    message,
                    request_id,
                });
            }
            return Err(QwenTtsError::OutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "Model Studio returned HTTP {} after TTS submission: {}",
                        response.status, message
                    ),
                },
                request_id,
            });
        }
        if response.header("content-type").is_some_and(|value| {
            !value
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
        }) {
            return Err(QwenTtsError::AcceptedInvalidResponse {
                message: "Qwen TTS streaming response is not text/event-stream".into(),
                request_id: response.header("x-request-id").map(str::to_owned),
            });
        }
        Ok(QwenTtsStream::new(response, pinned_service.scope.clone()))
    }

    fn http_request(
        &self,
        request: &QwenTtsRequest,
        options: &RequestOptions,
        streaming: bool,
    ) -> Result<HttpRequest, QwenTtsError> {
        self.scope.validate()?;
        request.validate()?;
        if options
            .account_scope
            .as_deref()
            .is_some_and(|value| value != self.scope.account_scope)
        {
            return Err(QwenTtsError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen TTS requires the API key for the selected Model Studio region"
                    .into(),
            })?;

        let mut input = serde_json::Map::new();
        input.insert("text".into(), json!(request.text));
        input.insert("voice".into(), json!(request.voice));
        if let Some(language) = request.language_type {
            input.insert("language_type".into(), json!(language.wire_value()));
        }
        let body = serde_json::to_vec(&json!({
            "model": "qwen3-tts-flash",
            "input": input,
        }))
        .map_err(|error| {
            QwenTtsError::InvalidInput(format!("could not encode synthesis request: {error}"))
        })?;

        let mut headers = vec![
            (
                "authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            ),
            ("content-type".into(), "application/json".into()),
            (
                "accept".into(),
                if streaming {
                    "text/event-stream"
                } else {
                    "application/json"
                }
                .into(),
            ),
        ];
        if streaming {
            headers.push(("X-DashScope-SSE".into(), "enable".into()));
        }
        Ok(HttpRequest {
            method: "POST".into(),
            url: self.scope.region.endpoint().into(),
            headers,
            body: Bytes::from(body),
            timeout: options.total_timeout,
        })
    }
}

fn decode_synthesis(value: Value, scope: &QwenTtsScope) -> Result<QwenTtsSynthesis, QwenTtsError> {
    let request_id = value
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(code) = value
        .get("code")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())
    {
        let message = value
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Model Studio reported a synthesis error")
            .to_owned();
        return Err(QwenTtsError::AcceptedInvalidResponse {
            message: format!("{code}: {message}"),
            request_id,
        });
    }
    let audio = value
        .pointer("/output/audio")
        .and_then(Value::as_object)
        .ok_or_else(|| QwenTtsError::AcceptedInvalidResponse {
            message: "missing output.audio object".into(),
            request_id: request_id.clone(),
        })?;
    let audio_url = audio
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| validate_audio_url(url))
        .ok_or_else(|| QwenTtsError::AcceptedInvalidResponse {
            message: "missing or invalid output.audio.url".into(),
            request_id: request_id.clone(),
        })?
        .to_owned();

    Ok(QwenTtsSynthesis {
        scope: scope.clone(),
        request_id,
        audio_url,
        audio_id: audio.get("id").and_then(Value::as_str).map(str::to_owned),
        expires_at_unix_seconds: audio.get("expires_at").and_then(Value::as_u64),
        finish_reason: value
            .pointer("/output/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        characters: value.pointer("/usage/characters").and_then(Value::as_u64),
    })
}

fn validate_audio_url(value: &str) -> bool {
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
        && url.fragment().is_none()
}

fn error_details(body: &[u8]) -> (Option<String>, String, Option<String>) {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        let code = value
            .get("code")
            .and_then(Value::as_str)
            .filter(|code| !code.is_empty())
            .map(str::to_owned);
        let message = value
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());
        let request_id = value
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        (code, message, request_id)
    } else {
        (None, String::from_utf8_lossy(body).into_owned(), None)
    }
}
