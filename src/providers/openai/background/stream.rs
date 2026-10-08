//! Native SSE events and resumable sequence cursors for background Responses.
use super::*;
use crate::{
    codecs::{CodecContext, RequestMode, StreamDecoder, WireCodec},
    framing::sse::SseFrameSplitter,
    protocol::{ChatResponse, ContinuationRef, StreamEvent},
    transport::{HttpResponse, StreamResponse, MAX_ERROR_BODY_SIZE},
};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackgroundEventCursor {
    pub reference: BackgroundJobRef,
    /// Last fully delivered provider event. Resume starts strictly after it.
    pub sequence_number: u64,
}

#[derive(Debug)]
pub struct BackgroundEvent {
    pub native: Value,
    pub cursor: BackgroundEventCursor,
    pub terminal: bool,
}

/// A native background event decoded into the crate's provider-neutral stream
/// events. Terminal response events also carry the complete decoded response.
#[derive(Debug)]
pub struct BackgroundChatEvent {
    /// The provider event is retained verbatim for replay and diagnostics.
    pub native: Value,
    pub cursor: BackgroundEventCursor,
    pub events: Vec<StreamEvent>,
    /// Set for completed and incomplete response events.
    pub response: Option<ChatResponse>,
    pub terminal: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BackgroundStreamError {
    #[error("background event stream interrupted: {source}")]
    Interrupted {
        cursor: Option<Box<BackgroundEventCursor>>,
        #[source]
        source: Box<LlmError>,
    },
    #[error("invalid background event: {message}")]
    InvalidEvent {
        message: String,
        cursor: Option<Box<BackgroundEventCursor>>,
        native: Box<Value>,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum BackgroundChatStreamError {
    #[error(transparent)]
    Native(#[from] BackgroundStreamError),
    #[error("could not decode background event: {source}")]
    Decode {
        cursor: BackgroundEventCursor,
        native: Box<Value>,
        events: Vec<StreamEvent>,
        #[source]
        source: Box<LlmError>,
    },
}

/// Provider-neutral view over the resumable native background event stream.
/// Previous events remain caller-owned so an interruption cannot silently
/// become a successful partial response.
pub struct BackgroundChatEventStream {
    native: BackgroundEventStream,
    codec: Arc<dyn WireCodec>,
    context: CodecContext,
    decoder: Box<dyn StreamDecoder>,
    account_scope: String,
    finished: bool,
    replay_until: Option<BackgroundEventCursor>,
}

impl BackgroundChatEventStream {
    fn new(
        native: BackgroundEventStream,
        codec: Arc<dyn WireCodec>,
        context: CodecContext,
        account_scope: String,
    ) -> Self {
        let decoder = codec.stream_decoder(&context);
        Self {
            native,
            codec,
            context,
            decoder,
            account_scope,
            finished: false,
            replay_until: None,
        }
    }

    /// Return one native event with its decoded stream events and cursor.
    /// On an interruption, already returned deltas remain valid partial output;
    /// the error retains the last safe cursor for a later resume.
    pub async fn next_event(
        &mut self,
    ) -> Result<Option<BackgroundChatEvent>, BackgroundChatStreamError> {
        loop {
            let event = match self.next_decoded_event().await {
                Ok(event) => event,
                Err(mut error) => {
                    // A failed state rebuild must not roll the caller's durable
                    // checkpoint backwards to an already delivered event.
                    if let Some(checkpoint) = &self.replay_until {
                        match &mut error {
                            BackgroundChatStreamError::Native(
                                BackgroundStreamError::Interrupted { cursor, .. }
                                | BackgroundStreamError::InvalidEvent { cursor, .. },
                            ) => *cursor = Some(Box::new(checkpoint.clone())),
                            BackgroundChatStreamError::Decode { cursor, events, .. } => {
                                *cursor = checkpoint.clone();
                                events.clear();
                            }
                        }
                    }
                    return Err(error);
                }
            };
            let Some(event) = event else {
                return Ok(None);
            };
            if self.replay_until.as_ref().is_some_and(|checkpoint| {
                event.cursor.sequence_number <= checkpoint.sequence_number
            }) {
                continue;
            }
            self.replay_until = None;
            return Ok(Some(event));
        }
    }

    async fn next_decoded_event(
        &mut self,
    ) -> Result<Option<BackgroundChatEvent>, BackgroundChatStreamError> {
        if self.finished {
            return Ok(None);
        }
        let native_event = match self.native.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                self.finished = true;
                return Ok(None);
            }
            Err(error) => {
                self.finished = true;
                return Err(error.into());
            }
        };
        let frame = format!("data: {}\n\n", native_event.native);
        let mut decoded = Vec::new();
        for result in self.decoder.push_bytes(frame.as_bytes()) {
            match result {
                Ok(event) => decoded.push(event),
                Err(source) => {
                    self.finished = true;
                    return Err(BackgroundChatStreamError::Decode {
                        cursor: native_event.cursor,
                        native: Box::new(native_event.native),
                        events: decoded,
                        source: Box::new(source),
                    });
                }
            }
        }

        let kind = native_event.native.get("type").and_then(Value::as_str);
        let response = if matches!(kind, Some("response.completed" | "response.incomplete")) {
            match self.decode_response(&native_event.native) {
                Ok(response) => Some(response),
                Err(source) => {
                    self.finished = true;
                    return Err(BackgroundChatStreamError::Decode {
                        cursor: native_event.cursor,
                        native: Box::new(native_event.native),
                        events: decoded,
                        source: Box::new(source),
                    });
                }
            }
        } else {
            None
        };
        self.finished = native_event.terminal;
        Ok(Some(BackgroundChatEvent {
            native: native_event.native,
            cursor: native_event.cursor,
            events: decoded,
            response,
            terminal: native_event.terminal,
        }))
    }

    fn decode_response(&self, event: &Value) -> Result<ChatResponse, LlmError> {
        let response = event
            .get("response")
            .ok_or_else(|| LlmError::ProviderInternal {
                message: "terminal background event has no response object".into(),
            })?;
        let http = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: serde_json::to_vec(response)
                .map_err(|error| LlmError::ProviderInternal {
                    message: format!("terminal background response cannot be serialized: {error}"),
                })?
                .into(),
        };
        let mut decoded = self.codec.decode_response(&http, &self.context)?;
        let profile = self.context.profile();
        decoded.executed_profile = Some(profile.profile_name.clone());
        if event["type"] == "response.completed"
            && profile
                .extra
                .get("supports_previous_response_id")
                .and_then(Value::as_bool)
                == Some(true)
        {
            if let Some(id) = decoded.response_id.clone() {
                decoded.continuation = Some(ContinuationRef::scoped(
                    id,
                    profile,
                    self.context.request_model(),
                    &self.account_scope,
                    None,
                ));
            }
        }
        Ok(decoded)
    }
}

/// One HTTP event stream. Dropping it closes the read; the provider job keeps running.
pub struct BackgroundEventStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    cursor: Option<BackgroundEventCursor>,
    reference_template: Option<BackgroundJobRef>,
    pending_error: Option<LlmError>,
    terminal: bool,
}

impl BackgroundEventStream {
    fn new(
        response: StreamResponse,
        cursor: Option<BackgroundEventCursor>,
        reference_template: Option<BackgroundJobRef>,
    ) -> Self {
        Self {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            cursor,
            reference_template,
            pending_error: None,
            terminal: false,
        }
    }

    /// Return the next complete native event and its durable resume cursor.
    /// An interruption retains the last delivered cursor when one exists.
    pub async fn next_event(&mut self) -> Result<Option<BackgroundEvent>, BackgroundStreamError> {
        if self.terminal {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                let event = self.decode(frame);
                if event.is_err() {
                    self.terminal = true;
                }
                return event.map(Some);
            }
            if let Some(source) = self.pending_error.take() {
                self.terminal = true;
                return Err(BackgroundStreamError::Interrupted {
                    cursor: self.cursor.clone().map(Box::new),
                    source: Box::new(source),
                });
            }
            match self.body.next().await {
                Some(Ok(bytes)) => {
                    let (frames, error) = self.splitter.push_batch(&bytes);
                    self.ready.extend(frames);
                    self.pending_error = error;
                }
                Some(Err(source)) => {
                    self.terminal = true;
                    return Err(BackgroundStreamError::Interrupted {
                        cursor: self.cursor.clone().map(Box::new),
                        source: Box::new(source),
                    });
                }
                None => {
                    if let Some(frame) = self.splitter.finish().map_err(|source| {
                        BackgroundStreamError::Interrupted {
                            cursor: self.cursor.clone().map(Box::new),
                            source: Box::new(source),
                        }
                    })? {
                        self.ready.push_back(frame);
                    } else {
                        self.terminal = true;
                        return Err(BackgroundStreamError::Interrupted {
                            cursor: self.cursor.clone().map(Box::new),
                            source: Box::new(LlmError::StreamInterrupted {
                                message: "background stream ended before a terminal event".into(),
                            }),
                        });
                    }
                }
            }
        }
    }

    fn decode(&mut self, frame: Vec<u8>) -> Result<BackgroundEvent, BackgroundStreamError> {
        let native: Value =
            serde_json::from_slice(&frame).map_err(|_| BackgroundStreamError::InvalidEvent {
                message: "SSE data is not a JSON event".into(),
                cursor: self.cursor.clone().map(Box::new),
                native: Box::new(Value::String(String::from_utf8_lossy(&frame).into_owned())),
            })?;
        let sequence = native
            .get("sequence_number")
            .and_then(Value::as_u64)
            .ok_or_else(|| self.invalid("missing sequence_number", native.clone()))?;
        if self
            .cursor
            .as_ref()
            .is_some_and(|cursor| sequence <= cursor.sequence_number)
        {
            return Err(self.invalid("non-increasing sequence_number", native));
        }
        let response_id = native
            .get("response")
            .and_then(|response| response.get("id"))
            .and_then(Value::as_str);
        let cursor = match (&self.cursor, response_id) {
            (Some(previous), Some(id)) if id != previous.reference.response_id => {
                return Err(self.invalid("response id differs from stream reference", native));
            }
            (Some(previous), _) => BackgroundEventCursor {
                reference: previous.reference.clone(),
                sequence_number: sequence,
            },
            (None, Some(id)) if valid_id(id) => {
                let mut reference = self.reference_template.take().ok_or_else(|| {
                    self.invalid("stream has no bound job reference", native.clone())
                })?;
                if !reference.response_id.is_empty() && reference.response_id != id {
                    return Err(self.invalid("response id differs from stream reference", native));
                }
                reference.response_id = id.into();
                BackgroundEventCursor {
                    reference,
                    sequence_number: sequence,
                }
            }
            (None, _) => return Err(self.invalid("stream event has no response id", native)),
        };
        let terminal = matches!(
            native.get("type").and_then(Value::as_str),
            Some(
                "response.completed"
                    | "response.incomplete"
                    | "response.failed"
                    | "response.cancelled"
            )
        );
        self.cursor = Some(cursor.clone());
        self.terminal = terminal;
        Ok(BackgroundEvent {
            native,
            cursor,
            terminal,
        })
    }

    fn invalid(&self, message: &str, native: Value) -> BackgroundStreamError {
        BackgroundStreamError::InvalidEvent {
            message: message.into(),
            cursor: self.cursor.clone().map(Box::new),
            native: Box::new(native),
        }
    }
}

impl BackgroundService<'_> {
    /// Submit with `background=true, stream=true`; the first event supplies the job id.
    pub async fn submit_stream(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundEventStream, BackgroundError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .submit_stream(profile_name, request, options)
            .await
    }

    /// Reconnect after the last delivered event, without replaying submission.
    pub async fn resume_stream(
        self,
        cursor: BackgroundEventCursor,
        options: &RequestOptions,
    ) -> Result<BackgroundEventStream, BackgroundError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .resume_stream(cursor, options)
            .await
    }

    /// Submit a background response and decode each native event into
    /// provider-neutral stream events. Terminal events also carry a decoded
    /// `ChatResponse`; interrupted streams can resume from their last cursor.
    pub async fn submit_chat_stream(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundChatEventStream, BackgroundError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .submit_chat_stream(profile_name, request, options)
            .await
    }

    /// Rebuild decoder state by replaying the existing response, then deliver
    /// only events after the cursor. This never resubmits the response.
    pub async fn resume_chat_stream(
        self,
        cursor: BackgroundEventCursor,
        options: &RequestOptions,
    ) -> Result<BackgroundChatEventStream, BackgroundError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .resume_chat_stream(cursor, options)
            .await
    }
}

impl Pinned<'_> {
    async fn submit_chat_stream(
        &self,
        profile_name: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundChatEventStream, BackgroundError> {
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown background profile"))?;
        let matches: Vec<_> = profile
            .models
            .iter()
            .filter(|model| {
                model.display_model == request.model
                    || model.request_model == request.model
                    || model.aliases.iter().any(|alias| alias == &request.model)
            })
            .collect();
        let [model] = matches.as_slice() else {
            return Err(invalid("background model is missing or ambiguous on this profile").into());
        };
        let account_scope = options
            .account_scope
            .as_deref()
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| invalid("background request requires a non-secret account_scope"))?
            .to_owned();
        let codec = self.codec_arc()?;
        let context = CodecContext::for_model(profile, model, RequestMode::Stream)
            .with_file_scope(options.file_account_scope.as_deref());
        let native = self.submit_stream(profile_name, request, options).await?;
        Ok(BackgroundChatEventStream::new(
            native,
            codec,
            context,
            account_scope,
        ))
    }

    async fn resume_chat_stream(
        &self,
        cursor: BackgroundEventCursor,
        options: &RequestOptions,
    ) -> Result<BackgroundChatEventStream, BackgroundError> {
        let profile = self
            .client
            .native_profile(&cursor.reference.profile_name)
            .ok_or_else(|| invalid("unknown background profile"))?;
        let codec = self.codec_arc()?;
        let context = CodecContext::new(profile, &cursor.reference.model, RequestMode::Stream);
        let account_scope = cursor.reference.account_scope.clone();
        let native = self
            .resume_stream_inner(cursor.clone(), options, true)
            .await?;
        let mut stream = BackgroundChatEventStream::new(native, codec, context, account_scope);
        stream.replay_until = Some(cursor);
        Ok(stream)
    }

    async fn submit_stream(
        &self,
        profile_name: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundEventStream, BackgroundError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        let matches: Vec<_> = profile
            .models
            .iter()
            .filter(|model| {
                model.display_model == request.model
                    || model.request_model == request.model
                    || model.aliases.iter().any(|alias| alias == &request.model)
            })
            .collect();
        let [model] = matches.as_slice() else {
            return Err(invalid("background model is missing or ambiguous on this profile").into());
        };
        if let Some(reference) = &request.continuation {
            reference.validate(profile, &model.request_model, Some(scope), None)?;
        }
        let codec = self.codec()?;
        let context = CodecContext::for_model(profile, model, RequestMode::Stream)
            .with_file_scope(options.file_account_scope.as_deref());
        crate::codecs::structured::validate(request, &context)?;
        codec.validate_request(request, &context)?;
        let mut http = codec.encode_request(EncodeRequest::new(request), &context)?;
        if http.url != route.endpoint {
            return Err(
                invalid("background route differs from the profile's Responses endpoint").into(),
            );
        }
        let mut body: Value = serde_json::from_slice(&http.body)
            .map_err(|_| invalid("background stream body is not JSON"))?;
        if body.get("stream") != Some(&Value::Bool(true)) {
            return Err(invalid("background stream requires stream=true").into());
        }
        if body
            .get("background")
            .is_some_and(|value| value != &Value::Bool(true))
        {
            return Err(invalid("background request conflicts with background=true").into());
        }
        body["background"] = Value::Bool(true);
        http.body = serde_json::to_vec(&body)
            .map_err(|_| invalid("background stream body cannot be serialized"))?
            .into();
        if http.body.len() > MAX_BODY {
            return Err(invalid("background stream body exceeds 64 MiB").into());
        }
        apply_auth(route, options.credential.as_ref(), &mut http)?;
        let deadline = Deadline::after(options.total_timeout);
        http.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send(http)
            .await
            .map_err(|source| match source {
                LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. } => {
                    BackgroundError::SubmitOutcomeUnknown { source }
                }
                other => BackgroundError::Llm(other),
            })?;
        let template = BackgroundJobRef {
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
            account_scope: scope.into(),
            model: model.request_model.clone(),
            response_id: String::new(),
        };
        open_stream(response, None, Some(template)).await
    }

    fn codec_arc(&self) -> Result<Arc<dyn WireCodec>, LlmError> {
        self.client
            .runtime
            .codecs
            .get(&crate::protocol::ProtocolFamily::OpenAiResponses)
            .cloned()
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: "OpenAI Responses codec is unavailable".into(),
            })
    }

    async fn resume_stream(
        &self,
        cursor: BackgroundEventCursor,
        options: &RequestOptions,
    ) -> Result<BackgroundEventStream, BackgroundError> {
        self.resume_stream_inner(cursor, options, false).await
    }

    async fn resume_stream_inner(
        &self,
        cursor: BackgroundEventCursor,
        options: &RequestOptions,
        replay: bool,
    ) -> Result<BackgroundEventStream, BackgroundError> {
        let reference = &cursor.reference;
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.account_scope != scope
            || reference.model.trim().is_empty()
            || !valid_id(&reference.response_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "background cursor belongs to another provider, profile, endpoint, model or account".into(),
            }
            .into());
        }
        let mut url =
            url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid background endpoint"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid background endpoint"))?
            .push(&reference.response_id);
        url.query_pairs_mut().append_pair("stream", "true");
        if !replay {
            url.query_pairs_mut()
                .append_pair("starting_after", &cursor.sequence_number.to_string());
        }
        let mut http = HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url: url.into(),
            headers: vec![("accept".into(), "text/event-stream".into())],
            body: Vec::new().into(),
            timeout: None,
        };
        apply_auth(route, options.credential.as_ref(), &mut http)?;
        let deadline = Deadline::after(options.total_timeout);
        http.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send(http)
            .await?;
        if replay {
            open_stream(response, None, Some(cursor.reference)).await
        } else {
            open_stream(response, Some(cursor), None).await
        }
    }
}

async fn open_stream(
    response: StreamResponse,
    cursor: Option<BackgroundEventCursor>,
    template: Option<BackgroundJobRef>,
) -> Result<BackgroundEventStream, BackgroundError> {
    if !(200..300).contains(&response.status) {
        let response = HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE)).await?;
        let native = parse_json(&response)?;
        return Err(provider_error(&response, native));
    }
    Ok(BackgroundEventStream::new(response, cursor, template))
}
