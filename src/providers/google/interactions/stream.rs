//! Native Gemini Interactions SSE events.
use super::*;
use crate::{framing::sse::SseFrameSplitter, transport::MAX_ERROR_BODY_SIZE};
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct InteractionEvent {
    pub event_type: String,
    pub event_id: Option<String>,
    pub native: Value,
    pub reference: Option<InteractionRef>,
}

#[derive(Debug, thiserror::Error)]
pub enum InteractionStreamError {
    #[error("interaction stream interrupted: {source}")]
    Interrupted {
        reference: Option<InteractionRef>,
        #[source]
        source: LlmError,
    },
    #[error("invalid interaction event: {message}")]
    InvalidEvent {
        message: String,
        reference: Option<InteractionRef>,
        native: Value,
    },
}

/// One SSE connection. The most recent event cursor is retained verbatim so a
/// caller can explicitly reopen a stored interaction with `resume_stream`.
pub struct InteractionEventStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_error: Option<LlmError>,
    ended: bool,
    done: bool,
    reference: Option<InteractionRef>,
    interaction_id: Option<String>,
    last_event_id: Option<String>,
    template: InteractionRef,
    stored: bool,
    pub request_id: Option<String>,
}

impl InteractionEventStream {
    pub fn reference(&self) -> Option<&InteractionRef> {
        self.reference.as_ref()
    }

    /// The most recent provider event cursor observed on this stream.
    /// Pass it to [`InteractionService::resume_stream`] after an interruption.
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<InteractionEvent>, InteractionStreamError> {
        if self.done {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                if frame == b"[DONE]" {
                    self.done = true;
                    if self.interaction_id.is_none() {
                        return Err(InteractionStreamError::InvalidEvent {
                            message: "stream completed without interaction id".into(),
                            reference: None,
                            native: Value::Null,
                        });
                    }
                    return Ok(None);
                }
                let native: Value = serde_json::from_slice(&frame).map_err(|_| {
                    self.done = true;
                    InteractionStreamError::InvalidEvent {
                        message: "SSE data is not JSON".into(),
                        reference: self.reference.clone(),
                        native: Value::String(String::from_utf8_lossy(&frame).into_owned()),
                    }
                })?;
                let event_type = native
                    .get("event_type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        self.done = true;
                        InteractionStreamError::InvalidEvent {
                            message: "missing event_type".into(),
                            reference: self.reference.clone(),
                            native: native.clone(),
                        }
                    })?
                    .to_owned();
                if event_type == "error" {
                    self.done = true;
                    return Err(InteractionStreamError::InvalidEvent {
                        message: "provider emitted an error event".into(),
                        reference: self.reference.clone(),
                        native,
                    });
                }
                if let Some(id) = native
                    .get("interaction")
                    .and_then(|v| v.get("id"))
                    .and_then(Value::as_str)
                {
                    if !valid_id(id)
                        || self
                            .interaction_id
                            .as_deref()
                            .is_some_and(|known| known != id)
                    {
                        self.done = true;
                        return Err(InteractionStreamError::InvalidEvent {
                            message: "invalid or changing interaction id".into(),
                            reference: self.reference.clone(),
                            native,
                        });
                    }
                    self.interaction_id = Some(id.into());
                    if self.stored && self.reference.is_none() {
                        let mut reference = self.template.clone();
                        reference.id = id.into();
                        self.reference = Some(reference);
                    }
                }
                let event_id = native
                    .get("event_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let Some(event_id) = &event_id {
                    self.last_event_id = Some(event_id.clone());
                }
                return Ok(Some(InteractionEvent {
                    event_type,
                    event_id,
                    native,
                    reference: self.reference.clone(),
                }));
            }
            if let Some(source) = self.pending_error.take() {
                self.done = true;
                return Err(InteractionStreamError::Interrupted {
                    reference: self.reference.clone(),
                    source,
                });
            }
            if self.ended {
                self.done = true;
                return Err(InteractionStreamError::Interrupted {
                    reference: self.reference.clone(),
                    source: LlmError::StreamInterrupted {
                        message: "interaction stream ended before [DONE]".into(),
                    },
                });
            }
            match self.body.next().await {
                Some(Ok(bytes)) => {
                    let (frames, error) = self.splitter.push_batch(&bytes);
                    self.ready.extend(frames);
                    self.pending_error = error;
                }
                Some(Err(source)) => self.pending_error = Some(source),
                None => {
                    self.ended = true;
                    match self.splitter.finish() {
                        Ok(Some(frame)) => self.ready.push_back(frame),
                        Ok(None) => {}
                        Err(source) => self.pending_error = Some(source),
                    }
                }
            }
        }
    }
}

impl InteractionService<'_> {
    /// Submit `stream=true` and return native step/lifecycle events.
    pub async fn create_stream(
        self,
        request: &InteractionRequest,
        options: &RequestOptions,
    ) -> Result<InteractionEventStream, InteractionError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .create_stream(profile_name, request, options)
            .await
    }

    /// Reopen a stored interaction's event stream after the given event cursor.
    /// The cursor is opaque and should be the most recent `event_id` received
    /// from the previous stream. This performs one GET request; callers decide
    /// whether and when to reconnect again.
    pub async fn resume_stream(
        self,
        reference: &InteractionRef,
        last_event_id: &str,
        options: &RequestOptions,
    ) -> Result<InteractionEventStream, InteractionError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .resume_stream(reference, last_event_id, options)
            .await
    }
}

impl Pinned<'_> {
    async fn create_stream(
        &self,
        profile_name: &str,
        request: &InteractionRequest,
        options: &RequestOptions,
    ) -> Result<InteractionEventStream, InteractionError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        if let Some(previous) = &request.previous {
            check_reference(profile, route, scope, previous)?;
        }
        let body = encode_create_body(request, true)?;
        let template = InteractionRef {
            provider_id: profile.provider_id.as_str().into(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
            account_scope: scope.into(),
            id: String::new(),
        };
        self.open_stream(
            route,
            options,
            "POST",
            route.endpoint.clone(),
            body,
            true,
            request.store,
            template,
            None,
            None,
        )
        .await
    }

    async fn resume_stream(
        &self,
        reference: &InteractionRef,
        last_event_id: &str,
        options: &RequestOptions,
    ) -> Result<InteractionEventStream, InteractionError> {
        if last_event_id.is_empty()
            || last_event_id.len() > 4096
            || last_event_id.chars().any(char::is_control)
        {
            return Err(invalid("invalid interaction event cursor").into());
        }
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        check_reference(profile, route, scope, reference)?;
        let mut url = interaction_url(route, reference)?;
        url.query_pairs_mut()
            .append_pair("stream", "true")
            .append_pair("last_event_id", last_event_id);
        self.open_stream(
            route,
            options,
            "GET",
            url.into(),
            vec![],
            false,
            true,
            reference.clone(),
            Some(reference.clone()),
            Some(last_event_id.to_owned()),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn open_stream(
        &self,
        route: &InteractionRoute,
        options: &RequestOptions,
        method: &str,
        url: String,
        body: Vec<u8>,
        submitting: bool,
        stored: bool,
        template: InteractionRef,
        reference: Option<InteractionRef>,
        initial_event_id: Option<String>,
    ) -> Result<InteractionEventStream, InteractionError> {
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Gemini Interactions requires a credential".into(),
            })?;
        let ServiceAuth::ApiKey { header } = &route.auth else {
            return Err(invalid("Interactions requires API key authentication").into());
        };
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let http = HttpRequest {
            method: method.into(),
            url,
            headers: vec![
                (header.clone(), credential.expose_secret().into()),
                ("content-type".into(), "application/json".into()),
                ("accept".into(), "text/event-stream".into()),
            ],
            body: body.into(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send(http)
            .await
            .map_err(|source| {
                if submitting
                    && matches!(
                        source,
                        LlmError::Transport { .. }
                            | LlmError::TransportTimeout { .. }
                            | LlmError::StreamInterrupted { .. }
                    )
                {
                    InteractionError::SubmitOutcomeUnknown { source }
                } else {
                    InteractionError::Llm(source)
                }
            })?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE)).await?;
            return Err(provider_error(response));
        }
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let interaction_id = reference.as_ref().map(|r| r.id.clone());
        Ok(InteractionEventStream {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_error: None,
            ended: false,
            done: false,
            reference,
            interaction_id,
            last_event_id: initial_event_id,
            template,
            stored,
            request_id,
        })
    }
}
