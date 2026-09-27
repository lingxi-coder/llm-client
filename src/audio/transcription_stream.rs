//! File transcription SSE events. The uploaded recording is still a one-shot stream.
use super::*;
use crate::{framing::sse::SseFrameSplitter, transport::MAX_ERROR_BODY_SIZE};
use std::collections::VecDeque;

/// Native event data is kept intact so callers can use provider extensions.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionEvent {
    pub event_type: String,
    pub native: Value,
    pub terminal: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum TranscriptionStreamError {
    #[error("transcription stream interrupted: {source}")]
    Interrupted {
        #[source]
        source: LlmError,
    },
    #[error("invalid transcription event: {message}")]
    InvalidEvent { message: String, native: Value },
}

/// A completed recording's SSE response. Dropping it cancels the read; it does
/// not resubmit or resume the uploaded file.
pub struct TranscriptionEventStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    ready: VecDeque<Vec<u8>>,
    pending_error: Option<LlmError>,
    ended: bool,
    terminal: bool,
    pub request_id: Option<String>,
}

impl TranscriptionEventStream {
    pub async fn next_event(
        &mut self,
    ) -> Result<Option<TranscriptionEvent>, TranscriptionStreamError> {
        if self.terminal {
            return Ok(None);
        }
        loop {
            if let Some(frame) = self.ready.pop_front() {
                let native: Value = serde_json::from_slice(&frame).map_err(|_| {
                    self.terminal = true;
                    TranscriptionStreamError::InvalidEvent {
                        message: "SSE data is not a JSON event".into(),
                        native: Value::String(String::from_utf8_lossy(&frame).into_owned()),
                    }
                })?;
                let event_type = native
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        self.terminal = true;
                        TranscriptionStreamError::InvalidEvent {
                            message: "missing event type".into(),
                            native: native.clone(),
                        }
                    })?
                    .to_owned();
                let required = match event_type.as_str() {
                    "transcript.text.delta" => "delta",
                    "transcript.text.done" | "transcript.text.segment" => "text",
                    _ => "",
                };
                if !required.is_empty() && !native.get(required).is_some_and(Value::is_string) {
                    self.terminal = true;
                    return Err(TranscriptionStreamError::InvalidEvent {
                        message: format!("{event_type} has no {required}"),
                        native,
                    });
                }
                let terminal = event_type == "transcript.text.done";
                self.terminal = terminal;
                return Ok(Some(TranscriptionEvent {
                    event_type,
                    native,
                    terminal,
                }));
            }
            if let Some(source) = self.pending_error.take() {
                self.terminal = true;
                return Err(TranscriptionStreamError::Interrupted { source });
            }
            if self.ended {
                self.terminal = true;
                return Err(TranscriptionStreamError::Interrupted {
                    source: LlmError::StreamInterrupted {
                        message: "transcription stream ended before transcript.text.done".into(),
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

impl AudioService<'_> {
    /// Stream transcription of a completed file; `whisper-1` is unsupported.
    pub async fn transcribe_stream(
        self,
        profile_name: &str,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<TranscriptionEventStream, AudioError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .transcribe_stream(profile_name, input, request, options)
            .await
    }
}

impl Pinned<'_> {
    async fn transcribe_stream(
        &self,
        profile_name: &str,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<TranscriptionEventStream, AudioError> {
        validate_transcription(request)?;
        validate_input(&input)?;
        if request.model == TranscriptionModel::Whisper1 {
            return Err(invalid("whisper-1 does not support file transcription streaming").into());
        }
        if !matches!(
            request.format,
            AudioTextFormat::Json | AudioTextFormat::DiarizedJson
        ) {
            return Err(
                invalid("file transcription streaming requires a JSON response format").into(),
            );
        }
        let route = self.route(profile_name)?;
        let secret = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "audio route requires a credential".into(),
            })?;
        let fields = transcription_fields(request, true);
        let boundary = multipart_boundary();
        let (prefix, suffix) = multipart_parts(&boundary, &fields, &input);
        let content_length = input
            .size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "audio multipart size overflows".into(),
            })?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let http = HttpStreamRequest {
            method: "POST".into(),
            url: route.transcriptions_endpoint.clone(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", secret.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
                ("accept".into(), "text/event-stream".into()),
            ],
            body: multipart_stream(prefix, input.body, input.size_bytes, suffix),
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .send_stream(http)
            .await
            .map_err(|source| match source {
                LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. } => AudioError::OutcomeUnknown { source },
                other => AudioError::Llm(other),
            })?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_ERROR_BODY_SIZE)).await?;
            let request_id = response
                .header("x-request-id")
                .or_else(|| response.header("request-id"))
                .map(str::to_owned);
            let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(AudioError::Provider {
                status: response.status,
                request_id,
                body,
            });
        }
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        Ok(TranscriptionEventStream {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            ready: VecDeque::new(),
            pending_error: None,
            ended: false,
            terminal: false,
            request_id,
        })
    }
}
