//! MiniMax's independent file transcription service.
//!
//! This module implements the synchronous `asr-1.0` speech-to-text endpoint.
//! It is deliberately separate from chat audio and from the OpenAI audio
//! routes: MiniMax has its own multipart fields, response variants, endpoint
//! hosts, and dispatch uncertainty.

use crate::{
    audio::AudioInput,
    client::RequestOptions,
    files::{append_field, multipart_boundary, sanitize_filename, validate_media_type},
    framing::sse::SseFrameSplitter,
    protocol::LlmError,
    runtime::Deadline,
    transport::{HttpExecutor, HttpStreamRequest, StreamResponse, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::VecDeque,
    pin::Pin,
    str::FromStr,
    task::{Context, Poll},
    time::Duration,
};
use thiserror::Error;
use url::Url;

/// Official international MiniMax ASR endpoint documented by MiniMax.
pub const MINIMAX_ASR_INTERNATIONAL_ENDPOINT: &str = "https://api.minimax.io/v1/speech_to_text";
/// Official mainland China MiniMax ASR endpoint documented by MiniMax.
pub const MINIMAX_ASR_CHINA_ENDPOINT: &str = "https://api.minimax.cn/v1/speech_to_text";

const MAX_INPUT_BYTES: u64 = 50_000_000;
const MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Which response MiniMax should return for an ASR request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTranscriptFormat {
    /// Full transcript and duration as JSON.
    #[default]
    Json,
    /// Transcript, duration, diarized timestamp segments, and speaker count.
    VerboseJson,
    /// SRT subtitles with timestamps and speaker diarization.
    Srt,
    /// WebVTT subtitles with timestamps and speaker diarization.
    Vtt,
}

impl MiniMaxTranscriptFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    fn is_json(self) -> bool {
        matches!(self, Self::Json | Self::VerboseJson)
    }
}

/// Timestamp granularity for verbose JSON and subtitle output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxTimestampLevel {
    /// Sentence/segment timestamps (MiniMax default).
    #[default]
    Sentence,
    /// Word timestamps for English and character timestamps for Chinese.
    Word,
}

impl MiniMaxTimestampLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sentence => "sentence",
            Self::Word => "word",
        }
    }
}

/// A language hint supported by MiniMax ASR 1.0.
///
/// `None` on [`MiniMaxTranscriptionRequest::language`] enables mixed-language
/// recognition. The values here match MiniMax's current published list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxSpeechLanguage {
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "yue")]
    Cantonese,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
    #[serde(rename = "ko")]
    Korean,
    #[serde(rename = "th")]
    Thai,
    #[serde(rename = "vi")]
    Vietnamese,
    #[serde(rename = "id")]
    Indonesian,
    #[serde(rename = "ms")]
    Malay,
    #[serde(rename = "fil")]
    Filipino,
    #[serde(rename = "ar")]
    Arabic,
    #[serde(rename = "tr")]
    Turkish,
    #[serde(rename = "fr")]
    French,
    #[serde(rename = "de")]
    German,
    #[serde(rename = "es")]
    Spanish,
    #[serde(rename = "it")]
    Italian,
    #[serde(rename = "pt")]
    Portuguese,
    #[serde(rename = "pl")]
    Polish,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "uk")]
    Ukrainian,
}

impl MiniMaxSpeechLanguage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chinese => "zh",
            Self::Cantonese => "yue",
            Self::English => "en",
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Thai => "th",
            Self::Vietnamese => "vi",
            Self::Indonesian => "id",
            Self::Malay => "ms",
            Self::Filipino => "fil",
            Self::Arabic => "ar",
            Self::Turkish => "tr",
            Self::French => "fr",
            Self::German => "de",
            Self::Spanish => "es",
            Self::Italian => "it",
            Self::Portuguese => "pt",
            Self::Polish => "pl",
            Self::Russian => "ru",
            Self::Ukrainian => "uk",
        }
    }
}

/// Options for MiniMax's synchronous `asr-1.0` transcription endpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxTranscriptionRequest {
    #[serde(default)]
    pub format: MiniMaxTranscriptFormat,
    #[serde(default)]
    pub timestamp_level: MiniMaxTimestampLevel,
    #[serde(default)]
    pub language: Option<MiniMaxSpeechLanguage>,
}

impl MiniMaxTranscriptionRequest {
    pub fn new() -> Self {
        Self::default()
    }
}

/// One diarized timestamp unit returned by MiniMax `verbose_json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMaxTranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub speaker: Option<String>,
}

/// A transcription result. `text` contains the transcript for JSON formats and
/// the complete subtitle document for SRT/VTT formats.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxTranscription {
    pub text: String,
    pub format: MiniMaxTranscriptFormat,
    pub duration_seconds: Option<f64>,
    pub speaker_count: Option<u32>,
    pub segments: Vec<MiniMaxTranscriptSegment>,
    /// Full native provider JSON for JSON responses; subtitle responses are
    /// plain text and therefore have no native JSON value.
    pub native: Option<Value>,
    /// MiniMax `trace_id`, falling back to an HTTP request ID header.
    pub request_id: Option<String>,
}

/// One incremental transcription event from MiniMax's documented SSE stream.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxTranscriptEvent {
    /// Zero-based sequence number supplied by MiniMax.
    pub index: u64,
    /// Newly recognized text for this event.
    pub delta: String,
    /// True for the terminal event. No further events are read after it.
    pub finish: bool,
    /// Audio duration, present only on the terminal event.
    pub duration_seconds: Option<f64>,
    /// Native JSON payload for fields added by MiniMax.
    pub native: Value,
}

/// Incremental MiniMax ASR result stream. The final event carries the duration.
pub struct MiniMaxTranscriptionStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    splitter: SseFrameSplitter,
    frames: VecDeque<Vec<u8>>,
    next_index: u64,
    request_id: Option<String>,
    terminal_seen: bool,
    eof: bool,
    done: bool,
}

impl MiniMaxTranscriptionStream {
    /// Provider request ID response header, when supplied.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }
}

impl Stream for MiniMaxTranscriptionStream {
    type Item = Result<MiniMaxTranscriptEvent, MiniMaxAudioError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        loop {
            if this.terminal_seen {
                this.done = true;
                return Poll::Ready(None);
            }
            if let Some(frame) = this.frames.pop_front() {
                match decode_stream_event(&frame, this.next_index, this.request_id.clone()) {
                    Ok(event) => {
                        if event.finish {
                            this.terminal_seen = true;
                        } else if let Some(next_index) = this.next_index.checked_add(1) {
                            this.next_index = next_index;
                        } else {
                            this.done = true;
                            return Poll::Ready(Some(Err(invalid_response(
                                "SSE event index overflowed before the terminal event",
                                this.request_id.clone(),
                            ))));
                        }
                        return Poll::Ready(Some(Ok(event)));
                    }
                    Err(error) => {
                        this.done = true;
                        return Poll::Ready(Some(Err(error)));
                    }
                }
            }
            if this.eof {
                this.done = true;
                return Poll::Ready(Some(Err(invalid_response(
                    "SSE stream ended before a terminal finish event",
                    this.request_id.clone(),
                ))));
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => match this.splitter.push(&chunk) {
                    Ok(frames) => this.frames.extend(frames),
                    Err(error) => {
                        this.done = true;
                        return Poll::Ready(Some(Err(map_transport_error(error))));
                    }
                },
                Poll::Ready(Some(Err(error))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(map_transport_error(error))));
                }
                Poll::Ready(None) => {
                    this.eof = true;
                    match this.splitter.finish() {
                        Ok(Some(frame)) => this.frames.push_back(frame),
                        Ok(None) => {}
                        Err(error) => {
                            this.done = true;
                            return Poll::Ready(Some(Err(map_transport_error(error))));
                        }
                    }
                }
            }
        }
    }
}

/// How far an ASR request may have progressed at the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxAudioDispatch {
    /// Local validation or transport capability checks prevented the send.
    NotSent,
    /// MiniMax returned an explicit rejection, normally an HTTP 4xx.
    Rejected,
    /// The connection ended without proof that MiniMax rejected the request.
    Unknown,
    /// MiniMax returned success, but its response could not be decoded.
    Accepted,
}

/// Errors from the MiniMax ASR operation, including billing-aware retry state.
#[derive(Debug, Error)]
pub enum MiniMaxAudioError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("MiniMax ASR returned HTTP {status}: {message}")]
    Provider {
        status: u16,
        code: Option<i64>,
        message: String,
        request_id: Option<String>,
        dispatch: MiniMaxAudioDispatch,
    },
    #[error("MiniMax ASR returned an invalid successful response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
    },
    #[error("MiniMax ASR upload outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

impl MiniMaxAudioError {
    /// Whether a failed transcription can safely be retried without risking a
    /// duplicate provider charge. This service never retries automatically.
    pub fn dispatch(&self) -> MiniMaxAudioDispatch {
        match self {
            Self::Llm(_) => MiniMaxAudioDispatch::NotSent,
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => MiniMaxAudioDispatch::Accepted,
            Self::OutcomeUnknown { .. } => MiniMaxAudioDispatch::Unknown,
        }
    }
}

/// MiniMax's file-based speech-to-text service.
///
/// The endpoint is explicit so callers can select the official international
/// or mainland route for the credential they supply. This service stores only
/// the endpoint and transport: credentials are taken from `RequestOptions` for
/// each call, and ASR requests are not cached or associated with an account
/// scope.
#[derive(Clone)]
pub struct MiniMaxAudioService<'a> {
    transport: &'a dyn Transport,
    endpoint: String,
}

impl<'a> MiniMaxAudioService<'a> {
    /// Construct against one of MiniMax's documented regional ASR endpoints.
    ///
    /// The full endpoint must be HTTPS, use port 443, contain no userinfo,
    /// query, or fragment, and exactly match one of the official ASR routes.
    /// This prevents a credential from being sent to an arbitrary configured
    /// host or a redirect-style URL.
    pub fn new(
        transport: &'a dyn Transport,
        endpoint: impl Into<String>,
    ) -> Result<Self, MiniMaxAudioError> {
        let endpoint = endpoint.into();
        validate_endpoint(&endpoint)?;
        Ok(Self {
            transport,
            endpoint,
        })
    }

    /// Upload and transcribe one completed audio file with MiniMax ASR 1.0.
    ///
    /// The upload stream is consumed once. No retry or regional failover is
    /// attempted. If a transport error occurs after the send begins, the
    /// returned error has [`MiniMaxAudioDispatch::Unknown`] because MiniMax
    /// may already have accepted and billed the request.
    pub async fn transcribe(
        &self,
        input: AudioInput,
        request: &MiniMaxTranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<MiniMaxTranscription, MiniMaxAudioError> {
        let response = self
            .send_transcription(input, request, options, false)
            .await?;
        let response = HttpExecutor::collect_response(response, Some(MAX_RESULT_BYTES))
            .await
            .map_err(map_transport_error)?;
        decode_response(response, request.format)
    }

    /// Upload and transcribe one audio file as incremental JSON SSE events.
    /// This operation always uses `response_format=json`; verbose diarization
    /// and subtitle formats cannot be combined with MiniMax streaming.
    pub async fn transcribe_stream(
        &self,
        input: AudioInput,
        request: &MiniMaxTranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<MiniMaxTranscriptionStream, MiniMaxAudioError> {
        if request.format != MiniMaxTranscriptFormat::Json {
            return Err(invalid(
                "MiniMax ASR streaming supports only response_format=json",
            ));
        }
        let response = self
            .send_transcription(input, request, options, true)
            .await?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_RESULT_BYTES))
                .await
                .map_err(map_transport_error)?;
            return match decode_response(response, MiniMaxTranscriptFormat::Json) {
                Err(error) => Err(error),
                Ok(_) => Err(invalid_response(
                    "provider error response was unexpectedly decoded as a transcript",
                    request_id,
                )),
            };
        }
        let is_event_stream = response
            .header("content-type")
            .and_then(|value| value.split(';').next())
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"));
        if !is_event_stream {
            return Err(invalid_response(
                "successful streaming response did not use text/event-stream",
                request_id,
            ));
        }
        Ok(MiniMaxTranscriptionStream {
            body: response.body,
            splitter: SseFrameSplitter::new(),
            frames: VecDeque::new(),
            next_index: 0,
            request_id,
            terminal_seen: false,
            eof: false,
            done: false,
        })
    }

    async fn send_transcription(
        &self,
        input: AudioInput,
        request: &MiniMaxTranscriptionRequest,
        options: &RequestOptions,
        streaming: bool,
    ) -> Result<StreamResponse, MiniMaxAudioError> {
        validate_input(&input)?;
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "MiniMax ASR requires an API key".into(),
            })?;

        let boundary = multipart_boundary();
        let (prefix, suffix) = multipart_parts(&boundary, &input, request, streaming);
        let content_length = input
            .size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "MiniMax ASR multipart size overflows".into(),
            })?;
        let timeout = options.total_timeout.unwrap_or(DEFAULT_TIMEOUT);
        let deadline = Deadline::after(Some(timeout));
        let body = multipart_stream(prefix, input.body, input.size_bytes, suffix);
        let http = HttpStreamRequest {
            method: "POST".into(),
            url: self.endpoint.clone(),
            headers: {
                let mut headers = vec![
                    (
                        "authorization".into(),
                        format!("Bearer {}", credential.expose_secret()),
                    ),
                    (
                        "content-type".into(),
                        format!("multipart/form-data; boundary={boundary}"),
                    ),
                ];
                if streaming {
                    headers.push(("accept".into(), "text/event-stream".into()));
                }
                if let Some(language) = request.language {
                    headers.push(("language".into(), language.as_str().into()));
                }
                headers
            },
            body,
            content_length,
            timeout: deadline.remaining()?,
        };

        HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .send_stream(http)
            .await
            .map_err(map_transport_error)
    }
}

fn validate_endpoint(endpoint: &str) -> Result<(), MiniMaxAudioError> {
    let url = Url::from_str(endpoint).map_err(|_| invalid("invalid MiniMax ASR endpoint URL"))?;
    let valid_host = matches!(url.host_str(), Some("api.minimax.io" | "api.minimax.cn"));
    if url.scheme() != "https"
        || !valid_host
        || url.port_or_known_default() != Some(443)
        || endpoint.contains('@')
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/v1/speech_to_text"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "MiniMax ASR endpoint must be the official HTTPS regional /v1/speech_to_text route",
        ));
    }
    Ok(())
}

fn validate_input(input: &AudioInput) -> Result<(), MiniMaxAudioError> {
    if input.size_bytes == 0 {
        return Err(invalid("MiniMax ASR audio input is empty"));
    }
    if input.size_bytes > MAX_INPUT_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: "MiniMax ASR audio exceeds the 50 MB upload limit".into(),
        }
        .into());
    }
    validate_media_type(&input.media_type)?;
    let filename = sanitize_filename(&input.filename);
    let extension = filename
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase());
    let supported_extension = matches!(
        extension.as_deref(),
        Some("wav" | "aiff" | "aif" | "flac" | "m4a" | "mp3" | "aac" | "opus" | "ogg")
    );
    if !supported_extension {
        return Err(invalid(
            "MiniMax ASR supports WAV, AIFF, FLAC, M4A/ALAC, MP3, AAC, Opus, and Ogg files",
        ));
    }
    if !input.media_type.starts_with("audio/")
        && !matches!(
            input.media_type.as_str(),
            "application/octet-stream" | "application/ogg"
        )
    {
        return Err(invalid("MiniMax ASR input must have an audio MIME type"));
    }
    Ok(())
}

fn multipart_parts(
    boundary: &str,
    input: &AudioInput,
    request: &MiniMaxTranscriptionRequest,
    streaming: bool,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    append_field(&mut prefix, boundary, "model", "asr-1.0");
    append_field(
        &mut prefix,
        boundary,
        "response_format",
        request.format.as_str(),
    );
    append_field(
        &mut prefix,
        boundary,
        "timestamp_level",
        request.timestamp_level.as_str(),
    );
    append_field(
        &mut prefix,
        boundary,
        "stream",
        if streaming { "true" } else { "false" },
    );
    let filename = sanitize_filename(&input.filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            input.media_type
        )
        .as_bytes(),
    );
    let suffix = Bytes::from(format!("\r\n--{boundary}--\r\n").into_bytes());
    (prefix.freeze(), suffix)
}

#[derive(Clone, Copy)]
enum MultipartPhase {
    Prefix,
    File,
    Done,
}

struct MultipartState {
    prefix: Option<Bytes>,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    suffix: Option<Bytes>,
    phase: MultipartPhase,
    declared_file_bytes: u64,
    actual_file_bytes: u64,
}

fn multipart_stream(
    prefix: Bytes,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    declared_file_bytes: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::unfold(
        MultipartState {
            prefix: Some(prefix),
            file,
            suffix: Some(suffix),
            phase: MultipartPhase::Prefix,
            declared_file_bytes,
            actual_file_bytes: 0,
        },
        |mut state| async move {
            match state.phase {
                MultipartPhase::Prefix => {
                    state.phase = MultipartPhase::File;
                    Some((
                        Ok(state.prefix.take().expect("prefix phase has bytes")),
                        state,
                    ))
                }
                MultipartPhase::File => match state.file.next().await {
                    Some(Ok(chunk)) => {
                        let Some(actual) = state.actual_file_bytes.checked_add(chunk.len() as u64)
                        else {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "MiniMax ASR input byte count overflowed".into(),
                                }),
                                state,
                            ));
                        };
                        if actual > state.declared_file_bytes {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "MiniMax ASR input exceeded its declared length"
                                        .into(),
                                }),
                                state,
                            ));
                        }
                        state.actual_file_bytes = actual;
                        Some((Ok(chunk), state))
                    }
                    Some(Err(_)) => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "MiniMax ASR input stream was interrupted".into(),
                            }),
                            state,
                        ))
                    }
                    None if state.actual_file_bytes != state.declared_file_bytes => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "MiniMax ASR input ended before its declared length"
                                    .into(),
                            }),
                            state,
                        ))
                    }
                    None => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Ok(state.suffix.take().expect("suffix phase has bytes")),
                            state,
                        ))
                    }
                },
                MultipartPhase::Done => None,
            }
        },
    )
    .boxed()
}

fn map_transport_error(source: LlmError) -> MiniMaxAudioError {
    match source {
        LlmError::Transport { .. }
        | LlmError::TransportTimeout { .. }
        | LlmError::StreamInterrupted { .. } => MiniMaxAudioError::OutcomeUnknown { source },
        other => MiniMaxAudioError::Llm(other),
    }
}

#[derive(Debug, Deserialize)]
struct AsrJsonResponse {
    text: String,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    n_speakers: Option<u32>,
    #[serde(default)]
    segments: Vec<MiniMaxTranscriptSegment>,
    #[serde(default)]
    trace_id: Option<String>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
}

#[derive(Debug, Deserialize)]
struct BaseResponse {
    status_code: i64,
    #[serde(default)]
    status_msg: Option<String>,
}

fn decode_response(
    response: crate::transport::HttpResponse,
    format: MiniMaxTranscriptFormat,
) -> Result<MiniMaxTranscription, MiniMaxAudioError> {
    let header_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    if !(200..300).contains(&response.status) {
        let (code, message, trace_id) = decode_provider_error(&response.body);
        let request_id = trace_id.or(header_id);
        return Err(MiniMaxAudioError::Provider {
            status: response.status,
            code,
            message,
            request_id,
            dispatch: if (400..500).contains(&response.status) {
                MiniMaxAudioDispatch::Rejected
            } else {
                MiniMaxAudioDispatch::Unknown
            },
        });
    }

    if format.is_json() {
        let decoded: AsrJsonResponse = serde_json::from_slice(&response.body).map_err(|_| {
            MiniMaxAudioError::InvalidResponse {
                message: "expected a JSON transcription response".into(),
                request_id: header_id.clone(),
            }
        })?;
        let request_id = decoded.trace_id.clone().or(header_id);
        if let Some(base_resp) = decoded.base_resp {
            if base_resp.status_code != 0 {
                return Err(MiniMaxAudioError::Provider {
                    status: response.status,
                    code: Some(base_resp.status_code),
                    message: base_resp
                        .status_msg
                        .unwrap_or_else(|| "MiniMax reported a request error".into()),
                    request_id,
                    dispatch: MiniMaxAudioDispatch::Rejected,
                });
            }
        }
        if decoded
            .duration
            .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
        {
            return Err(MiniMaxAudioError::InvalidResponse {
                message: "transcription duration is not a valid non-negative number".into(),
                request_id,
            });
        }
        let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
            MiniMaxAudioError::InvalidResponse {
                message: "expected a JSON transcription response".into(),
                request_id: None,
            }
        })?;
        return Ok(MiniMaxTranscription {
            text: decoded.text,
            format,
            duration_seconds: decoded.duration,
            speaker_count: decoded.n_speakers,
            segments: decoded.segments,
            native: Some(native),
            request_id,
        });
    }

    let text = String::from_utf8(response.body.to_vec()).map_err(|_| {
        MiniMaxAudioError::InvalidResponse {
            message: "subtitle response is not valid UTF-8".into(),
            request_id: header_id.clone(),
        }
    })?;
    Ok(MiniMaxTranscription {
        text,
        format,
        duration_seconds: None,
        speaker_count: None,
        segments: vec![],
        native: None,
        request_id: header_id,
    })
}

fn decode_stream_event(
    frame: &[u8],
    expected_index: u64,
    request_id: Option<String>,
) -> Result<MiniMaxTranscriptEvent, MiniMaxAudioError> {
    let native: Value = serde_json::from_slice(frame)
        .map_err(|_| invalid_response("SSE data is not valid JSON", request_id.clone()))?;
    let object = native
        .as_object()
        .ok_or_else(|| invalid_response("SSE data must be a JSON object", request_id.clone()))?;
    let index = object.get("index").and_then(Value::as_u64).ok_or_else(|| {
        invalid_response(
            "SSE event index must be an unsigned integer",
            request_id.clone(),
        )
    })?;
    if index != expected_index {
        return Err(invalid_response(
            "SSE event indexes must start at zero and advance by one",
            request_id,
        ));
    }
    let delta = object
        .get("delta")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_response("SSE event delta must be a string", request_id.clone()))?
        .to_owned();
    let finish = object
        .get("finish")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            invalid_response("SSE event finish must be a boolean", request_id.clone())
        })?;
    let duration = object.get("duration");
    let duration_seconds = match (finish, duration) {
        (true, Some(value)) => {
            let seconds = value.as_f64().ok_or_else(|| {
                invalid_response(
                    "terminal SSE event duration must be numeric",
                    request_id.clone(),
                )
            })?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(invalid_response(
                    "terminal SSE event duration must be a finite non-negative number",
                    request_id,
                ));
            }
            Some(seconds)
        }
        (true, None) => {
            return Err(invalid_response(
                "terminal SSE event is missing duration",
                request_id,
            ));
        }
        (false, Some(_)) => {
            return Err(invalid_response(
                "duration is only valid on the terminal SSE event",
                request_id,
            ));
        }
        (false, None) => None,
    };
    Ok(MiniMaxTranscriptEvent {
        index,
        delta,
        finish,
        duration_seconds,
        native,
    })
}

fn decode_provider_error(body: &[u8]) -> (Option<i64>, String, Option<String>) {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return (
            None,
            "provider returned a non-JSON error response".into(),
            None,
        );
    };
    let base = value.get("base_resp");
    let code = base
        .and_then(|value| value.get("status_code"))
        .and_then(Value::as_i64)
        .or_else(|| {
            value
                .get("error")
                .and_then(|value| value.get("code"))
                .and_then(Value::as_i64)
        });
    let message = base
        .and_then(|value| value.get("status_msg"))
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .get("error")
                .and_then(|value| value.get("message"))
                .and_then(Value::as_str)
        })
        .unwrap_or("provider rejected the transcription request")
        .chars()
        .take(512)
        .collect();
    let trace_id = value
        .get("trace_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    (code, message, trace_id)
}

fn invalid(message: impl Into<String>) -> MiniMaxAudioError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
    .into()
}

fn invalid_response(message: impl Into<String>, request_id: Option<String>) -> MiniMaxAudioError {
    MiniMaxAudioError::InvalidResponse {
        message: message.into(),
        request_id,
    }
}
