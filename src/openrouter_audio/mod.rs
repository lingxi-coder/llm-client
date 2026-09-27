//! OpenRouter's independent speech-to-text and text-to-speech endpoints.
//!
//! These routes use OpenRouter's own request and response contracts. STT can
//! send either base64 JSON or multipart form data; TTS returns raw audio bytes.

use crate::{
    audio::AudioInput,
    client::RequestOptions,
    files::{
        append_field, multipart_boundary, multipart_upload_stream, sanitize_filename,
        validate_media_type,
    },
    protocol::LlmError,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpStreamRequest, Transport},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt, time::Duration};
use thiserror::Error;

/// OpenRouter's speech-to-text route.
pub const OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT: &str =
    "https://openrouter.ai/api/v1/audio/transcriptions";
/// OpenRouter's text-to-speech route.
pub const OPENROUTER_AUDIO_SPEECH_ENDPOINT: &str = "https://openrouter.ai/api/v1/audio/speech";

const MAX_MULTIPART_INPUT_BYTES: u64 = 25_000_000;
const MAX_JSON_INPUT_BYTES: u64 = 50_000_000;
const MAX_SPEECH_INPUT_BYTES: usize = 1_000_000;
const MAX_SPEECH_REFERENCE_BASE64_BYTES: usize = 20 * 1024 * 1024;
const MAX_SPEECH_REFERENCE_AUDIO_BYTES: usize = 15 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Selects the wire representation used by OpenRouter's transcription route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterTranscriptionEncoding {
    /// Sends `model` and `input_audio` in a JSON object with base64 audio.
    Base64Json,
    /// Sends an OpenAI-style multipart form containing the file and model.
    #[default]
    Multipart,
}

/// Audio formats accepted by the OpenRouter STT JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterInputAudioFormat {
    /// Waveform Audio File Format.
    Wav,
    /// MPEG Layer III audio.
    Mp3,
    /// Free Lossless Audio Codec.
    Flac,
    /// MPEG-4 audio container.
    M4a,
    /// Ogg audio container.
    Ogg,
    /// WebM audio container.
    Webm,
    /// Advanced Audio Coding.
    Aac,
}

impl OpenRouterInputAudioFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::M4a => "m4a",
            Self::Ogg => "ogg",
            Self::Webm => "webm",
            Self::Aac => "aac",
        }
    }

    fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Flac => "audio/flac",
            Self::M4a => "audio/mp4",
            Self::Ogg => "audio/ogg",
            Self::Webm => "audio/webm",
            Self::Aac => "audio/aac",
        }
    }
}

/// Response detail returned by a transcription request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterTranscriptionResponseFormat {
    /// Return transcript text and usage.
    #[default]
    Json,
    /// Also request provider-supported language, duration, and timestamps.
    VerboseJson,
}

impl OpenRouterTranscriptionResponseFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
        }
    }
}

/// Timestamp granularity requested with `verbose_json` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterTimestampGranularity {
    /// Request timestamps for transcript segments.
    Segment,
    /// Request timestamps for individual words when supported.
    Word,
}

impl OpenRouterTimestampGranularity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Segment => "segment",
            Self::Word => "word",
        }
    }
}

/// Options for one OpenRouter transcription request.
#[derive(Debug, Clone)]
pub struct OpenRouterTranscriptionRequest {
    /// OpenRouter model slug, for example `openai/whisper-1`.
    pub model: String,
    /// Selects multipart upload or base64 JSON.
    pub encoding: OpenRouterTranscriptionEncoding,
    /// Declares the input file format for the base64 JSON representation.
    pub input_format: OpenRouterInputAudioFormat,
    /// Optional ISO-639-1 language hint.
    pub language: Option<String>,
    /// Optional sampling temperature in the documented 0.0–1.0 range.
    pub temperature: Option<f64>,
    /// Requested STT response shape.
    pub response_format: OpenRouterTranscriptionResponseFormat,
    /// Timestamp detail requested with `verbose_json`.
    pub timestamp_granularities: Vec<OpenRouterTimestampGranularity>,
    /// Provider-keyed options passed through OpenRouter's `provider` field.
    pub provider: Option<Value>,
}

impl OpenRouterTranscriptionRequest {
    /// Create a transcription request for the given model and audio format.
    pub fn new(model: impl Into<String>, input_format: OpenRouterInputAudioFormat) -> Self {
        Self {
            model: model.into(),
            encoding: OpenRouterTranscriptionEncoding::Multipart,
            input_format,
            language: None,
            temperature: None,
            response_format: OpenRouterTranscriptionResponseFormat::Json,
            timestamp_granularities: Vec::new(),
            provider: None,
        }
    }
}

/// Normalized OpenRouter transcription usage.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterAudioUsage {
    /// Input duration in seconds when supplied by the provider.
    pub seconds: Option<f64>,
    /// Combined input and output tokens when supplied by the provider.
    pub total_tokens: Option<u64>,
    /// Input tokens when supplied by the provider.
    pub input_tokens: Option<u64>,
    /// Output tokens when supplied by the provider.
    pub output_tokens: Option<u64>,
    /// Actual cost in USD when supplied by OpenRouter.
    pub cost: Option<f64>,
}

/// Result of an OpenRouter transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouterTranscription {
    /// Recognized text.
    pub text: String,
    /// Usage details returned by OpenRouter, if present.
    pub usage: Option<OpenRouterAudioUsage>,
    /// OpenRouter generation ID from `X-Generation-Id`.
    pub generation_id: Option<String>,
    /// Caller-provided stable identity of the account used for this request.
    pub account_scope: Option<String>,
    /// Full response JSON, including model-specific verbose fields.
    pub native: Value,
}

/// Output encoding accepted by OpenRouter's TTS route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterSpeechFormat {
    /// Compressed MPEG audio.
    Mp3,
    /// Raw PCM audio.
    #[default]
    Pcm,
}

impl OpenRouterSpeechFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Pcm => "pcm",
        }
    }

    fn media_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Pcm => "audio/pcm",
        }
    }
}

/// Options for one OpenRouter speech synthesis request.
#[derive(Debug, Clone)]
pub struct OpenRouterSpeechRequest {
    /// OpenRouter TTS model slug.
    pub model: String,
    /// Text to synthesize.
    pub input: String,
    /// Voice identifier supported by the selected model, if required.
    pub voice: Option<String>,
    /// Requested raw audio encoding.
    pub response_format: OpenRouterSpeechFormat,
    /// Optional playback speed. Support is model-dependent.
    pub speed: Option<f64>,
    /// Provider-keyed options passed through OpenRouter's `provider` field.
    pub provider: Option<Value>,
    /// Optional Base64 reference audio and transcript for stateless voice cloning.
    pub input_references: Option<OpenRouterSpeechInputReferences>,
}

/// Reference audio for stateless voice cloning. The service emits one
/// `input_audio` part and, if supplied, one following `text` transcript part.
#[derive(Clone, PartialEq, Eq)]
pub struct OpenRouterSpeechInputReferences {
    /// Audio container format. Model/provider support remains provider-owned.
    pub audio_format: OpenRouterInputAudioFormat,
    /// Standard Base64 encoded audio bytes, without a `data:` URI prefix.
    pub audio_base64: String,
    /// Optional transcript corresponding to the reference audio.
    pub transcript: Option<String>,
}

impl fmt::Debug for OpenRouterSpeechInputReferences {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenRouterSpeechInputReferences")
            .field("audio_format", &self.audio_format)
            .field("audio_base64", &"<redacted>")
            .field("audio_base64_bytes", &self.audio_base64.len())
            .field(
                "transcript",
                &self.transcript.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl OpenRouterSpeechInputReferences {
    pub fn new(audio_format: OpenRouterInputAudioFormat, audio_base64: impl Into<String>) -> Self {
        Self {
            audio_format,
            audio_base64: audio_base64.into(),
            transcript: None,
        }
    }

    pub fn with_transcript(mut self, transcript: impl Into<String>) -> Self {
        self.transcript = Some(transcript.into());
        self
    }
}

impl OpenRouterSpeechRequest {
    /// Create a speech request with the required text and model.
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            voice: None,
            response_format: OpenRouterSpeechFormat::Pcm,
            speed: None,
            provider: None,
            input_references: None,
        }
    }
}

/// Raw audio returned by OpenRouter's TTS route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenRouterSpeechOutput {
    /// Audio bytes exactly as returned by OpenRouter.
    pub bytes: Bytes,
    /// The provider's `Content-Type`, or the requested format's known type.
    pub content_type: String,
    /// Requested wire format.
    pub format: OpenRouterSpeechFormat,
    /// OpenRouter generation ID from `X-Generation-Id`.
    pub generation_id: Option<String>,
    /// Caller-provided stable identity of the account used for this request.
    pub account_scope: Option<String>,
}

/// How far an audio request may have progressed at OpenRouter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRouterAudioDispatch {
    /// Local validation or a missing transport capability prevented sending.
    NotSent,
    /// OpenRouter returned an explicit client-side rejection.
    Rejected,
    /// The transport ended without proof that the provider rejected the call.
    Unknown,
    /// OpenRouter returned success, but its response could not be decoded.
    Accepted,
}

/// Errors returned by OpenRouter STT and TTS operations.
#[derive(Debug, Error)]
pub enum OpenRouterAudioError {
    /// A local or transport error described by the shared protocol layer.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// OpenRouter returned a non-success status.
    #[error("OpenRouter audio endpoint returned HTTP {status}: {message}")]
    Provider {
        /// HTTP status code.
        status: u16,
        /// Provider's best available error message.
        message: String,
        /// OpenRouter generation ID, if supplied.
        generation_id: Option<String>,
        /// Whether the request was rejected or may have been processed.
        dispatch: OpenRouterAudioDispatch,
        /// Parsed provider error JSON, when available.
        body: Option<Value>,
    },
    /// A successful response did not match the documented response shape.
    #[error("OpenRouter audio endpoint returned an invalid response: {message}")]
    InvalidResponse {
        /// Description of the malformed response.
        message: String,
        /// OpenRouter generation ID, if supplied.
        generation_id: Option<String>,
    },
    /// The connection failed after a request may have reached OpenRouter.
    #[error("OpenRouter audio request outcome is unknown: {source}")]
    OutcomeUnknown {
        /// Underlying transport failure.
        #[source]
        source: LlmError,
    },
}

impl OpenRouterAudioError {
    /// Return the provider dispatch state without retrying automatically.
    pub fn dispatch(&self) -> OpenRouterAudioDispatch {
        match self {
            Self::Llm(_) => OpenRouterAudioDispatch::NotSent,
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => OpenRouterAudioDispatch::Accepted,
            Self::OutcomeUnknown { .. } => OpenRouterAudioDispatch::Unknown,
        }
    }
}

/// Stateless OpenRouter STT and TTS client.
///
/// This service is pinned to OpenRouter's two documented HTTPS endpoints. It
/// retains only the transport; every request supplies its own bearer key and
/// account scope through [`RequestOptions`]. The scope is returned with results
/// and is never sent to OpenRouter.
#[derive(Clone, Copy)]
pub struct OpenRouterAudioService<'a> {
    transport: &'a dyn Transport,
}

impl<'a> OpenRouterAudioService<'a> {
    /// Construct a service using the documented OpenRouter audio endpoints.
    pub fn new(transport: &'a dyn Transport) -> Self {
        Self { transport }
    }

    /// Transcribe one audio input through OpenRouter's STT endpoint.
    pub async fn transcribe(
        &self,
        input: AudioInput,
        request: &OpenRouterTranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<OpenRouterTranscription, OpenRouterAudioError> {
        validate_transcription(input.size_bytes, request)?;
        if request.encoding == OpenRouterTranscriptionEncoding::Multipart
            && request.provider.is_some()
        {
            return Err(invalid(
                "OpenRouter provider options require base64 JSON transcription encoding",
            ));
        }
        if request.encoding == OpenRouterTranscriptionEncoding::Multipart {
            validate_multipart_input(&input.filename, &input.media_type, input.size_bytes)?;
        }
        let credential = credential(options)?;
        let account_scope = options.account_scope.clone();
        let deadline = Deadline::after(Some(options.total_timeout.unwrap_or(DEFAULT_TIMEOUT)));

        let response = match request.encoding {
            OpenRouterTranscriptionEncoding::Base64Json => {
                let audio = collect_input(input, MAX_JSON_INPUT_BYTES, deadline).await?;
                let body = build_json_transcription_body(&audio.bytes, request)?;
                HttpExecutor::new(self.transport)
                    .with_deadline(deadline)
                    .execute_bounded(
                        HttpRequest {
                            method: "POST".into(),
                            url: OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT.into(),
                            headers: vec![
                                ("authorization".into(), format!("Bearer {credential}")),
                                ("content-type".into(), "application/json".into()),
                            ],
                            body,
                            timeout: deadline.remaining()?,
                        },
                        MAX_RESPONSE_BYTES,
                    )
                    .await
                    .map_err(map_transport_error)?
            }
            OpenRouterTranscriptionEncoding::Multipart => {
                let boundary = multipart_boundary();
                let (prefix, suffix) = multipart_transcription_parts(
                    &input.filename,
                    &input.media_type,
                    &boundary,
                    request,
                );
                let content_length = exact_multipart_length(input.size_bytes, &prefix, &suffix)?;
                let input_body = mark_audio_source_errors_unknown(input.body);
                let body = multipart_upload_stream(prefix, input_body, input.size_bytes, suffix);
                HttpExecutor::new(self.transport)
                    .with_deadline(deadline)
                    .execute_stream_bounded(
                        HttpStreamRequest {
                            method: "POST".into(),
                            url: OPENROUTER_AUDIO_TRANSCRIPTIONS_ENDPOINT.into(),
                            headers: vec![
                                ("authorization".into(), format!("Bearer {credential}")),
                                (
                                    "content-type".into(),
                                    format!("multipart/form-data; boundary={boundary}"),
                                ),
                            ],
                            body,
                            content_length,
                            timeout: deadline.remaining()?,
                        },
                        MAX_RESPONSE_BYTES,
                    )
                    .await
                    .map_err(map_transport_error)?
            }
        };

        let generation_id = response.header("x-generation-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(provider_error(
                response.status,
                &response.body,
                generation_id,
            ));
        }
        let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
            OpenRouterAudioError::InvalidResponse {
                message: "expected a JSON transcription response".into(),
                generation_id: generation_id.clone(),
            }
        })?;
        let text = native
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| OpenRouterAudioError::InvalidResponse {
                message: "transcription response is missing a string `text` field".into(),
                generation_id: generation_id.clone(),
            })?
            .to_owned();
        let usage = native
            .get("usage")
            .filter(|value| !value.is_null())
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| OpenRouterAudioError::InvalidResponse {
                message: "transcription response contains invalid usage fields".into(),
                generation_id: generation_id.clone(),
            })?;
        validate_usage(usage.as_ref()).map_err(|message| {
            OpenRouterAudioError::InvalidResponse {
                message,
                generation_id: generation_id.clone(),
            }
        })?;
        Ok(OpenRouterTranscription {
            text,
            usage,
            generation_id,
            account_scope,
            native,
        })
    }

    /// Synthesize text and return OpenRouter's raw audio response bytes.
    pub async fn speak(
        &self,
        request: &OpenRouterSpeechRequest,
        options: &RequestOptions,
    ) -> Result<OpenRouterSpeechOutput, OpenRouterAudioError> {
        validate_speech(request)?;
        let credential = credential(options)?;
        let account_scope = options.account_scope.clone();
        let deadline = Deadline::after(Some(options.total_timeout.unwrap_or(DEFAULT_TIMEOUT)));
        let body =
            serde_json::to_vec(&SpeechWireRequest::from_request(request)).map_err(|error| {
                LlmError::InvalidRequest {
                    message: format!("could not encode OpenRouter speech request: {error}"),
                }
            })?;
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_bounded(
                HttpRequest {
                    method: "POST".into(),
                    url: OPENROUTER_AUDIO_SPEECH_ENDPOINT.into(),
                    headers: vec![
                        ("authorization".into(), format!("Bearer {credential}")),
                        ("content-type".into(), "application/json".into()),
                    ],
                    body: Bytes::from(body),
                    timeout: deadline.remaining()?,
                },
                MAX_RESPONSE_BYTES,
            )
            .await
            .map_err(map_transport_error)?;
        let generation_id = response.header("x-generation-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(provider_error(
                response.status,
                &response.body,
                generation_id,
            ));
        }
        if response.body.is_empty() {
            return Err(OpenRouterAudioError::InvalidResponse {
                message: "successful speech response contained no audio bytes".into(),
                generation_id,
            });
        }
        let content_type = match response.header("content-type") {
            Some(value) if value.starts_with("audio/") => value.to_owned(),
            Some(_) => {
                return Err(OpenRouterAudioError::InvalidResponse {
                    message: "successful speech response did not contain an audio content type"
                        .into(),
                    generation_id,
                });
            }
            None => request.response_format.media_type().into(),
        };
        Ok(OpenRouterSpeechOutput {
            bytes: response.body,
            content_type,
            format: request.response_format,
            generation_id,
            account_scope,
        })
    }
}

#[derive(Serialize)]
struct JsonTranscriptionWireRequest<'a> {
    model: &'a str,
    input_audio: InputAudioWire<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    response_format: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    timestamp_granularities: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: &'a Option<Value>,
}

#[derive(Serialize)]
struct InputAudioWire<'a> {
    data: String,
    format: &'a str,
}

fn build_json_transcription_body(
    audio: &[u8],
    request: &OpenRouterTranscriptionRequest,
) -> Result<Bytes, OpenRouterAudioError> {
    let wire = JsonTranscriptionWireRequest {
        model: &request.model,
        input_audio: InputAudioWire {
            data: BASE64.encode(audio),
            format: request.input_format.as_str(),
        },
        language: &request.language,
        temperature: request.temperature,
        response_format: request.response_format.as_str(),
        timestamp_granularities: request
            .timestamp_granularities
            .iter()
            .map(|value| value.as_str())
            .collect(),
        provider: &request.provider,
    };
    serde_json::to_vec(&wire).map(Bytes::from).map_err(|error| {
        OpenRouterAudioError::Llm(LlmError::InvalidRequest {
            message: format!("could not encode OpenRouter transcription request: {error}"),
        })
    })
}

fn multipart_transcription_parts(
    filename: &str,
    media_type: &str,
    boundary: &str,
    request: &OpenRouterTranscriptionRequest,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    append_field(&mut prefix, boundary, "model", &request.model);
    if let Some(language) = request.language.as_deref() {
        append_field(&mut prefix, boundary, "language", language);
    }
    if let Some(temperature) = request.temperature {
        append_field(
            &mut prefix,
            boundary,
            "temperature",
            &temperature.to_string(),
        );
    }
    append_field(
        &mut prefix,
        boundary,
        "response_format",
        request.response_format.as_str(),
    );
    for granularity in &request.timestamp_granularities {
        append_field(
            &mut prefix,
            boundary,
            "timestamp_granularities[]",
            granularity.as_str(),
        );
    }
    let filename = sanitize_filename(filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    (
        prefix.freeze(),
        Bytes::from(format!("\r\n--{boundary}--\r\n")),
    )
}

fn exact_multipart_length(
    file_size: u64,
    prefix: &Bytes,
    suffix: &Bytes,
) -> Result<u64, OpenRouterAudioError> {
    let prefix_len = u64::try_from(prefix.len()).map_err(|_| LlmError::RequestTooLarge {
        message: "multipart prefix length overflows".into(),
    })?;
    let suffix_len = u64::try_from(suffix.len()).map_err(|_| LlmError::RequestTooLarge {
        message: "multipart suffix length overflows".into(),
    })?;
    file_size
        .checked_add(prefix_len)
        .and_then(|length| length.checked_add(suffix_len))
        .ok_or_else(|| {
            LlmError::RequestTooLarge {
                message: "multipart request size overflows".into(),
            }
            .into()
        })
}

fn mark_audio_source_errors_unknown(
    input: futures::stream::BoxStream<'static, Result<Bytes, LlmError>>,
) -> futures::stream::BoxStream<'static, Result<Bytes, LlmError>> {
    input
        .map(|chunk| {
            chunk.map_err(|source| LlmError::StreamInterrupted {
                message: format!("OpenRouter multipart STT audio stream failed: {source}"),
            })
        })
        .boxed()
}

#[derive(Serialize)]
struct SpeechWireRequest<'a> {
    model: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: &'a Option<String>,
    response_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: &'a Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_references: Option<Vec<SpeechInputReferenceWire>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SpeechInputReferenceWire {
    InputAudio { input_audio: SpeechInputAudioWire },
    Text { text: String },
}

#[derive(Serialize)]
struct SpeechInputAudioWire {
    data: String,
}

impl<'a> SpeechWireRequest<'a> {
    fn from_request(request: &'a OpenRouterSpeechRequest) -> Self {
        let input_references = request.input_references.as_ref().map(|references| {
            let mut parts = vec![SpeechInputReferenceWire::InputAudio {
                input_audio: SpeechInputAudioWire {
                    data: format!(
                        "data:{};base64,{}",
                        references.audio_format.mime_type(),
                        references.audio_base64
                    ),
                },
            }];
            if let Some(transcript) = &references.transcript {
                parts.push(SpeechInputReferenceWire::Text {
                    text: transcript.clone(),
                });
            }
            parts
        });
        Self {
            model: &request.model,
            input: &request.input,
            voice: &request.voice,
            response_format: request.response_format.as_str(),
            speed: request.speed,
            provider: &request.provider,
            input_references,
        }
    }
}

async fn collect_input(
    mut input: AudioInput,
    limit: u64,
    deadline: Deadline,
) -> Result<BufferedAudio, OpenRouterAudioError> {
    validate_media_type(&input.media_type)?;
    if !input.media_type.starts_with("audio/") && input.media_type != "application/ogg" {
        return Err(invalid("OpenRouter STT input must have an audio MIME type"));
    }
    if sanitize_filename(&input.filename).is_empty() {
        return Err(invalid("OpenRouter STT input filename must not be empty"));
    }
    if input.size_bytes == 0 {
        return Err(invalid("audio input must not be empty"));
    }
    if input.size_bytes > limit {
        return Err(LlmError::RequestTooLarge {
            message: format!("audio input exceeds the {limit}-byte endpoint limit"),
        }
        .into());
    }
    let mut bytes = BytesMut::with_capacity(input.size_bytes.min(usize::MAX as u64) as usize);
    while let Some(chunk) = deadline
        .run(input.body.next())
        .await
        .map_err(OpenRouterAudioError::Llm)?
    {
        let chunk = chunk.map_err(OpenRouterAudioError::Llm)?;
        if (bytes.len() as u64).saturating_add(chunk.len() as u64) > limit
            || (bytes.len() as u64).saturating_add(chunk.len() as u64) > input.size_bytes
        {
            return Err(invalid("audio input exceeded its declared size"));
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() as u64 != input.size_bytes {
        return Err(invalid("audio input ended before its declared size"));
    }
    Ok(BufferedAudio {
        bytes: bytes.freeze(),
    })
}

struct BufferedAudio {
    bytes: Bytes,
}

fn validate_multipart_input(
    filename: &str,
    media_type: &str,
    size_bytes: u64,
) -> Result<(), OpenRouterAudioError> {
    validate_media_type(media_type)?;
    if !media_type.starts_with("audio/") && media_type != "application/ogg" {
        return Err(invalid("OpenRouter STT input must have an audio MIME type"));
    }
    if sanitize_filename(filename).is_empty() {
        return Err(invalid("OpenRouter STT input filename must not be empty"));
    }
    if size_bytes == 0 {
        return Err(invalid("audio input must not be empty"));
    }
    if size_bytes > MAX_MULTIPART_INPUT_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: format!(
                "audio input exceeds the {MAX_MULTIPART_INPUT_BYTES}-byte endpoint limit"
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_transcription(
    input_bytes: u64,
    request: &OpenRouterTranscriptionRequest,
) -> Result<(), OpenRouterAudioError> {
    if request.model.trim().is_empty() {
        return Err(invalid("OpenRouter STT model slug must not be empty"));
    }
    if request
        .language
        .as_deref()
        .is_some_and(|language| language.trim().is_empty())
    {
        return Err(invalid("OpenRouter STT language hint must not be empty"));
    }
    if request
        .temperature
        .is_some_and(|temperature| !temperature.is_finite() || !(0.0..=1.0).contains(&temperature))
    {
        return Err(invalid(
            "OpenRouter STT temperature must be between 0 and 1",
        ));
    }
    if request.response_format == OpenRouterTranscriptionResponseFormat::Json
        && !request.timestamp_granularities.is_empty()
    {
        return Err(invalid(
            "timestamp granularities require verbose_json transcription output",
        ));
    }
    let limit = match request.encoding {
        OpenRouterTranscriptionEncoding::Base64Json => MAX_JSON_INPUT_BYTES,
        OpenRouterTranscriptionEncoding::Multipart => MAX_MULTIPART_INPUT_BYTES,
    };
    if input_bytes == 0 {
        return Err(invalid("audio input must not be empty"));
    }
    if input_bytes > limit {
        return Err(LlmError::RequestTooLarge {
            message: format!("audio input exceeds the {limit}-byte endpoint limit"),
        }
        .into());
    }
    Ok(())
}

fn validate_speech(request: &OpenRouterSpeechRequest) -> Result<(), OpenRouterAudioError> {
    if request.model.trim().is_empty() {
        return Err(invalid("OpenRouter TTS model slug must not be empty"));
    }
    if request.input.trim().is_empty() {
        return Err(invalid("OpenRouter TTS input text must not be empty"));
    }
    if request.input.len() > MAX_SPEECH_INPUT_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: format!("OpenRouter TTS input exceeds {MAX_SPEECH_INPUT_BYTES} bytes"),
        }
        .into());
    }
    if request
        .voice
        .as_deref()
        .is_some_and(|voice| voice.trim().is_empty())
    {
        return Err(invalid(
            "OpenRouter TTS voice must not be empty when supplied",
        ));
    }
    if request
        .speed
        .is_some_and(|speed| !speed.is_finite() || speed <= 0.0)
    {
        return Err(invalid(
            "OpenRouter TTS speed must be a positive finite number",
        ));
    }
    if let Some(references) = &request.input_references {
        if references.audio_base64.is_empty() {
            return Err(invalid("OpenRouter TTS reference audio must not be empty"));
        }
        if references.audio_base64.len() > MAX_SPEECH_REFERENCE_BASE64_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: format!(
                    "OpenRouter TTS reference audio exceeds {MAX_SPEECH_REFERENCE_BASE64_BYTES} Base64 bytes"
                ),
            }
            .into());
        }
        let decoded = BASE64
            .decode(&references.audio_base64)
            .map_err(|_| invalid("OpenRouter TTS reference audio must be standard Base64"))?;
        if decoded.is_empty() {
            return Err(invalid("OpenRouter TTS reference audio must not be empty"));
        }
        if decoded.len() > MAX_SPEECH_REFERENCE_AUDIO_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: format!(
                    "OpenRouter TTS reference audio exceeds {MAX_SPEECH_REFERENCE_AUDIO_BYTES} decoded bytes"
                ),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_usage(usage: Option<&OpenRouterAudioUsage>) -> Result<(), String> {
    let Some(usage) = usage else {
        return Ok(());
    };
    if usage
        .seconds
        .is_some_and(|value| !value.is_finite() || value < 0.0)
        || usage
            .cost
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err("transcription usage has a negative or non-finite value".into());
    }
    Ok(())
}

fn credential(options: &RequestOptions) -> Result<&str, OpenRouterAudioError> {
    let credential = options
        .credential
        .as_ref()
        .map(|value| value.expose_secret().as_str())
        .ok_or_else(|| {
            OpenRouterAudioError::Llm(LlmError::Authentication {
                message: "OpenRouter audio request requires an API key".into(),
            })
        })?;
    if credential.trim().is_empty() {
        return Err(LlmError::Authentication {
            message: "OpenRouter audio request requires a non-empty API key".into(),
        }
        .into());
    }
    Ok(credential)
}

fn provider_error(status: u16, body: &[u8], generation_id: Option<String>) -> OpenRouterAudioError {
    let parsed = serde_json::from_slice::<Value>(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
        })
        .map(str::to_owned)
        .unwrap_or_else(|| "OpenRouter returned an audio request error".into());
    OpenRouterAudioError::Provider {
        status,
        message,
        generation_id,
        dispatch: if (400..500).contains(&status) {
            OpenRouterAudioDispatch::Rejected
        } else {
            OpenRouterAudioDispatch::Unknown
        },
        body: parsed,
    }
}

fn map_transport_error(source: LlmError) -> OpenRouterAudioError {
    match source {
        LlmError::UnsupportedCapability { .. }
        | LlmError::InvalidRequest { .. }
        | LlmError::RequestTooLarge { .. } => OpenRouterAudioError::Llm(source),
        source => OpenRouterAudioError::OutcomeUnknown { source },
    }
}

fn invalid(message: &str) -> OpenRouterAudioError {
    OpenRouterAudioError::Llm(LlmError::InvalidRequest {
        message: message.into(),
    })
}
