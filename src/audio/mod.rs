//! Independent OpenAI file transcription and translation routes.
mod speech;
mod transcription_stream;
mod voices;
use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    files::{append_field, multipart_boundary, sanitize_filename, validate_media_type},
    protocol::{LlmError, ProviderProfile, ServiceAuth, ServiceSetting},
    runtime::Deadline,
    transport::{HttpExecutor, HttpResponse, HttpStreamRequest},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::{Bytes, BytesMut};
use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use speech::{
    SpeechEvent, SpeechEventStream, SpeechEventStreamError, SpeechFormat, SpeechModel,
    SpeechRequest, SpeechStream, SpeechStreamError, SpeechVoice,
};
use std::time::Duration;
pub use transcription_stream::{
    TranscriptionEvent, TranscriptionEventStream, TranscriptionStreamError,
};
pub use voices::{
    CustomVoice, CustomVoiceCreateRequest, CustomVoiceRef, VoiceConsent, VoiceConsentCreateRequest,
    VoiceConsentDeleteReceipt, VoiceConsentListQuery, VoiceConsentPage, VoiceConsentRef,
    VoiceConsentUpdateRequest, VoiceResourceError, VoiceResourceService,
};

const MAX_INPUT: u64 = 25_000_000;
const MAX_RESULT: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioRoute {
    pub transcriptions_endpoint: String,
    pub translations_endpoint: String,
    #[serde(default)]
    pub speech_endpoint: Option<String>,
    pub auth: ServiceAuth,
}

pub struct AudioInput {
    pub filename: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub body: BoxStream<'static, Result<Bytes, LlmError>>,
}
impl AudioInput {
    pub fn from_bytes(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        bytes: impl Into<Bytes>,
    ) -> Self {
        let bytes = bytes.into();
        Self {
            filename: filename.into(),
            media_type: media_type.into(),
            size_bytes: bytes.len() as u64,
            body: stream::once(async move { Ok(bytes) }).boxed(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionModel {
    GptTranscribe,
    Gpt4oTranscribe,
    Gpt4oMiniTranscribe,
    Gpt4oMiniTranscribe2025_12_15,
    Gpt4oTranscribeDiarize,
    Whisper1,
}
impl TranscriptionModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GptTranscribe => "gpt-transcribe",
            Self::Gpt4oTranscribe => "gpt-4o-transcribe",
            Self::Gpt4oMiniTranscribe => "gpt-4o-mini-transcribe",
            Self::Gpt4oMiniTranscribe2025_12_15 => "gpt-4o-mini-transcribe-2025-12-15",
            Self::Gpt4oTranscribeDiarize => "gpt-4o-transcribe-diarize",
            Self::Whisper1 => "whisper-1",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTextFormat {
    #[default]
    Json,
    Text,
    Srt,
    Vtt,
    VerboseJson,
    DiarizedJson,
}
impl AudioTextFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::VerboseJson => "verbose_json",
            Self::DiarizedJson => "diarized_json",
        }
    }
    fn is_json(self) -> bool {
        matches!(self, Self::Json | Self::VerboseJson | Self::DiarizedJson)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampGranularity {
    Word,
    Segment,
}
impl TimestampGranularity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Segment => "segment",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionRequest {
    pub model: TranscriptionModel,
    #[serde(default)]
    pub format: AudioTextFormat,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub timestamp_granularities: Vec<TimestampGranularity>,
    #[serde(default)]
    pub chunking_auto: bool,
    /// Known-speaker references for `gpt-4o-transcribe-diarize`.
    #[serde(default)]
    pub known_speakers: Vec<KnownSpeakerReference>,
}
impl TranscriptionRequest {
    pub fn new(model: TranscriptionModel) -> Self {
        Self {
            model,
            format: AudioTextFormat::Json,
            language: None,
            prompt: None,
            temperature: None,
            timestamp_granularities: vec![],
            chunking_auto: false,
            known_speakers: vec![],
        }
    }

    pub fn with_known_speaker(mut self, reference: KnownSpeakerReference) -> Self {
        self.known_speakers.push(reference);
        self
    }
}

/// A short audio sample used to associate diarized segments with a known speaker.
/// The duration is caller-supplied because the client does not decode audio files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnownSpeakerReference {
    pub name: String,
    pub filename: String,
    pub media_type: String,
    pub duration_seconds: f64,
    pub audio_bytes: Vec<u8>,
}

impl KnownSpeakerReference {
    pub fn new(
        name: impl Into<String>,
        filename: impl Into<String>,
        media_type: impl Into<String>,
        duration_seconds: f64,
        audio_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            filename: filename.into(),
            media_type: media_type.into(),
            duration_seconds,
            audio_bytes: audio_bytes.into(),
        }
    }

    fn data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.media_type,
            BASE64.encode(&self.audio_bytes)
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationRequest {
    #[serde(default)]
    pub format: AudioTextFormat,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioTextResult {
    pub text: String,
    pub language: Option<String>,
    pub languages: Vec<String>,
    pub duration_seconds: Option<f64>,
    pub words: Vec<TimedWord>,
    pub segments: Vec<TimedSegment>,
    /// Full provider JSON when the requested format was JSON.
    pub native: Option<Value>,
    pub request_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("audio provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid audio response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("audio submission outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

#[derive(Clone, Copy)]
pub struct AudioService<'a> {
    source: ClientSource<'a>,
}
impl<'a> AudioService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    /// Access the documented OpenAI custom voice and consent resource APIs.
    pub fn voices(self) -> VoiceResourceService<'a> {
        VoiceResourceService::new(self.source)
    }

    pub async fn transcribe(
        self,
        profile_name: &str,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<AudioTextResult, AudioError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .transcribe(profile_name, input, request, options)
            .await
    }
    pub async fn translate(
        self,
        profile_name: &str,
        input: AudioInput,
        request: &TranslationRequest,
        options: &RequestOptions,
    ) -> Result<AudioTextResult, AudioError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .translate(profile_name, input, request, options)
            .await
    }
}

struct Pinned<'a> {
    client: &'a ClientSnapshot,
}
impl Pinned<'_> {
    fn route(&self, profile_name: &str) -> Result<&AudioRoute, AudioError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown audio profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("audio profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.audio else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no audio route".into(),
            }
            .into());
        };
        Ok(route)
    }

    async fn transcribe(
        &self,
        profile_name: &str,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<AudioTextResult, AudioError> {
        validate_transcription(request)?;
        let route = self.route(profile_name)?;
        let fields = transcription_fields(request, false);
        self.execute(
            &route.transcriptions_endpoint,
            input,
            fields,
            request.format,
            options,
        )
        .await
    }

    async fn translate(
        &self,
        profile_name: &str,
        input: AudioInput,
        request: &TranslationRequest,
        options: &RequestOptions,
    ) -> Result<AudioTextResult, AudioError> {
        validate_translation(request)?;
        let route = self.route(profile_name)?;
        let mut fields = vec![
            ("model".into(), "whisper-1".into()),
            ("response_format".into(), request.format.as_str().into()),
        ];
        if let Some(prompt) = &request.prompt {
            fields.push(("prompt".into(), prompt.clone()));
        }
        if let Some(temperature) = request.temperature {
            fields.push(("temperature".into(), temperature.to_string()));
        }
        self.execute(
            &route.translations_endpoint,
            input,
            fields,
            request.format,
            options,
        )
        .await
    }

    async fn execute(
        &self,
        endpoint: &str,
        input: AudioInput,
        fields: Vec<(String, String)>,
        format: AudioTextFormat,
        options: &RequestOptions,
    ) -> Result<AudioTextResult, AudioError> {
        validate_input(&input)?;
        let secret = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "audio route requires a credential".into(),
            })?;
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
            url: endpoint.into(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", secret.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: multipart_stream(prefix, input.body, input.size_bytes, suffix),
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_stream_bounded(http, MAX_RESULT)
            .await
            .map_err(|source| match source {
                LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. } => AudioError::OutcomeUnknown { source },
                other => AudioError::Llm(other),
            })?;
        decode_result(&response, format)
    }
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn validate_input(input: &AudioInput) -> Result<(), LlmError> {
    validate_audio_file(&input.filename, &input.media_type, input.size_bytes)
}

fn validate_audio_file(filename: &str, media_type: &str, size_bytes: u64) -> Result<(), LlmError> {
    if size_bytes == 0 || size_bytes > MAX_INPUT {
        return Err(LlmError::RequestTooLarge {
            message: "audio file must contain 1–25000000 bytes".into(),
        });
    }
    validate_media_type(media_type)?;
    let ext = filename
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase());
    if !matches!(
        ext.as_deref(),
        Some("flac" | "mp3" | "mp4" | "mpeg" | "mpga" | "m4a" | "ogg" | "wav" | "webm")
    ) {
        return Err(invalid("audio filename needs a supported extension"));
    }
    if !media_type.starts_with("audio/") && !matches!(media_type, "video/mp4" | "video/webm") {
        return Err(invalid(
            "audio file needs an audio or supported video media type",
        ));
    }
    Ok(())
}
fn validate_temperature(value: Option<f64>) -> Result<(), LlmError> {
    if value.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err(invalid("audio temperature must be between 0 and 1"));
    }
    Ok(())
}
fn validate_prompt(prompt: Option<&str>) -> Result<(), LlmError> {
    if prompt.is_some_and(|p| p.trim().is_empty() || p.len() > 10_000) {
        return Err(invalid("audio prompt must contain 1–10000 bytes"));
    }
    Ok(())
}
fn validate_transcription(request: &TranscriptionRequest) -> Result<(), LlmError> {
    validate_temperature(request.temperature)?;
    validate_prompt(request.prompt.as_deref())?;
    if request
        .language
        .as_ref()
        .is_some_and(|l| l.len() != 2 || !l.bytes().all(|b| b.is_ascii_alphabetic()))
    {
        return Err(invalid("transcription language must be an ISO-639-1 code"));
    }
    match request.model {
        TranscriptionModel::Whisper1 => {
            if matches!(request.format, AudioTextFormat::DiarizedJson) || request.chunking_auto {
                return Err(invalid(
                    "Whisper does not support diarization or chunking_strategy",
                ));
            }
            if !request.timestamp_granularities.is_empty()
                && request.format != AudioTextFormat::VerboseJson
            {
                return Err(invalid("Whisper timestamps require verbose_json"));
            }
        }
        TranscriptionModel::Gpt4oTranscribeDiarize => {
            if !matches!(
                request.format,
                AudioTextFormat::Json | AudioTextFormat::Text | AudioTextFormat::DiarizedJson
            ) || request.prompt.is_some()
                || !request.timestamp_granularities.is_empty()
            {
                return Err(invalid(
                    "diarization supports only json, text or diarized_json without prompt or timestamp_granularities",
                ));
            }
        }
        _ => {
            if request.format != AudioTextFormat::Json
                || request.chunking_auto
                || !request.timestamp_granularities.is_empty()
            {
                return Err(invalid(
                    "this transcription model supports only json without timestamp or chunking options",
                ));
            }
        }
    }
    validate_known_speakers(request)?;
    if request.timestamp_granularities.len() > 2
        || (request.timestamp_granularities.len() == 2
            && request.timestamp_granularities[0] == request.timestamp_granularities[1])
    {
        return Err(invalid("timestamp granularities must be unique"));
    }
    Ok(())
}

fn validate_known_speakers(request: &TranscriptionRequest) -> Result<(), LlmError> {
    if request.known_speakers.is_empty() {
        return Ok(());
    }
    if request.model != TranscriptionModel::Gpt4oTranscribeDiarize
        || request.format != AudioTextFormat::DiarizedJson
    {
        return Err(invalid(
            "known speakers require gpt-4o-transcribe-diarize with diarized_json",
        ));
    }
    if request.known_speakers.len() > 4 {
        return Err(invalid("at most four known speakers are supported"));
    }
    let mut names = std::collections::BTreeSet::new();
    for speaker in &request.known_speakers {
        if speaker.name.trim().is_empty()
            || speaker.name.chars().any(char::is_control)
            || !names.insert(speaker.name.as_str())
        {
            return Err(invalid(
                "known speaker names must be non-empty, unique identifiers without control characters",
            ));
        }
        if !speaker.duration_seconds.is_finite()
            || !(2.0..=10.0).contains(&speaker.duration_seconds)
        {
            return Err(invalid("known speaker audio must be 2–10 seconds long"));
        }
        validate_audio_file(
            &speaker.filename,
            &speaker.media_type,
            speaker.audio_bytes.len() as u64,
        )?;
    }
    Ok(())
}

fn transcription_fields(request: &TranscriptionRequest, stream: bool) -> Vec<(String, String)> {
    let mut fields = vec![
        ("model".into(), request.model.as_str().into()),
        ("response_format".into(), request.format.as_str().into()),
    ];
    if stream {
        fields.push(("stream".into(), "true".into()));
    }
    if let Some(language) = &request.language {
        fields.push(("language".into(), language.clone()));
    }
    if let Some(prompt) = &request.prompt {
        fields.push(("prompt".into(), prompt.clone()));
    }
    if let Some(temperature) = request.temperature {
        fields.push(("temperature".into(), temperature.to_string()));
    }
    for granularity in &request.timestamp_granularities {
        fields.push((
            "timestamp_granularities[]".into(),
            granularity.as_str().into(),
        ));
    }
    if request.chunking_auto {
        fields.push(("chunking_strategy".into(), "auto".into()));
    }
    for speaker in &request.known_speakers {
        fields.push(("known_speaker_names[]".into(), speaker.name.clone()));
        fields.push(("known_speaker_references[]".into(), speaker.data_url()));
    }
    fields
}
fn validate_translation(request: &TranslationRequest) -> Result<(), LlmError> {
    validate_temperature(request.temperature)?;
    validate_prompt(request.prompt.as_deref())?;
    if request.format == AudioTextFormat::DiarizedJson {
        return Err(invalid("translation does not support diarized_json"));
    }
    Ok(())
}

fn multipart_parts(
    boundary: &str,
    fields: &[(String, String)],
    input: &AudioInput,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    for (name, value) in fields {
        append_field(&mut prefix, boundary, name, value);
    }
    prefix.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n", sanitize_filename(&input.filename), input.media_type).as_bytes());
    let suffix = Bytes::from(format!("\r\n--{boundary}--\r\n"));
    (prefix.freeze(), suffix)
}
struct MultipartState {
    prefix: Option<Bytes>,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    remaining: u64,
    done: bool,
    suffix: Option<Bytes>,
}
fn multipart_stream(
    prefix: Bytes,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    size_bytes: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::try_unfold(
        MultipartState {
            prefix: Some(prefix),
            file,
            remaining: size_bytes,
            done: false,
            suffix: Some(suffix),
        },
        |mut state| async move {
            if let Some(prefix) = state.prefix.take() {
                return Ok(Some((prefix, state)));
            }
            if !state.done {
                match state.file.next().await {
                    Some(Ok(chunk)) => {
                        if chunk.len() as u64 > state.remaining {
                            return Err(invalid("audio stream exceeds declared size"));
                        }
                        state.remaining -= chunk.len() as u64;
                        return Ok(Some((chunk, state)));
                    }
                    Some(Err(error)) => return Err(error),
                    None => {
                        state.done = true;
                        if state.remaining != 0 {
                            return Err(invalid("audio stream is shorter than declared size"));
                        }
                    }
                }
            }
            Ok(state.suffix.take().map(|suffix| (suffix, state)))
        },
    )
    .boxed()
}

fn decode_result(
    response: &HttpResponse,
    format: AudioTextFormat,
) -> Result<AudioTextResult, AudioError> {
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    if !(200..300).contains(&response.status) {
        let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        });
        return Err(AudioError::Provider {
            status: response.status,
            request_id,
            body,
        });
    }
    if !format.is_json() {
        let text =
            String::from_utf8(response.body.to_vec()).map_err(|_| AudioError::InvalidResponse {
                message: "text response is not UTF-8".into(),
                native: Value::Null,
            })?;
        return Ok(AudioTextResult {
            text,
            language: None,
            languages: vec![],
            duration_seconds: None,
            words: vec![],
            segments: vec![],
            native: None,
            request_id,
        });
    }
    let native: Value =
        serde_json::from_slice(&response.body).map_err(|_| AudioError::InvalidResponse {
            message: "JSON response is not valid JSON".into(),
            native: Value::String(String::from_utf8_lossy(&response.body).into_owned()),
        })?;
    let text = native
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| AudioError::InvalidResponse {
            message: "JSON response has no text".into(),
            native: native.clone(),
        })?
        .to_owned();
    let words: Vec<TimedWord> = decode_timed(&native, "words")?;
    let segments: Vec<TimedSegment> = decode_timed(&native, "segments")?;
    if words
        .iter()
        .any(|word| !valid_interval(word.start, word.end))
        || segments
            .iter()
            .any(|segment| !valid_interval(segment.start, segment.end))
    {
        return Err(AudioError::InvalidResponse {
            message: "invalid audio timestamp interval".into(),
            native,
        });
    }
    let duration_seconds = native.get("duration").and_then(Value::as_f64);
    if duration_seconds.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(AudioError::InvalidResponse {
            message: "invalid duration".into(),
            native,
        });
    }
    let language = native
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let languages = match native.get("languages") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| {
                row.get("code")
                    .and_then(Value::as_str)
                    .filter(|code| !code.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| AudioError::InvalidResponse {
                        message: "invalid detected language entry".into(),
                        native: native.clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(AudioError::InvalidResponse {
                message: "invalid detected languages array".into(),
                native,
            });
        }
    };
    Ok(AudioTextResult {
        text,
        language,
        languages,
        duration_seconds,
        words,
        segments,
        native: Some(native),
        request_id,
    })
}
fn valid_interval(start: f64, end: f64) -> bool {
    start.is_finite() && end.is_finite() && start >= 0.0 && end >= start
}
fn decode_timed<T: for<'de> Deserialize<'de>>(
    native: &Value,
    key: &str,
) -> Result<Vec<T>, AudioError> {
    match native.get(key) {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| {
                serde_json::from_value(row.clone()).map_err(|_| AudioError::InvalidResponse {
                    message: format!("invalid {key} entry"),
                    native: native.clone(),
                })
            })
            .collect(),
        _ => Err(AudioError::InvalidResponse {
            message: format!("invalid {key} array"),
            native: native.clone(),
        }),
    }
}

pub fn validate_route(profile: &ProviderProfile, route: &AudioRoute) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "openai" || route.auth != ServiceAuth::Bearer {
        return Err(invalid(
            "audio route requires OpenAI provider identity and bearer authentication",
        ));
    }
    let transcriptions = url::Url::parse(&route.transcriptions_endpoint)
        .map_err(|_| invalid("invalid transcriptions endpoint"))?;
    let translations = url::Url::parse(&route.translations_endpoint)
        .map_err(|_| invalid("invalid translations endpoint"))?;
    for url in [&transcriptions, &translations] {
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(invalid(
                "audio endpoint has an invalid origin or URL component",
            ));
        }
    }
    if transcriptions.scheme() != translations.scheme()
        || transcriptions.host_str() != translations.host_str()
        || transcriptions.port_or_known_default() != translations.port_or_known_default()
        || transcriptions.path().strip_suffix("/transcriptions")
            != translations.path().strip_suffix("/translations")
        || !transcriptions.path().ends_with("/audio/transcriptions")
    {
        return Err(invalid(
            "audio endpoints must share one origin and documented paths",
        ));
    }
    if let Some(speech) = &route.speech_endpoint {
        let speech = url::Url::parse(speech).map_err(|_| invalid("invalid speech endpoint"))?;
        if speech.scheme() != transcriptions.scheme()
            || speech.host_str() != transcriptions.host_str()
            || speech.port_or_known_default() != transcriptions.port_or_known_default()
            || speech.path() != transcriptions.path().replace("/transcriptions", "/speech")
            || !speech.username().is_empty()
            || speech.password().is_some()
            || speech.query().is_some()
            || speech.fragment().is_some()
        {
            return Err(invalid(
                "speech endpoint must share the documented audio route origin",
            ));
        }
    }
    Ok(())
}
