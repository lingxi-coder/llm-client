//! Hosted GLM audio APIs from Zhipu's BigModel and Z.AI platforms.
//!
//! This module is separate from [`crate::providers::zhipu::audio`], which calls a
//! self-hosted SGLang endpoint. Hosted ASR supports the documented
//! `glm-asr-2512` transcription route in both mainland China and international
//! regions. Hosted GLM-TTS is currently exposed only on the mainland route.
//! Credentials are borrowed per operation so a long-lived service never pins
//! a stale API key.

use crate::{
    files::{multipart_boundary, sanitize_filename, validate_media_type},
    protocol::{LlmError, Secret},
    providers::openai::audio::AudioInput,
    transport::{HttpExecutor, HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt, time::Duration};
use thiserror::Error;

/// Mainland BigModel API root documented by Zhipu.
pub const GLM_MAINLAND_API_BASE: &str = "https://open.bigmodel.cn/api/paas/v4";
/// International Z.AI API root documented by Z.AI.
pub const GLM_INTERNATIONAL_API_BASE: &str = "https://api.z.ai/api/paas/v4";
/// Hosted transcription model documented by the first-party ASR reference.
pub const GLM_ASR_2512_MODEL: &str = "glm-asr-2512";
/// Hosted text-to-speech model published in the BigModel audio API reference.
pub const GLM_TTS_MODEL: &str = "glm-tts";

const MAX_AUDIO_BYTES: u64 = 25_000_000;
const MAX_JSON_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TTS_TEXT_CHARS: usize = 100_000;
const MAX_METADATA_CHARS: usize = 4096;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Region of the hosted GLM audio account. API keys are region-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmCloudAudioRegion {
    MainlandChina,
    International,
}

impl GlmCloudAudioRegion {
    fn api_base(self) -> &'static str {
        match self {
            Self::MainlandChina => GLM_MAINLAND_API_BASE,
            Self::International => GLM_INTERNATIONAL_API_BASE,
        }
    }
}

/// Non-secret caller identity bound to one region. There are no asynchronous
/// references in these synchronous audio operations, but responses retain the
/// scope so callers can keep account provenance with the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmCloudAudioScope {
    profile_name: String,
    account_scope: String,
    region: GlmCloudAudioRegion,
}

impl GlmCloudAudioScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: GlmCloudAudioRegion,
    ) -> Result<Self, GlmCloudAudioError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
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

    pub fn region(&self) -> GlmCloudAudioRegion {
        self.region
    }

    fn validate(&self) -> Result<(), GlmCloudAudioError> {
        if self.profile_name.trim().is_empty() || self.account_scope.trim().is_empty() {
            return Err(invalid("profile name and account scope must be non-empty"));
        }
        Ok(())
    }
}

/// Options for one GLM-ASR-2512 transcription. The request intentionally
/// exposes only fields confirmed by the published regional contracts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmCloudTranscriptionRequest {
    /// `false` returns a complete transcription; `true` returns the provider's
    /// streaming response as raw bytes for caller-managed event decoding.
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
}

impl GlmCloudTranscriptionRequest {
    pub fn new() -> Self {
        Self::default()
    }

    fn validate(&self) -> Result<(), GlmCloudAudioError> {
        validate_identifier(self.request_id.as_deref(), 6, 64, "request_id")?;
        validate_identifier(self.user_id.as_deref(), 6, 128, "user_id")?;
        Ok(())
    }
}

/// One mainland hosted GLM-TTS request. `voice`, `response_format`, and
/// `encode_format` are passed through using the first-party SDK's JSON field
/// names; callers remain responsible for selecting values enabled for their
/// model/account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmCloudSpeechRequest {
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The first-party SDK requires this field for its speech operation.
    pub encode_format: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl GlmCloudSpeechRequest {
    pub fn new(input: impl Into<String>, encode_format: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            voice: None,
            response_format: None,
            encode_format: encode_format.into(),
            stream: false,
            request_id: None,
            user_id: None,
        }
    }

    fn validate(&self) -> Result<(), GlmCloudAudioError> {
        if self.input.trim().is_empty()
            || self.input.chars().count() > MAX_TTS_TEXT_CHARS
            || self.input.contains('\0')
        {
            return Err(invalid(
                "TTS input must contain 1–100000 non-NUL characters",
            ));
        }
        validate_token(&self.encode_format, "encode_format")?;
        if let Some(voice) = &self.voice {
            validate_token(voice, "voice")?;
        }
        if let Some(format) = &self.response_format {
            validate_token(format, "response_format")?;
        }
        validate_identifier(self.request_id.as_deref(), 6, 64, "request_id")?;
        validate_identifier(self.user_id.as_deref(), 6, 128, "user_id")?;
        Ok(())
    }
}

/// Outcome class for a billable hosted audio operation. The service never
/// retries; callers must treat `Unknown` as possibly processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmCloudAudioDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmCloudAudioOperation {
    Transcription,
    Speech,
}

#[derive(Debug, Error)]
pub enum GlmCloudAudioError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid hosted GLM audio input: {0}")]
    InvalidInput(String),
    #[error("hosted GLM {operation:?} is not documented for the {region:?} region")]
    UnsupportedRegion {
        operation: GlmCloudAudioOperation,
        region: GlmCloudAudioRegion,
    },
    #[error("hosted GLM {operation:?} returned HTTP {status}: {message}")]
    Provider {
        operation: GlmCloudAudioOperation,
        status: u16,
        message: String,
        request_id: Option<String>,
        dispatch: GlmCloudAudioDispatch,
    },
    #[error(
        "hosted GLM {operation:?} accepted the request but returned an invalid response: {message}"
    )]
    InvalidResponse {
        operation: GlmCloudAudioOperation,
        message: String,
        request_id: Option<String>,
    },
    #[error("hosted GLM {operation:?} outcome is unknown: {source}")]
    OutcomeUnknown {
        operation: GlmCloudAudioOperation,
        #[source]
        source: LlmError,
    },
}

impl GlmCloudAudioError {
    pub fn dispatch(&self) -> GlmCloudAudioDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::UnsupportedRegion { .. } => {
                GlmCloudAudioDispatch::NotSent
            }
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => GlmCloudAudioDispatch::Accepted,
            Self::OutcomeUnknown { .. } => GlmCloudAudioDispatch::Unknown,
        }
    }
}

/// Normalized transcript plus the provider response for forward compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmCloudTranscript {
    pub text: String,
    pub model: Option<String>,
    pub response_id: Option<String>,
    pub request_id: Option<String>,
    pub scope: GlmCloudAudioScope,
    pub native: Value,
}

/// Result of transcription. Streaming responses are deliberately opaque so
/// the caller can parse the exact event format indicated by `content_type`.
#[derive(Debug)]
pub enum GlmCloudTranscriptionOutput {
    Complete(GlmCloudTranscript),
    Streaming(GlmCloudAudioByteStream),
}

/// Raw response body stream used by streaming ASR and all TTS responses.
/// Arbitrary transport chunks are preserved; no UTF-8 or JSON conversion is
/// applied to audio bytes.
pub struct GlmCloudAudioByteStream {
    content_type: Option<String>,
    request_id: Option<String>,
    scope: GlmCloudAudioScope,
    body: BoxStream<'static, Result<Bytes, GlmCloudAudioError>>,
}

impl GlmCloudAudioByteStream {
    pub fn content_type(&self) -> Option<&str> {
        self.content_type.as_deref()
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn scope(&self) -> &GlmCloudAudioScope {
        &self.scope
    }

    pub fn into_body(self) -> BoxStream<'static, Result<Bytes, GlmCloudAudioError>> {
        self.body
    }
}

impl fmt::Debug for GlmCloudAudioByteStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmCloudAudioByteStream")
            .field("content_type", &self.content_type)
            .field("request_id", &self.request_id)
            .field("scope", &self.scope)
            .field("body", &"<stream>")
            .finish()
    }
}

/// Typed adapter for the documented hosted GLM audio operations.
#[derive(Clone)]
pub struct GlmCloudAudioService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: GlmCloudAudioScope,
}

impl<'a> GlmCloudAudioService<'a> {
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

    pub fn new(
        http: &'a dyn Transport,
        scope: GlmCloudAudioScope,
    ) -> Result<Self, GlmCloudAudioError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &GlmCloudAudioScope {
        &self.scope
    }

    /// Transcribe one WAV or MP3 upload. International API documentation
    /// states a 25 MB and 30-second ceiling. The size is checked locally; the
    /// caller/provider must enforce duration because this layer does not
    /// decode media.
    pub async fn transcribe(
        &self,
        credential: &Secret<String>,
        input: AudioInput,
        request: &GlmCloudTranscriptionRequest,
    ) -> Result<GlmCloudTranscriptionOutput, GlmCloudAudioError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        request.validate()?;
        validate_audio_input(&input)?;

        let operation = GlmCloudAudioOperation::Transcription;
        let boundary = multipart_boundary();
        let fields = transcription_fields(request);
        let (prefix, suffix) = multipart_parts(&boundary, &fields, &input);
        let content_length = input
            .size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| invalid("multipart content length overflows"))?;
        let http_request = HttpStreamRequest {
            method: "POST".into(),
            url: format!(
                "{}/audio/transcriptions",
                pinned_service.scope.region.api_base()
            ),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: multipart_stream(prefix, input.body, input.size_bytes, suffix),
            content_length,
            timeout: Some(DEFAULT_TIMEOUT),
        };
        let response = HttpExecutor::new(pinned_service.http)
            .send_stream(http_request)
            .await
            .map_err(|source| outcome_unknown(operation, source))?;
        let status = response.status;
        let request_id = response
            .header("x-request-id")
            .map(str::to_owned)
            .or_else(|| response.header("request-id").map(str::to_owned));
        let content_type = response.header("content-type").map(str::to_owned);

        if !(200..300).contains(&status) {
            return Err(provider_error(operation, response, request_id).await);
        }

        if request.stream {
            return Ok(GlmCloudTranscriptionOutput::Streaming(
                GlmCloudAudioByteStream {
                    content_type,
                    request_id,
                    scope: pinned_service.scope.clone(),
                    body: wrap_response_body(response.body, operation),
                },
            ));
        }

        let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
            .await
            .map_err(|source| outcome_unknown(operation, source))?;
        let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
            GlmCloudAudioError::InvalidResponse {
                operation,
                message: "response was not valid JSON".into(),
                request_id: request_id.clone(),
            }
        })?;
        decode_transcript(
            pinned_service.scope.region,
            pinned_service.scope.clone(),
            request_id,
            native,
        )
        .map(GlmCloudTranscriptionOutput::Complete)
    }

    /// Synthesize speech in mainland China. The binary response remains a
    /// stream and is never buffered or decoded as text. International TTS is
    /// rejected before any request because no corresponding route is listed
    /// in the first-party international API index.
    pub async fn synthesize(
        &self,
        credential: &Secret<String>,
        request: &GlmCloudSpeechRequest,
    ) -> Result<GlmCloudAudioByteStream, GlmCloudAudioError> {
        let pinned_service = self.pin()?;
        if pinned_service.scope.region != GlmCloudAudioRegion::MainlandChina {
            return Err(GlmCloudAudioError::UnsupportedRegion {
                operation: GlmCloudAudioOperation::Speech,
                region: pinned_service.scope.region,
            });
        }
        validate_credential(credential)?;
        request.validate()?;

        let operation = GlmCloudAudioOperation::Speech;
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(GLM_TTS_MODEL));
        body.insert("input".into(), json!(request.input));
        body.insert("encode_format".into(), json!(request.encode_format));
        body.insert("stream".into(), json!(request.stream));
        if let Some(value) = &request.voice {
            body.insert("voice".into(), json!(value));
        }
        if let Some(value) = &request.response_format {
            body.insert("response_format".into(), json!(value));
        }
        if let Some(value) = &request.request_id {
            body.insert("request_id".into(), json!(value));
        }
        if let Some(value) = &request.user_id {
            body.insert("user_id".into(), json!(value));
        }
        let body = serde_json::to_vec(&Value::Object(body))
            .map_err(|_| invalid("could not encode hosted TTS request"))?;
        let http_request = HttpRequest {
            method: "POST".into(),
            url: format!("{}/audio/speech", pinned_service.scope.region.api_base()),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                ),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(body),
            timeout: Some(DEFAULT_TIMEOUT),
        };
        let response = HttpExecutor::new(pinned_service.http)
            .send(http_request)
            .await
            .map_err(|source| outcome_unknown(operation, source))?;
        let request_id = response
            .header("x-request-id")
            .map(str::to_owned)
            .or_else(|| response.header("request-id").map(str::to_owned));
        let content_type = response.header("content-type").map(str::to_owned);
        let status = response.status;
        if !(200..300).contains(&status) {
            return Err(provider_error(operation, response, request_id).await);
        }

        Ok(GlmCloudAudioByteStream {
            content_type,
            request_id,
            scope: pinned_service.scope.clone(),
            body: wrap_response_body(response.body, operation),
        })
    }
}

fn transcription_fields(request: &GlmCloudTranscriptionRequest) -> Vec<(String, String)> {
    let mut fields = vec![
        ("model".into(), GLM_ASR_2512_MODEL.into()),
        ("stream".into(), request.stream.to_string()),
    ];
    if let Some(id) = &request.request_id {
        fields.push(("request_id".into(), id.clone()));
    }
    if let Some(id) = &request.user_id {
        fields.push(("user_id".into(), id.clone()));
    }
    fields
}

fn multipart_parts(
    boundary: &str,
    fields: &[(String, String)],
    input: &AudioInput,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    for (name, value) in fields {
        crate::files::append_field(&mut prefix, boundary, name, value);
    }
    let filename = sanitize_filename(&input.filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            input.media_type
        )
        .as_bytes(),
    );
    let suffix = Bytes::from(format!("\r\n--{boundary}--\r\n"));
    (prefix.freeze(), suffix)
}

fn multipart_stream(
    prefix: Bytes,
    input: BoxStream<'static, Result<Bytes, LlmError>>,
    expected_size: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    enum Part {
        Prefix,
        Audio(u64),
        Done,
    }
    stream::unfold(
        (Part::Prefix, Some(prefix), input, expected_size, suffix),
        |(part, prefix, mut input, expected_size, suffix)| async move {
            match part {
                Part::Prefix => Some((
                    Ok(prefix.expect("prefix is present")),
                    (Part::Audio(0), None, input, expected_size, suffix),
                )),
                Part::Audio(mut sent) => match input.next().await {
                    Some(Ok(chunk)) => {
                        sent = sent.saturating_add(chunk.len() as u64);
                        if sent > expected_size {
                            let error = LlmError::InvalidRequest {
                                message: "audio stream exceeded its declared byte size".into(),
                            };
                            Some((Err(error), (Part::Done, None, input, expected_size, suffix)))
                        } else {
                            Some((
                                Ok(chunk),
                                (Part::Audio(sent), None, input, expected_size, suffix),
                            ))
                        }
                    }
                    Some(Err(error)) => {
                        Some((Err(error), (Part::Done, None, input, expected_size, suffix)))
                    }
                    None if sent != expected_size => {
                        let error = LlmError::InvalidRequest {
                            message: "audio stream was shorter than its declared byte size".into(),
                        };
                        Some((Err(error), (Part::Done, None, input, expected_size, suffix)))
                    }
                    None => Some((
                        Ok(suffix),
                        (Part::Done, None, input, expected_size, Bytes::new()),
                    )),
                },
                Part::Done => None,
            }
        },
    )
    .boxed()
}

fn validate_audio_input(input: &AudioInput) -> Result<(), GlmCloudAudioError> {
    if input.size_bytes == 0 || input.size_bytes > MAX_AUDIO_BYTES {
        return Err(invalid("audio must contain 1–25000000 bytes"));
    }
    validate_media_type(&input.media_type).map_err(|_| invalid("audio media type is invalid"))?;
    let extension = input
        .filename
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase());
    if !matches!(extension.as_deref(), Some("wav" | "mp3")) {
        return Err(invalid("GLM-ASR accepts WAV or MP3 audio files"));
    }
    if input.filename.trim().is_empty()
        || input.filename.chars().any(char::is_control)
        || input.media_type.chars().any(char::is_control)
    {
        return Err(invalid("audio filename and media type must be valid"));
    }
    Ok(())
}

fn decode_transcript(
    region: GlmCloudAudioRegion,
    scope: GlmCloudAudioScope,
    header_request_id: Option<String>,
    native: Value,
) -> Result<GlmCloudTranscript, GlmCloudAudioError> {
    let text = match region {
        GlmCloudAudioRegion::International => native.get("text").and_then(Value::as_str),
        GlmCloudAudioRegion::MainlandChina => native
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str),
    }
    .filter(|text| !text.is_empty())
    .ok_or_else(|| GlmCloudAudioError::InvalidResponse {
        operation: GlmCloudAudioOperation::Transcription,
        message: "success response did not contain transcript text".into(),
        request_id: header_request_id.clone(),
    })?;

    Ok(GlmCloudTranscript {
        text: text.to_owned(),
        model: native
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        response_id: native.get("id").and_then(Value::as_str).map(str::to_owned),
        request_id: native
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or(header_request_id),
        scope,
        native,
    })
}

async fn provider_error(
    operation: GlmCloudAudioOperation,
    response: StreamResponse,
    header_request_id: Option<String>,
) -> GlmCloudAudioError {
    let status = response.status;
    let response =
        match HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES)).await {
            Ok(response) => response,
            Err(source) => return outcome_unknown(operation, source),
        };
    let body: Value = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
    let message = body
        .get("message")
        .or_else(|| body.get("msg"))
        .or_else(|| body.pointer("/error/message"))
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty())
        .unwrap_or("provider returned an error")
        .chars()
        .take(MAX_METADATA_CHARS)
        .collect::<String>();
    let request_id = body
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(header_request_id);
    let dispatch = if (400..500).contains(&status) {
        GlmCloudAudioDispatch::Rejected
    } else {
        GlmCloudAudioDispatch::Unknown
    };
    GlmCloudAudioError::Provider {
        operation,
        status,
        message,
        request_id,
        dispatch,
    }
}

fn validate_credential(credential: &Secret<String>) -> Result<(), GlmCloudAudioError> {
    if credential.expose_secret().trim().is_empty() {
        return Err(invalid("hosted GLM bearer credential must be non-empty"));
    }
    Ok(())
}

fn validate_identifier(
    value: Option<&str>,
    min_chars: usize,
    max_chars: usize,
    name: &str,
) -> Result<(), GlmCloudAudioError> {
    if value.is_some_and(|value| {
        let length = value.chars().count();
        length < min_chars || length > max_chars || value.chars().any(char::is_control)
    }) {
        return Err(invalid(&format!(
            "{name} must contain {min_chars}–{max_chars} non-control characters"
        )));
    }
    Ok(())
}

fn validate_token(value: &str, name: &str) -> Result<(), GlmCloudAudioError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(invalid(&format!(
            "{name} must be a 1–128 character ASCII token"
        )));
    }
    Ok(())
}

fn outcome_unknown(operation: GlmCloudAudioOperation, source: LlmError) -> GlmCloudAudioError {
    if matches!(
        &source,
        LlmError::InvalidRequest { .. }
            | LlmError::UnsupportedCapability { .. }
            | LlmError::Authentication { .. }
            | LlmError::PermissionDenied { .. }
    ) {
        GlmCloudAudioError::Llm(source)
    } else {
        GlmCloudAudioError::OutcomeUnknown { operation, source }
    }
}

fn wrap_response_body(
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    operation: GlmCloudAudioOperation,
) -> BoxStream<'static, Result<Bytes, GlmCloudAudioError>> {
    body.map(move |chunk| chunk.map_err(|source| outcome_unknown(operation, source)))
        .boxed()
}

fn invalid(message: &str) -> GlmCloudAudioError {
    GlmCloudAudioError::InvalidInput(message.into())
}
