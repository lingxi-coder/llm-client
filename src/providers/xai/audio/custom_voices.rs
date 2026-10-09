//! Native xAI custom-voice metadata and reference-audio lifecycle.

use super::{
    invalid, invalid_response, map_outcome, parse_error_body, response_id, validate_credential,
    XaiAudioCredentials, XaiAudioError, XaiAudioScope, XaiAudioService, XaiAudioUnknownResponse,
    MAX_JSON_RESPONSE_BYTES,
};
use crate::{
    files::{append_field, multipart_boundary, sanitize_filename, validate_media_type},
    protocol::LlmError,
    providers::openai::audio::AudioInput,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest, StreamResponse},
};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use url::Url;

/// A voice identifier bound to the profile, API endpoint, and account that
/// created or listed it. References from another service scope are rejected
/// before any request is sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCustomVoiceRef {
    scope: XaiAudioScope,
    voice_id: String,
}

impl XaiCustomVoiceRef {
    pub fn new(scope: XaiAudioScope, voice_id: impl Into<String>) -> Result<Self, XaiAudioError> {
        let voice_id = voice_id.into();
        validate_voice_id(&voice_id)?;
        Ok(Self { scope, voice_id })
    }

    pub fn scope(&self) -> &XaiAudioScope {
        &self.scope
    }

    pub fn voice_id(&self) -> &str {
        &self.voice_id
    }
}

/// Metadata returned for one xAI custom voice. `native` retains all provider
/// fields, including fields not yet modeled by this client.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiCustomVoice {
    pub reference: XaiCustomVoiceRef,
    pub name: Option<String>,
    pub description: Option<String>,
    pub gender: Option<String>,
    pub accent: Option<String>,
    pub age: Option<String>,
    pub language: Option<String>,
    pub use_case: Option<String>,
    pub tone: Option<String>,
    pub created_at: Option<String>,
    pub native: Value,
    pub request_id: Option<String>,
}

/// One opaque continuation token bound to the account that returned it.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCustomVoiceCursor {
    scope: XaiAudioScope,
    token: String,
}

impl XaiCustomVoiceCursor {
    pub fn scope(&self) -> &XaiAudioScope {
        &self.scope
    }
}

impl fmt::Debug for XaiCustomVoiceCursor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiCustomVoiceCursor")
            .field("scope", &self.scope)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// Parameters for one page of `GET /v1/custom-voices`.
#[derive(Debug, Clone, Default)]
pub struct XaiCustomVoiceListRequest {
    limit: Option<u16>,
    cursor: Option<XaiCustomVoiceCursor>,
}

impl XaiCustomVoiceListRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limit(mut self, limit: u16) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn after(mut self, cursor: XaiCustomVoiceCursor) -> Self {
        self.cursor = Some(cursor);
        self
    }
}

/// One page returned by xAI's team-scoped custom-voice list endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiCustomVoiceList {
    pub scope: XaiAudioScope,
    pub voices: Vec<XaiCustomVoice>,
    pub next_page: Option<XaiCustomVoiceCursor>,
    pub native: Value,
    pub request_id: Option<String>,
}

/// Documented gender values accepted when creating or updating a custom voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum XaiCustomVoiceGender {
    Male,
    Female,
    Neutral,
}

/// Documented age values accepted when creating or updating a custom voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XaiCustomVoiceAge {
    #[serde(rename = "young")]
    Young,
    #[serde(rename = "middle-aged")]
    MiddleAged,
    #[serde(rename = "old")]
    Old,
}

/// Documented use-case values accepted when creating or updating a custom voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiCustomVoiceUseCase {
    Conversational,
    Narration,
    Characters,
    Educational,
    Advertisement,
    SocialMedia,
    Entertainment,
}

/// Documented tone values accepted when creating or updating a custom voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiCustomVoiceTone {
    Warm,
    Casual,
    Professional,
    Friendly,
    Authoritative,
    Expressive,
    Calm,
}

/// Multipart request for creating one custom voice.
///
/// The optional duration is caller supplied metadata for local validation; the
/// client does not inspect or decode the audio stream to estimate duration.
pub struct XaiCustomVoiceCreateRequest {
    audio: AudioInput,
    duration_seconds: Option<f64>,
    name: Option<String>,
    description: Option<String>,
    gender: Option<XaiCustomVoiceGender>,
    accent: Option<String>,
    age: Option<XaiCustomVoiceAge>,
    language: Option<String>,
    use_case: Option<XaiCustomVoiceUseCase>,
    tone: Option<XaiCustomVoiceTone>,
}

impl XaiCustomVoiceCreateRequest {
    pub fn new(audio: AudioInput) -> Self {
        Self {
            audio,
            duration_seconds: None,
            name: None,
            description: None,
            gender: None,
            accent: None,
            age: None,
            language: None,
            use_case: None,
            tone: None,
        }
    }

    pub fn with_duration_seconds(mut self, duration_seconds: f64) -> Self {
        self.duration_seconds = Some(duration_seconds);
        self
    }

    pub fn with_name(mut self, value: impl Into<String>) -> Self {
        self.name = Some(value.into());
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_gender(mut self, value: XaiCustomVoiceGender) -> Self {
        self.gender = Some(value);
        self
    }

    pub fn with_accent(mut self, value: impl Into<String>) -> Self {
        self.accent = Some(value.into());
        self
    }

    pub fn with_age(mut self, value: XaiCustomVoiceAge) -> Self {
        self.age = Some(value);
        self
    }

    pub fn with_language(mut self, value: impl Into<String>) -> Self {
        self.language = Some(value.into());
        self
    }

    pub fn with_use_case(mut self, value: XaiCustomVoiceUseCase) -> Self {
        self.use_case = Some(value);
        self
    }

    pub fn with_tone(mut self, value: XaiCustomVoiceTone) -> Self {
        self.tone = Some(value);
        self
    }
}

impl fmt::Debug for XaiCustomVoiceCreateRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiCustomVoiceCreateRequest")
            .field("filename", &self.audio.filename)
            .field("media_type", &self.audio.media_type)
            .field("size_bytes", &self.audio.size_bytes)
            .field("duration_seconds", &self.duration_seconds)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("gender", &self.gender)
            .field("accent", &self.accent)
            .field("age", &self.age)
            .field("language", &self.language)
            .field("use_case", &self.use_case)
            .field("tone", &self.tone)
            .finish()
    }
}

/// Partial metadata update. Unset fields are omitted; `clear_*` methods encode
/// JSON `null`, as required by the documented PATCH contract.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct XaiCustomVoicePatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gender: Option<Option<XaiCustomVoiceGender>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accent: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    age: Option<Option<XaiCustomVoiceAge>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_case: Option<Option<XaiCustomVoiceUseCase>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tone: Option<Option<XaiCustomVoiceTone>>,
}

impl XaiCustomVoicePatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, value: impl Into<String>) -> Self {
        self.name = Some(Some(value.into()));
        self
    }

    pub fn clear_name(mut self) -> Self {
        self.name = Some(None);
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(Some(value.into()));
        self
    }

    pub fn clear_description(mut self) -> Self {
        self.description = Some(None);
        self
    }

    pub fn with_gender(mut self, value: XaiCustomVoiceGender) -> Self {
        self.gender = Some(Some(value));
        self
    }

    pub fn clear_gender(mut self) -> Self {
        self.gender = Some(None);
        self
    }

    pub fn with_accent(mut self, value: impl Into<String>) -> Self {
        self.accent = Some(Some(value.into()));
        self
    }

    pub fn clear_accent(mut self) -> Self {
        self.accent = Some(None);
        self
    }

    pub fn with_age(mut self, value: XaiCustomVoiceAge) -> Self {
        self.age = Some(Some(value));
        self
    }

    pub fn clear_age(mut self) -> Self {
        self.age = Some(None);
        self
    }

    pub fn with_language(mut self, value: impl Into<String>) -> Self {
        self.language = Some(Some(value.into()));
        self
    }

    pub fn clear_language(mut self) -> Self {
        self.language = Some(None);
        self
    }

    pub fn with_use_case(mut self, value: XaiCustomVoiceUseCase) -> Self {
        self.use_case = Some(Some(value));
        self
    }

    pub fn clear_use_case(mut self) -> Self {
        self.use_case = Some(None);
        self
    }

    pub fn with_tone(mut self, value: XaiCustomVoiceTone) -> Self {
        self.tone = Some(Some(value));
        self
    }

    pub fn clear_tone(mut self) -> Self {
        self.tone = Some(None);
        self
    }

    fn validate(&self) -> Result<(), XaiAudioError> {
        for value in [
            self.name.as_ref().and_then(Option::as_ref),
            self.description.as_ref().and_then(Option::as_ref),
            self.accent.as_ref().and_then(Option::as_ref),
            self.language.as_ref().and_then(Option::as_ref),
        ]
        .into_iter()
        .flatten()
        {
            validate_patch_text(value)?;
            if value.is_empty() {
                return Err(invalid(
                    "custom voice PATCH string values must not be empty; use clear_* to send null",
                ));
            }
        }
        Ok(())
    }
}

/// Receipt returned by `DELETE /v1/custom-voices/{voice_id}`.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiCustomVoiceDeleteReceipt {
    pub reference: XaiCustomVoiceRef,
    pub deleted: bool,
    pub native: Value,
    pub request_id: Option<String>,
}

/// Streaming access to the original reference audio. `max_bytes` is supplied
/// by the caller and enforced while consuming the stream; xAI does not publish
/// a download byte-size limit.
pub struct XaiCustomVoiceAudioStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    pub reference: XaiCustomVoiceRef,
    pub content_type: String,
    pub request_id: Option<String>,
    max_bytes: u64,
    delivered_bytes: u64,
    finished: bool,
}

impl XaiCustomVoiceAudioStream {
    fn stop_body(&mut self) {
        self.finished = true;
        self.body = stream::empty().boxed();
    }

    /// Return the next audio chunk. Dropping the stream cancels the HTTP read.
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, XaiCustomVoiceAudioStreamError> {
        if self.finished {
            return Ok(None);
        }
        match self.body.next().await {
            Some(Ok(bytes)) => {
                let Some(total) = self.delivered_bytes.checked_add(bytes.len() as u64) else {
                    self.stop_body();
                    return Err(XaiCustomVoiceAudioStreamError {
                        delivered_bytes: self.delivered_bytes,
                        source: Box::new(LlmError::RequestTooLarge {
                            message: "custom voice audio byte count overflowed".into(),
                        }),
                    });
                };
                if total > self.max_bytes {
                    self.stop_body();
                    return Err(XaiCustomVoiceAudioStreamError {
                        delivered_bytes: self.delivered_bytes,
                        source: Box::new(LlmError::RequestTooLarge {
                            message: "custom voice audio exceeded the caller's byte limit".into(),
                        }),
                    });
                }
                self.delivered_bytes = total;
                Ok(Some(bytes))
            }
            Some(Err(source)) => {
                self.stop_body();
                Err(XaiCustomVoiceAudioStreamError {
                    delivered_bytes: self.delivered_bytes,
                    source: Box::new(source),
                })
            }
            None => {
                self.stop_body();
                if self.delivered_bytes == 0 {
                    Err(XaiCustomVoiceAudioStreamError {
                        delivered_bytes: 0,
                        source: Box::new(LlmError::ProviderInternal {
                            message: "xAI returned an empty custom voice audio body".into(),
                        }),
                    })
                } else {
                    Ok(None)
                }
            }
        }
    }
}

impl fmt::Debug for XaiCustomVoiceAudioStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiCustomVoiceAudioStream")
            .field("reference", &self.reference)
            .field("content_type", &self.content_type)
            .field("request_id", &self.request_id)
            .field("max_bytes", &self.max_bytes)
            .field("delivered_bytes", &self.delivered_bytes)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("xAI custom voice audio stream interrupted after {delivered_bytes} bytes: {source}")]
pub struct XaiCustomVoiceAudioStreamError {
    pub delivered_bytes: u64,
    #[source]
    pub source: Box<LlmError>,
}

impl<'a> XaiAudioService<'a> {
    /// Create a custom voice from a streamed reference audio file.
    ///
    /// The request is sent once. Transport failure after dispatch is reported
    /// as `OutcomeUnknown`; the client never retries a create automatically.
    pub async fn create_custom_voice(
        &self,
        request: XaiCustomVoiceCreateRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoice, XaiAudioError> {
        let pinned_service = self.pin()?;
        validate_create_request(&request)?;
        validate_credential(&credentials.api_key)?;

        let boundary = multipart_boundary();
        let (prefix, suffix) = create_multipart_parts(&boundary, &request);
        let content_length = request
            .audio
            .size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| invalid("custom voice multipart request size overflows"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let body =
            create_multipart_stream(prefix, request.audio.body, request.audio.size_bytes, suffix);
        let http = HttpStreamRequest {
            http1_header_layout: None,
            method: "POST".into(),
            url: pinned_service.route_url("custom-voices")?,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body,
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .execute_stream_bounded(http, MAX_JSON_RESPONSE_BYTES)
            .await
            .map_err(|source| map_outcome("create-custom-voice", source))?;
        let result = decode_voice_http_response(
            response,
            "create-custom-voice",
            pinned_service.scope.clone(),
            None,
        );
        uncertain_mutation_result("create-custom-voice", result)
    }

    /// Fetch one page of team-owned custom voices.
    pub async fn list_custom_voices(
        &self,
        request: &XaiCustomVoiceListRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoiceList, XaiAudioError> {
        let pinned_service = self.pin()?;
        validate_list_request(request, &pinned_service.scope)?;
        validate_credential(&credentials.api_key)?;
        let mut url = Url::parse(&pinned_service.route_url("custom-voices")?)
            .map_err(|_| invalid("configured custom voices route URL is invalid"))?;
        if request.limit.is_some() || request.cursor.is_some() {
            let mut query = url.query_pairs_mut();
            if let Some(limit) = request.limit {
                query.append_pair("limit", &limit.to_string());
            }
            if let Some(cursor) = &request.cursor {
                query.append_pair("pagination_token", &cursor.token);
            }
        }
        let previous_token = request.cursor.as_ref().map(|cursor| cursor.token.as_str());
        let response = pinned_service
            .send_read_request("GET", url.into(), credentials)
            .await?;
        decode_voice_list_response(response, pinned_service.scope.clone(), previous_token).await
    }

    /// Fetch metadata for a voice created or listed under this exact scope.
    pub async fn get_custom_voice(
        &self,
        reference: &XaiCustomVoiceRef,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoice, XaiAudioError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_ref(reference)?;
        validate_credential(&credentials.api_key)?;
        let url = pinned_service.voice_url(reference, None)?;
        let response = pinned_service
            .send_read_request("GET", url, credentials)
            .await?;
        decode_voice_stream_response(
            response,
            "get-custom-voice",
            pinned_service.scope.clone(),
            Some(reference.voice_id()),
            false,
        )
        .await
    }

    /// Update custom voice metadata with documented PATCH semantics.
    pub async fn update_custom_voice(
        &self,
        reference: &XaiCustomVoiceRef,
        patch: &XaiCustomVoicePatch,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoice, XaiAudioError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_ref(reference)?;
        patch.validate()?;
        validate_credential(&credentials.api_key)?;
        let body = serde_json::to_vec(patch)
            .map_err(|_| invalid("custom voice PATCH body could not be serialized"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let request = HttpRequest {
            http1_header_layout: None,
            method: "PATCH".into(),
            url: pinned_service.voice_url(reference, None)?,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(body),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .send(request)
            .await
            .map_err(|source| map_outcome("update-custom-voice", source))?;
        let result = decode_voice_stream_response(
            response,
            "update-custom-voice",
            pinned_service.scope.clone(),
            Some(reference.voice_id()),
            true,
        )
        .await;
        uncertain_mutation_result("update-custom-voice", result)
    }

    /// Delete a custom voice and its underlying reference audio.
    pub async fn delete_custom_voice(
        &self,
        reference: &XaiCustomVoiceRef,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoiceDeleteReceipt, XaiAudioError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_ref(reference)?;
        validate_credential(&credentials.api_key)?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let request = HttpRequest {
            http1_header_layout: None,
            method: "DELETE".into(),
            url: pinned_service.voice_url(reference, None)?,
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credentials.api_key.expose_secret()),
            )],
            body: Bytes::new(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .send(request)
            .await
            .map_err(|source| map_outcome("delete-custom-voice", source))?;
        decode_delete_response(response, reference.clone()).await
    }

    /// Stream the original reference audio without buffering or saving it.
    /// `max_bytes` is an explicit caller-side bound, not an xAI upload or
    /// download limit. The actual `Content-Type` response header is preserved.
    pub async fn get_custom_voice_audio(
        &self,
        reference: &XaiCustomVoiceRef,
        max_bytes: u64,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiCustomVoiceAudioStream, XaiAudioError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_ref(reference)?;
        if max_bytes == 0 {
            return Err(invalid("custom voice audio max_bytes must be positive"));
        }
        validate_credential(&credentials.api_key)?;
        let url = pinned_service.voice_url(reference, Some("audio"))?;
        let deadline = Deadline::after(Some(pinned_service.config.request_timeout));
        let request = HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url,
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credentials.api_key.expose_secret()),
            )],
            body: Bytes::new(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(pinned_service.transport)
            .with_deadline(deadline)
            .send(request)
            .await?;
        let request_id = response_id(&response.headers);
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES)).await?;
            return Err(XaiAudioError::Provider {
                operation: "get-custom-voice-audio",
                status: response.status,
                request_id,
                body: Box::new(parse_error_body(&response.body)),
            });
        }
        let content_type = response
            .header("content-type")
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                invalid_response(
                    "get-custom-voice-audio",
                    "successful audio response has no Content-Type header",
                    request_id.clone(),
                    Value::Null,
                )
            })?
            .to_owned();
        Ok(XaiCustomVoiceAudioStream {
            body: response.body,
            reference: reference.clone(),
            content_type,
            request_id,
            max_bytes,
            delivered_bytes: 0,
            finished: false,
        })
    }

    async fn send_read_request(
        &self,
        method: &str,
        url: String,
        credentials: &XaiAudioCredentials,
    ) -> Result<StreamResponse, XaiAudioError> {
        let deadline = Deadline::after(Some(self.config.request_timeout));
        let request = HttpRequest {
            http1_header_layout: None,
            method: method.into(),
            url,
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credentials.api_key.expose_secret()),
            )],
            body: Bytes::new(),
            timeout: deadline.remaining()?,
        };
        HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .send(request)
            .await
            .map_err(XaiAudioError::Llm)
    }

    fn validate_ref(&self, reference: &XaiCustomVoiceRef) -> Result<(), XaiAudioError> {
        if reference.scope != self.scope {
            return Err(invalid(
                "custom voice reference belongs to a different xAI profile, endpoint, or account",
            ));
        }
        validate_voice_id(&reference.voice_id)
    }

    fn voice_url(
        &self,
        reference: &XaiCustomVoiceRef,
        suffix: Option<&str>,
    ) -> Result<String, XaiAudioError> {
        let mut url = Url::parse(&self.route_url("custom-voices")?)
            .map_err(|_| invalid("configured custom voices route URL is invalid"))?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| invalid("configured custom voices URL cannot accept path segments"))?;
            segments.push(&reference.voice_id);
            if let Some(suffix) = suffix {
                segments.push(suffix);
            }
        }
        Ok(url.into())
    }
}

fn validate_create_request(request: &XaiCustomVoiceCreateRequest) -> Result<(), XaiAudioError> {
    if request.audio.size_bytes == 0 {
        return Err(invalid("custom voice reference audio must not be empty"));
    }
    if request.audio.filename.trim().is_empty()
        || request.audio.filename.chars().any(char::is_control)
    {
        return Err(invalid(
            "custom voice audio filename must be nonempty and contain no control characters",
        ));
    }
    validate_media_type(&request.audio.media_type)?;
    if let Some(duration) = request.duration_seconds {
        if !duration.is_finite() || duration <= 0.0 || duration > 120.0 {
            return Err(invalid(
                "caller-supplied custom voice audio duration must be finite, greater than zero, and at most 120 seconds",
            ));
        }
    }
    for value in [
        request.name.as_deref(),
        request.description.as_deref(),
        request.accent.as_deref(),
        request.language.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_form_text(value)?;
    }
    Ok(())
}

fn validate_form_text(value: &str) -> Result<(), XaiAudioError> {
    if value.chars().any(char::is_control) {
        Err(invalid(
            "custom voice multipart metadata must not contain control characters",
        ))
    } else {
        Ok(())
    }
}

fn validate_patch_text(value: &str) -> Result<(), XaiAudioError> {
    if value.chars().any(char::is_control) {
        Err(invalid(
            "custom voice PATCH metadata must not contain control characters",
        ))
    } else {
        Ok(())
    }
}

fn validate_list_request(
    request: &XaiCustomVoiceListRequest,
    scope: &XaiAudioScope,
) -> Result<(), XaiAudioError> {
    if request
        .limit
        .is_some_and(|limit| !(1..=1000).contains(&limit))
    {
        return Err(invalid(
            "custom voice list limit must be between 1 and 1000",
        ));
    }
    if let Some(cursor) = &request.cursor {
        if &cursor.scope != scope {
            return Err(invalid(
                "custom voice cursor belongs to a different xAI profile, endpoint, or account",
            ));
        }
        if cursor.token.trim().is_empty() || cursor.token.chars().any(char::is_control) {
            return Err(invalid("custom voice pagination token is malformed"));
        }
    }
    Ok(())
}

fn validate_voice_id(voice_id: &str) -> Result<(), XaiAudioError> {
    if voice_id.len() != 8
        || !voice_id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    {
        return Err(invalid(
            "xAI custom voice ID must be 8 lowercase alphanumeric characters",
        ));
    }
    Ok(())
}

fn create_multipart_parts(boundary: &str, request: &XaiCustomVoiceCreateRequest) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    if let Some(name) = request.name.as_deref() {
        append_field(&mut prefix, boundary, "name", name);
    }
    if let Some(description) = request.description.as_deref() {
        append_field(&mut prefix, boundary, "description", description);
    }
    if let Some(gender) = request.gender {
        append_field(&mut prefix, boundary, "gender", gender.into_form_value());
    }
    if let Some(accent) = request.accent.as_deref() {
        append_field(&mut prefix, boundary, "accent", accent);
    }
    if let Some(age) = request.age {
        append_field(&mut prefix, boundary, "age", age.into_form_value());
    }
    if let Some(language) = request.language.as_deref() {
        append_field(&mut prefix, boundary, "language", language);
    }
    if let Some(use_case) = request.use_case {
        append_field(
            &mut prefix,
            boundary,
            "use_case",
            use_case.into_form_value(),
        );
    }
    if let Some(tone) = request.tone {
        append_field(&mut prefix, boundary, "tone", tone.into_form_value());
    }
    let filename = sanitize_filename(&request.audio.filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            request.audio.media_type
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

fn create_multipart_stream(
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
                        Ok(state.prefix.take().expect("multipart prefix is present")),
                        state,
                    ))
                }
                MultipartPhase::File => match state.file.next().await {
                    Some(Ok(chunk)) => {
                        let Some(total) = state.actual_file_bytes.checked_add(chunk.len() as u64)
                        else {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "xAI custom voice upload byte count overflowed".into(),
                                }),
                                state,
                            ));
                        };
                        if total > state.declared_file_bytes {
                            state.phase = MultipartPhase::Done;
                            return Some((
                                Err(LlmError::StreamInterrupted {
                                    message: "xAI custom voice upload exceeded its declared length"
                                        .into(),
                                }),
                                state,
                            ));
                        }
                        state.actual_file_bytes = total;
                        Some((Ok(chunk), state))
                    }
                    Some(Err(_)) => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "xAI custom voice upload stream was interrupted".into(),
                            }),
                            state,
                        ))
                    }
                    None if state.actual_file_bytes != state.declared_file_bytes => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Err(LlmError::StreamInterrupted {
                                message: "xAI custom voice upload ended before its declared length"
                                    .into(),
                            }),
                            state,
                        ))
                    }
                    None => {
                        state.phase = MultipartPhase::Done;
                        Some((
                            Ok(state.suffix.take().expect("multipart suffix is present")),
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

async fn decode_voice_stream_response(
    response: StreamResponse,
    operation: &'static str,
    scope: XaiAudioScope,
    expected_voice_id: Option<&str>,
    mutation: bool,
) -> Result<XaiCustomVoice, XaiAudioError> {
    let request_id = response_id(&response.headers);
    if !(200..300).contains(&response.status) {
        let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
            .await
            .map_err(XaiAudioError::Llm)?;
        return Err(XaiAudioError::Provider {
            operation,
            status: response.status,
            request_id,
            body: Box::new(parse_error_body(&response.body)),
        });
    }
    let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
        .await
        .map_err(|source| {
            if mutation {
                map_outcome(operation, source)
            } else {
                XaiAudioError::Llm(source)
            }
        })?;
    let result = decode_voice_http_response(response, operation, scope, expected_voice_id);
    if mutation {
        uncertain_mutation_result(operation, result)
    } else {
        result
    }
}

fn decode_voice_http_response(
    response: HttpResponse,
    operation: &'static str,
    scope: XaiAudioScope,
    expected_voice_id: Option<&str>,
) -> Result<XaiCustomVoice, XaiAudioError> {
    let request_id = response_id(&response.headers);
    if !(200..300).contains(&response.status) {
        return Err(XaiAudioError::Provider {
            operation,
            status: response.status,
            request_id,
            body: Box::new(parse_error_body(&response.body)),
        });
    }
    let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
        invalid_response(
            operation,
            "response body is not JSON",
            request_id.clone(),
            Value::String(String::from_utf8_lossy(&response.body).into_owned()),
        )
    })?;
    let voice = decode_voice(native, scope, request_id, operation)?;
    if expected_voice_id.is_some_and(|expected| expected != voice.reference.voice_id) {
        return Err(invalid_response(
            operation,
            "response voice_id does not match the requested resource",
            voice.request_id.clone(),
            voice.native,
        ));
    }
    Ok(voice)
}

async fn decode_voice_list_response(
    response: StreamResponse,
    scope: XaiAudioScope,
    previous_token: Option<&str>,
) -> Result<XaiCustomVoiceList, XaiAudioError> {
    let request_id = response_id(&response.headers);
    if !(200..300).contains(&response.status) {
        let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
            .await
            .map_err(XaiAudioError::Llm)?;
        return Err(XaiAudioError::Provider {
            operation: "list-custom-voices",
            status: response.status,
            request_id,
            body: Box::new(parse_error_body(&response.body)),
        });
    }
    let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
        .await
        .map_err(XaiAudioError::Llm)?;
    let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
        invalid_response(
            "list-custom-voices",
            "response body is not JSON",
            request_id.clone(),
            Value::Null,
        )
    })?;
    let Some(voice_values) = native.get("voices").and_then(Value::as_array) else {
        return Err(invalid_response(
            "list-custom-voices",
            "response has no voices array",
            request_id,
            native,
        ));
    };
    let token = match native.get("pagination_token") {
        Some(Value::Null) => None,
        Some(Value::String(token)) if !token.trim().is_empty() => Some(token.clone()),
        Some(Value::String(_)) => {
            return Err(invalid_response(
                "list-custom-voices",
                "pagination_token must be null or a nonempty string",
                request_id,
                native,
            ));
        }
        _ => {
            return Err(invalid_response(
                "list-custom-voices",
                "response has no valid pagination_token",
                request_id,
                native,
            ));
        }
    };
    if token
        .as_deref()
        .is_some_and(|token| Some(token) == previous_token)
    {
        return Err(invalid_response(
            "list-custom-voices",
            "pagination_token did not advance from the requested cursor",
            request_id,
            native,
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(voice_values.len());
    let mut voices = Vec::with_capacity(voice_values.len());
    for value in voice_values {
        let voice = decode_voice(
            value.clone(),
            scope.clone(),
            request_id.clone(),
            "list-custom-voices",
        )?;
        if !seen.insert(voice.reference.voice_id.clone()) {
            return Err(invalid_response(
                "list-custom-voices",
                &format!(
                    "voices contains duplicate voice_id `{}`",
                    voice.reference.voice_id
                ),
                request_id,
                native,
            ));
        }
        voices.push(voice);
    }
    Ok(XaiCustomVoiceList {
        scope: scope.clone(),
        voices,
        next_page: token.map(|token| XaiCustomVoiceCursor { scope, token }),
        native,
        request_id,
    })
}

async fn decode_delete_response(
    response: StreamResponse,
    reference: XaiCustomVoiceRef,
) -> Result<XaiCustomVoiceDeleteReceipt, XaiAudioError> {
    let operation = "delete-custom-voice";
    let request_id = response_id(&response.headers);
    if !(200..300).contains(&response.status) {
        let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
            .await
            .map_err(XaiAudioError::Llm)?;
        return Err(XaiAudioError::Provider {
            operation,
            status: response.status,
            request_id,
            body: Box::new(parse_error_body(&response.body)),
        });
    }
    let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
        .await
        .map_err(|source| map_outcome(operation, source))?;
    let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
        outcome_unknown_response(
            operation,
            "successful DELETE response body is not JSON",
            request_id.clone(),
            Value::String(String::from_utf8_lossy(&response.body).into_owned()),
        )
    })?;
    if native.get("deleted") != Some(&Value::Bool(true)) {
        return Err(outcome_unknown_response(
            operation,
            "successful DELETE response does not confirm deleted=true",
            request_id,
            native,
        ));
    }
    Ok(XaiCustomVoiceDeleteReceipt {
        reference,
        deleted: true,
        native,
        request_id,
    })
}

fn decode_voice(
    native: Value,
    scope: XaiAudioScope,
    request_id: Option<String>,
    operation: &'static str,
) -> Result<XaiCustomVoice, XaiAudioError> {
    let Some(voice_id) = native.get("voice_id").and_then(Value::as_str) else {
        return Err(invalid_response(
            operation,
            "voice object has no string voice_id",
            request_id,
            native,
        ));
    };
    if validate_voice_id(voice_id).is_err() {
        return Err(invalid_response(
            operation,
            "voice object has an invalid voice_id",
            request_id,
            native,
        ));
    }
    let name = optional_string(&native, "name", operation, request_id.clone())?;
    let description = optional_string(&native, "description", operation, request_id.clone())?;
    let gender = optional_string(&native, "gender", operation, request_id.clone())?;
    let accent = optional_string(&native, "accent", operation, request_id.clone())?;
    let age = optional_string(&native, "age", operation, request_id.clone())?;
    let language = optional_string(&native, "language", operation, request_id.clone())?;
    let use_case = optional_string(&native, "use_case", operation, request_id.clone())?;
    let tone = optional_string(&native, "tone", operation, request_id.clone())?;
    let created_at = optional_string(&native, "created_at", operation, request_id.clone())?;
    let reference = XaiCustomVoiceRef {
        scope,
        voice_id: voice_id.to_owned(),
    };
    Ok(XaiCustomVoice {
        reference,
        name,
        description,
        gender,
        accent,
        age,
        language,
        use_case,
        tone,
        created_at,
        native,
        request_id,
    })
}

fn optional_string(
    object: &Value,
    field: &'static str,
    operation: &'static str,
    request_id: Option<String>,
) -> Result<Option<String>, XaiAudioError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid_response(
            operation,
            &format!("voice object field `{field}` must be a string or null"),
            request_id,
            object.clone(),
        )),
    }
}

fn uncertain_mutation_result<T>(
    operation: &'static str,
    result: Result<T, XaiAudioError>,
) -> Result<T, XaiAudioError> {
    result.map_err(|error| match error {
        XaiAudioError::InvalidResponse {
            message,
            request_id,
            native,
            ..
        } => outcome_unknown_response(
            operation,
            &format!("provider returned success but acknowledgement was unusable: {message}"),
            request_id,
            *native,
        ),
        other => other,
    })
}

fn outcome_unknown_response(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> XaiAudioError {
    XaiAudioError::OutcomeUnknown {
        operation,
        source: LlmError::ProviderInternal {
            message: message.to_owned(),
        },
        response: Some(Box::new(XaiAudioUnknownResponse {
            request_id,
            native: Box::new(native),
        })),
    }
}

trait CustomVoiceEnumValue {
    fn into_form_value(self) -> &'static str;
}

impl CustomVoiceEnumValue for XaiCustomVoiceGender {
    fn into_form_value(self) -> &'static str {
        match self {
            Self::Male => "male",
            Self::Female => "female",
            Self::Neutral => "neutral",
        }
    }
}

impl CustomVoiceEnumValue for XaiCustomVoiceAge {
    fn into_form_value(self) -> &'static str {
        match self {
            Self::Young => "young",
            Self::MiddleAged => "middle-aged",
            Self::Old => "old",
        }
    }
}

impl CustomVoiceEnumValue for XaiCustomVoiceUseCase {
    fn into_form_value(self) -> &'static str {
        match self {
            Self::Conversational => "conversational",
            Self::Narration => "narration",
            Self::Characters => "characters",
            Self::Educational => "educational",
            Self::Advertisement => "advertisement",
            Self::SocialMedia => "social_media",
            Self::Entertainment => "entertainment",
        }
    }
}

impl CustomVoiceEnumValue for XaiCustomVoiceTone {
    fn into_form_value(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Casual => "casual",
            Self::Professional => "professional",
            Self::Friendly => "friendly",
            Self::Authoritative => "authoritative",
            Self::Expressive => "expressive",
            Self::Calm => "calm",
        }
    }
}
