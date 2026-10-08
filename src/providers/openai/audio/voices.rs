//! OpenAI custom voice and consent resource lifecycle.
//!
//! This module deliberately exposes only documented resource operations. The
//! API currently documents creation for voices and CRUD for voice consents.
use super::{
    speech::{validate_voice_id, OPENAI_SPEECH_ENDPOINT},
    AudioInput,
};
use crate::{
    client::RequestOptions,
    files::{
        append_field, multipart_boundary, provider_file_endpoint_fingerprint, sanitize_filename,
    },
    protocol::{LlmError, ServiceSetting},
    runtime::Deadline,
    runtime::{ClientSnapshot, ClientSource},
    transport::{HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest},
};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, StreamExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

const MAX_RECORDING_BYTES: u64 = 10 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const RECORDING_MEDIA_TYPES: &[&str] = &[
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/aac",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
];

/// A custom voice creation and consent-resource API handle.
#[derive(Clone, Copy)]
pub struct VoiceResourceService<'a> {
    source: ClientSource<'a>,
}

impl<'a> VoiceResourceService<'a> {
    pub(super) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    /// Import an already approved custom voice ID into the selected profile
    /// and caller-declared account scope. This is a local reference operation;
    /// it does not call OpenAI or prove that the voice ID exists.
    pub fn import_approved_voice(
        self,
        voice_id: impl Into<String>,
        options: &RequestOptions,
    ) -> Result<CustomVoiceRef, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        let scope = custom_voice_scope(&snapshot, profile_name, options)?;
        custom_voice_ref(voice_id.into(), scope)
    }

    /// Fetch the current consent-phrase catalog.
    ///
    /// OpenAI documents the endpoint but not a stable response schema, so this
    /// returns the original JSON value without projecting it into guessed fields.
    pub async fn list_consent_phrases(
        self,
        options: &RequestOptions,
    ) -> Result<Value, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        require_account_scope(options)?;
        let snapshot = self.source.pin()?;
        let endpoint = audio_resource_endpoint(&snapshot, profile_name, "consent_phrases")?;
        let response = execute_json(&snapshot, endpoint, "GET", None, options, None, false).await?;
        decode_json_value(&response)
    }

    /// Upload a speaker's consent recording.
    pub async fn create_consent(
        self,
        request: &VoiceConsentCreateRequest,
        recording: AudioInput,
        options: &RequestOptions,
    ) -> Result<VoiceConsent, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        validate_scalar(&request.name, "consent name")?;
        validate_scalar(&request.language, "consent language")?;
        validate_recording(&recording)?;
        require_account_scope(options)?;

        let snapshot = self.source.pin()?;
        let endpoint = audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        let scope = consent_scope(profile_name, &endpoint, options)?;
        let fields = vec![
            ("name", request.name.as_str()),
            ("language", request.language.as_str()),
        ];
        let response = execute_upload(
            &snapshot,
            endpoint,
            recording,
            "recording",
            &fields,
            options,
            "creating a voice consent",
        )
        .await?;
        let wire: VoiceConsentResponse = decode_response(&response, "voice consent")?;
        consent_from_wire(wire, scope, None)
    }

    /// Create a voice consent list page. `after` and `limit` follow the
    /// documented cursor parameters; the caller owns pagination.
    pub async fn list_consents(
        self,
        query: &VoiceConsentListQuery,
        options: &RequestOptions,
    ) -> Result<VoiceConsentPage, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        if query.limit.is_some_and(|limit| !(1..=100).contains(&limit)) {
            return Err(invalid("consent list limit must be between 1 and 100"));
        }
        if let Some(after) = &query.after {
            validate_scalar(after, "consent cursor")?;
        }
        require_account_scope(options)?;

        let snapshot = self.source.pin()?;
        let collection_endpoint =
            audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        let endpoint = append_list_query(&collection_endpoint, query)?;
        let response = execute_json(&snapshot, endpoint, "GET", None, options, None, false).await?;
        let page: VoiceConsentPageResponse = decode_response(&response, "voice consent list")?;
        if page.object != "list" {
            return Err(invalid_response(
                "voice consent list has an unexpected object type",
                response,
            ));
        }
        let scope = consent_scope(profile_name, &collection_endpoint, options)?;
        let data = page
            .data
            .into_iter()
            .map(|consent| consent_from_wire(consent, scope.clone(), None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VoiceConsentPage {
            data,
            first_id: page.first_id,
            has_more: page.has_more,
            last_id: page.last_id,
            object: page.object,
        })
    }

    /// Retrieve consent metadata by ID.
    pub async fn get_consent(
        self,
        consent: &VoiceConsentRef,
        options: &RequestOptions,
    ) -> Result<VoiceConsent, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        let collection_endpoint =
            audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        validate_consent_reference(consent, profile_name, &collection_endpoint, options)?;
        let endpoint = append_path_segment(&collection_endpoint, consent.id())?;
        let response = execute_json(&snapshot, endpoint, "GET", None, options, None, false).await?;
        let wire: VoiceConsentResponse = decode_response(&response, "voice consent")?;
        let scope = consent_scope(profile_name, &collection_endpoint, options)?;
        consent_from_wire(wire, scope, Some(consent.id()))
    }

    /// Update the documented consent label only; the uploaded audio is not replaced.
    pub async fn update_consent(
        self,
        consent: &VoiceConsentRef,
        request: &VoiceConsentUpdateRequest,
        options: &RequestOptions,
    ) -> Result<VoiceConsent, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        validate_scalar(&request.name, "consent name")?;
        let snapshot = self.source.pin()?;
        let collection_endpoint =
            audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        validate_consent_reference(consent, profile_name, &collection_endpoint, options)?;
        let endpoint = append_path_segment(&collection_endpoint, consent.id())?;
        let body = serde_json::to_vec(request).map_err(|_| invalid("could not encode update"))?;
        let response = execute_json(
            &snapshot,
            endpoint,
            "POST",
            Some(body),
            options,
            Some("updating a voice consent"),
            true,
        )
        .await?;
        let wire: VoiceConsentResponse = decode_response(&response, "voice consent")?;
        consent_from_wire(
            wire,
            consent_scope(profile_name, &collection_endpoint, options)?,
            Some(consent.id()),
        )
    }

    /// Delete consent metadata and receive OpenAI's deletion confirmation.
    pub async fn delete_consent(
        self,
        consent: &VoiceConsentRef,
        options: &RequestOptions,
    ) -> Result<VoiceConsentDeleteReceipt, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        let collection_endpoint =
            audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        validate_consent_reference(consent, profile_name, &collection_endpoint, options)?;
        let endpoint = append_path_segment(&collection_endpoint, consent.id())?;
        let response = execute_json(
            &snapshot,
            endpoint,
            "DELETE",
            None,
            options,
            Some("deleting a voice consent"),
            true,
        )
        .await?;
        let receipt: VoiceConsentDeleteReceipt = decode_response(&response, "delete receipt")?;
        if receipt.id != consent.id() || !receipt.deleted || receipt.object != "audio.voice_consent"
        {
            return Err(invalid_response(
                "voice consent deletion was not confirmed for the requested ID",
                response,
            ));
        }
        Ok(receipt)
    }

    /// Create a custom voice from the sample and a previously uploaded consent.
    pub async fn create_voice(
        self,
        request: &CustomVoiceCreateRequest,
        audio_sample: AudioInput,
        options: &RequestOptions,
    ) -> Result<CustomVoice, VoiceResourceError> {
        let profile_name = self.source.profile_name()?;
        validate_scalar(&request.name, "voice name")?;
        validate_recording(&audio_sample)?;

        let snapshot = self.source.pin()?;
        let voice_scope = custom_voice_scope(&snapshot, profile_name, options)?;
        let consent_endpoint = audio_resource_endpoint(&snapshot, profile_name, "voice_consents")?;
        validate_consent_reference(&request.consent, profile_name, &consent_endpoint, options)?;
        let endpoint = audio_resource_endpoint(&snapshot, profile_name, "voices")?;
        let fields = vec![
            ("name", request.name.as_str()),
            ("consent", request.consent.id()),
        ];
        let response = execute_upload(
            &snapshot,
            endpoint,
            audio_sample,
            "audio_sample",
            &fields,
            options,
            "creating a custom voice",
        )
        .await?;
        let wire: CustomVoiceResponse = decode_response(&response, "custom voice")
            .map_err(custom_voice_creation_outcome_unknown)?;
        custom_voice_from_wire(wire, voice_scope).map_err(custom_voice_creation_outcome_unknown)
    }
}

/// Fields for `POST /audio/voice_consents`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceConsentCreateRequest {
    pub name: String,
    /// BCP 47 tag matching the phrase the speaker read, such as `en-US`.
    pub language: String,
}

/// Cursor options for `GET /audio/voice_consents`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceConsentListQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u8>,
}

/// The metadata returned for a stored consent recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceConsent {
    pub id: String,
    pub created_at: u64,
    pub language: String,
    pub name: String,
    pub object: String,
    reference: VoiceConsentRef,
}

impl VoiceConsent {
    /// Return the opaque identity bound to the profile, endpoint, and account
    /// scope that produced this consent metadata.
    pub fn reference(&self) -> &VoiceConsentRef {
        &self.reference
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct VoiceConsentResponse {
    id: String,
    created_at: u64,
    language: String,
    name: String,
    object: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct VoiceConsentPageResponse {
    data: Vec<VoiceConsentResponse>,
    #[serde(default)]
    first_id: Option<String>,
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
    object: String,
}

/// A page from the documented voice consent list endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceConsentPage {
    pub data: Vec<VoiceConsent>,
    pub first_id: Option<String>,
    pub has_more: bool,
    pub last_id: Option<String>,
    pub object: String,
}

/// Opaque consent identity tied to the OpenAI profile, consent endpoint, and
/// stable caller-provided project/account scope used when it was returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceConsentRef {
    consent_id: String,
    profile_name: String,
    endpoint_fingerprint: String,
    account_scope: String,
}

impl VoiceConsentRef {
    pub fn id(&self) -> &str {
        &self.consent_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }
}

/// Fields for the documented metadata-only consent update endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceConsentUpdateRequest {
    pub name: String,
}

/// Confirmation returned by `DELETE /audio/voice_consents/{consent_id}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceConsentDeleteReceipt {
    pub id: String,
    pub deleted: bool,
    pub object: String,
}

/// Fields for `POST /audio/voices`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomVoiceCreateRequest {
    pub name: String,
    pub consent: VoiceConsentRef,
}

/// Metadata and a scoped reference for a successfully created custom voice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomVoice {
    pub id: String,
    pub created_at: u64,
    pub name: String,
    pub object: String,
    reference: CustomVoiceRef,
}

impl CustomVoice {
    /// Return the opaque handle required to select this voice for synthesis.
    pub fn reference(&self) -> &CustomVoiceRef {
        &self.reference
    }
}

/// Opaque serializable custom voice identity bound to its OpenAI provider,
/// profile, Speech endpoint, and caller-declared account scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomVoiceRef {
    voice_id: String,
    provider_id: String,
    profile_name: String,
    endpoint_fingerprint: String,
    account_scope: String,
}

impl CustomVoiceRef {
    pub fn id(&self) -> &str {
        &self.voice_id
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub(super) fn validate_for(
        &self,
        provider_id: &str,
        profile_name: &str,
        speech_endpoint: &str,
        account_scope: Option<&str>,
    ) -> Result<(), LlmError> {
        validate_voice_id(&self.voice_id)?;
        let account_scope = account_scope.ok_or_else(|| LlmError::InvalidRequest {
            message: "custom voice use requires RequestOptions.account_scope".into(),
        })?;
        validate_account_scope(account_scope)?;
        if self.provider_id != provider_id
            || self.profile_name != profile_name
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(speech_endpoint)
            || self.account_scope != account_scope
        {
            return Err(LlmError::InvalidRequest {
                message: "custom voice reference belongs to a different provider, profile, speech endpoint, or account scope".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct CustomVoiceResponse {
    id: String,
    created_at: u64,
    name: String,
    object: String,
}

/// Failure from an OpenAI custom voice lifecycle operation.
#[derive(Debug, thiserror::Error)]
pub enum VoiceResourceError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("OpenAI audio resource request returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid OpenAI audio resource response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("OpenAI audio {operation} outcome is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
}

async fn execute_json(
    client: &ClientSnapshot,
    endpoint: String,
    method: &str,
    body: Option<Vec<u8>>,
    options: &RequestOptions,
    mutation: Option<&'static str>,
    is_mutation: bool,
) -> Result<HttpResponse, VoiceResourceError> {
    let secret = options
        .credential
        .as_ref()
        .ok_or_else(|| LlmError::Authentication {
            message: "OpenAI voice resources require a bearer credential".into(),
        })?;
    let deadline = request_deadline(options);
    let mut headers = vec![(
        "authorization".into(),
        format!("Bearer {}", secret.expose_secret()),
    )];
    let request_body = if let Some(body) = body {
        headers.push(("content-type".into(), "application/json".into()));
        Bytes::from(body)
    } else {
        Bytes::new()
    };
    let request = HttpRequest {
        http1_header_layout: None,
        method: method.into(),
        url: endpoint,
        headers,
        body: request_body,
        timeout: deadline.remaining()?,
    };
    let response = HttpExecutor::new(client.runtime.http.as_ref())
        .with_deadline(deadline)
        .execute_bounded(request, MAX_RESPONSE_BYTES)
        .await
        .map_err(|source| map_execution_error(source, mutation, is_mutation))?;
    ensure_success(&response)?;
    Ok(response)
}

async fn execute_upload(
    client: &ClientSnapshot,
    endpoint: String,
    input: AudioInput,
    file_field: &str,
    fields: &[(&str, &str)],
    options: &RequestOptions,
    operation: &'static str,
) -> Result<HttpResponse, VoiceResourceError> {
    let secret = options
        .credential
        .as_ref()
        .ok_or_else(|| LlmError::Authentication {
            message: "OpenAI voice resources require a bearer credential".into(),
        })?;
    let boundary = multipart_boundary();
    let (prefix, suffix) = multipart_parts(&boundary, fields, file_field, &input);
    let content_length = input
        .size_bytes
        .checked_add(prefix.len() as u64)
        .and_then(|length| length.checked_add(suffix.len() as u64))
        .ok_or_else(|| invalid("voice upload multipart size overflows"))?;
    let deadline = request_deadline(options);
    let request = HttpStreamRequest {
        http1_header_layout: None,
        method: "POST".into(),
        url: endpoint,
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
    let response = HttpExecutor::new(client.runtime.http.as_ref())
        .with_deadline(deadline)
        .execute_stream_bounded(request, MAX_RESPONSE_BYTES)
        .await
        .map_err(|source| map_execution_error(source, Some(operation), true))?;
    ensure_success(&response)?;
    Ok(response)
}

fn audio_resource_endpoint(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    resource: &str,
) -> Result<String, VoiceResourceError> {
    let profile = snapshot
        .native_profile(profile_name)
        .ok_or_else(|| invalid("unknown audio profile"))?;
    if !profile.supports_region(snapshot.region()) {
        return Err(invalid("audio profile is unavailable in this region"));
    }
    let ServiceSetting::Enabled(route) = &profile.audio else {
        return Err(LlmError::UnsupportedCapability {
            message: "profile has no audio route".into(),
        }
        .into());
    };
    super::validate_route(profile, route)?;

    let mut url = url::Url::parse(&route.transcriptions_endpoint)
        .map_err(|_| invalid("invalid audio transcription endpoint"))?;
    let prefix = url
        .path()
        .strip_suffix("/transcriptions")
        .ok_or_else(|| invalid("audio endpoint has no documented resource root"))?;
    url.set_path(&format!("{prefix}/{resource}"));
    Ok(url.into())
}

fn append_list_query(
    endpoint: &str,
    query: &VoiceConsentListQuery,
) -> Result<String, VoiceResourceError> {
    if query.after.is_none() && query.limit.is_none() {
        return Ok(endpoint.to_owned());
    }
    let mut url = url::Url::parse(endpoint).map_err(|_| invalid("invalid consent endpoint"))?;
    {
        let mut pairs = url.query_pairs_mut();
        if let Some(after) = &query.after {
            pairs.append_pair("after", after);
        }
        if let Some(limit) = query.limit {
            pairs.append_pair("limit", &limit.to_string());
        }
    }
    Ok(url.into())
}

fn append_path_segment(endpoint: &str, id: &str) -> Result<String, VoiceResourceError> {
    let mut url = url::Url::parse(endpoint).map_err(|_| invalid("invalid consent endpoint"))?;
    url.path_segments_mut()
        .map_err(|_| invalid("consent endpoint cannot contain a resource ID"))?
        .push(id);
    Ok(url.into())
}

fn multipart_parts(
    boundary: &str,
    fields: &[(&str, &str)],
    file_field: &str,
    input: &AudioInput,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    for (name, value) in fields {
        append_field(&mut prefix, boundary, name, value);
    }
    let filename = sanitize_filename(&input.filename);
    prefix.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            input.media_type
        )
        .as_bytes(),
    );
    (
        prefix.freeze(),
        Bytes::from(format!("\r\n--{boundary}--\r\n")),
    )
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
                            return Err(llm_invalid(
                                "voice recording stream exceeds its declared size",
                            ));
                        }
                        state.remaining -= chunk.len() as u64;
                        return Ok(Some((chunk, state)));
                    }
                    Some(Err(error)) => return Err(error),
                    None => {
                        state.done = true;
                        if state.remaining != 0 {
                            return Err(llm_invalid(
                                "voice recording stream is shorter than its declared size",
                            ));
                        }
                    }
                }
            }
            Ok(state.suffix.take().map(|suffix| (suffix, state)))
        },
    )
    .boxed()
}

struct MultipartState {
    prefix: Option<Bytes>,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    remaining: u64,
    done: bool,
    suffix: Option<Bytes>,
}

fn validate_recording(input: &AudioInput) -> Result<(), VoiceResourceError> {
    if input.size_bytes == 0 || input.size_bytes > MAX_RECORDING_BYTES {
        return Err(invalid("voice recordings must contain 1–10485760 bytes"));
    }
    if !RECORDING_MEDIA_TYPES.contains(&input.media_type.as_str()) {
        return Err(invalid("voice recording MIME type is not supported"));
    }
    if input.filename.trim().is_empty() {
        return Err(invalid("voice recording filename is required"));
    }
    Ok(())
}

fn validate_scalar(value: &str, label: &str) -> Result<(), VoiceResourceError> {
    if value.trim().is_empty()
        || value
            .chars()
            .any(|ch| ch == '\r' || ch == '\n' || ch == '\0')
    {
        return Err(invalid(&format!(
            "{label} must be non-empty and contain no line breaks"
        )));
    }
    Ok(())
}

fn require_account_scope(options: &RequestOptions) -> Result<&str, VoiceResourceError> {
    let scope = options.account_scope.as_deref().ok_or_else(|| {
        invalid("OpenAI voice resource calls require RequestOptions.account_scope")
    })?;
    validate_account_scope(scope)?;
    Ok(scope)
}

pub(super) fn validate_account_scope(scope: &str) -> Result<(), LlmError> {
    if scope.trim().is_empty()
        || scope != scope.trim()
        || scope
            .chars()
            .any(|ch| ch == '\r' || ch == '\n' || ch == '\0')
    {
        return Err(LlmError::InvalidRequest {
            message: "account scope must be non-empty, trimmed, and contain no line breaks".into(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConsentScope {
    profile_name: String,
    endpoint_fingerprint: String,
    account_scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomVoiceScope {
    provider_id: String,
    profile_name: String,
    endpoint_fingerprint: String,
    account_scope: String,
}

fn custom_voice_scope(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    options: &RequestOptions,
) -> Result<CustomVoiceScope, VoiceResourceError> {
    validate_scalar(profile_name, "profile name")?;
    let profile = snapshot
        .native_profile(profile_name)
        .ok_or_else(|| invalid("unknown audio profile"))?;
    if !profile.supports_region(snapshot.region()) {
        return Err(invalid("audio profile is unavailable in this region"));
    }
    let ServiceSetting::Enabled(route) = &profile.audio else {
        return Err(LlmError::UnsupportedCapability {
            message: "profile has no audio route".into(),
        }
        .into());
    };
    super::validate_route(profile, route)?;
    let speech_endpoint =
        route
            .speech_endpoint
            .as_deref()
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: "profile has no speech route".into(),
            })?;
    if profile.provider_id.as_str() != "openai" || speech_endpoint != OPENAI_SPEECH_ENDPOINT {
        return Err(LlmError::UnsupportedCapability {
            message: "custom voice references require the official OpenAI Speech route".into(),
        }
        .into());
    }
    Ok(CustomVoiceScope {
        provider_id: profile.provider_id.as_str().to_owned(),
        profile_name: profile_name.to_owned(),
        // This ID is later consumed by Speech synthesis; do not bind it to
        // the separate /audio/voices creation endpoint.
        endpoint_fingerprint: provider_file_endpoint_fingerprint(speech_endpoint),
        account_scope: require_account_scope(options)?.to_owned(),
    })
}

fn custom_voice_ref(
    voice_id: String,
    scope: CustomVoiceScope,
) -> Result<CustomVoiceRef, VoiceResourceError> {
    validate_voice_id(&voice_id)?;
    Ok(CustomVoiceRef {
        voice_id,
        provider_id: scope.provider_id,
        profile_name: scope.profile_name,
        endpoint_fingerprint: scope.endpoint_fingerprint,
        account_scope: scope.account_scope,
    })
}

fn custom_voice_from_wire(
    wire: CustomVoiceResponse,
    scope: CustomVoiceScope,
) -> Result<CustomVoice, VoiceResourceError> {
    if wire.object != "audio.voice" || wire.name.trim().is_empty() {
        return Err(invalid("custom voice response has an invalid identity"));
    }
    let reference = custom_voice_ref(wire.id.clone(), scope)?;
    Ok(CustomVoice {
        id: wire.id,
        created_at: wire.created_at,
        name: wire.name,
        object: wire.object,
        reference,
    })
}

fn consent_scope(
    profile_name: &str,
    consent_endpoint: &str,
    options: &RequestOptions,
) -> Result<ConsentScope, VoiceResourceError> {
    validate_scalar(profile_name, "profile name")?;
    Ok(ConsentScope {
        profile_name: profile_name.to_owned(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(consent_endpoint),
        account_scope: require_account_scope(options)?.to_owned(),
    })
}

fn validate_consent_reference(
    consent: &VoiceConsentRef,
    profile_name: &str,
    consent_endpoint: &str,
    options: &RequestOptions,
) -> Result<(), VoiceResourceError> {
    validate_scalar(consent.id(), "consent ID")?;
    let expected = consent_scope(profile_name, consent_endpoint, options)?;
    if consent.profile_name != expected.profile_name
        || consent.endpoint_fingerprint != expected.endpoint_fingerprint
        || consent.account_scope != expected.account_scope
    {
        return Err(invalid(
            "voice consent reference belongs to a different profile, endpoint, or account scope",
        ));
    }
    Ok(())
}

fn consent_from_wire(
    wire: VoiceConsentResponse,
    scope: ConsentScope,
    expected_id: Option<&str>,
) -> Result<VoiceConsent, VoiceResourceError> {
    if wire.id.trim().is_empty()
        || wire.name.trim().is_empty()
        || wire.language.trim().is_empty()
        || wire.object != "audio.voice_consent"
        || expected_id.is_some_and(|id| wire.id != id)
    {
        return Err(invalid("voice consent response has an invalid identity"));
    }
    let reference = VoiceConsentRef {
        consent_id: wire.id.clone(),
        profile_name: scope.profile_name,
        endpoint_fingerprint: scope.endpoint_fingerprint,
        account_scope: scope.account_scope,
    };
    Ok(VoiceConsent {
        id: wire.id,
        created_at: wire.created_at,
        language: wire.language,
        name: wire.name,
        object: wire.object,
        reference,
    })
}

fn request_deadline(options: &RequestOptions) -> Deadline {
    Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ))
}

fn map_execution_error(
    source: LlmError,
    operation: Option<&'static str>,
    is_mutation: bool,
) -> VoiceResourceError {
    if is_mutation
        && matches!(
            source,
            LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. }
        )
    {
        return VoiceResourceError::OutcomeUnknown {
            operation: operation.unwrap_or("resource mutation"),
            source,
        };
    }
    VoiceResourceError::Llm(source)
}

fn ensure_success(response: &HttpResponse) -> Result<(), VoiceResourceError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let body = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    Err(VoiceResourceError::Provider {
        status: response.status,
        request_id: request_id(response),
        body,
    })
}

fn decode_json_value(response: &HttpResponse) -> Result<Value, VoiceResourceError> {
    serde_json::from_slice(&response.body)
        .map_err(|_| invalid_response("consent phrase response was not JSON", response.clone()))
}

fn decode_response<T: DeserializeOwned>(
    response: &HttpResponse,
    label: &str,
) -> Result<T, VoiceResourceError> {
    let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
        invalid_response(&format!("{label} response was not JSON"), response.clone())
    })?;
    serde_json::from_value(native.clone()).map_err(|_| VoiceResourceError::InvalidResponse {
        message: format!("{label} response did not match the documented shape"),
        native,
    })
}

fn invalid_response(message: &str, response: HttpResponse) -> VoiceResourceError {
    let native = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    VoiceResourceError::InvalidResponse {
        message: message.into(),
        native,
    }
}

fn request_id(response: &HttpResponse) -> Option<String> {
    response
        .header("x-request-id")
        .or_else(|| response.header("openai-request-id"))
        .or_else(|| response.header("request-id"))
        .map(str::to_owned)
}

fn invalid(message: &str) -> VoiceResourceError {
    llm_invalid(message).into()
}

fn custom_voice_creation_outcome_unknown(error: VoiceResourceError) -> VoiceResourceError {
    VoiceResourceError::OutcomeUnknown {
        operation: "creating a custom voice",
        source: LlmError::ProviderInternal {
            message: format!("the successful response could not be validated: {error}"),
        },
    }
}

fn llm_invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
