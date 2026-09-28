//! Gemini Developer API custom and prebuilt voice catalog operations.
//!
//! The Voices API is a separate REST resource from Interactions TTS and Live.
//! Stored custom-voice references bind the Google profile, account, and exact
//! endpoint that produced them. Replication audio and consent recordings are
//! encoded only for the one create request and are never retained by the
//! service.

use super::*;
use std::fmt;

const VOICES_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/voices";
const VOICES_PATH: &str = "/v1beta/voices";
const MAX_VOICE_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const MAX_VOICE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_VOICE_PAGE_SIZE: u32 = 1_000;

/// Google Gemini Developer API Voices collection endpoint.
pub const GEMINI_VOICES_ENDPOINT: &str = VOICES_ENDPOINT;

/// Profile and account identity for one Google Voices endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiVoicesScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint: String,
    endpoint_fingerprint: String,
}

impl GeminiVoicesScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        endpoint: impl AsRef<str>,
    ) -> Result<Self, GeminiVoicesError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !voices_identity(&profile_name) || !voices_identity(&account_scope) {
            return Err(voices_invalid(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        let endpoint = normalize_voices_endpoint(endpoint.as_ref())?;
        let endpoint_fingerprint = provider_file_endpoint_fingerprint(&endpoint);
        Ok(Self {
            provider_id: ProviderId::from("google"),
            profile_name,
            account_scope,
            endpoint,
            endpoint_fingerprint,
        })
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    /// Bind a Google-managed stored voice ID from this account and endpoint.
    pub fn voice_ref(
        &self,
        voice_id: impl Into<String>,
    ) -> Result<GeminiVoiceRef, GeminiVoicesError> {
        GeminiVoiceRef::from_scope(self, voice_id)
    }

    fn validate(&self) -> Result<(), GeminiVoicesError> {
        let endpoint = normalize_voices_endpoint(&self.endpoint)?;
        if self.provider_id.as_str() != "google"
            || !voices_identity(&self.profile_name)
            || !voices_identity(&self.account_scope)
            || endpoint != self.endpoint
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.endpoint)
        {
            return Err(voices_invalid("Gemini Voices scope identity is invalid"));
        }
        Ok(())
    }
}

/// Opaque identity for a Google-managed custom voice (`store = true`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiVoiceRef {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
    voice_id: String,
}

impl GeminiVoiceRef {
    pub fn from_scope(
        scope: &GeminiVoicesScope,
        voice_id: impl Into<String>,
    ) -> Result<Self, GeminiVoicesError> {
        scope.validate()?;
        let voice_id = voice_id.into();
        if !valid_stored_voice_id(&voice_id) {
            return Err(voices_invalid(
                "voice ID must be a path-safe Google stored voice ID beginning with voice_",
            ));
        }
        Ok(Self {
            provider_id: scope.provider_id.clone(),
            profile_name: scope.profile_name.clone(),
            account_scope: scope.account_scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint.clone(),
            voice_id,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
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

    pub fn voice_id(&self) -> &str {
        &self.voice_id
    }
}

/// Voice type as returned by Google's catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum GeminiVoiceType {
    Prompted,
    Replicated,
    Prebuilt,
    Other(String),
}

impl From<String> for GeminiVoiceType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "prompted" => Self::Prompted,
            "replicated" => Self::Replicated,
            "prebuilt" => Self::Prebuilt,
            _ => Self::Other(value),
        }
    }
}

impl From<GeminiVoiceType> for String {
    fn from(value: GeminiVoiceType) -> Self {
        match value {
            GeminiVoiceType::Prompted => "prompted".into(),
            GeminiVoiceType::Replicated => "replicated".into(),
            GeminiVoiceType::Prebuilt => "prebuilt".into(),
            GeminiVoiceType::Other(value) => value,
        }
    }
}

/// Pitch classification accepted by the Voices API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeminiVoicePitch {
    Low,
    Medium,
    High,
}

impl GeminiVoicePitch {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// One raw audio recording with the IANA MIME type sent to Google.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiVoiceAudioData {
    data: Bytes,
    mime_type: String,
}

impl GeminiVoiceAudioData {
    pub fn new(data: impl Into<Bytes>, mime_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn mime_type(&self) -> &str {
        &self.mime_type
    }

    fn validate(&self, label: &str) -> Result<(), GeminiVoicesError> {
        if self.data.is_empty() {
            return Err(voices_invalid(format!("{label} audio must not be empty")));
        }
        if self.mime_type.trim().is_empty() || self.mime_type.chars().any(char::is_control) {
            return Err(voices_invalid(format!(
                "{label} audio MIME type must be a non-empty IANA media type"
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for GeminiVoiceAudioData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiVoiceAudioData")
            .field("data", &"<redacted audio bytes>")
            .field("size_bytes", &self.data.len())
            .field("mime_type", &self.mime_type)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct GeminiVoiceMetadata {
    display_name: Option<String>,
    description: Option<String>,
    accent: Option<String>,
    context: Option<String>,
    gender: Option<String>,
    language_code: Option<String>,
    model: Option<String>,
    persona: Option<String>,
    pitch: Option<GeminiVoicePitch>,
    region_code: Option<String>,
}

impl GeminiVoiceMetadata {
    fn insert_into(&self, object: &mut serde_json::Map<String, Value>) {
        insert_optional(object, "display_name", self.display_name.as_deref());
        insert_optional(object, "description", self.description.as_deref());
        insert_optional(object, "accent", self.accent.as_deref());
        insert_optional(object, "context", self.context.as_deref());
        insert_optional(object, "gender", self.gender.as_deref());
        insert_optional(object, "language_code", self.language_code.as_deref());
        insert_optional(object, "model", self.model.as_deref());
        insert_optional(object, "persona", self.persona.as_deref());
        if let Some(pitch) = self.pitch {
            object.insert("pitch".into(), json!(pitch.as_str()));
        }
        insert_optional(object, "region_code", self.region_code.as_deref());
    }

    fn validate(&self) -> Result<(), GeminiVoicesError> {
        for value in [
            self.display_name.as_deref(),
            self.description.as_deref(),
            self.accent.as_deref(),
            self.context.as_deref(),
            self.gender.as_deref(),
            self.language_code.as_deref(),
            self.model.as_deref(),
            self.persona.as_deref(),
            self.region_code.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if value.chars().any(char::is_control) {
                return Err(voices_invalid(
                    "voice metadata fields must not contain control characters",
                ));
            }
        }
        Ok(())
    }
}

/// Typed create request for prompted Voice Design or replicated voice creation.
///
/// The storage choice is explicit on the wire. Prompted voices require stored
/// mode; replication supports stored IDs or caller-managed seven-day keys.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiVoiceCreateRequest {
    store: bool,
    creation: GeminiVoiceCreation,
    metadata: GeminiVoiceMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GeminiVoiceCreation {
    Prompted {
        input: String,
    },
    Replicated {
        source_audio: GeminiVoiceAudioData,
        consent_audio: GeminiVoiceAudioData,
    },
}

impl GeminiVoiceCreateRequest {
    /// Create a prompted voice. Google requires `store = true` for this type.
    pub fn prompted(input: impl Into<String>) -> Self {
        Self {
            store: true,
            creation: GeminiVoiceCreation::Prompted {
                input: input.into(),
            },
            metadata: GeminiVoiceMetadata::default(),
        }
    }

    /// Create a replicated voice using separate reference and speaker-consent recordings.
    pub fn replicated(
        source_audio: GeminiVoiceAudioData,
        consent_audio: GeminiVoiceAudioData,
        store: bool,
    ) -> Self {
        Self {
            store,
            creation: GeminiVoiceCreation::Replicated {
                source_audio,
                consent_audio,
            },
            metadata: GeminiVoiceMetadata::default(),
        }
    }

    pub fn store(&self) -> bool {
        self.store
    }

    pub fn with_display_name(mut self, value: impl Into<String>) -> Self {
        self.metadata.display_name = Some(value.into());
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.metadata.description = Some(value.into());
        self
    }

    pub fn with_accent(mut self, value: impl Into<String>) -> Self {
        self.metadata.accent = Some(value.into());
        self
    }

    pub fn with_context(mut self, value: impl Into<String>) -> Self {
        self.metadata.context = Some(value.into());
        self
    }

    pub fn with_gender(mut self, value: impl Into<String>) -> Self {
        self.metadata.gender = Some(value.into());
        self
    }

    pub fn with_language_code(mut self, value: impl Into<String>) -> Self {
        self.metadata.language_code = Some(value.into());
        self
    }

    pub fn with_model(mut self, value: impl Into<String>) -> Self {
        self.metadata.model = Some(value.into());
        self
    }

    pub fn with_persona(mut self, value: impl Into<String>) -> Self {
        self.metadata.persona = Some(value.into());
        self
    }

    pub fn with_pitch(mut self, value: GeminiVoicePitch) -> Self {
        self.metadata.pitch = Some(value);
        self
    }

    pub fn with_region_code(mut self, value: impl Into<String>) -> Self {
        self.metadata.region_code = Some(value.into());
        self
    }

    fn validate(&self) -> Result<(), GeminiVoicesError> {
        self.metadata.validate()?;
        match &self.creation {
            GeminiVoiceCreation::Prompted { input } => {
                if input.trim().is_empty() || !self.store {
                    return Err(voices_invalid(
                        "prompted voice creation requires non-empty input and store = true",
                    ));
                }
            }
            GeminiVoiceCreation::Replicated {
                source_audio,
                consent_audio,
            } => {
                source_audio.validate("source")?;
                consent_audio.validate("consent")?;
            }
        }
        if self
            .estimated_json_bytes()
            .is_none_or(|size| size > MAX_VOICE_REQUEST_BYTES)
        {
            return Err(voices_invalid(
                "Gemini Voice request exceeds the 32 MiB local JSON limit",
            ));
        }
        Ok(())
    }

    fn estimated_json_bytes(&self) -> Option<usize> {
        let mut estimate = 4_096usize;
        for value in [
            self.metadata.display_name.as_deref(),
            self.metadata.description.as_deref(),
            self.metadata.accent.as_deref(),
            self.metadata.context.as_deref(),
            self.metadata.gender.as_deref(),
            self.metadata.language_code.as_deref(),
            self.metadata.model.as_deref(),
            self.metadata.persona.as_deref(),
            self.metadata.region_code.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            estimate = estimate.checked_add(value.len().checked_mul(2)?)?;
        }
        match &self.creation {
            GeminiVoiceCreation::Prompted { input } => {
                estimate = estimate.checked_add(input.len().checked_mul(6)?)?;
            }
            GeminiVoiceCreation::Replicated {
                source_audio,
                consent_audio,
            } => {
                for audio in [source_audio, consent_audio] {
                    estimate = estimate.checked_add(base64_encoded_len(audio.data.len())?)?;
                    estimate = estimate.checked_add(audio.mime_type.len().checked_mul(2)?)?;
                }
            }
        }
        Some(estimate)
    }

    fn to_body(&self) -> Value {
        let mut voice = serde_json::Map::new();
        self.metadata.insert_into(&mut voice);
        match &self.creation {
            GeminiVoiceCreation::Prompted { input } => {
                voice.insert("type".into(), json!("prompted"));
                voice.insert("prompted".into(), json!({ "input": input }));
            }
            GeminiVoiceCreation::Replicated {
                source_audio,
                consent_audio,
            } => {
                voice.insert("type".into(), json!("replicated"));
                voice.insert(
                    "replicated".into(),
                    json!({
                        "source_audio": audio_json(source_audio),
                        "consent_audio": audio_json(consent_audio),
                    }),
                );
            }
        }
        json!({ "store": self.store, "voice": Value::Object(voice) })
    }
}

impl fmt::Debug for GeminiVoiceCreateRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.creation {
            GeminiVoiceCreation::Prompted { .. } => "prompted",
            GeminiVoiceCreation::Replicated { .. } => "replicated",
        };
        f.debug_struct("GeminiVoiceCreateRequest")
            .field("store", &self.store)
            .field("creation", &kind)
            .field("metadata", &self.metadata)
            .field("audio", &"<redacted>")
            .finish()
    }
}

/// Filters and page controls for `GET /v1beta/voices`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiVoiceListOptions {
    page_size: Option<u32>,
    page_token: Option<String>,
    accent: Vec<String>,
    context: Vec<String>,
    gender: Vec<String>,
    language_code: Vec<String>,
    persona: Vec<String>,
    pitch: Vec<String>,
    region_code: Vec<String>,
    voice_type: Vec<String>,
    search: Option<String>,
}

impl GeminiVoiceListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_page_size(mut self, page_size: u32) -> Self {
        self.page_size = Some(page_size);
        self
    }

    pub fn with_page_token(mut self, page_token: impl Into<String>) -> Self {
        self.page_token = Some(page_token.into());
        self
    }

    pub fn with_accents(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.accent.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_contexts(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.context.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_genders(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.gender.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_language_codes(
        mut self,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.language_code
            .extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_personas(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.persona.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_pitches(mut self, values: impl IntoIterator<Item = GeminiVoicePitch>) -> Self {
        self.pitch
            .extend(values.into_iter().map(|value| value.as_str().to_owned()));
        self
    }

    pub fn with_region_codes(
        mut self,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.region_code.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_types(mut self, values: impl IntoIterator<Item = GeminiVoiceType>) -> Self {
        self.voice_type.extend(values.into_iter().map(String::from));
        self
    }

    pub fn with_search(mut self, search: impl Into<String>) -> Self {
        self.search = Some(search.into());
        self
    }

    fn validate(&self) -> Result<(), GeminiVoicesError> {
        if self
            .page_size
            .is_some_and(|size| size == 0 || size > MAX_VOICE_PAGE_SIZE)
        {
            return Err(voices_invalid(
                "page_size must be between 1 and Google's maximum of 1000",
            ));
        }
        if self
            .page_token
            .as_ref()
            .is_some_and(|token| token.is_empty() || token.chars().any(char::is_control))
        {
            return Err(voices_invalid(
                "page_token must be non-empty and contain no controls",
            ));
        }
        for value in self
            .accent
            .iter()
            .chain(&self.context)
            .chain(&self.gender)
            .chain(&self.language_code)
            .chain(&self.persona)
            .chain(&self.pitch)
            .chain(&self.region_code)
            .chain(&self.voice_type)
        {
            if value.trim().is_empty() || value.chars().any(char::is_control) {
                return Err(voices_invalid(
                    "voice filter values must be non-empty and contain no controls",
                ));
            }
        }
        if self
            .voice_type
            .iter()
            .any(|value| !matches!(value.as_str(), "prompted" | "replicated" | "prebuilt"))
        {
            return Err(voices_invalid(
                "voice type filters must be prompted, replicated, or prebuilt",
            ));
        }
        if self
            .search
            .as_ref()
            .is_some_and(|search| search.len() > 2_048 || search.chars().any(char::is_control))
        {
            return Err(voices_invalid(
                "search must be at most 2048 bytes and contain no controls",
            ));
        }
        Ok(())
    }
}

/// One voice from the catalog. `reference` is present only for stored custom
/// voices; prebuilt IDs are catalog names and cannot be fetched or deleted.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiVoice {
    pub scope: GeminiVoicesScope,
    pub reference: Option<GeminiVoiceRef>,
    pub id: Option<String>,
    /// Client-managed `voicekey_...`, returned only for stateless replication.
    pub key: Option<String>,
    pub voice_type: GeminiVoiceType,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub accent: Option<String>,
    pub context: Option<String>,
    pub gender: Option<String>,
    pub language_code: Option<String>,
    pub model: Option<String>,
    pub persona: Option<String>,
    pub pitch: Option<GeminiVoicePitch>,
    pub region_code: Option<String>,
    pub expire_time: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Debug for GeminiVoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiVoice")
            .field("scope", &self.scope)
            .field("reference", &self.reference)
            .field("id", &self.id)
            .field("key", &self.key.as_ref().map(|_| "<redacted>"))
            .field("voice_type", &self.voice_type)
            .field("display_name", &self.display_name)
            .field("description", &self.description)
            .field("accent", &self.accent)
            .field("context", &self.context)
            .field("gender", &self.gender)
            .field("language_code", &self.language_code)
            .field("model", &self.model)
            .field("persona", &self.persona)
            .field("pitch", &self.pitch)
            .field("region_code", &self.region_code)
            .field("expire_time", &self.expire_time)
            .field("request_id", &self.request_id)
            .field("native", &"<provider response omitted from Debug>")
            .finish()
    }
}

impl GeminiVoice {
    /// Return the ID or client-managed key to pass to a Gemini speech request.
    pub fn synthesis_voice(&self) -> Option<&str> {
        self.key.as_deref().or(self.id.as_deref())
    }

    /// Build a TTS request from this voice after checking the project/profile
    /// identity shared by the Voices and Interactions services.
    pub fn speech_request(
        &self,
        scope: &GeminiSpeechScope,
        model: GeminiSpeechModel,
        text: impl Into<String>,
    ) -> Result<GeminiSpeechRequest, GeminiSpeechError> {
        self.scope
            .validate()
            .map_err(|_| super::invalid("Gemini voice resource scope is invalid"))?;
        scope
            .validate()
            .map_err(|_| super::invalid("Gemini speech scope is invalid"))?;
        if self.scope.provider_id != scope.provider_id
            || self.scope.profile_name != scope.profile_name
            || self.scope.account_scope != scope.account_scope
        {
            return Err(super::invalid(
                "Gemini voice belongs to another provider profile or account",
            ));
        }
        let voice = self
            .synthesis_voice()
            .ok_or_else(|| super::invalid("Gemini voice response had no synthesis ID or key"))?;
        GeminiSpeechRequest::new(model, text, voice)
    }
}

/// A provider-paginated voices response. Pass `next_page_token` unchanged to
/// the next request and keep all filter values the same between pages.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiVoicePage {
    pub voices: Vec<GeminiVoice>,
    pub next_page_token: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Debug for GeminiVoicePage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiVoicePage")
            .field("voices", &self.voices)
            .field("next_page_token", &self.next_page_token)
            .field("request_id", &self.request_id)
            .field("native", &"<provider response omitted from Debug>")
            .finish()
    }
}

/// Dispatch state for mutations to the Voices resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiVoicesDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(thiserror::Error)]
pub enum GeminiVoicesError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Gemini Voices input: {0}")]
    InvalidInput(String),
    #[error("Gemini Voices returned HTTP {status}: {message}")]
    Provider {
        status: u16,
        message: String,
        request_id: Option<String>,
        dispatch: GeminiVoicesDispatch,
        native: Box<Value>,
    },
    #[error("Gemini Voices accepted a request but returned an invalid response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
        dispatch: GeminiVoicesDispatch,
        native: Box<Value>,
    },
    #[error("Gemini Voices {operation} outcome is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
}

impl fmt::Debug for GeminiVoicesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(error) => f.debug_tuple("Llm").field(error).finish(),
            Self::InvalidInput(message) => f.debug_tuple("InvalidInput").field(message).finish(),
            Self::Provider {
                status,
                message,
                request_id,
                dispatch,
                ..
            } => f
                .debug_struct("Provider")
                .field("status", status)
                .field("message", message)
                .field("request_id", request_id)
                .field("dispatch", dispatch)
                .field("native", &"<provider response omitted from Debug>")
                .finish(),
            Self::InvalidResponse {
                message,
                request_id,
                dispatch,
                ..
            } => f
                .debug_struct("InvalidResponse")
                .field("message", message)
                .field("request_id", request_id)
                .field("dispatch", dispatch)
                .field("native", &"<provider response omitted from Debug>")
                .finish(),
            Self::OutcomeUnknown { operation, source } => f
                .debug_struct("OutcomeUnknown")
                .field("operation", operation)
                .field("source", source)
                .finish(),
        }
    }
}

impl GeminiVoicesError {
    pub fn dispatch(&self) -> GeminiVoicesDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) => GeminiVoicesDispatch::NotSent,
            Self::Provider { dispatch, .. } | Self::InvalidResponse { dispatch, .. } => *dispatch,
            Self::OutcomeUnknown { .. } => GeminiVoicesDispatch::Unknown,
        }
    }
}

/// Caller-keyed Gemini Voices resource client. The service stores no API key
/// and never retries, polls, or downloads provider resources.
#[derive(Clone)]
pub struct GeminiVoicesService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: GeminiVoicesScope,
}

impl<'a> GeminiVoicesService<'a> {
    pub(crate) fn with_binding(
        mut self,
        binding: &crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding.clone());
        self
    }
    fn pin(&self) -> Result<Self, LlmError> {
        let mut pinned = self.clone();
        pinned.binding = self
            .binding
            .as_ref()
            .map(crate::providers::binding::ProviderBinding::pinned)
            .transpose()?;
        Ok(pinned)
    }

    pub fn new(
        http: &'a dyn Transport,
        scope: GeminiVoicesScope,
    ) -> Result<Self, GeminiVoicesError> {
        scope.validate()?;
        Ok(Self {
            http,
            scope,
            binding: None,
        })
    }

    pub fn scope(&self) -> &GeminiVoicesScope {
        &self.scope
    }

    /// Create a prompted or replicated custom voice once.
    pub async fn create(
        &self,
        credential: &Secret<String>,
        request: &GeminiVoiceCreateRequest,
    ) -> Result<GeminiVoice, GeminiVoicesError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_voices_credential(credential)?;
        request.validate()?;
        let body = serde_json::to_vec(&request.to_body())
            .map_err(|_| voices_invalid("could not encode Gemini Voice request"))?;
        if body.len() > MAX_VOICE_REQUEST_BYTES {
            return Err(voices_invalid(
                "Gemini Voice request exceeds the 32 MiB local JSON limit",
            ));
        }
        let response = pinned_service
            .send(
                "POST",
                pinned_service.collection_url()?,
                body,
                credential,
                true,
                "create voice",
            )
            .await?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(voices_provider_error(response, request_id, true));
        }
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            voices_invalid_response(
                "create voice response was not valid JSON",
                request_id.clone(),
                GeminiVoicesDispatch::Accepted,
                Value::Null,
            )
        })?;
        let mut voice =
            decode_voice(native, &pinned_service.scope, request_id.clone()).map_err(|error| {
                match error {
                    GeminiVoicesError::InvalidResponse {
                        message, native, ..
                    } => voices_invalid_response(
                        message,
                        request_id.clone(),
                        GeminiVoicesDispatch::Accepted,
                        *native,
                    ),
                    other => other,
                }
            })?;
        let expected_type = match &request.creation {
            GeminiVoiceCreation::Prompted { .. } => GeminiVoiceType::Prompted,
            GeminiVoiceCreation::Replicated { .. } => GeminiVoiceType::Replicated,
        };
        if voice.voice_type != expected_type {
            return Err(voices_invalid_response(
                "create response voice type did not match the request",
                request_id,
                GeminiVoicesDispatch::Accepted,
                voice.native,
            ));
        }
        if request.store && (voice.reference.is_none() || voice.key.is_some()) {
            return Err(voices_invalid_response(
                "stored voice response must include a voice_ ID and omit the stateless key",
                request_id,
                GeminiVoicesDispatch::Accepted,
                voice.native,
            ));
        }
        if !request.store
            && (voice.id.is_some()
                || voice.reference.is_some()
                || voice.key.as_deref().is_none_or(|key| !valid_voice_key(key)))
        {
            return Err(voices_invalid_response(
                "stateless replicated voice response must include a valid voicekey_ key and omit an ID",
                request_id,
                GeminiVoicesDispatch::Accepted,
                voice.native,
            ));
        }
        voice.request_id = request_id;
        Ok(voice)
    }

    /// List one page of stored and prebuilt voices. Pagination is caller-managed.
    pub async fn list(
        &self,
        credential: &Secret<String>,
        options: &GeminiVoiceListOptions,
    ) -> Result<GeminiVoicePage, GeminiVoicesError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_voices_credential(credential)?;
        options.validate()?;
        let mut url = pinned_service.collection_url()?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(page_size) = options.page_size {
                query.append_pair("page_size", &page_size.to_string());
            }
            if let Some(page_token) = &options.page_token {
                query.append_pair("page_token", page_token);
            }
            for value in &options.accent {
                query.append_pair("accent", value);
            }
            for value in &options.context {
                query.append_pair("context", value);
            }
            for value in &options.gender {
                query.append_pair("gender", value);
            }
            for value in &options.language_code {
                query.append_pair("language_code", value);
            }
            for value in &options.persona {
                query.append_pair("persona", value);
            }
            for value in &options.pitch {
                query.append_pair("pitch", value);
            }
            for value in &options.region_code {
                query.append_pair("region_code", value);
            }
            for value in &options.voice_type {
                query.append_pair("type", value);
            }
            if let Some(search) = &options.search {
                query.append_pair("search", search);
            }
        }
        let response = pinned_service
            .send("GET", url, Vec::new(), credential, false, "list voices")
            .await?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(voices_provider_error(response, request_id, false));
        }
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            voices_invalid_response(
                "list voices response was not valid JSON",
                request_id.clone(),
                GeminiVoicesDispatch::Accepted,
                Value::Null,
            )
        })?;
        let rows = native
            .get("voices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let voices = rows
            .into_iter()
            .map(|row| decode_voice(row, &pinned_service.scope, request_id.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let next_page_token = native
            .get("next_page_token")
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty())
            .map(str::to_owned);
        Ok(GeminiVoicePage {
            voices,
            next_page_token,
            request_id,
            native,
        })
    }

    /// Get a stored custom voice. Prebuilt catalog voices are list-only.
    pub async fn get(
        &self,
        credential: &Secret<String>,
        reference: &GeminiVoiceRef,
    ) -> Result<GeminiVoice, GeminiVoicesError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_voices_credential(credential)?;
        pinned_service.validate_ref(reference)?;
        let url = pinned_service.voice_url(reference)?;
        let response = pinned_service
            .send("GET", url, Vec::new(), credential, false, "get voice")
            .await?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(voices_provider_error(response, request_id, false));
        }
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            voices_invalid_response(
                "get voice response was not valid JSON",
                request_id.clone(),
                GeminiVoicesDispatch::Accepted,
                Value::Null,
            )
        })?;
        let voice = decode_voice(native, &pinned_service.scope, request_id)?;
        if voice.id.as_deref() != Some(reference.voice_id()) {
            return Err(voices_invalid_response(
                "voice lookup returned a different resource ID",
                voice.request_id.clone(),
                GeminiVoicesDispatch::Accepted,
                voice.native,
            ));
        }
        Ok(voice)
    }

    /// Delete one Google-managed custom voice. No implicit delete is performed
    /// for stateless voice keys or prebuilt catalog voices.
    pub async fn delete(
        &self,
        credential: &Secret<String>,
        reference: &GeminiVoiceRef,
    ) -> Result<(), GeminiVoicesError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        validate_voices_credential(credential)?;
        pinned_service.validate_ref(reference)?;
        let response = pinned_service
            .send(
                "DELETE",
                pinned_service.voice_url(reference)?,
                Vec::new(),
                credential,
                true,
                "delete voice",
            )
            .await?;
        let request_id = header(&response.headers, "x-goog-request-id").map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(voices_provider_error(response, request_id, true));
        }
        Ok(())
    }

    fn validate_ref(&self, reference: &GeminiVoiceRef) -> Result<(), GeminiVoicesError> {
        let expected = GeminiVoiceRef::from_scope(&self.scope, reference.voice_id())?;
        if expected != *reference {
            return Err(voices_invalid(
                "voice reference belongs to another provider, profile, account, or endpoint",
            ));
        }
        Ok(())
    }

    fn collection_url(&self) -> Result<Url, GeminiVoicesError> {
        Url::parse(&self.scope.endpoint)
            .map_err(|_| voices_invalid("Gemini Voices endpoint is invalid"))
    }

    fn voice_url(&self, reference: &GeminiVoiceRef) -> Result<Url, GeminiVoicesError> {
        let mut url = self.collection_url()?;
        url.path_segments_mut()
            .map_err(|_| voices_invalid("Gemini Voices endpoint cannot accept path segments"))?
            .push(reference.voice_id());
        Ok(url)
    }

    async fn send(
        &self,
        method: &str,
        url: Url,
        body: Vec<u8>,
        credential: &Secret<String>,
        mutation: bool,
        operation: &'static str,
    ) -> Result<HttpResponse, GeminiVoicesError> {
        let request = HttpRequest {
            method: method.into(),
            url: url.into(),
            headers: {
                let mut headers =
                    vec![("x-goog-api-key".into(), credential.expose_secret().clone())];
                if !body.is_empty() {
                    headers.push(("content-type".into(), "application/json".into()));
                }
                headers
            },
            body: Bytes::from(body),
            timeout: Some(REQUEST_TIMEOUT),
        };
        HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_VOICE_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if mutation && is_unknown_mutation_transport(&source) {
                    GeminiVoicesError::OutcomeUnknown { operation, source }
                } else {
                    GeminiVoicesError::Llm(source)
                }
            })
    }
}

fn decode_voice(
    native: Value,
    scope: &GeminiVoicesScope,
    request_id: Option<String>,
) -> Result<GeminiVoice, GeminiVoicesError> {
    if !native.is_object() {
        return Err(voices_invalid_response(
            "voice response was not an object",
            request_id,
            GeminiVoicesDispatch::Accepted,
            native,
        ));
    }
    let Some(voice_type) = native
        .get("type")
        .and_then(Value::as_str)
        .map(|value| GeminiVoiceType::from(value.to_owned()))
    else {
        return Err(voices_invalid_response(
            "voice response omitted its required type",
            request_id,
            GeminiVoicesDispatch::Accepted,
            native,
        ));
    };
    let id_value = native.get("id").filter(|value| !value.is_null());
    let key_value = native.get("key").filter(|value| !value.is_null());
    let id = id_value.and_then(Value::as_str).map(str::to_owned);
    let key = key_value.and_then(Value::as_str).map(str::to_owned);
    if (id_value.is_some() && id.is_none()) || (key_value.is_some() && key.is_none()) {
        return Err(voices_invalid_response(
            "voice response contained a non-string ID or key",
            request_id,
            GeminiVoicesDispatch::Accepted,
            native,
        ));
    }
    if id.is_some() && key.is_some() {
        return Err(voices_invalid_response(
            "voice response contained both a stored ID and a stateless key",
            request_id,
            GeminiVoicesDispatch::Accepted,
            native,
        ));
    }
    if key.is_some() && voice_type != GeminiVoiceType::Replicated {
        return Err(voices_invalid_response(
            "voice response returned a stateless key for a non-replicated voice",
            request_id,
            GeminiVoicesDispatch::Accepted,
            native,
        ));
    }
    let reference = id
        .as_deref()
        .filter(|id| valid_stored_voice_id(id))
        .and_then(|id| GeminiVoiceRef::from_scope(scope, id).ok());
    let pitch = native
        .get("pitch")
        .and_then(Value::as_str)
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "low" | "pitch_low" => Some(GeminiVoicePitch::Low),
            "medium" | "pitch_medium" => Some(GeminiVoicePitch::Medium),
            "high" | "pitch_high" => Some(GeminiVoicePitch::High),
            _ => None,
        });
    Ok(GeminiVoice {
        scope: scope.clone(),
        reference,
        id,
        key,
        voice_type,
        display_name: optional_string(&native, "display_name"),
        description: optional_string(&native, "description"),
        accent: optional_string(&native, "accent"),
        context: optional_string(&native, "context"),
        gender: optional_string(&native, "gender"),
        language_code: optional_string(&native, "language_code"),
        model: optional_string(&native, "model"),
        persona: optional_string(&native, "persona"),
        pitch,
        region_code: optional_string(&native, "region_code"),
        expire_time: optional_string(&native, "expire_time"),
        request_id,
        native,
    })
}

fn voices_provider_error(
    response: HttpResponse,
    header_request_id: Option<String>,
    mutation: bool,
) -> GeminiVoicesError {
    let status = response.status;
    let native = parse_body(&response.body);
    let message = native
        .pointer("/error/message")
        .or_else(|| native.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Google returned an error")
        .to_owned();
    let request_id = header_request_id.or_else(|| {
        native
            .pointer("/error/details")
            .and_then(Value::as_array)
            .and_then(|details| {
                details.iter().find_map(|detail| {
                    detail
                        .get("requestId")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
            })
    });
    let dispatch = if mutation && (status == 408 || status >= 500) {
        GeminiVoicesDispatch::Unknown
    } else {
        GeminiVoicesDispatch::Rejected
    };
    GeminiVoicesError::Provider {
        status,
        message,
        request_id,
        dispatch,
        native: Box::new(native),
    }
}

fn voices_invalid_response(
    message: impl Into<String>,
    request_id: Option<String>,
    dispatch: GeminiVoicesDispatch,
    native: Value,
) -> GeminiVoicesError {
    GeminiVoicesError::InvalidResponse {
        message: message.into(),
        request_id,
        dispatch,
        native: Box::new(native),
    }
}

fn normalize_voices_endpoint(endpoint: &str) -> Result<String, GeminiVoicesError> {
    let mut url = Url::parse(endpoint)
        .map_err(|_| voices_invalid("Gemini Voices endpoint must be an absolute HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str() != Some("generativelanguage.googleapis.com")
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != VOICES_PATH
    {
        return Err(voices_invalid(
            "endpoint must be Google's HTTPS /v1beta/voices collection without query or fragment",
        ));
    }
    url.set_path(VOICES_PATH);
    Ok(url.into())
}

fn validate_voices_credential(credential: &Secret<String>) -> Result<(), GeminiVoicesError> {
    let key = credential.expose_secret();
    if key.trim().is_empty() || key.contains('\r') || key.contains('\n') {
        return Err(voices_invalid(
            "Google API key must be non-empty and contain no CR/LF",
        ));
    }
    Ok(())
}

fn voices_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_stored_voice_id(value: &str) -> bool {
    value.starts_with("voice_")
        && value.len() > "voice_".len()
        && value.len() <= 256
        && !value
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
}

fn audio_json(audio: &GeminiVoiceAudioData) -> Value {
    json!({
        "data": BASE64.encode(&audio.data),
        "mime_type": audio.mime_type,
    })
}

fn base64_encoded_len(bytes: usize) -> Option<usize> {
    bytes.checked_add(2)?.checked_div(3)?.checked_mul(4)
}

fn insert_optional(object: &mut serde_json::Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_owned(), json!(value));
    }
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn voices_invalid(message: impl Into<String>) -> GeminiVoicesError {
    GeminiVoicesError::InvalidInput(message.into())
}

fn is_unknown_mutation_transport(error: &LlmError) -> bool {
    matches!(
        error,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    )
}
