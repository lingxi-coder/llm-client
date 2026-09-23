//! `LlmClientBuilder` / `LlmClient`: the provider-neutral machine.
//!
//! `resolve` turns a model id into a route and its failover chain; `complete`
//! and `stream` walk that chain (gate 32). Nothing here knows a provider by
//! name — a new OpenAI-compatible provider is a `ProviderProfile` in settings
//! and no code change (gate 30). The builder refuses to build a client whose
//! profile names a protocol with no codec (gate 33).

pub mod account;
mod failover;
pub mod files;
pub mod options;
pub mod pricing;
mod resolve;
pub mod route;
mod store;
mod stream;
pub(crate) mod usage;

pub use account::{
    AccountBalance, AccountCostBucket, AccountCostUsage, AccountFailure, AccountIdentity,
    AccountMetric, AccountQuery, AccountQuotaWindow, AccountScope, AccountScopeKind,
    AccountSelector, AccountSnapshot, AccountSubscription, AccountTokenBucket, AccountTokenUsage,
    AccountUsageError, AccountUsageSource, AlibabaAccessKey, SubscriptionStatus,
};
pub use options::RequestOptions;
pub use resolve::ResolveError;
pub use store::{ProviderStoreError, ProviderSyncOperation, ProviderSyncResult};
pub use stream::ModelStream;

use crate::auth::Authenticator;
use crate::codecs::WireCodec;
use crate::directory::ModelDirectory;
use crate::transport::{Clock, HttpTransport, SystemClock, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream::iter, StreamExt};
use lingxi_agent_api::protocol::{
    AttachmentRef, AuthStrategy, CompletionRequest, ContentBlock, CredentialConfig, DocumentSource,
    ImageSource, LlmError, ModelListing, ProtocolFamily, ProviderListing, ProviderProfile,
    Submission, Usage, VideoSource,
};
use route::ResolvedRoute;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;

/// Maximum app attachment content resolved into one in-memory model request.
pub const MAX_ATTACHMENT_BYTES: u64 = 64 * 1024 * 1024;

const MAX_OPENAI_INPUT_FILE_BYTES: u64 = 50_000_000;
const MAX_PROVIDER_FILE_CACHE_ENTRIES: usize = 256;
pub(crate) const INLINE_IMAGE_PREFERENCE_LIMIT: usize = 5 * 1024 * 1024;

/// Application-owned source of attachment bytes. A resolver can read from a
/// local object store, a remote task service, or another authenticated host
/// application. It must return the exact immutable bytes identified by the
/// attachment id and revision.
#[async_trait]
pub trait AttachmentResolver: Send + Sync + 'static {
    async fn resolve(&self, attachment: &AttachmentRef) -> Result<Bytes, LlmError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BuildError {
    #[error(
        "provider profile {profile_name:?} uses protocol {family:?} but no codec for it is registered"
    )]
    MissingCodec {
        profile_name: String,
        family: ProtocolFamily,
    },
    #[error(
        "provider profile {profile_name:?} uses auth {strategy:?} but no authenticator for it is registered"
    )]
    MissingAuthenticator {
        profile_name: String,
        strategy: AuthStrategy,
    },
    #[error("provider profile name {profile_name:?} is declared twice")]
    DuplicateProfile { profile_name: String },
    #[error("connection id {connection_id:?} is declared twice in group {group:?}")]
    DuplicateConnectionId {
        group: String,
        connection_id: String,
    },
    #[error("provider profile {profile_name:?} has an invalid peak price schedule: {reason}")]
    InvalidPeakSchedule {
        profile_name: String,
        reason: String,
    },
}

pub struct LlmClientBuilder {
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    profile_account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    attachment_resolver: Option<Arc<dyn AttachmentResolver>>,
    profiles: Vec<ProviderProfile>,
}

impl LlmClientBuilder {
    /// Use the built-in HTTP/HTTPS transport and system clock, with API-key
    /// and bearer authentication registered. No custom services are needed.
    /// Requests require a Tokio runtime with I/O and time enabled.
    pub fn new(profiles: &[ProviderProfile]) -> Result<Self, LlmError> {
        Ok(Self::with_transport(
            Arc::new(HttpTransport::new()?),
            profiles,
        ))
    }

    /// Use a custom transport with the system clock. Registers all built-in
    /// codecs, model directories, API-key and bearer authenticators, just like
    /// [`Self::new`]. This constructor does not create a network client.
    pub fn with_transport(http: Arc<dyn Transport>, profiles: &[ProviderProfile]) -> Self {
        let mut codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>> = BTreeMap::new();
        for codec in crate::codecs::builtin() {
            codecs.insert(codec.family(), codec);
        }
        let mut directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>> = BTreeMap::new();
        for directory in crate::directory::builtin() {
            directories.insert(directory.shape(), directory);
        }
        let mut builder = Self {
            http,
            clock: Arc::new(SystemClock),
            codecs,
            directories,
            authenticators: BTreeMap::new(),
            account_sources: account::builtin_sources(),
            profile_account_sources: BTreeMap::new(),
            attachment_resolver: None,
            profiles: profiles.to_vec(),
        };
        builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(crate::ApiKeyAuthenticator));
        builder.register_authenticator(AuthStrategy::Bearer, Arc::new(crate::BearerAuthenticator));
        builder
    }

    /// Replace the default system clock, for example to test price schedules.
    pub fn with_clock(&mut self, clock: Arc<dyn Clock>) -> &mut Self {
        self.clock = clock;
        self
    }

    /// Later registrations for the same family replace earlier ones; the
    /// composition root decides the order.
    pub fn register_codec(&mut self, codec: Arc<dyn WireCodec>) -> &mut Self {
        self.codecs.insert(codec.family(), codec);
        self
    }

    /// Replace or add the directory for a shape. Unlike a codec, a shape with
    /// none is allowed — see `ModelDirectory` for why the asymmetry is
    /// deliberate.
    pub fn register_directory(&mut self, directory: Arc<dyn ModelDirectory>) -> &mut Self {
        self.directories.insert(directory.shape(), directory);
        self
    }

    pub fn register_authenticator(
        &mut self,
        strategy: AuthStrategy,
        auth: Arc<dyn Authenticator>,
    ) -> &mut Self {
        self.authenticators.insert(strategy, auth);
        self
    }

    /// Register an account source for one provider and principal kind.
    /// Host-owned OAuth and local service sessions can be supplied here.
    pub fn register_account_source(
        &mut self,
        provider_id: impl Into<String>,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> &mut Self {
        self.account_sources
            .insert((provider_id.into(), identity), source);
        self
    }

    /// Register a session-bound account source for exactly one connection.
    /// Use this for separate Codex/Copilot logins under the same provider.
    pub fn register_profile_account_source(
        &mut self,
        profile_name: impl Into<String>,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> &mut Self {
        self.profile_account_sources
            .insert((profile_name.into(), identity), source);
        self
    }

    /// Resolve app-owned attachments before provider-specific request
    /// preparation. The application remains responsible for storage and
    /// cross-device authorization.
    pub fn with_attachment_resolver(&mut self, resolver: Arc<dyn AttachmentResolver>) -> &mut Self {
        self.attachment_resolver = Some(resolver);
        self
    }

    pub fn add_profile(&mut self, profile: ProviderProfile) -> &mut Self {
        self.profiles.push(profile);
        self
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.codecs.keys().copied().collect()
    }

    /// Every profile's protocol must have a codec and its auth strategy an
    /// authenticator (`AuthStrategy::None` needs none). Fails before the first
    /// request, naming the profile (gate 33).
    pub fn build(self) -> Result<LlmClient, BuildError> {
        validate_profiles(&self.profiles, &self.codecs, &self.authenticators)?;
        let store = store::ProviderStore::new(self.profiles.clone());
        Ok(LlmClient {
            http: self.http,
            clock: self.clock,
            codecs: self.codecs,
            directories: self.directories,
            authenticators: self.authenticators,
            account_sources: self.account_sources,
            profile_account_sources: self.profile_account_sources,
            attachment_resolver: self.attachment_resolver,
            provider_file_cache: Mutex::new(BTreeMap::new()),
            provider_file_upload_locks: Mutex::new(BTreeMap::new()),
            qwen_file_rate_limiters: Mutex::new(BTreeMap::new()),
            profiles: self.profiles,
            store,
        })
    }
}

fn validate_profiles(
    profiles: &[ProviderProfile],
    codecs: &BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    authenticators: &BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
) -> Result<(), BuildError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut connections = std::collections::BTreeSet::new();
    for p in profiles {
        if !seen.insert(p.profile_name.clone()) {
            return Err(BuildError::DuplicateProfile {
                profile_name: p.profile_name.clone(),
            });
        }
        if let Some(connection_id) = &p.connection.connection_id {
            let identity = (p.group().to_owned(), connection_id.clone());
            if !connections.insert(identity.clone()) {
                return Err(BuildError::DuplicateConnectionId {
                    group: identity.0,
                    connection_id: identity.1,
                });
            }
        }
    }
    for p in profiles {
        if let Some(peak) = &p.pricing.peak {
            peak.validate()
                .map_err(|reason| BuildError::InvalidPeakSchedule {
                    profile_name: p.profile_name.clone(),
                    reason,
                })?;
        }
        if !codecs.contains_key(&p.protocol) {
            return Err(BuildError::MissingCodec {
                profile_name: p.profile_name.clone(),
                family: p.protocol,
            });
        }
        if p.auth != AuthStrategy::None && !authenticators.contains_key(&p.auth) {
            return Err(BuildError::MissingAuthenticator {
                profile_name: p.profile_name.clone(),
                strategy: p.auth,
            });
        }
    }
    // A profile whose `model_list` names a shape with no directory is
    // deliberately NOT an error here. It can run every turn it could run
    // before; all it cannot do is refresh its own list.
    Ok(())
}

/// The provider-neutral client. Holds every registered codec and profile;
/// routing and requests are M1.
pub struct LlmClient {
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    profile_account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    attachment_resolver: Option<Arc<dyn AttachmentResolver>>,
    provider_file_cache: Mutex<BTreeMap<FileCacheKey, CachedProviderFile>>,
    provider_file_upload_locks:
        Mutex<BTreeMap<FileCacheKey, std::sync::Weak<futures::lock::Mutex<()>>>>,
    qwen_file_rate_limiters: Mutex<BTreeMap<(String, String), Arc<files::QwenFileRateLimiter>>>,
    profiles: Vec<ProviderProfile>,
    store: store::ProviderStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttachmentKind {
    Image,
    Document,
    Video,
}

/// Resolved app bytes and their location in the request. Provider file
/// preparation uses this to upload the original filename and bytes, while the
/// fallback inline request remains a normal Base64 source.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedAttachmentPayload {
    pub message_index: usize,
    pub block_index: usize,
    pub kind: AttachmentKind,
    pub attachment: AttachmentRef,
    pub bytes: Bytes,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedRequest {
    pub request: CompletionRequest,
    pub attachments: Vec<ResolvedAttachmentPayload>,
}

pub(crate) struct ProviderFilePreparation<'a> {
    pub opts: &'a RequestOptions,
    pub stable_account_scope: Option<&'a str>,
    pub inline_image_data_budget_bytes: Option<usize>,
    pub authenticator: Option<&'a dyn Authenticator>,
    pub credential: Option<&'a lingxi_agent_api::protocol::Secret<String>>,
    pub automatic_cleanup: Option<Arc<files::AutomaticFileCleanup>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FileCacheKey {
    attachment_id: String,
    revision: String,
    filename: String,
    media_type: String,
    profile_name: String,
    provider_id: lingxi_agent_api::protocol::ProviderId,
    protocol: ProtocolFamily,
    base_url: String,
    account_scope: String,
    purpose: String,
}

#[derive(Debug, Clone)]
struct CachedProviderFile {
    file: files::ProviderFileRef,
    expires_at: Option<Instant>,
    cached_at: Instant,
}

#[derive(Clone)]
pub(crate) struct PreparedProviderFileUse {
    // An unscoped request has no cache entry, but still needs a 404 retry.
    key: Option<FileCacheKey>,
    file_id: String,
    uri: Option<String>,
    automatic_cleanup: Option<Arc<files::AutomaticFileCleanup>>,
}

fn is_qwen_long(profile: &ProviderProfile, model: &str) -> bool {
    profile.provider_id.as_str() == "qwen" && files::is_qwen_long_model(model)
}

fn preflight_qwen_long_files(
    profile: &ProviderProfile,
    model: &str,
    request: &CompletionRequest,
    attachments: &[ResolvedAttachmentPayload],
    opts: &RequestOptions,
) -> Result<(), LlmError> {
    if !is_qwen_long(profile, model) {
        return Ok(());
    }
    if !files::qwen_long_region_supported(profile) {
        return Err(LlmError::UnsupportedCapability {
            message: "Qwen-Long is supported only on a Beijing Qwen endpoint".into(),
        });
    }
    let resolved_positions = attachments
        .iter()
        .map(|payload| (payload.message_index, payload.block_index))
        .collect::<Vec<_>>();
    files::validate_qwen_long_inputs(request, &resolved_positions)?;
    let mut total_references = 0;
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let file = match block {
            ContentBlock::Document {
                source: DocumentSource::ProviderFile { file },
                ..
            }
            | ContentBlock::Image {
                source: ImageSource::ProviderFile { file },
            } => file,
            _ => continue,
        };
        let file = crate::codecs::validate_provider_file(file, profile, opts)?;
        if file.protocol != ProtocolFamily::OpenAiChat
            || file.purpose.as_deref() != Some("file-extract")
        {
            return Err(crate::codecs::provider_file_protocol_error());
        }
        if !files::valid_qwen_file_id(&file.file_id) {
            return Err(LlmError::InvalidRequest {
                message:
                    "Qwen-Long file IDs must contain only letters, digits, hyphens, or underscores"
                        .into(),
            });
        }
        total_references += 1;
    }
    for payload in attachments {
        if payload.kind == AttachmentKind::Video {
            continue;
        }
        let caps = files::capabilities_for_purpose(
            profile,
            model,
            &payload.attachment.media_type,
            files::FilePurpose::ModelInput,
        );
        if !caps.upload || caps.model_input == files::ModelFileReference::Unsupported {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "Qwen-Long has no documented file input for media type {:?}",
                    payload.attachment.media_type
                ),
            });
        }
        if let Some(limit) = caps.max_upload_bytes {
            if u64::try_from(payload.bytes.len()).unwrap_or(u64::MAX) > limit {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "Qwen-Long file {:?} exceeds the provider limit of {limit} bytes",
                        payload.attachment.filename
                    ),
                });
            }
        }
        total_references += 1;
    }
    if total_references > files::QWEN_LONG_MAX_FILE_REFERENCES {
        return Err(LlmError::InvalidRequest {
            message: format!(
                "Qwen-Long accepts at most {} file references per request",
                files::QWEN_LONG_MAX_FILE_REFERENCES
            ),
        });
    }
    Ok(())
}

fn is_image_media_type(media_type: &str) -> bool {
    media_type.trim().to_ascii_lowercase().starts_with("image/")
}

fn validate_image_attachment_media_types(request: &CompletionRequest) -> Result<(), LlmError> {
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Image {
            source: ImageSource::Attachment { attachment },
        } = block
        else {
            continue;
        };
        if !is_image_media_type(&attachment.media_type) {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "image attachment {:?} has non-image media type {:?}",
                    attachment.attachment_id, attachment.media_type
                ),
            });
        }
    }
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Video {
            source: VideoSource::Attachment { attachment },
        } = block
        else {
            continue;
        };
        if !attachment
            .media_type
            .trim()
            .to_ascii_lowercase()
            .starts_with("video/")
        {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "video attachment {:?} has non-video media type {:?}",
                    attachment.attachment_id, attachment.media_type
                ),
            });
        }
    }
    Ok(())
}

fn validate_anthropic_document_media_types(request: &CompletionRequest) -> Result<(), LlmError> {
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Document { source, .. } = block else {
            continue;
        };
        let media_type = match source {
            DocumentSource::Base64 { media_type, .. } | DocumentSource::Text { media_type, .. } => {
                Some(media_type.as_str())
            }
            DocumentSource::Attachment { attachment } => Some(attachment.media_type.as_str()),
            DocumentSource::ProviderFile { file } => file.media_type.as_deref(),
            DocumentSource::Url { .. } => None,
        };
        if media_type.is_some_and(is_image_media_type) {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic document blocks cannot use an image media type".into(),
            });
        }
    }
    Ok(())
}

fn uses_first_party_openai_file_inputs(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() == "openai"
        && matches!(
            profile.protocol,
            ProtocolFamily::OpenAiResponses | ProtocolFamily::OpenAiChat
        )
        && reqwest::Url::parse(&profile.base_url)
            .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some("api.openai.com"))
}

fn first_party_image_formats(
    profile: &ProviderProfile,
) -> Option<(&'static str, &'static [&'static str])> {
    const OPENAI: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
    const ANTHROPIC: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
    const GEMINI: &[&str] = &[
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ];
    if uses_first_party_openai_file_inputs(profile) {
        return Some(("OpenAI", OPENAI));
    }
    if uses_first_party_anthropic_messages(profile) {
        return Some(("Anthropic", ANTHROPIC));
    }
    if profile.provider_id.as_str() == "google"
        && profile.protocol == ProtocolFamily::GeminiGenerateContent
        && reqwest::Url::parse(&profile.base_url).is_ok_and(|url| {
            url.scheme() == "https" && url.host_str() == Some("generativelanguage.googleapis.com")
        })
    {
        return Some(("Gemini", GEMINI));
    }
    None
}

fn validate_first_party_image_media_types(
    request: &CompletionRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    let Some((provider, formats)) = first_party_image_formats(profile) else {
        return Ok(());
    };
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Image { source } = block else {
            continue;
        };
        let media_type = match source {
            ImageSource::Base64 { media_type, .. } => Some(media_type.as_str()),
            ImageSource::Attachment { attachment } => Some(attachment.media_type.as_str()),
            ImageSource::ProviderFile { file } => file.media_type.as_deref(),
            ImageSource::Url { .. } => None,
        };
        if let Some(media_type) = media_type {
            if !formats
                .iter()
                .any(|format| media_type.eq_ignore_ascii_case(format))
            {
                return Err(LlmError::UnsupportedCapability {
                    message: format!("{provider} does not accept image media type {media_type:?}"),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn uses_first_party_anthropic_messages(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages
        && reqwest::Url::parse(&profile.base_url)
            .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some("api.anthropic.com"))
}

fn base64_decoded_len(data: &str) -> Result<u64, LlmError> {
    let bytes = data.as_bytes();
    let invalid = || LlmError::InvalidRequest {
        message: "OpenAI document data must be valid standard Base64".into(),
    };
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid());
    }
    let padding = bytes.iter().rev().take_while(|byte| **byte == b'=').count();
    if padding > 2
        || bytes[..bytes.len().saturating_sub(padding)]
            .iter()
            .any(|byte| !byte.is_ascii_alphanumeric() && *byte != b'+' && *byte != b'/')
    {
        return Err(invalid());
    }
    let decoded_len = (bytes.len() / 4).saturating_mul(3).saturating_sub(padding);
    u64::try_from(decoded_len).map_err(|_| LlmError::RequestTooLarge {
        message: "OpenAI document size cannot be represented".into(),
    })
}

fn validate_first_party_openai_documents(
    request: &CompletionRequest,
    protocol: ProtocolFamily,
) -> Result<(), LlmError> {
    let api = if protocol == ProtocolFamily::OpenAiChat {
        "OpenAI Chat"
    } else {
        "OpenAI Responses"
    };
    let mut total_bytes = 0_u64;
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Document { source, .. } = block else {
            continue;
        };
        let (media_type, size_bytes) = match source {
            DocumentSource::Base64 { media_type, data } => {
                (Some(media_type.as_str()), Some(base64_decoded_len(data)?))
            }
            DocumentSource::Text { media_type, data } => {
                (Some(media_type.as_str()), Some(data.len() as u64))
            }
            DocumentSource::Attachment { attachment } => (
                Some(attachment.media_type.as_str()),
                Some(attachment.size_bytes),
            ),
            DocumentSource::ProviderFile { file } => (file.media_type.as_deref(), None),
            DocumentSource::Url { .. } => (None, None),
        };
        if media_type.is_some_and(is_image_media_type) {
            return Err(LlmError::InvalidRequest {
                message: format!("{api} document blocks cannot use an image media type"),
            });
        }
        if let Some(size_bytes) = size_bytes {
            if size_bytes > MAX_OPENAI_INPUT_FILE_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "{api} input file is {size_bytes} bytes; the per-file limit is {MAX_OPENAI_INPUT_FILE_BYTES} bytes"
                    ),
                });
            }
            total_bytes = total_bytes.saturating_add(size_bytes);
            if total_bytes > MAX_OPENAI_INPUT_FILE_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "{api} input files total {total_bytes} bytes; the request limit is {MAX_OPENAI_INPUT_FILE_BYTES} bytes"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn prune_expired_provider_files(
    cache: &mut BTreeMap<FileCacheKey, CachedProviderFile>,
    now: Instant,
) {
    cache.retain(|_, entry| entry.expires_at.is_none_or(|expiry| expiry > now));
}

/// Evict the oldest insertion first, using cache-key order to break ties.
fn cap_provider_file_cache(cache: &mut BTreeMap<FileCacheKey, CachedProviderFile>) {
    while cache.len() > MAX_PROVIDER_FILE_CACHE_ENTRIES {
        let oldest_key = cache
            .iter()
            .min_by(|(left_key, left), (right_key, right)| {
                left.cached_at
                    .cmp(&right.cached_at)
                    .then_with(|| left_key.cmp(right_key))
            })
            .map(|(key, _)| key.clone());
        let Some(oldest_key) = oldest_key else {
            break;
        };
        cache.remove(&oldest_key);
    }
}

fn validate_attachment_metadata(attachment: &AttachmentRef) -> Result<(), LlmError> {
    if attachment.attachment_id.trim().is_empty()
        || attachment.revision.trim().is_empty()
        || attachment.filename.trim().is_empty()
        || attachment.media_type.trim().is_empty()
    {
        return Err(LlmError::InvalidRequest {
            message: "attachment id, revision, filename, and media type must be non-empty".into(),
        });
    }
    if attachment.size_bytes > MAX_ATTACHMENT_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: format!(
                "attachment {:?} is {} bytes; the limit is {} bytes",
                attachment.attachment_id, attachment.size_bytes, MAX_ATTACHMENT_BYTES
            ),
        });
    }
    Ok(())
}

impl LlmClient {
    /// Materialize durable application references into a per-call request
    /// copy. The transcript supplied by the caller keeps the stable refs.
    pub(crate) async fn resolve_attachments(
        &self,
        req: &CompletionRequest,
    ) -> Result<ResolvedRequest, LlmError> {
        validate_image_attachment_media_types(req)?;
        let has_attachments = req.messages.iter().any(|message| {
            message.content.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::Image {
                        source: ImageSource::Attachment { .. },
                    } | ContentBlock::Document {
                        source: DocumentSource::Attachment { .. },
                        ..
                    } | ContentBlock::Video {
                        source: VideoSource::Attachment { .. },
                    }
                )
            })
        });
        if !has_attachments {
            return Ok(ResolvedRequest {
                request: req.clone(),
                attachments: Vec::new(),
            });
        }
        let resolver =
            self.attachment_resolver
                .as_ref()
                .ok_or_else(|| LlmError::UnsupportedCapability {
                    message:
                        "request contains app attachments but no AttachmentResolver is configured"
                            .into(),
                })?;

        use base64::Engine as _;
        let mut request = req.clone();
        let mut attachments = Vec::new();
        let mut resolved_by_revision = BTreeMap::<(String, String), (AttachmentRef, Bytes)>::new();
        let mut total_bytes = 0_u64;
        for (message_index, message) in request.messages.iter_mut().enumerate() {
            for (block_index, block) in message.content.iter_mut().enumerate() {
                let (attachment, kind) = match block {
                    ContentBlock::Image {
                        source: ImageSource::Attachment { attachment },
                    } => (attachment.clone(), AttachmentKind::Image),
                    ContentBlock::Document {
                        source: DocumentSource::Attachment { attachment },
                        ..
                    } => (attachment.clone(), AttachmentKind::Document),
                    ContentBlock::Video {
                        source: VideoSource::Attachment { attachment },
                    } => (attachment.clone(), AttachmentKind::Video),
                    _ => continue,
                };
                validate_attachment_metadata(&attachment)?;
                total_bytes = total_bytes.saturating_add(attachment.size_bytes);
                if total_bytes > MAX_ATTACHMENT_BYTES {
                    return Err(LlmError::RequestTooLarge {
                        message: format!(
                            "resolved attachments exceed the {} MiB request limit",
                            MAX_ATTACHMENT_BYTES / (1024 * 1024)
                        ),
                    });
                }
                let key = (
                    attachment.attachment_id.clone(),
                    attachment.revision.clone(),
                );
                let bytes = if let Some((cached_ref, bytes)) = resolved_by_revision.get(&key) {
                    if cached_ref != &attachment {
                        return Err(LlmError::InvalidRequest {
                            message: format!(
                                "attachment {:?} revision {:?} has inconsistent metadata",
                                attachment.attachment_id, attachment.revision
                            ),
                        });
                    }
                    bytes.clone()
                } else {
                    let bytes = resolver.resolve(&attachment).await?;
                    if bytes.len() as u64 != attachment.size_bytes {
                        return Err(LlmError::InvalidRequest {
                            message: format!(
                                "attachment resolver returned {} bytes for {:?} revision {:?}, expected {}",
                                bytes.len(),
                                attachment.attachment_id,
                                attachment.revision,
                                attachment.size_bytes
                            ),
                        });
                    }
                    resolved_by_revision.insert(key, (attachment.clone(), bytes.clone()));
                    bytes
                };
                let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                match block {
                    ContentBlock::Image { source } => {
                        *source = ImageSource::Base64 {
                            media_type: attachment.media_type.clone(),
                            data,
                        };
                    }
                    ContentBlock::Document { source, title } => {
                        *source = DocumentSource::Base64 {
                            media_type: attachment.media_type.clone(),
                            data,
                        };
                        if title.is_none() {
                            *title = Some(attachment.filename.clone());
                        }
                    }
                    ContentBlock::Video { source } => {
                        *source = VideoSource::Base64 {
                            media_type: attachment.media_type.clone(),
                            data,
                        };
                    }
                    _ => unreachable!("attachment block kind was checked above"),
                }
                attachments.push(ResolvedAttachmentPayload {
                    message_index,
                    block_index,
                    kind,
                    attachment,
                    bytes,
                });
            }
        }
        Ok(ResolvedRequest {
            request,
            attachments,
        })
    }

    pub(crate) fn qwen_file_rate_limiter(
        &self,
        profile: &ProviderProfile,
        stable_account_scope: Option<&str>,
    ) -> Arc<files::QwenFileRateLimiter> {
        let key = (
            profile.base_url.clone(),
            stable_account_scope
                .unwrap_or(&profile.profile_name)
                .to_owned(),
        );
        let mut limiters = self
            .qwen_file_rate_limiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(
            limiters
                .entry(key)
                .or_insert_with(|| Arc::new(files::QwenFileRateLimiter::new())),
        )
    }

    /// Replace eligible model inputs with files uploaded for this exact
    /// failover attempt. The resolved source bytes stay app-owned and are
    /// never overwritten in the caller's request.
    pub(crate) async fn prepare_provider_file_inputs(
        &self,
        profile: &ProviderProfile,
        model: &str,
        request: &mut CompletionRequest,
        attachments: &[ResolvedAttachmentPayload],
        preparation: ProviderFilePreparation<'_>,
    ) -> Result<Vec<PreparedProviderFileUse>, LlmError> {
        const MAX_INLINE_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;
        let ProviderFilePreparation {
            opts,
            stable_account_scope,
            inline_image_data_budget_bytes,
            authenticator,
            credential,
            automatic_cleanup,
        } = preparation;
        validate_first_party_image_media_types(request, profile)?;
        preflight_qwen_long_files(profile, model, request, attachments, opts)?;
        if uses_first_party_openai_file_inputs(profile) {
            validate_first_party_openai_documents(request, profile.protocol)?;
        }
        let mut prepared_file_uses = Vec::new();
        if profile.protocol == ProtocolFamily::AnthropicMessages {
            validate_anthropic_document_media_types(request)?;
        }
        let mut local_files = BTreeMap::<(String, String, String), files::ProviderFileRef>::new();
        let mut inline_image_data_bytes = attachments
            .iter()
            .filter(|payload| payload.kind == AttachmentKind::Image)
            .map(|payload| payload.bytes.len().div_ceil(3) * 4)
            .sum::<usize>();
        // Promote images that already exceed the inline preference first, so
        // they free budget before deciding whether smaller images need files.
        let mut ordered_attachments = attachments.iter().collect::<Vec<_>>();
        ordered_attachments.sort_by_key(|payload| {
            !(payload.kind == AttachmentKind::Image
                && payload.bytes.len() > INLINE_IMAGE_PREFERENCE_LIMIT)
        });
        for payload in ordered_attachments {
            let mut model_matches = profile
                .models
                .iter()
                .filter(|candidate| candidate.request_model == model);
            let model_profile = match (model_matches.next(), model_matches.next()) {
                (Some(model), None) => model,
                (None, _) => {
                    return Err(LlmError::ModelUnavailable {
                        message: format!(
                            "profile {:?} no longer contains model {model:?}",
                            profile.profile_name
                        ),
                    });
                }
                (Some(_), Some(_)) => {
                    return Err(LlmError::UnsupportedCapability {
                        message: format!(
                            "profile {:?} has ambiguous metadata for model {model:?}",
                            profile.profile_name
                        ),
                    });
                }
            };
            let model_capability = match payload.kind {
                AttachmentKind::Image => lingxi_agent_api::protocol::ModelCapability::Vision,
                AttachmentKind::Document => lingxi_agent_api::protocol::ModelCapability::Documents,
                AttachmentKind::Video => lingxi_agent_api::protocol::ModelCapability::Vision,
            };
            let video_supported = model_profile
                .metadata
                .input_modalities
                .iter()
                .any(|modality| modality.eq_ignore_ascii_case("video"));
            if (payload.kind == AttachmentKind::Video && !video_supported)
                || (payload.kind != AttachmentKind::Video
                    && model_profile.capability_support_for(model_capability)
                        == lingxi_agent_api::protocol::CapabilitySupport::Unsupported)
            {
                return Err(LlmError::UnsupportedCapability {
                    message: format!(
                        "model {model:?} explicitly does not accept this attachment type"
                    ),
                });
            }
            // Images remain inline when comfortably within common provider
            // request limits; documents use native references when the exact
            // provider/model pair explicitly supports them.
            if payload.kind == AttachmentKind::Image
                && !is_qwen_long(profile, model)
                && payload.bytes.len() <= INLINE_IMAGE_PREFERENCE_LIMIT
                && inline_image_data_budget_bytes
                    .is_none_or(|budget| inline_image_data_bytes <= budget)
            {
                continue;
            }
            let purpose = if payload.kind == AttachmentKind::Video {
                files::FilePurpose::VideoUnderstanding
            } else {
                files::FilePurpose::ModelInput
            };
            let service = files::FileService::new(
                self.http.as_ref(),
                profile,
                authenticator,
                credential,
                opts.file_account_scope.as_deref(),
            );
            let service = if let Some(cleanup) = automatic_cleanup.as_ref() {
                service.with_qwen_rate_limiter(cleanup.rate_limiter())
            } else {
                service
            };
            let capabilities =
                service.capabilities_for_purpose(model, &payload.attachment.media_type, purpose);
            if !capabilities.upload
                || capabilities.model_input == files::ModelFileReference::Unsupported
            {
                if payload.bytes.len() > MAX_INLINE_ATTACHMENT_BYTES {
                    return Err(LlmError::RequestTooLarge {
                        message: format!(
                            "{} attachment {:?} has no supported provider file reference and exceeds the inline limit of {} bytes",
                            match payload.kind {
                                AttachmentKind::Image => "image",
                                AttachmentKind::Document => "document",
                                AttachmentKind::Video => "video",
                            },
                            payload.attachment.filename,
                            MAX_INLINE_ATTACHMENT_BYTES
                        ),
                    });
                }
                continue;
            }

            let purpose_key = match payload.kind {
                AttachmentKind::Image => "model_input:image",
                AttachmentKind::Document => "model_input:document",
                AttachmentKind::Video => "model_input:video",
            };
            let local_key = (
                payload.attachment.attachment_id.clone(),
                payload.attachment.revision.clone(),
                purpose_key.to_owned(),
            );
            let file = if let Some(file) = local_files.get(&local_key) {
                file.clone()
            } else {
                let cache_key = stable_account_scope
                    .filter(|_| automatic_cleanup.is_none())
                    .map(|scope| FileCacheKey {
                        attachment_id: payload.attachment.attachment_id.clone(),
                        revision: payload.attachment.revision.clone(),
                        filename: payload.attachment.filename.clone(),
                        media_type: payload.attachment.media_type.clone(),
                        profile_name: profile.profile_name.clone(),
                        provider_id: profile.provider_id.clone(),
                        protocol: profile.protocol,
                        base_url: profile.base_url.clone(),
                        account_scope: scope.to_owned(),
                        purpose: purpose_key.to_owned(),
                    });
                let upload_lock = cache_key.as_ref().map(|key| {
                    let mut locks = self
                        .provider_file_upload_locks
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    locks.retain(|_, weak| weak.strong_count() > 0);
                    if let Some(lock) = locks.get(key).and_then(std::sync::Weak::upgrade) {
                        lock
                    } else {
                        let lock = Arc::new(futures::lock::Mutex::new(()));
                        locks.insert(key.clone(), Arc::downgrade(&lock));
                        lock
                    }
                });
                let _upload_guard = match upload_lock.as_ref() {
                    Some(lock) => Some(lock.lock().await),
                    None => None,
                };
                let cached = cache_key.as_ref().and_then(|key| {
                    let now = Instant::now();
                    let mut cache = self
                        .provider_file_cache
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    prune_expired_provider_files(&mut cache, now);
                    cache.get(key).map(|entry| entry.file.clone())
                });
                let file = if let Some(file) = cached {
                    local_files.insert(local_key.clone(), file.clone());
                    file
                } else {
                    let file = service
                        .upload_automatic(
                            &files::UploadFile {
                                filename: payload.attachment.filename.clone(),
                                media_type: payload.attachment.media_type.clone(),
                                bytes: payload.bytes.clone(),
                            },
                            purpose,
                        )
                        .await?;
                    if let Some(cleanup) = automatic_cleanup.as_ref() {
                        cleanup.track(file.clone());
                        service
                            .wait_for_qwen_file_ready(&file, opts.total_timeout)
                            .await?;
                    }
                    if let Some(cache_key) = cache_key.as_ref() {
                        let now = Instant::now();
                        let retention =
                            files::automatic_file_cache_ttl(profile).or(capabilities.retention);
                        let expires_at = retention.and_then(|retention| now.checked_add(retention));
                        let mut cache = self
                            .provider_file_cache
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        prune_expired_provider_files(&mut cache, now);
                        cache.insert(
                            cache_key.clone(),
                            CachedProviderFile {
                                file: file.clone(),
                                expires_at,
                                cached_at: now,
                            },
                        );
                        cap_provider_file_cache(&mut cache);
                    }
                    local_files.insert(local_key.clone(), file.clone());
                    file
                };
                prepared_file_uses.push(PreparedProviderFileUse {
                    key: cache_key,
                    file_id: file.file_id.clone(),
                    uri: file.uri.clone(),
                    automatic_cleanup: automatic_cleanup.clone(),
                });
                file
            };

            let block = request
                .messages
                .get_mut(payload.message_index)
                .and_then(|message| message.content.get_mut(payload.block_index))
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "resolved attachment no longer matches the request".into(),
                })?;
            match (payload.kind, block) {
                (AttachmentKind::Image, ContentBlock::Image { source }) => {
                    *source = ImageSource::ProviderFile {
                        file: file.model_reference(),
                    };
                }
                (AttachmentKind::Document, ContentBlock::Document { source, .. }) => {
                    *source = DocumentSource::ProviderFile {
                        file: file.model_reference(),
                    };
                }
                (AttachmentKind::Video, ContentBlock::Video { source }) => {
                    *source = VideoSource::ProviderFile {
                        file: file.model_reference(),
                    };
                }
                _ => {
                    return Err(LlmError::InvalidRequest {
                        message: "resolved attachment kind does not match its request block".into(),
                    });
                }
            }
            if payload.kind == AttachmentKind::Image {
                inline_image_data_bytes =
                    inline_image_data_bytes.saturating_sub(payload.bytes.len().div_ceil(3) * 4);
            }
        }
        Ok(prepared_file_uses)
    }

    /// Invalidate cached provider references that this attempt actually used.
    /// Entries already removed by another caller and newer replacements are
    /// left alone; the caller can still retry with the current cache state.
    pub(crate) async fn invalidate_provider_file_cache(
        &self,
        prepared_file_uses: &[PreparedProviderFileUse],
    ) {
        for used_file in prepared_file_uses {
            let Some(key) = used_file.key.as_ref() else {
                continue;
            };
            let lock = {
                let mut locks = self
                    .provider_file_upload_locks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                locks.retain(|_, weak| weak.strong_count() > 0);
                if let Some(lock) = locks.get(key).and_then(std::sync::Weak::upgrade) {
                    lock
                } else {
                    let lock = Arc::new(futures::lock::Mutex::new(()));
                    locks.insert(key.clone(), Arc::downgrade(&lock));
                    lock
                }
            };
            let _guard = lock.lock().await;
            let now = Instant::now();
            let mut cache = self
                .provider_file_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            prune_expired_provider_files(&mut cache, now);
            let still_same_file = cache
                .get(key)
                .is_some_and(|entry| entry.file.file_id == used_file.file_id);
            if still_same_file {
                cache.remove(key);
            }
        }
    }

    /// Read an account's available balance, token history, quota windows and
    /// subscription evidence without sending a model request.
    pub async fn account_usage(
        &self,
        profile_name: &str,
        query: &account::AccountQuery,
    ) -> Result<account::AccountSnapshot, account::AccountUsageError> {
        let profile = self
            .profile(profile_name)
            .ok_or_else(|| account::AccountUsageError::UnknownProfile(profile_name.to_owned()))?;
        let now = self
            .clock
            .now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let range = query.range(now)?;
        let profile_key = (profile.profile_name.clone(), query.identity);
        let provider_key = (profile.provider_id.as_str().to_owned(), query.identity);
        let profile_source = self.profile_account_sources.get(&profile_key);
        let source = profile_source.or_else(|| self.account_sources.get(&provider_key));
        if profile_source.is_none()
            && source.is_some_and(|source| source.requires_profile_binding(query))
            && self
                .profiles
                .iter()
                .filter(|candidate| candidate.provider_id == profile.provider_id)
                .count()
                > 1
        {
            return Err(account::AccountUsageError::AmbiguousAccountSource(
                profile_name.to_owned(),
            ));
        }
        let mut snapshot = if let Some(source) = source {
            source
                .fetch(profile, query, range, now, self.http.as_ref())
                .await
        } else {
            account::AccountSnapshot::unsupported(profile, query.identity, now)
        };
        // Profile identity comes from the client, not a host-provided source.
        snapshot.profile_name = profile.profile_name.clone();
        snapshot.provider_id = profile.provider_id.clone();
        snapshot.identity = query.identity;
        snapshot.fetched_at_unix = now;
        Ok(snapshot)
    }

    /// Query every configured connection independently, including hidden ones.
    /// Missing query entries and individual provider failures stay local to
    /// their profile's result.
    pub async fn accounts_usage(
        &self,
        queries_by_profile: &BTreeMap<String, account::AccountQuery>,
    ) -> Vec<(
        String,
        Result<account::AccountSnapshot, account::AccountUsageError>,
    )> {
        iter(self.profiles.iter().map(|profile| async move {
            let name = profile.profile_name.clone();
            let result = match queries_by_profile.get(&name) {
                Some(query) => self.account_usage(&name, query).await,
                None => Err(account::AccountUsageError::MissingQuery(name.clone())),
            };
            (name, result)
        }))
        .buffered(4)
        .collect()
        .await
    }

    /// Every model a picker may offer. A hidden connection is skipped: those
    /// exist only to be failed over onto, so offering them would let a user
    /// pick the spare key directly.
    ///
    /// `id` is the display model, which is the ref a session stores and
    /// `resolve` accepts back.
    pub fn models(&self) -> Vec<ModelListing> {
        self.profiles
            .iter()
            .filter(|p| !p.connection.hidden)
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| !m.hidden && self.tracks(p, m))
                    .map(move |m| ModelListing {
                        id: m.display_model.clone(),
                        profile_name: p.profile_name.clone(),
                        request_model: m.request_model.clone(),
                        billing_mode: m.billing_mode_on(&p.pricing),
                        pricing: m.pricing.clone(),
                        // The catalog's display name, not its description: the
                        // latter is a paragraph of vendor prose and was never a
                        // name. It keeps its own field below.
                        display_name: m.display_model.clone(),
                        description: m.description.clone(),
                        provider_id: p.provider_id.clone(),
                        context_window: m.metadata.context_window_tokens,
                        max_output_tokens: m.metadata.max_output_tokens,
                        capabilities: m.capabilities,
                    })
            })
            .collect()
    }

    /// Every configured provider, including the ones with no credential and the
    /// spare connections a picker should not offer.
    ///
    /// Unfiltered on purpose, which is the opposite of `models()`: an app needs
    /// the unconfigured ones to offer "set this up", and it needs to see the
    /// spares to explain a group. `hidden` says which is which.
    pub fn providers(&self) -> Vec<ProviderListing> {
        self.profiles
            .iter()
            .map(|p| ProviderListing {
                provider_id: p.provider_id.clone(),
                profile_name: p.profile_name.clone(),
                group: p.group().to_owned(),
                info: p.info.clone(),
                protocol: p.protocol,
                auth: p.auth,
                credential_env: match &p.credential {
                    CredentialConfig::Env { var } => Some(var.clone()),
                    _ => None,
                },
                billing_mode: p.pricing.billing_mode,
                model_count: p.models.iter().filter(|m| self.tracks(p, m)).count(),
                hidden: p.connection.hidden,
            })
            .collect()
    }

    pub fn profiles(&self) -> &[ProviderProfile] {
        &self.profiles
    }

    fn tracks(
        &self,
        profile: &ProviderProfile,
        model: &lingxi_agent_api::protocol::ModelProfile,
    ) -> bool {
        self.store.tracks(profile, model)
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.codecs.keys().copied().collect()
    }

    /// How to ask this connection what it serves, when it publishes that at
    /// all and this crate can read the shape it publishes it in.
    ///
    /// `None` is an ordinary answer, not a failure: a connection that declares
    /// no directory, and one whose directory speaks a shape with no reader,
    /// both keep working from the shipped catalog.
    #[must_use]
    pub fn directory_for(&self, profile: &ProviderProfile) -> Option<Arc<dyn ModelDirectory>> {
        let shape = profile.model_list.shape(profile.protocol)?;
        self.directories.get(&shape).cloned()
    }

    pub fn directory_shapes(&self) -> Vec<ProtocolFamily> {
        self.directories.keys().copied().collect()
    }

    pub(super) fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.profiles.iter().find(|p| p.profile_name == name)
    }

    /// Preflight estimate for the route's first connection. For a completed
    /// request that may have failed over, use [`Self::estimate_actual_cost`].
    /// What a finished request on `route` cost, priced from the catalog at
    /// the rates in force now. `Ok(None)` is a model the catalog does not
    /// price on a connection that tolerates that; a connection that set
    /// `require_priced` gets the error instead.
    pub fn estimate_cost(
        &self,
        route: &ResolvedRoute,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        let unpriced = || LlmError::CostUnavailable {
            message: format!(
                "{:?} publishes no price for {:?}",
                route.profile_name, route.display_model
            ),
        };
        let profile = self.profile(&route.profile_name).ok_or_else(unpriced)?;
        if profile.provider_id != route.provider_id
            || profile.provider_id != route.pricing_model.pricing_provider_id
            || route.request_model != route.pricing_model.request_model
            || route.display_model != route.pricing_model.display_model
        {
            return Err(unpriced());
        }
        let mut models = profile.models.iter().filter(|m| {
            m.request_model == route.request_model
                && m.display_model == route.pricing_model.display_model
                && m.billing_model == route.pricing_model.billing_model
        });
        let model = match (models.next(), models.next()) {
            (Some(model), None) => model,
            _ => return Err(unpriced()),
        };
        let Some(pricing) = &model.pricing else {
            return if profile.pricing.require_priced {
                Err(unpriced())
            } else {
                Ok(None)
            };
        };
        let now = self
            .clock
            .now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        pricing::estimate(
            pricing,
            submission,
            profile.pricing.peak.as_ref(),
            now,
            usage,
            &route.pricing_model,
        )
        .map(Some)
    }

    /// Price usage against the connection that actually served a response.
    pub fn estimate_actual_cost(
        &self,
        route: &ResolvedRoute,
        response: &lingxi_agent_api::protocol::CompletionResponse,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        let name =
            response
                .executed_profile
                .as_deref()
                .ok_or_else(|| LlmError::CostUnavailable {
                    message: "response has no executed profile".into(),
                })?;
        self.estimate_cost_for_profile(route, name, usage, submission)
    }

    /// Price usage from a streamed request using `ModelStream::executed_profile`.
    pub fn estimate_cost_for_profile(
        &self,
        route: &ResolvedRoute,
        name: &str,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        if name != route.profile_name
            && !route
                .connection_chain
                .iter()
                .any(|hop| hop.profile_name == name)
        {
            return Err(LlmError::CostUnavailable {
                message: format!("profile {name:?} is not on this route"),
            });
        }
        let profile = self
            .profile(name)
            .ok_or_else(|| LlmError::CostUnavailable {
                message: format!("executed profile {name:?} is unavailable"),
            })?;
        if name == route.profile_name {
            return self.estimate_cost(route, usage, submission);
        }
        let mut candidates = profile
            .models
            .iter()
            .filter(|m| m.request_model == route.request_model);
        let model = match (candidates.next(), candidates.next()) {
            (Some(model), None) => model,
            (None, _) => {
                return Err(LlmError::CostUnavailable {
                    message: format!(
                        "executed profile {name:?} does not serve {:?}",
                        route.request_model
                    ),
                });
            }
            (Some(_), Some(_)) => {
                return Err(LlmError::CostUnavailable {
                    message: format!(
                        "executed profile {name:?} ambiguously serves {:?}",
                        route.request_model
                    ),
                });
            }
        };
        let mut actual_route = route.clone();
        actual_route.profile_name = name.to_owned();
        actual_route.provider_id = profile.provider_id.clone();
        actual_route.display_model = model.display_model.clone();
        actual_route.pricing_model.pricing_provider_id = profile.provider_id.clone();
        actual_route.pricing_model.billing_model = model.billing_model.clone();
        actual_route.pricing_model.display_model = model.display_model.clone();
        self.estimate_cost(&actual_route, usage, submission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{
        HttpRequest, HttpResponse, StreamResponse, Transport, WebSocketSession,
    };
    use async_trait::async_trait;
    use lingxi_agent_api::protocol::LlmError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MemoryAttachments {
        bytes: Bytes,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AttachmentResolver for MemoryAttachments {
        async fn resolve(&self, _attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.bytes.clone())
        }
    }

    struct MissingAttachments;

    #[async_trait]
    impl AttachmentResolver for MissingAttachments {
        async fn resolve(&self, attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
            Err(LlmError::ModelUnavailable {
                message: format!(
                    "application attachment {:?} revision {:?} is unavailable",
                    attachment.attachment_id, attachment.revision
                ),
            })
        }
    }

    fn request_with_attachment_blocks() -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m1",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "attachment", "attachment": {
                        "attachment_id": "att-1", "revision": "rev-1", "filename": "image.png",
                        "media_type": "image/png", "size_bytes": 3
                    }}},
                    {"type": "document", "source": {"type": "attachment", "attachment": {
                        "attachment_id": "att-1", "revision": "rev-1", "filename": "image.png",
                        "media_type": "image/png", "size_bytes": 3
                    }}}
                ]
            }]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn attachment_refs_are_resolved_once_and_only_in_a_request_copy() {
        let resolver = Arc::new(MemoryAttachments {
            bytes: Bytes::from_static(b"png"),
            calls: AtomicUsize::new(0),
        });
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(resolver.clone());
        let client = builder.build().unwrap();
        let original = request_with_attachment_blocks();

        let resolved = client.resolve_attachments(&original).await.unwrap();

        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
        assert_eq!(resolved.attachments.len(), 2);
        assert_eq!(original, request_with_attachment_blocks());
        assert!(matches!(
            &resolved.request.messages[0].content[0],
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data }
            } if media_type == "image/png" && data == "cG5n"
        ));
        assert!(matches!(
            &resolved.request.messages[0].content[1],
            ContentBlock::Document {
                source: DocumentSource::Base64 { media_type, data },
                title: Some(title),
            } if media_type == "image/png" && data == "cG5n" && title == "image.png"
        ));
    }

    #[tokio::test]
    async fn attachment_refs_without_a_resolver_fail_clearly() {
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[])
            .build()
            .unwrap();
        assert!(matches!(
            client.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::UnsupportedCapability { message })
                if message.contains("no AttachmentResolver")
        ));
    }

    #[tokio::test]
    async fn resolver_byte_count_must_match_attachment_metadata() {
        let resolver = Arc::new(MemoryAttachments {
            bytes: Bytes::from_static(b"different"),
            calls: AtomicUsize::new(0),
        });
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(resolver);
        let client = builder.build().unwrap();
        assert!(matches!(
            client.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::InvalidRequest { message }) if message.contains("expected 3")
        ));
    }

    #[tokio::test]
    async fn a_missing_remote_attachment_revision_returns_the_resolver_error() {
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(Arc::new(MissingAttachments));
        let client = builder.build().unwrap();

        assert!(matches!(
            client.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::ModelUnavailable { message })
                if message.contains("attachment \"att-1\" revision \"rev-1\" is unavailable")
        ));
    }

    struct NoHttp;
    #[async_trait]
    impl Transport for NoHttp {
        async fn execute(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
        async fn open_stream(&self, _req: HttpRequest) -> Result<StreamResponse, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
        async fn open_responses_websocket_session(
            &self,
            _req: HttpRequest,
        ) -> Result<Box<dyn WebSocketSession>, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
    }
    fn profile(name: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(serde_json::json!({
            "provider_id": "acme", "profile_name": name, "base_url": "https://x.test",
            "protocol": protocol, "auth": "none",
            "models": [{"display_model": "m1", "request_model": "m1", "billing_model": "m1"}]
        }))
        .unwrap()
    }

    /// Gate 33's error path. Every family in the closed set now ships a codec,
    /// so this is only reachable by removing one or by adding a tenth family —
    /// which is exactly what it is here to catch. `tests/families.rs` asserts
    /// the covering invariant that keeps it unreachable in practice.
    #[test]
    fn build_names_the_profile_and_the_family_when_a_codec_is_missing() {
        let mut b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "gemini_generate_content")],
        );
        b.codecs.remove(&ProtocolFamily::GeminiGenerateContent);
        assert_eq!(
            b.build()
                .err()
                .expect("a profile with no codec cannot build"),
            BuildError::MissingCodec {
                profile_name: "p1".into(),
                family: ProtocolFamily::GeminiGenerateContent
            }
        );
    }

    #[test]
    fn a_family_this_crate_speaks_needs_no_capability_entry() {
        let b =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("p1", "open_ai_chat")]);
        assert!(
            b.build().is_ok(),
            "an OpenAI-compatible provider is a settings entry and nothing else (gate 30)"
        );
    }

    #[test]
    fn build_rejects_duplicate_profile_names() {
        let b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "open_ai_chat"), profile("p1", "open_ai_chat")],
        );
        assert_eq!(
            b.build().err().unwrap(),
            BuildError::DuplicateProfile {
                profile_name: "p1".into()
            }
        );
    }

    #[test]
    fn empty_builder_builds_and_lists_nothing() {
        let c = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[])
            .build()
            .unwrap();
        assert!(c.models().is_empty());
    }

    #[test]
    fn build_rejects_empty_and_malformed_peak_windows() {
        use lingxi_agent_api::protocol::PeakSchedule;

        for windows in [vec![], vec!["25:00-26:00".to_owned()]] {
            let mut profile = profile("p1", "open_ai_chat");
            profile.pricing.peak = Some(PeakSchedule {
                utc_windows: windows,
                weekdays_only: false,
                off_peak_multiplier: 0.5,
            });
            assert!(matches!(
                LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile]).build(),
                Err(BuildError::InvalidPeakSchedule { .. })
            ));
        }
    }

    #[test]
    fn clock_override_controls_time_based_pricing() {
        use lingxi_agent_api::protocol::{PeakSchedule, TokenPricing};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        struct FixedClock(u64);
        impl Clock for FixedClock {
            fn now(&self) -> SystemTime {
                UNIX_EPOCH + Duration::from_secs(self.0)
            }
        }

        let mut p = profile("p1", "open_ai_chat");
        p.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            ..Default::default()
        });
        p.pricing.peak = Some(PeakSchedule {
            weekdays_only: false,
            utc_windows: vec!["09:00-17:00".into()],
            off_peak_multiplier: 0.5,
        });
        for (hour, expected) in [(12, 2.0), (20, 1.0)] {
            let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p.clone()]);
            builder.with_clock(Arc::new(FixedClock(hour * 3600)));
            let client = builder.build().unwrap();
            let route = client.resolve("m1").unwrap();
            let cost = client
                .estimate_cost(
                    &route,
                    &Usage {
                        input_tokens: 1_000_000,
                        ..Default::default()
                    },
                    Submission::Interactive,
                )
                .unwrap()
                .unwrap();
            assert_eq!(cost.total_usd, expected);
        }
    }

    #[test]
    fn duplicate_display_models_use_the_resolved_model_for_pricing() {
        use lingxi_agent_api::protocol::{ModelProfile, TokenPricing};

        let mut p = profile("p1", "open_ai_chat");
        p.models = vec![
            serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-a",
                "billing_model": "bill-a",
                "aliases": ["first"]
            }))
            .unwrap(),
            serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-b",
                "billing_model": "bill-b",
                "aliases": ["second"]
            }))
            .unwrap(),
        ];
        p.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            ..Default::default()
        });
        p.models[1].pricing = Some(TokenPricing {
            input_per_million: Some(7.0),
            ..Default::default()
        });

        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p])
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();
        assert_eq!(route.display_model, "shared");
        assert_eq!(route.request_model, "wire-b");
        assert_eq!(route.pricing_model.billing_model, "bill-b");
        let estimate = client
            .estimate_cost(
                &route,
                &Usage {
                    input_tokens: 1_000_000,
                    ..Usage::default()
                },
                Submission::Interactive,
            )
            .unwrap()
            .unwrap();
        assert_eq!(estimate.input_usd, 7.0);
    }

    #[test]
    fn identical_resolved_model_identities_do_not_guess_between_prices() {
        use lingxi_agent_api::protocol::{ModelProfile, TokenPricing};

        let model = |alias: &str, rate: f64| {
            let mut model = serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-shared",
                "billing_model": "bill-shared",
                "aliases": [alias]
            }))
            .unwrap();
            model.pricing = Some(TokenPricing {
                input_per_million: Some(rate),
                ..Default::default()
            });
            model
        };
        let mut p = profile("p1", "open_ai_chat");
        p.models = vec![model("first", 2.0), model("second", 7.0)];
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p])
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();

        assert!(matches!(
            client.estimate_cost(
                &route,
                &Usage {
                    input_tokens: 1_000_000,
                    ..Usage::default()
                },
                Submission::Interactive,
            ),
            Err(LlmError::CostUnavailable { .. })
        ));
    }

    #[test]
    fn actual_cost_estimation_uses_the_resolved_identity_or_rejects_ambiguous_failover() {
        use lingxi_agent_api::protocol::{ModelProfile, TokenPricing};

        let model = |display: &str, billing: &str, alias: &str| {
            let mut model = serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": display,
                "request_model": "wire-shared",
                "billing_model": billing,
                "aliases": [alias]
            }))
            .unwrap();
            model.pricing = Some(TokenPricing {
                input_per_million: Some(if billing == "bill-second" { 7.0 } else { 2.0 }),
                ..Default::default()
            });
            model
        };
        let mut primary = profile("p1", "open_ai_chat");
        primary.models = vec![
            model("first-label", "bill-first", "first"),
            model("second-label", "bill-second", "second"),
        ];
        let mut sibling = profile("p2", "open_ai_chat");
        sibling.models = vec![
            model("other-first", "other-bill-first", "other-first"),
            model("other-second", "other-bill-second", "other-second"),
        ];
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[primary, sibling])
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();
        let usage = Usage {
            input_tokens: 1_000_000,
            ..Usage::default()
        };

        let cost = client
            .estimate_cost_for_profile(&route, "p1", &usage, Submission::Interactive)
            .unwrap()
            .unwrap();
        assert_eq!(cost.input_usd, 7.0);

        let mut ambiguous_route = route;
        ambiguous_route.connection_chain.push(route::ConnectionHop {
            profile_name: "p2".into(),
            request_model: "wire-shared".into(),
        });
        assert!(matches!(
            client.estimate_cost_for_profile(
                &ambiguous_route,
                "p2",
                &usage,
                Submission::Interactive
            ),
            Err(LlmError::CostUnavailable { message }) if message.contains("ambiguously")
        ));
    }
}
