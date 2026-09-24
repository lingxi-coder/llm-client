//! Attachment resolution, upload preparation and request-local resources.

use super::{files, RequestOptions};
use super::{AttachmentResolver, MAX_ATTACHMENT_BYTES};
use crate::{
    auth::Authenticator,
    protocol::{
        AttachmentRef, CompletionRequest, ContentBlock, DocumentSource, ImageSource, LlmError,
        ProtocolFamily, ProviderProfile, VideoSource,
    },
    transport::Transport,
};
use bytes::Bytes;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};
mod cache;
mod plan;
mod prepare;
mod resolve;
pub(crate) use plan::uses_first_party_anthropic_messages;
use plan::*;

pub(crate) struct AttachmentManager {
    http: Arc<dyn Transport>,
    attachment_resolver: Option<Arc<dyn AttachmentResolver>>,
    provider_file_cache: Mutex<BTreeMap<FileCacheKey, CachedProviderFile>>,
    provider_file_upload_locks:
        Mutex<BTreeMap<FileCacheKey, std::sync::Weak<futures::lock::Mutex<()>>>>,
    qwen_file_rate_limiters: Mutex<BTreeMap<(String, String), Arc<files::QwenFileRateLimiter>>>,
}
impl AttachmentManager {
    pub(crate) async fn resolve_image(
        &self,
        attachment: &AttachmentRef,
    ) -> Result<Bytes, LlmError> {
        if attachment.size_bytes > MAX_ATTACHMENT_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: "image attachment exceeds 64 MiB".into(),
            });
        }
        let resolver =
            self.attachment_resolver
                .as_ref()
                .ok_or_else(|| LlmError::UnsupportedCapability {
                    message: "image attachment requires an AttachmentResolver".into(),
                })?;
        let bytes = resolver.resolve(attachment).await?;
        if bytes.len() as u64 != attachment.size_bytes || bytes.len() as u64 > MAX_ATTACHMENT_BYTES
        {
            return Err(LlmError::InvalidRequest {
                message: "resolved image attachment size does not match its reference".into(),
            });
        }
        Ok(bytes)
    }
    pub(super) fn new(
        http: Arc<dyn Transport>,
        attachment_resolver: Option<Arc<dyn AttachmentResolver>>,
    ) -> Self {
        Self {
            http,
            attachment_resolver,
            provider_file_cache: Default::default(),
            provider_file_upload_locks: Default::default(),
            qwen_file_rate_limiters: Default::default(),
        }
    }
    pub(super) fn invalidate_profiles(&self, names: &BTreeSet<String>) {
        self.provider_file_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|key, _| !names.contains(&key.profile_name));
    }
}

pub(crate) const INLINE_IMAGE_PREFERENCE_LIMIT: usize = 5 * 1024 * 1024;
const MAX_PROVIDER_FILE_CACHE_ENTRIES: usize = 256;
const MAX_OPENAI_INPUT_FILE_BYTES: u64 = 50_000_000;
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
pub(crate) struct ResolvedRequest<'a> {
    pub request: &'a CompletionRequest,
    pub attachments: Vec<ResolvedAttachmentPayload>,
}

pub(crate) struct ProviderFilePreparation<'a> {
    /// Selected-model data for capability checks; authentication uses the original profile.
    pub planning_profile: &'a ProviderProfile,
    pub opts: &'a RequestOptions,
    pub request_deadline: Option<Instant>,
    pub stable_account_scope: Option<&'a str>,
    pub inline_image_data_budget_bytes: Option<usize>,
    pub authenticator: Option<&'a dyn Authenticator>,
    pub credential: Option<&'a crate::protocol::Secret<String>>,
    pub automatic_cleanup: Option<Arc<files::AutomaticFileCleanup>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FileCacheKey {
    attachment_id: String,
    revision: String,
    filename: String,
    media_type: String,
    profile_name: String,
    provider_id: crate::protocol::ProviderId,
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
    pub(crate) file_id: String,
    pub(crate) uri: Option<String>,
}

struct PlannedUpload<'a> {
    payload: &'a ResolvedAttachmentPayload,
    purpose: files::FilePurpose,
    retention: Option<std::time::Duration>,
}
struct AttachmentPlan<'a> {
    uploads: Vec<PlannedUpload<'a>>,
}
pub(crate) struct PreparedAttachments<'a> {
    pub uses: Vec<PreparedProviderFileUse>,
    pub bindings: Vec<crate::codecs::ContentBinding<'a>>,
    pub cleanup: Option<Arc<files::AutomaticFileCleanup>>,
}
pub(crate) fn provider_file_binding<'a>(
    request: &'a CompletionRequest,
    payload: &ResolvedAttachmentPayload,
    file: crate::protocol::ProviderFileSource,
) -> Result<crate::codecs::ContentBinding<'a>, LlmError> {
    let original = request
        .messages
        .get(payload.message_index)
        .and_then(|m| m.content.get(payload.block_index))
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "resolved attachment no longer matches the request".into(),
        })?;
    let replacement = match (payload.kind, original) {
        (AttachmentKind::Image, ContentBlock::Image { .. }) => ContentBlock::Image {
            source: ImageSource::ProviderFile { file },
        },
        (AttachmentKind::Document, ContentBlock::Document { title, .. }) => {
            ContentBlock::Document {
                source: DocumentSource::ProviderFile { file },
                title: Some(
                    title
                        .as_ref()
                        .unwrap_or(&payload.attachment.filename)
                        .clone(),
                ),
            }
        }
        (AttachmentKind::Video, ContentBlock::Video { .. }) => ContentBlock::Video {
            source: VideoSource::ProviderFile { file },
        },
        _ => {
            return Err(LlmError::InvalidRequest {
                message: "resolved attachment kind does not match its request block".into(),
            })
        }
    };
    Ok(crate::codecs::ContentBinding {
        original,
        replacement,
    })
}
