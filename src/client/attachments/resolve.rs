//! Resolve immutable attachment revisions once and share their Bytes.
use super::*;
use futures::{StreamExt, TryStreamExt};

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

impl AttachmentManager {
    pub(super) async fn resolve_content(
        &self,
        resolver: &dyn AttachmentResolver,
        attachment: &AttachmentRef,
    ) -> Result<Bytes, LlmError> {
        // Every caller must pass the host's authority check before looking up
        // cached bytes or waiting for another caller's read. Existing resolvers
        // opt out by default and keep resolve() as their per-request check.
        let reuse = resolver.validate_content_reuse(attachment).await?;
        let cache_key = reuse.then(|| ContentCacheKey::from(attachment));
        let read_lock = cache_key.as_ref().map(|key| self.content_read_lock(key));
        let _read_guard = match read_lock.as_ref() {
            Some(lock) => Some(lock.lock().await),
            None => None,
        };
        if let Some(bytes) = cache_key.as_ref().and_then(|key| self.cached_content(key)) {
            return Ok(bytes);
        }
        let bytes = resolver.resolve(attachment).await?;
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
        if let Some(key) = cache_key {
            return Ok(self.cache_content(key, bytes));
        }
        Ok(bytes)
    }

    pub(crate) async fn resolve_attachments<'a>(
        &self,
        req: &'a ChatRequest,
    ) -> Result<ResolvedRequest<'a>, LlmError> {
        validate_image_attachment_media_types(req)?;
        let mut locations = Vec::new();
        let mut unique = Vec::<&AttachmentRef>::new();
        let mut by_revision = BTreeMap::<(&str, &str), usize>::new();
        let mut total_bytes = 0_u64;
        // Validate the complete request before initiating any resolver I/O.
        for (message_index, message) in req.messages.iter().enumerate() {
            for (block_index, block) in message.content.iter().enumerate() {
                let (attachment, kind) = match block {
                    ContentBlock::Image {
                        source: ImageSource::Attachment { attachment },
                    } => (attachment, AttachmentKind::Image),
                    ContentBlock::Document {
                        source: DocumentSource::Attachment { attachment },
                        ..
                    } => (attachment, AttachmentKind::Document),
                    ContentBlock::Video {
                        source: VideoSource::Attachment { attachment },
                    } => (attachment, AttachmentKind::Video),
                    _ => continue,
                };
                validate_attachment_metadata(attachment)?;
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
                    attachment.attachment_id.as_str(),
                    attachment.revision.as_str(),
                );
                let unique_index = if let Some(&index) = by_revision.get(&key) {
                    if unique[index] != attachment {
                        return Err(LlmError::InvalidRequest {
                            message: format!(
                                "attachment {:?} revision {:?} has inconsistent metadata",
                                attachment.attachment_id, attachment.revision
                            ),
                        });
                    }
                    index
                } else {
                    let index = unique.len();
                    by_revision.insert(key, index);
                    unique.push(attachment);
                    index
                };
                locations.push((message_index, block_index, kind, unique_index));
            }
        }
        if unique.is_empty() {
            return Ok(ResolvedRequest {
                request: req,
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
        // Read independent revisions concurrently and propagate failures as
        // soon as they arrive. Restore occurrence order after completion.
        let pending: Vec<_> = unique
            .iter()
            .enumerate()
            .map(|(index, attachment)| async move {
                self.resolve_content(resolver.as_ref(), attachment)
                    .await
                    .map(|bytes| (index, bytes))
            })
            .collect();
        let mut resolved: Vec<(usize, Bytes)> = futures::stream::iter(pending)
            .buffer_unordered(ATTACHMENT_CONCURRENCY)
            .try_collect()
            .await?;
        resolved.sort_unstable_by_key(|(index, _)| *index);
        let attachments = locations
            .into_iter()
            .map(
                |(message_index, block_index, kind, index)| ResolvedAttachmentPayload {
                    message_index,
                    block_index,
                    kind,
                    attachment: unique[index].clone(),
                    bytes: resolved[index].1.clone(),
                },
            )
            .collect();
        Ok(ResolvedRequest {
            request: req,
            attachments,
        })
    }
}
