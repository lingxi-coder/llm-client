//! Resolve immutable attachment revisions once and share their Bytes.
use super::*;
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
    pub(crate) async fn resolve_attachments<'a>(
        &self,
        req: &'a CompletionRequest,
    ) -> Result<ResolvedRequest<'a>, LlmError> {
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

        let request = req;
        let mut attachments = Vec::new();
        let mut resolved_by_revision = BTreeMap::<(String, String), (AttachmentRef, Bytes)>::new();
        let mut total_bytes = 0_u64;
        for (message_index, message) in request.messages.iter().enumerate() {
            for (block_index, block) in message.content.iter().enumerate() {
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
}
