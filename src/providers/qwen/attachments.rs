//! Qwen-Long attachment reference limits and preflight authority checks.
use crate::protocol::{
    ChatRequest, ContentBlock, DocumentSource, ImageSource, LlmError, ProtocolFamily,
    ProviderProfile,
};
use crate::{
    client::{AttachmentKind, RequestOptions, ResolvedAttachmentPayload},
    files,
};

pub(crate) fn is_qwen_long(profile: &ProviderProfile, model: &str) -> bool {
    profile.provider_id.as_str() == "qwen" && files::is_qwen_long_model(model)
}

pub(crate) fn preflight_qwen_long_files(
    profile: &ProviderProfile,
    model: &str,
    request: &ChatRequest,
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
        let file = crate::files::validate_provider_file_identity(
            file,
            profile,
            opts.file_account_scope.as_deref(),
        )?;
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
