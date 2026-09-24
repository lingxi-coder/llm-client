//! Validate every attachment and choose transfer modes before any I/O.
use super::*;
pub(super) fn is_qwen_long(profile: &ProviderProfile, model: &str) -> bool {
    profile.provider_id.as_str() == "qwen" && files::is_qwen_long_model(model)
}

pub(super) fn preflight_qwen_long_files(
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
        let file = crate::files::validate_provider_file(
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

pub(super) fn is_image_media_type(media_type: &str) -> bool {
    media_type.trim().to_ascii_lowercase().starts_with("image/")
}

pub(super) fn validate_image_attachment_media_types(
    request: &CompletionRequest,
) -> Result<(), LlmError> {
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

pub(super) fn validate_anthropic_document_media_types(
    request: &CompletionRequest,
) -> Result<(), LlmError> {
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

pub(super) fn uses_first_party_openai_file_inputs(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() == "openai"
        && matches!(
            profile.protocol,
            ProtocolFamily::OpenAiResponses | ProtocolFamily::OpenAiChat
        )
        && url::Url::parse(&profile.base_url)
            .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some("api.openai.com"))
}

pub(super) fn first_party_image_formats(
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
        && url::Url::parse(&profile.base_url).is_ok_and(|url| {
            url.scheme() == "https" && url.host_str() == Some("generativelanguage.googleapis.com")
        })
    {
        return Some(("Gemini", GEMINI));
    }
    None
}

pub(super) fn validate_first_party_image_media_types(
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

pub(super) fn validate_gemini_video_media_types(
    request: &CompletionRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "google"
        || profile.protocol != ProtocolFamily::GeminiGenerateContent
    {
        return Ok(());
    }
    for block in request.messages.iter().flat_map(|message| &message.content) {
        let ContentBlock::Video { source } = block else {
            continue;
        };
        let media_type = match source {
            VideoSource::Base64 { media_type, .. } => Some(media_type.as_str()),
            VideoSource::Attachment { attachment } => Some(attachment.media_type.as_str()),
            VideoSource::ProviderFile { file } => file.media_type.as_deref(),
            VideoSource::Url { .. } => None,
        };
        if media_type.is_some_and(|media_type| !files::is_gemini_video_type(media_type)) {
            return Err(LlmError::UnsupportedCapability {
                message: format!("Gemini does not accept video media type {media_type:?}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn uses_first_party_anthropic_messages(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages
        && url::Url::parse(&profile.base_url)
            .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some("api.anthropic.com"))
}

pub(super) fn base64_decoded_len(data: &str) -> Result<u64, LlmError> {
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

pub(super) fn validate_first_party_openai_documents(
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
impl AttachmentManager {
    pub(super) fn plan_provider_files<'a>(
        &self,
        profile: &ProviderProfile,
        model: &str,
        request: &CompletionRequest,
        attachments: &'a [ResolvedAttachmentPayload],
        opts: &RequestOptions,
        inline_image_data_budget_bytes: Option<usize>,
    ) -> Result<AttachmentPlan<'a>, LlmError> {
        const MAX_INLINE_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;
        let mut uploads = Vec::new();
        crate::files::validate_direct_provider_file_inputs(
            request.messages.iter().flat_map(|m| &m.content),
            profile,
            model,
            opts.file_account_scope.as_deref(),
        )?;
        validate_first_party_image_media_types(request, profile)?;
        validate_gemini_video_media_types(request, profile)?;
        preflight_qwen_long_files(profile, model, request, attachments, opts)?;
        if uses_first_party_openai_file_inputs(profile) {
            validate_first_party_openai_documents(request, profile.protocol)?;
        }
        if profile.protocol == ProtocolFamily::AnthropicMessages {
            validate_anthropic_document_media_types(request)?;
        }
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
            files::validate_media_type(&payload.attachment.media_type)?;
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
                AttachmentKind::Image => crate::protocol::ModelCapability::Vision,
                AttachmentKind::Document => crate::protocol::ModelCapability::Documents,
                AttachmentKind::Video => crate::protocol::ModelCapability::Vision,
            };
            let video_supported = model_profile
                .metadata
                .input_modalities
                .iter()
                .any(|modality| modality.eq_ignore_ascii_case("video"));
            if (payload.kind == AttachmentKind::Video && !video_supported)
                || (payload.kind != AttachmentKind::Video
                    && model_profile.capability_support_for(model_capability)
                        == crate::protocol::CapabilitySupport::Unsupported)
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
            let purpose = if payload.kind == AttachmentKind::Video
                && profile.provider_id.as_str() == "minimax"
            {
                files::FilePurpose::VideoUnderstanding
            } else {
                files::FilePurpose::ModelInput
            };
            let capabilities = files::capabilities_for_purpose(
                profile,
                model,
                &payload.attachment.media_type,
                purpose,
            );
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

            if capabilities
                .max_upload_bytes
                .is_some_and(|limit| payload.bytes.len() as u64 > limit)
            {
                return Err(LlmError::RequestTooLarge {
                    message: "attachment exceeds the provider upload limit".into(),
                });
            }
            uploads.push(PlannedUpload {
                payload,
                purpose,
                retention: capabilities.retention,
            });
            if payload.kind == AttachmentKind::Image {
                inline_image_data_bytes =
                    inline_image_data_bytes.saturating_sub(payload.bytes.len().div_ceil(3) * 4);
            }
        }
        Ok(AttachmentPlan { uploads })
    }
}
