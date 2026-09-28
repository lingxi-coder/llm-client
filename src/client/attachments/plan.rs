//! Validate every attachment and choose transfer modes before any I/O.
use super::*;

pub(super) fn is_image_media_type(media_type: &str) -> bool {
    media_type.trim().to_ascii_lowercase().starts_with("image/")
}

pub(super) fn validate_image_attachment_media_types(request: &ChatRequest) -> Result<(), LlmError> {
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

pub(super) struct FilePlanOptions {
    pub file_validation_time: std::time::SystemTime,
    pub endpoint: FirstPartyEndpoint,
    pub inline_image_data_budget_bytes: Option<usize>,
}

pub(super) fn validate_first_party_image_media_types(
    request: &ChatRequest,
    endpoint: FirstPartyEndpoint,
) -> Result<(), LlmError> {
    let Some((provider, formats)) = crate::providers::attachment_policy::image_formats(endpoint)
    else {
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

impl AttachmentManager {
    pub(super) fn plan_provider_files<'a>(
        &self,
        profile: &ProviderProfile,
        model: &str,
        request: &ChatRequest,
        attachments: &'a [ResolvedAttachmentPayload],
        opts: &RequestOptions,
        planning: FilePlanOptions,
    ) -> Result<AttachmentPlan<'a>, LlmError> {
        let FilePlanOptions {
            file_validation_time,
            endpoint,
            inline_image_data_budget_bytes,
        } = planning;
        const MAX_INLINE_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;
        let mut uploads = Vec::new();
        crate::files::validate_direct_provider_file_inputs_at(
            request.messages.iter().flat_map(|m| &m.content),
            profile,
            model,
            opts.file_account_scope.as_deref(),
            file_validation_time,
        )?;
        validate_first_party_image_media_types(request, endpoint)?;
        crate::providers::attachment_policy::validate_request(
            profile,
            model,
            request,
            attachments,
            opts,
            endpoint,
        )?;
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
                && !crate::providers::attachment_policy::requires_file_reference(profile, model)
                && payload.bytes.len() <= INLINE_IMAGE_PREFERENCE_LIMIT
                && inline_image_data_budget_bytes
                    .is_none_or(|budget| inline_image_data_bytes <= budget)
            {
                continue;
            }
            let purpose = crate::providers::attachment_policy::file_purpose(profile, payload.kind);
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
