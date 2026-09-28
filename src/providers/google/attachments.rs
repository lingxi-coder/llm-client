//! Google media types and first-party endpoint authority.
use crate::files;
use crate::protocol::{
    ChatRequest, ContentBlock, LlmError, ProtocolFamily, ProviderProfile, VideoSource,
};
pub(crate) fn validate_gemini_video_media_types(
    request: &ChatRequest,
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

pub(crate) fn first_party(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::GeminiGenerateContent
        && super::super::attachment_policy::official_https_host(
            profile,
            "generativelanguage.googleapis.com",
        )
}
pub(crate) const IMAGE_FORMATS: &[&str] = &[
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
];
