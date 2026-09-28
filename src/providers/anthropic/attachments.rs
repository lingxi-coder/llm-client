//! Anthropic attachment and document policy.
use crate::protocol::{
    ChatRequest, ContentBlock, DocumentSource, LlmError, ProtocolFamily, ProviderProfile,
};
fn is_image_media_type(media_type: &str) -> bool {
    media_type.trim().to_ascii_lowercase().starts_with("image/")
}
pub(crate) fn validate_anthropic_document_media_types(
    request: &ChatRequest,
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

pub(crate) fn first_party(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::AnthropicMessages
        && super::super::attachment_policy::official_https_host(profile, "api.anthropic.com")
}
pub(crate) const IMAGE_FORMATS: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
