//! OpenAI attachment formats and aggregate document limits.
use crate::protocol::{
    ChatRequest, ContentBlock, DocumentSource, LlmError, ProtocolFamily, ProviderProfile,
};
fn is_image_media_type(media_type: &str) -> bool {
    media_type.trim().to_ascii_lowercase().starts_with("image/")
}
const MAX_OPENAI_INPUT_FILE_BYTES: u64 = 50_000_000;
pub(crate) fn base64_decoded_len(data: &str) -> Result<u64, LlmError> {
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

pub(crate) fn validate_first_party_openai_documents(
    request: &ChatRequest,
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

pub(crate) fn first_party(profile: &ProviderProfile) -> bool {
    matches!(
        profile.protocol,
        ProtocolFamily::OpenAiChat | ProtocolFamily::OpenAiResponses
    ) && super::super::attachment_policy::official_https_host(profile, "api.openai.com")
}
pub(crate) const IMAGE_FORMATS: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
