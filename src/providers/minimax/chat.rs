//! minimax chat policy composed with the registered wire codec.
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {
    fn validate_sealed_body(
        &self,
        profile: &crate::protocol::ProviderProfile,
        size: usize,
    ) -> Result<(), crate::protocol::LlmError> {
        if profile.protocol == crate::protocol::ProtocolFamily::OpenAiChat {
            if size > 64_000_000 {
                return Err(crate::protocol::LlmError::RequestTooLarge {
                    message: "MiniMax Chat request body exceeds 64 MB".into(),
                });
            }
            Ok(())
        } else if profile.protocol == crate::protocol::ProtocolFamily::AnthropicMessages {
            super::super::anthropic::chat::validate_body_size(size)
        } else {
            Ok(())
        }
    }
}

pub(crate) fn validate_chat_video(
    context: &crate::codecs::CodecContext,
    media_type: Option<&str>,
    size: Option<usize>,
) -> Result<(), crate::protocol::LlmError> {
    use crate::protocol::{LlmError, ProtocolFamily};
    if context.profile().provider_id.as_str() != "minimax"
        || context.profile().protocol != ProtocolFamily::OpenAiChat
        || !matches!(
            context.request_model().to_ascii_lowercase().as_str(),
            "minimax-m3" | "minimax-m3.1-flash-preview"
        )
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Chat video input requires a documented MiniMax M3 model".into(),
        });
    }
    if media_type.is_some_and(|mime| {
        !matches!(
            mime,
            "video/mp4" | "video/x-msvideo" | "video/avi" | "video/quicktime" | "video/x-matroska"
        )
    }) {
        return Err(LlmError::UnsupportedCapability {
            message: "MiniMax Chat video accepts MP4, AVI, MOV or MKV".into(),
        });
    }
    if size.is_some_and(|bytes| bytes > 50_000_000) {
        return Err(LlmError::RequestTooLarge {
            message: "MiniMax inline video exceeds 50 MB; use a scoped uploaded file".into(),
        });
    }
    Ok(())
}

pub(crate) fn chat_video_uri(
    source: &crate::protocol::VideoSource,
    context: &crate::codecs::CodecContext,
) -> Result<String, crate::protocol::LlmError> {
    use crate::protocol::{LlmError, VideoSource};
    use base64::Engine;
    validate_chat_video(context, None, None)?;
    match source {
        VideoSource::Url { url } => {
            let parsed = url::Url::parse(url).map_err(|_| LlmError::InvalidRequest {
                message: "invalid MiniMax video URL".into(),
            })?;
            if parsed.scheme() != "https"
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
            {
                return Err(LlmError::InvalidRequest {message:"MiniMax video URLs must use HTTPS without embedded credentials; uploaded files require a scoped reference".into()});
            }
            Ok(url.clone())
        }
        VideoSource::Base64 { media_type, data } => {
            let padding = if data.ends_with("==") {
                2
            } else if data.ends_with('=') {
                1
            } else {
                0
            };
            validate_chat_video(
                context,
                Some(media_type),
                Some((data.len().saturating_mul(3) / 4).saturating_sub(padding)),
            )?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|_| LlmError::InvalidRequest {
                    message: "invalid base64 video input".into(),
                })?;
            if bytes.is_empty() {
                return Err(LlmError::InvalidRequest {
                    message: "video input is empty".into(),
                });
            }
            validate_chat_video(context, Some(media_type), Some(bytes.len()))?;
            Ok(format!("data:{media_type};base64,{data}"))
        }
        VideoSource::ProviderFile { .. } => video_uri(source, context, context.request_model()),
        VideoSource::Attachment { .. } => Err(crate::codecs::unresolved_attachment_error()),
    }
}

pub(crate) fn validate_video_file(
    profile: &crate::protocol::ProviderProfile,
    model: &str,
    file: &crate::protocol::ProviderFileSource,
) -> Result<(), crate::protocol::LlmError> {
    if profile.provider_id.as_str() != "minimax"
        || !matches!(
            model.to_ascii_lowercase().as_str(),
            "minimax-m3" | "minimax-m3.1-flash-preview"
        )
        || !matches!(
            file.protocol,
            crate::protocol::ProtocolFamily::AnthropicMessages
                | crate::protocol::ProtocolFamily::OpenAiChat
        )
        || file.purpose.as_deref() != Some("video_understanding")
    {
        return Err(crate::protocol::LlmError::UnsupportedCapability {
            message: "video file references are supported only by MiniMax M3".into(),
        });
    }
    Ok(())
}

pub(crate) fn video_uri(
    source: &crate::protocol::VideoSource,
    context: &crate::codecs::CodecContext,
    model: &str,
) -> Result<String, crate::protocol::LlmError> {
    let crate::protocol::VideoSource::ProviderFile { file } = source else {
        return Err(crate::protocol::LlmError::UnsupportedCapability {
            message:
                "MiniMax video input requires a provider file uploaded for video_understanding"
                    .into(),
        });
    };
    let file = crate::codecs::validate_provider_file(file, context.profile(), context)?;
    validate_video_file(context.profile(), model, file)?;
    Ok(format!("mm_file://{}", file.file_id))
}
