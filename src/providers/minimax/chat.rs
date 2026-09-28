//! minimax chat policy composed with the registered wire codec.
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {}

pub(crate) fn validate_video_file(
    profile: &crate::protocol::ProviderProfile,
    model: &str,
    file: &crate::protocol::ProviderFileSource,
) -> Result<(), crate::protocol::LlmError> {
    if profile.provider_id.as_str() != "minimax"
        || !model.eq_ignore_ascii_case("minimax-m3")
        || file.protocol != crate::protocol::ProtocolFamily::AnthropicMessages
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
