//! kimi chat policy composed with the registered wire codec.
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {}

pub(crate) fn validate(
    req: &crate::protocol::ChatRequest,
    context: &crate::codecs::CodecContext,
) -> Result<(), crate::protocol::LlmError> {
    use crate::protocol::{ContentBlock, ImageSource, LlmError, ProtocolFamily};
    let p = context.profile();
    if p.provider_id.as_str() != "kimi"
        || p.protocol != ProtocolFamily::OpenAiChat
        || context.request_model() != "kimi-k3"
    {
        return Ok(());
    }
    if req.temperature.is_some_and(|value| value != 1.0) {
        return Err(LlmError::InvalidRequest {
            message: "Kimi K3 fixes temperature at 1.0; omit it or use 1.0".into(),
        });
    }
    if req
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| {
            matches!(
                block,
                ContentBlock::Image {
                    source: ImageSource::Url { .. }
                }
            )
        })
    {
        return Err(LlmError::UnsupportedCapability { message: "Kimi K3 Chat vision requires inline base64 or a scoped provider file; public image URLs are unsupported".into() });
    }
    Ok(())
}
