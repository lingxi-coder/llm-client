//! Google chat validation before attachment side effects.
use crate::codecs::{CodecContext, EncodeRequest, PreparedMedia, WireCodec};
use crate::protocol::{ChatRequest, ContentBlock, LlmError, ProtocolFamily};
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {}
pub(crate) fn has_hosted_tools(req: &ChatRequest) -> bool {
    crate::providers::google::hosted_tools::has_gemini_hosted_tools(req)
}
pub(crate) fn validate_host(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if req
        .native_options
        .iter()
        .any(|option| option.is::<super::computer::GeminiComputerToolConfig>())
        && context.profile().protocol != ProtocolFamily::GeminiInteractions
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Gemini desktop computer declaration requires the Interactions protocol"
                .into(),
        });
    }
    if has_hosted_tools(req) {
        crate::providers::google::hosted_tools::validate_hosted_tool_request(
            req,
            context.profile(),
            context.request_model(),
        )?;
    }
    Ok(())
}
pub(crate) fn accepts_audio(context: &CodecContext) -> bool {
    matches!(
        context.profile().protocol,
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
    )
}
pub(crate) fn preflight_media(
    req: &ChatRequest,
    media: &[PreparedMedia<'_>],
    context: &CodecContext,
    codec: &dyn WireCodec,
) -> Result<(), LlmError> {
    if accepts_audio(context)
        && req
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .any(|b| matches!(b, ContentBlock::Audio { .. }))
    {
        codec.encoded_body_len(EncodeRequest::new(req).with_media(media), context)?;
    }
    Ok(())
}
