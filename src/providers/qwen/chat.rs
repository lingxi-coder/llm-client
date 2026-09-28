//! Qwen chat validation shared by unified and typed clients.
use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily};
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {}
pub(crate) fn validate_host(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    crate::providers::dispatch::validate_output_contract(
        req,
        context.profile(),
        context.request_model(),
    )?;
    if context.profile().provider_id.as_str() == "qwen"
        && context.profile().protocol == ProtocolFamily::OpenAiChat
    {
        super::cache::validate(req, context)?;
    }
    Ok(())
}
