//! deepseek chat policy composed with the registered wire codec.
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {}
pub(crate) fn validate(
    req: &crate::protocol::ChatRequest,
    context: &crate::codecs::CodecContext,
) -> Result<(), crate::protocol::LlmError> {
    if context.profile().provider_id.as_str() != "deepseek" {
        return Ok(());
    }
    for message in &req.messages {
        if message
            .content
            .iter()
            .any(|b| matches!(b, crate::protocol::ContentBlock::Image { .. }))
            && message.role != crate::protocol::MessageRole::User
        {
            return Err(crate::protocol::LlmError::InvalidRequest {
                message: "DeepSeek images belong to user messages".into(),
            });
        }
    }
    Ok(())
}
