//! MiniMax supports explicit short-lived prompt cache markers only.
use crate::codecs::CodecContext;
use crate::protocol::{CacheTtl, ChatRequest, LlmError};
pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let policy = &req.prompt_cache;
    if context.profile().provider_id.as_str() == "minimax"
        && (policy.automatic.is_some()
            || policy
                .breakpoints
                .iter()
                .any(|b| b.ttl == CacheTtl::OneHour))
    {
        return Err(LlmError::UnsupportedCapability {
            message: "MiniMax supports explicit five-minute cache breakpoints only".into(),
        });
    }
    Ok(())
}
