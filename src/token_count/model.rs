//! Dispatch optional local tokenization to provider-owned mappings.
use super::{backends::Encoder, LocalTokenCountError};
use crate::protocol::ProviderId;
pub(super) fn encoder_for(
    provider: &ProviderId,
    model: &str,
) -> Result<Option<Encoder>, LocalTokenCountError> {
    match provider.as_str() {
        "openai" => crate::providers::openai::token_count::encoder_for(model),
        "deepseek" => crate::providers::deepseek::token_count::encoder_for(model),
        "qwen" => crate::providers::qwen::token_count::encoder_for(model),
        "kimi" => crate::providers::kimi::token_count::encoder_for(model),
        "zhipu" => crate::providers::zhipu::token_count::encoder_for(model),
        _ => Ok(None),
    }
}
