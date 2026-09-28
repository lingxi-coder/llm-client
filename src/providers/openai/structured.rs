//! OpenAI model-specific structured output restrictions.
use crate::codecs::structured::unsupported;
use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, LlmError, OutputFormat};
pub(crate) fn validate_output_contract(
    req: &ChatRequest,
    context: &CodecContext,
) -> Result<(), LlmError> {
    // OpenAI's May 2024 GPT-4o snapshot supports JSON mode but predates
    // json_schema response formatting (introduced with the August snapshot).
    if context.profile().provider_id.as_str() == "openai"
        && context.request_model() == "gpt-4o-2024-05-13"
        && matches!(req.output_format, OutputFormat::JsonSchema { .. })
    {
        return Err(unsupported(
            "gpt-4o-2024-05-13 does not support JSON Schema output",
        ));
    }
    Ok(())
}
