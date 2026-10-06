//! OpenAI model-specific structured output restrictions.
use crate::codecs::structured::unsupported;
use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, LlmError, OutputFormat, ProviderProfile};

pub(crate) fn supports_json_schema_model(profile: &ProviderProfile, request_model: &str) -> bool {
    // JSON Object mode is available on this snapshot; JSON Schema is not.
    profile.provider_id.as_str() != "openai" || request_model != "gpt-4o-2024-05-13"
}

pub(crate) fn validate_output_contract(
    req: &ChatRequest,
    context: &CodecContext,
) -> Result<(), LlmError> {
    // OpenAI's May 2024 GPT-4o snapshot supports JSON mode but predates
    // json_schema response formatting (introduced with the August snapshot).
    if !supports_json_schema_model(context.profile(), context.request_model())
        && matches!(req.output_format, OutputFormat::JsonSchema { .. })
    {
        return Err(unsupported(
            "gpt-4o-2024-05-13 does not support JSON Schema output",
        ));
    }
    Ok(())
}
