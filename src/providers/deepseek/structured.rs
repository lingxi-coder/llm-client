//! First-party DeepSeek JSON Output prompt requirement.
use crate::codecs::structured::{has_json_prompt_keyword, invalid};
use crate::protocol::{ChatRequest, LlmError, OutputFormat, ProtocolFamily, ProviderProfile};
pub(crate) fn validate_output_contract(
    req: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    if is_deepseek_chat_profile(profile)
        && matches!(req.output_format, OutputFormat::JsonObject)
        && !has_json_prompt_keyword(req)
    {
        return Err(invalid(
            "DeepSeek JSON Output requires the word JSON in system or user text",
        ));
    }
    Ok(())
}
fn is_deepseek_chat_profile(profile: &crate::protocol::ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "deepseek" || profile.protocol != ProtocolFamily::OpenAiChat
    {
        return false;
    }
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("api.deepseek.com")
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && matches!(url.path(), "/" | "/v1" | "/v1/")
        && url.query().is_none()
        && url.fragment().is_none()
}
