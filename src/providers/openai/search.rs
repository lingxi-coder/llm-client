//! openai hosted search adapter policy.
use crate::codecs::web_search::{self, ChoicePolicy, ChoiceWire, DomainPolicy, SearchPolicy};
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile};
use serde_json::{Map, Value};
pub(crate) fn responses(
    req: &ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    web_search::apply_policy(
        req,
        profile,
        body,
        &SearchPolicy {
            protocols: &[ProtocolFamily::OpenAiResponses],
            domains: DomainPolicy::AllowedOnly,
            max_uses: false,
            choice: ChoicePolicy::Any,
            reserved_tool_name: false,
            choice_wire: ChoiceWire::Chat,
        },
        |search, body| {
            let _ = body;
            Ok(Some(web_search::responses_tool(search)))
        },
    )
}
pub(crate) fn chat(
    req: &ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    web_search::apply_policy(
        req,
        profile,
        body,
        &SearchPolicy {
            protocols: &[ProtocolFamily::OpenAiChat],
            domains: DomainPolicy::None("this adapter does not support domain filters"),
            max_uses: false,
            choice: ChoicePolicy::Auto,
            reserved_tool_name: false,
            choice_wire: ChoiceWire::None,
        },
        |search, body| {
            let _ = search;
            body.insert("web_search_options".into(), serde_json::json!({}));
            Ok(None)
        },
    )
}
