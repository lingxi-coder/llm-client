//! kimi hosted search adapter policy.
use crate::codecs::web_search::{self, ChoicePolicy, ChoiceWire, DomainPolicy, SearchPolicy};
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile};
use serde_json::{Map, Value};
pub(crate) fn apply(
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
            choice: ChoicePolicy::Auto,
            reserved_tool_name: false,
            choice_wire: ChoiceWire::Chat,
        },
        |search, body| {
            if req.temperature.is_some()
                || req
                    .thinking
                    .as_ref()
                    .is_some_and(|t| *t != crate::protocol::ThinkingConfig::default())
            {
                return Err(web_search::unsupported(
                    profile,
                    "Kimi Responses search does not support temperature or reasoning controls",
                ));
            }
            body.insert(
                "include".into(),
                serde_json::json!(["web_search_call.action.sources"]),
            );
            Ok(Some(web_search::responses_tool(search)))
        },
    )
}
