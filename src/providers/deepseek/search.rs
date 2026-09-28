//! deepseek hosted search adapter policy.
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
            protocols: &[ProtocolFamily::AnthropicMessages],
            domains: DomainPolicy::None("this adapter does not support domain filters"),
            max_uses: false,
            choice: ChoicePolicy::Auto,
            reserved_tool_name: true,
            choice_wire: ChoiceWire::Messages,
        },
        |search, body| {
            let _ = body;
            Ok(Some(web_search::messages_tool(search)))
        },
    )
}
