//! xai hosted search adapter policy.
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
            domains: DomainPolicy::Five,
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
