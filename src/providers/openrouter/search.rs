//! openrouter hosted search adapter policy.
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
            protocols: &[ProtocolFamily::OpenAiChat],
            domains: DomainPolicy::Any,
            max_uses: false,
            choice: ChoicePolicy::Any,
            reserved_tool_name: true,
            choice_wire: ChoiceWire::Chat,
        },
        |search, body| {
            let _ = body;
            let mut tool = serde_json::json!({"type":"openrouter:web_search"});
            if !search.allowed_domains.is_empty() {
                tool["parameters"] = serde_json::json!({"allowed_domains":search.allowed_domains});
            }
            if !search.blocked_domains.is_empty() {
                tool["parameters"] = serde_json::json!({"excluded_domains":search.blocked_domains});
            }
            Ok(Some(tool))
        },
    )
}
