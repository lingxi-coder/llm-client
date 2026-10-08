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
            let allowed_domains = web_search::domain_strings(&search.allowed_domains)?;
            let blocked_domains = web_search::domain_strings(&search.blocked_domains)?;
            let mut tool = serde_json::json!({"type":"openrouter:web_search"});
            if !allowed_domains.is_empty() {
                tool["parameters"] = serde_json::json!({"allowed_domains":allowed_domains});
            }
            if !blocked_domains.is_empty() {
                tool["parameters"] = serde_json::json!({"excluded_domains":blocked_domains});
            }
            Ok(Some(tool))
        },
    )
}
