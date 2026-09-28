//! zhipu hosted search adapter policy.
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
            domains: DomainPolicy::OneAllowed,
            max_uses: false,
            choice: ChoicePolicy::Auto,
            reserved_tool_name: true,
            choice_wire: ChoiceWire::Chat,
        },
        |search, body| {
            let _ = body;
            let engine = match profile.extra.get("web_search_engine") {
                None => "search_pro",
                Some(Value::String(engine)) if !engine.trim().is_empty() => engine,
                _ => {
                    return Err(web_search::invalid(
                        "extra.web_search_engine must be a nonempty string",
                    ))
                }
            };
            let mut tool = serde_json::json!({"type":"web_search", "web_search":{"enable":true, "search_result":true, "search_engine":engine}});
            if let Some(domain) = search.allowed_domains.first() {
                tool["web_search"]["search_domain_filter"] = serde_json::json!(domain);
            }
            Ok(Some(tool))
        },
    )
}
