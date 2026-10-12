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
            let mut tool = web_search::responses_tool(search);
            if codex_responses(profile) {
                tool["external_web_access"] = Value::Bool(true);
            }
            Ok(Some(tool))
        },
    )
}

// Hosts may defer authentication while preparing the request, so recognize
// the Codex endpoint rather than relying on the temporary auth strategy.
pub(crate) fn codex_responses(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::OpenAiResponses
        && url::Url::parse(&profile.base_url).is_ok_and(|url| {
            url.scheme() == "https"
                && url.host_str() == Some("chatgpt.com")
                && url.path().trim_end_matches('/') == "/backend-api/codex"
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        })
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
