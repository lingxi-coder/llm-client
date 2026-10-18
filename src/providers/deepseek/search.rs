//! deepseek hosted search adapter policy.
use crate::codecs::web_search::{self, ChoicePolicy, ChoiceWire, DomainPolicy, SearchPolicy};
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile};
use serde_json::{Map, Value};
pub(crate) fn apply(
    req: &ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    if req.hosted_web_search().is_some() && profile.protocol == ProtocolFamily::OpenAiResponses {
        if body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&req.model)
            != "deepseek-v4-pro"
        {
            return Err(web_search::unsupported(
                profile,
                "Responses web_search is documented only for deepseek-v4-pro",
            ));
        }
        return web_search::apply_policy(
            req,
            profile,
            body,
            &SearchPolicy {
                protocols: &[ProtocolFamily::OpenAiResponses],
                domains: DomainPolicy::None("DeepSeek Responses does not document domain filters"),
                max_uses: false,
                choice: ChoicePolicy::Auto,
                reserved_tool_name: true,
                choice_wire: ChoiceWire::None,
            },
            |_, _| Ok(Some(serde_json::json!({"type":"web_search"}))),
        );
    }
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
            Ok(Some(web_search::messages_tool(search)?))
        },
    )
}
