//! Shared web-search validation and wire helpers for explicit endpoint adapters.
use crate::protocol::{
    ChatRequest, LlmError, ProtocolFamily, ProviderProfile, ToolChoice, WebSearchConfig,
};
use serde_json::{json, Map, Value};

pub(crate) fn apply(
    req: &ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    if req.hosted_web_search().is_none() {
        return Ok(());
    }
    let adapter = profile
        .extra
        .get("web_search")
        .and_then(Value::as_str)
        .ok_or_else(|| unsupported(profile, "no extra.web_search adapter declared"))?;
    // Explicit adapter configuration is authoritative, including on compatible
    // custom provider identities. Stable provider IDs never override it.
    match adapter {
        "openai_responses" => crate::providers::openai::search::responses(req, profile, body),
        "openai_chat" => crate::providers::openai::search::chat(req, profile, body),
        "anthropic" => crate::providers::anthropic::search::apply(req, profile, body),
        "gemini" => crate::providers::google::search::apply(req, profile, body),
        "qwen" => crate::providers::qwen::search::apply(req, profile, body),
        "minimax" => crate::providers::minimax::search::apply(req, profile, body),
        "glm" => crate::providers::zhipu::search::apply(req, profile, body),
        "kimi" => crate::providers::kimi::search::apply(req, profile, body),
        "xai" => crate::providers::xai::search::apply(req, profile, body),
        "openrouter" => crate::providers::openrouter::search::apply(req, profile, body),
        "deepseek" => crate::providers::deepseek::search::apply(req, profile, body),
        _ => Err(unsupported(
            profile,
            "unknown adapter or incompatible protocol",
        )),
    }
}

pub(crate) fn unsupported(profile: &ProviderProfile, message: &str) -> LlmError {
    LlmError::UnsupportedCapability {
        message: format!(
            "web search on profile {:?}: {message}",
            profile.profile_name
        ),
    }
}
pub(crate) fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.to_owned(),
    }
}

pub(crate) enum DomainPolicy {
    Any,
    None(&'static str),
    AllowedOnly,
    OneAllowed,
    Five,
}
pub(crate) enum ChoicePolicy {
    Any,
    Auto,
    AutoOrNone,
}
#[derive(Clone, Copy)]
pub(crate) enum ChoiceWire {
    Chat,
    Messages,
    None,
}
pub(crate) struct SearchPolicy {
    pub(crate) protocols: &'static [ProtocolFamily],
    pub(crate) domains: DomainPolicy,
    pub(crate) max_uses: bool,
    pub(crate) choice: ChoicePolicy,
    pub(crate) reserved_tool_name: bool,
    pub(crate) choice_wire: ChoiceWire,
}

pub(crate) fn apply_policy(
    req: &ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
    policy: &SearchPolicy,
    encode: impl FnOnce(&WebSearchConfig, &mut Map<String, Value>) -> Result<Option<Value>, LlmError>,
) -> Result<(), LlmError> {
    let Some(search) = req.hosted_web_search() else {
        return Ok(());
    };
    if !policy.protocols.contains(&profile.protocol) {
        return Err(unsupported(
            profile,
            "unknown adapter or incompatible protocol",
        ));
    }
    validate_domains(search)?;
    if search.max_uses.is_some() && !policy.max_uses {
        return Err(unsupported(
            profile,
            "max_uses is supported only by the anthropic adapter",
        ));
    }
    match policy.domains {
        DomainPolicy::None(message)
            if !search.allowed_domains.is_empty() || !search.blocked_domains.is_empty() =>
        {
            return Err(unsupported(profile, message))
        }
        DomainPolicy::AllowedOnly if !search.blocked_domains.is_empty() => {
            return Err(unsupported(
                profile,
                "this adapter supports allowed_domains only",
            ))
        }
        DomainPolicy::AllowedOnly if search.allowed_domains.len() > 100 => {
            return Err(invalid(
                "this web search adapter accepts at most 100 allowed domains",
            ))
        }
        DomainPolicy::OneAllowed
            if !search.blocked_domains.is_empty() || search.allowed_domains.len() > 1 =>
        {
            return Err(unsupported(
                profile,
                "GLM search supports one allowed domain and no blocked domains",
            ))
        }
        DomainPolicy::Five
            if search
                .allowed_domains
                .len()
                .max(search.blocked_domains.len())
                > 5 =>
        {
            return Err(invalid(
                "xAI web search accepts at most five domain filters",
            ))
        }
        _ => {}
    }
    match policy.choice {
        ChoicePolicy::Auto if !matches!(req.tool_choice, ToolChoice::Auto) => {
            return Err(unsupported(
                profile,
                "search requires automatic tool choice on this adapter",
            ))
        }
        ChoicePolicy::AutoOrNone
            if !matches!(req.tool_choice, ToolChoice::Auto | ToolChoice::None) =>
        {
            return Err(unsupported(
                profile,
                "MiniMax server search supports only automatic or disabled tool choice",
            ))
        }
        _ => {}
    }
    if policy.reserved_tool_name && req.tools.iter().any(|tool| tool.name == "web_search") {
        return Err(invalid(
            "a client tool named web_search conflicts with hosted search",
        ));
    }
    let Some(tool) = encode(search, body)? else {
        return Ok(());
    };
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("codec tools are always an array")
        .push(tool);
    if body.contains_key("tool_choice") {
        return Ok(());
    }
    let choice = match policy.choice_wire {
        ChoiceWire::None => return Ok(()),
        ChoiceWire::Messages => match &req.tool_choice {
            ToolChoice::Auto => json!({"type":"auto"}),
            ToolChoice::None => json!({"type":"none"}),
            ToolChoice::Any => json!({"type":"any"}),
            ToolChoice::Tool { name } => json!({"type":"tool", "name":name}),
        },
        ChoiceWire::Chat => match &req.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Any => json!("required"),
            ToolChoice::Tool { name } => {
                return Err(invalid(&format!(
                    "named client tool {name:?} is not advertised"
                )))
            }
        },
    };
    body.insert("tool_choice".into(), choice);
    Ok(())
}
fn validate_domains(search: &WebSearchConfig) -> Result<(), LlmError> {
    if !search.allowed_domains.is_empty() && !search.blocked_domains.is_empty() {
        return Err(invalid(
            "web search accepts allowed_domains or blocked_domains, not both",
        ));
    }
    if search.max_uses == Some(0) {
        return Err(invalid("web search max_uses must be positive"));
    }
    for domain in search.allowed_domains.iter().chain(&search.blocked_domains) {
        if domain.trim().is_empty()
            || domain.contains("://")
            || domain.chars().any(char::is_whitespace)
            || domain.contains('/')
        {
            return Err(invalid(
                "web search domain filters must be domain names without schemes or paths",
            ));
        }
    }
    Ok(())
}
pub(crate) fn responses_tool(search: &WebSearchConfig) -> Value {
    let mut tool = json!({"type":"web_search"});
    if !search.allowed_domains.is_empty() {
        tool["filters"] = json!({"allowed_domains": search.allowed_domains});
    }
    if !search.blocked_domains.is_empty() {
        tool["filters"] = json!({"excluded_domains": search.blocked_domains});
    }
    tool
}
pub(crate) fn messages_tool(search: &WebSearchConfig) -> Value {
    let mut tool = json!({"type":"web_search_20250305", "name":"web_search"});
    if let Some(max) = search.max_uses {
        tool["max_uses"] = json!(max);
    }
    if !search.allowed_domains.is_empty() {
        tool["allowed_domains"] = json!(search.allowed_domains);
    }
    if !search.blocked_domains.is_empty() {
        tool["blocked_domains"] = json!(search.blocked_domains);
    }
    tool
}
