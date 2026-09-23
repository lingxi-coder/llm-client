//! Explicit endpoint adapters: compatible JSON alone does not imply search.
use lingxi_agent_api::protocol::{
    CompletionRequest, LlmError, ProtocolFamily, ProviderProfile, ToolChoice,
};
use serde_json::{json, Map, Value};

/// Add the configured hosted tool after ordinary tools, before profile extras.
/// Adapter selection is data-driven; provider names never select behavior.
pub(crate) fn apply(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    let Some(search) = &req.web_search else {
        return Ok(());
    };
    let unsupported = |message: &str| LlmError::UnsupportedCapability {
        message: format!(
            "web search on profile {:?}: {message}",
            profile.profile_name
        ),
    };
    let invalid = |message: &str| LlmError::InvalidRequest {
        message: message.to_owned(),
    };
    let adapter = profile
        .extra
        .get("web_search")
        .and_then(Value::as_str)
        .ok_or_else(|| unsupported("no extra.web_search adapter declared"))?;
    use ProtocolFamily::*;
    let compatible = match adapter {
        "openai_responses" | "xai" | "kimi" => profile.protocol == OpenAiResponses,
        "deepseek" => profile.protocol == AnthropicMessages,
        "openai_chat" | "openrouter" | "glm" => profile.protocol == OpenAiChat,
        "anthropic" => matches!(
            profile.protocol,
            AnthropicMessages | FoundryClaude | VertexClaude
        ),
        "gemini" => matches!(profile.protocol, GeminiGenerateContent | VertexGemini),
        _ => false,
    };
    if !compatible {
        return Err(unsupported("unknown adapter or incompatible protocol"));
    }
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
    if search.max_uses.is_some() && adapter != "anthropic" {
        return Err(unsupported(
            "max_uses is supported only by the anthropic adapter",
        ));
    }
    if matches!(adapter, "gemini" | "openai_chat" | "deepseek")
        && (!search.allowed_domains.is_empty() || !search.blocked_domains.is_empty())
    {
        return Err(unsupported("this adapter does not support domain filters"));
    }
    if matches!(adapter, "openai_responses" | "kimi") && !search.blocked_domains.is_empty() {
        return Err(unsupported("this adapter supports allowed_domains only"));
    }
    if adapter == "glm" && (!search.blocked_domains.is_empty() || search.allowed_domains.len() > 1)
    {
        return Err(unsupported(
            "GLM search supports one allowed domain and no blocked domains",
        ));
    }
    if adapter == "xai"
        && search
            .allowed_domains
            .len()
            .max(search.blocked_domains.len())
            > 5
    {
        return Err(invalid(
            "xAI web search accepts at most five domain filters",
        ));
    }
    if matches!(adapter, "openai_responses" | "kimi") && search.allowed_domains.len() > 100 {
        return Err(invalid(
            "this web search adapter accepts at most 100 allowed domains",
        ));
    }
    // Gemini's functionCallingConfig does not control its hosted search tool;
    // Chat search models always search. Do not claim to honor 'none' there.
    if matches!(
        adapter,
        "gemini" | "openai_chat" | "glm" | "kimi" | "deepseek"
    ) && !matches!(req.tool_choice, ToolChoice::Auto)
    {
        return Err(unsupported(
            "search requires automatic tool choice on this adapter",
        ));
    }
    if matches!(adapter, "anthropic" | "openrouter" | "glm" | "deepseek")
        && req.tools.iter().any(|tool| tool.name == "web_search")
    {
        return Err(invalid(
            "a client tool named web_search conflicts with hosted search",
        ));
    }
    let tool = match adapter {
        "openai_responses" | "xai" | "kimi" => {
            if adapter == "kimi" {
                if req.temperature.is_some() || req.thinking.is_some() {
                    return Err(unsupported("Kimi Responses search does not support temperature or a thinking token budget"));
                }
                body.insert("include".into(), json!(["web_search_call.action.sources"]));
            }
            let mut tool = json!({"type": "web_search"});
            if !search.allowed_domains.is_empty() {
                tool["filters"] = json!({"allowed_domains": search.allowed_domains});
            }
            if !search.blocked_domains.is_empty() {
                tool["filters"] = json!({"excluded_domains": search.blocked_domains});
            }
            tool
        }
        "anthropic" | "deepseek" => {
            let mut tool = json!({"type": "web_search_20250305", "name": "web_search"});
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
        "gemini" => json!({"googleSearch": {}}),
        "glm" => {
            let engine = match profile.extra.get("web_search_engine") {
                None => "search_pro",
                Some(Value::String(engine)) if !engine.trim().is_empty() => engine,
                _ => return Err(invalid("extra.web_search_engine must be a nonempty string")),
            };
            let mut tool = json!({"type": "web_search", "web_search": {
                "enable": true, "search_result": true, "search_engine": engine
            }});
            if let Some(domain) = search.allowed_domains.first() {
                tool["web_search"]["search_domain_filter"] = json!(domain);
            }
            tool
        }
        "openrouter" => {
            let mut tool = json!({"type": "openrouter:web_search"});
            if !search.allowed_domains.is_empty() {
                tool["parameters"] = json!({"allowed_domains": search.allowed_domains});
            }
            if !search.blocked_domains.is_empty() {
                tool["parameters"] = json!({"excluded_domains": search.blocked_domains});
            }
            tool
        }
        "openai_chat" => {
            body.insert("web_search_options".into(), json!({}));
            return Ok(());
        }
        _ => unreachable!("adapter validated above"),
    };
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("codec tools are always an array")
        .push(tool);
    // The encoders normally emit tool_choice only when client tools exist.
    // Hosted tools also need it, including explicit None to disable execution.
    if adapter != "gemini" && !body.contains_key("tool_choice") {
        let choice = if matches!(adapter, "anthropic" | "deepseek") {
            match &req.tool_choice {
                ToolChoice::Auto => json!({"type": "auto"}),
                ToolChoice::None => json!({"type": "none"}),
                ToolChoice::Any => json!({"type": "any"}),
                ToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
            }
        } else {
            match &req.tool_choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Any => json!("required"),
                ToolChoice::Tool { name } => {
                    return Err(invalid(&format!(
                        "named client tool {name:?} is not advertised"
                    )))
                }
            }
        };
        body.insert("tool_choice".into(), choice);
    }
    Ok(())
}
