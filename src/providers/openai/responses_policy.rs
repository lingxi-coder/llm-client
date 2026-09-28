//! First-party Responses endpoint, search and model policy.
use crate::protocol::{LlmError, ProviderProfile};
use serde_json::{json, Map, Value};

pub(crate) fn apply_web_search(
    req: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    let Some(search) = req.hosted_web_search() else {
        return Ok(());
    };
    let adapter = profile
        .extra
        .get("web_search")
        .and_then(Value::as_str)
        .ok_or_else(|| LlmError::UnsupportedCapability {
            message: format!(
                "web search on profile {:?}: no extra.web_search adapter declared",
                profile.profile_name
            ),
        })?;
    if adapter != "openai_responses" || !is_official_openai_responses_profile(profile) {
        return crate::codecs::web_search::apply(req, profile, body);
    }
    if profile.protocol != crate::protocol::ProtocolFamily::OpenAiResponses {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI hosted web search requires the Responses protocol".into(),
        });
    }
    if search.max_uses.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses web_search does not expose max_uses".into(),
        });
    }
    if search.allowed_domains.len() > 100 || search.blocked_domains.len() > 100 {
        return Err(LlmError::InvalidRequest {
            message: "OpenAI Responses web_search accepts at most 100 domains per filter".into(),
        });
    }
    for domain in search.allowed_domains.iter().chain(&search.blocked_domains) {
        if domain.trim().is_empty()
            || domain.contains("://")
            || domain.contains('/')
            || domain.chars().any(char::is_whitespace)
        {
            return Err(LlmError::InvalidRequest {
                message: "web search domain filters must be host names without schemes or paths"
                    .into(),
            });
        }
    }
    let mut tool = json!({"type": "web_search"});
    if !search.allowed_domains.is_empty() || !search.blocked_domains.is_empty() {
        let mut filters = Map::new();
        if !search.allowed_domains.is_empty() {
            filters.insert("allowed_domains".into(), json!(search.allowed_domains));
        }
        if !search.blocked_domains.is_empty() {
            filters.insert("blocked_domains".into(), json!(search.blocked_domains));
        }
        tool["filters"] = Value::Object(filters);
    }
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Responses hosted tools must encode as an array".into(),
        })?
        .push(tool);
    // Include the provider's source records as well as message citations so
    // open_page/fetch actions remain available in native output metadata.
    body.insert("include".into(), json!(["web_search_call.action.sources"]));
    Ok(())
}

pub(crate) fn is_official_openai_responses_profile(profile: &ProviderProfile) -> bool {
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    profile.provider_id.as_str() == "openai"
        && profile.protocol == crate::protocol::ProtocolFamily::OpenAiResponses
        && url.scheme() == "https"
        && url.host_str() == Some("api.openai.com")
        && url.path().trim_end_matches('/') == "/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

pub(crate) fn supports_openai_tool_search_model(model: &str) -> bool {
    let Some(version) = model
        .to_ascii_lowercase()
        .strip_prefix("gpt-")
        .map(str::to_owned)
    else {
        return false;
    };
    let major_end = version
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(version.len());
    if major_end == 0
        || version.get(major_end..).is_some_and(|tail| {
            !tail.is_empty() && !tail.starts_with('.') && !tail.starts_with('-')
        })
    {
        return false;
    }
    let Ok(major) = version[..major_end].parse::<u32>() else {
        return false;
    };
    if major > 5 {
        return true;
    }
    if major < 5 || version.as_bytes().get(major_end) != Some(&b'.') {
        return false;
    }
    let minor = &version[major_end + 1..];
    let minor_end = minor
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(minor.len());
    if minor_end == 0
        || minor
            .get(minor_end..)
            .is_some_and(|tail| !tail.is_empty() && !tail.starts_with('-'))
    {
        return false;
    }
    minor[..minor_end]
        .parse::<u32>()
        .is_ok_and(|minor| minor >= 4)
}

pub(crate) fn chat_pdf_only_files(profile: &ProviderProfile) -> bool {
    profile.extra.get("chat_pdf_only").and_then(Value::as_bool) == Some(true)
        || url::Url::parse(&profile.base_url)
            .ok()
            .is_some_and(|url| url.host_str() == Some("api.openai.com"))
}

pub(crate) fn validate_tool_search(
    req: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<(), LlmError> {
    use super::types::OpenAiToolSearchExecution;
    let openai_tool_search = req.hosted_openai_tool_search();
    let has_openai_tool_search = openai_tool_search.is_some();
    let deferred_mcp = req
        .remote_mcp_servers()
        .any(|server| server.defer_loading());
    if deferred_mcp && !has_openai_tool_search {
        return Err(LlmError::UnsupportedCapability {
            message: "deferred OpenAI MCP servers require OpenAI Responses tool search".into(),
        });
    }
    if let Some(config) = openai_tool_search {
        if !is_official_openai_responses_profile(profile) {
            return Err(LlmError::UnsupportedCapability {
                message: "OpenAI tool search requires the official OpenAI Responses profile".into(),
            });
        }
        if !supports_openai_tool_search_model(model) {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "OpenAI Responses tool search requires GPT-5.4 or later; model {:?} is not supported",
                    model
                ),
            });
        }
        config.validate()?;
        if deferred_mcp && config.execution != OpenAiToolSearchExecution::Server {
            return Err(LlmError::UnsupportedCapability {
                message: "deferred OpenAI MCP servers require server-executed tool search".into(),
            });
        }
        if config.execution == OpenAiToolSearchExecution::Server
            && !req.tools.iter().any(|tool| tool.defer_loading)
            && !deferred_mcp
        {
            return Err(LlmError::InvalidRequest {
                message:
                    "server-executed OpenAI tool search requires a deferred function or MCP server"
                        .into(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_mcp_enabled(profile: &ProviderProfile) -> Result<(), LlmError> {
    if !is_official_openai_responses_profile(profile)
        || profile.extra.get("remote_mcp").and_then(Value::as_str) != Some("openai_responses")
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "remote MCP is not enabled for OpenAI Responses profile {:?}",
                profile.profile_name
            ),
        });
    }
    Ok(())
}
