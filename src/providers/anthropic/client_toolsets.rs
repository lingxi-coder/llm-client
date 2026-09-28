//! Request-side declarations for Anthropic's stable Browser and Computer
//! client toolsets. The host executes member calls; these are never hosted
//! provider tools.

use crate::codecs::CodecContext;
use crate::protocol::{
    CacheTtl, ChatRequest, LlmError, ProtocolFamily, ProviderProfile, ToolChoice,
};
use crate::providers::anthropic::types::{AnthropicClientToolset, AnthropicMcpCacheTtl};
use serde_json::Value;

const SUPPORTED_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
];

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

pub(crate) fn supports_profile(profile: &ProviderProfile) -> bool {
    crate::providers::anthropic::code_execution::is_official_profile(profile)
        || profile.protocol == ProtocolFamily::VertexClaude
}

fn has_beta(profile: &ProviderProfile, requested: &str) -> bool {
    let listed = profile
        .extra
        .get("betas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let headers = profile
        .extra
        .get("headers")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|headers| headers.iter())
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .filter_map(|(_, value)| value.as_str())
        .flat_map(|value| value.split(','));
    listed
        .chain(headers)
        .map(str::trim)
        .any(|beta| beta == requested)
}

fn reject_raw_toolsets(profile: &ProviderProfile) -> Result<(), LlmError> {
    if !supports_profile(profile) {
        return Ok(());
    }
    let raw_toolset = profile
        .extra
        .get("body")
        .and_then(|body| body.get("tools"))
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                matches!(
                    tool.get("type").and_then(Value::as_str),
                    Some("browser_toolset_20260801" | "computer_toolset_20260801")
                )
            })
        });
    if raw_toolset {
        return Err(invalid(
            "Anthropic Browser and Computer toolsets must use typed request configuration",
        ));
    }
    Ok(())
}

/// Number of client toolset declarations that are deferred as a whole. A
/// Browser or Computer toolset counts as one search definition, regardless of
/// how many enabled member tools it expands into.
pub(crate) fn deferred_count(request: &ChatRequest) -> usize {
    request
        .anthropic_client_toolsets()
        .iter()
        .filter(|toolset| {
            let enabled = toolset
                .members()
                .iter()
                .filter(|member| toolset.member_enabled(member).unwrap_or(false))
                .collect::<Vec<_>>();
            !enabled.is_empty()
                && enabled
                    .iter()
                    .all(|member| toolset.member_deferred(member).unwrap_or(false))
        })
        .count()
}

/// Cache markers placed on client toolset entries, in request order.
pub(crate) fn cache_ttls(request: &ChatRequest) -> Vec<CacheTtl> {
    request
        .anthropic_client_toolsets()
        .iter()
        .filter_map(AnthropicClientToolset::cache_control)
        .map(|control| match control.ttl {
            Some(AnthropicMcpCacheTtl::OneHour) => CacheTtl::OneHour,
            None | Some(AnthropicMcpCacheTtl::FiveMinutes) => CacheTtl::FiveMinutes,
        })
        .collect()
}

fn validate_toolset(
    request: &ChatRequest,
    toolset: &AnthropicClientToolset,
) -> Result<bool, LlmError> {
    if let Some(callers) = toolset.allowed_callers() {
        if callers != [crate::providers::anthropic::types::AnthropicToolCaller::Direct] {
            return Err(invalid(
                "Anthropic Browser and Computer toolsets accept only allowed_callers=['direct']",
            ));
        }
    }

    let enabled_members = toolset
        .members()
        .iter()
        .filter(|member| toolset.member_enabled(member).unwrap_or(false))
        .copied()
        .collect::<Vec<_>>();
    if enabled_members.is_empty() {
        return Err(invalid(format!(
            "Anthropic {} toolset must leave at least one member enabled",
            toolset.kind_name()
        )));
    }

    let deferred = enabled_members
        .iter()
        .map(|member| toolset.member_deferred(member).unwrap_or(false))
        .collect::<Vec<_>>();
    if deferred.iter().any(|value| *value != deferred[0]) {
        return Err(invalid(format!(
            "all enabled Anthropic {} members must use the same defer_loading value",
            toolset.kind_name()
        )));
    }
    let is_deferred = deferred[0];
    if is_deferred && toolset.cache_control().is_some() {
        return Err(invalid(format!(
            "deferred Anthropic {} toolsets cannot carry cache_control",
            toolset.kind_name()
        )));
    }

    if request
        .tools
        .iter()
        .any(|tool| tool.name == toolset.kind_name())
        || request.messages.iter().flat_map(|message| &message.content).any(|block| {
            matches!(block, crate::protocol::ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages, value,
            } if value.get("type").and_then(Value::as_str) == Some("tool_addition")
                && value.pointer("/tool/type").and_then(Value::as_str) == Some("tool_definition")
                && value.pointer("/tool/definition/name").and_then(Value::as_str) == Some(toolset.kind_name()))
        })
    {
        return Err(invalid(format!(
            "client tool name {:?} conflicts with the Anthropic {} toolset",
            toolset.kind_name(),
            toolset.kind_name()
        )));
    }
    if let ToolChoice::Tool { name } = &request.tool_choice {
        if name == toolset.kind_name() || toolset.members().contains(&name.as_str()) {
            return Err(invalid(format!(
                "tool_choice cannot force an Anthropic {} toolset or member; use auto, any, or none",
                toolset.kind_name()
            )));
        }
    }

    Ok(is_deferred)
}

pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let profile = context.profile();

    // Validate history even when the current request has no top-level
    // declarations; an earlier member call still carries this namespace.
    crate::providers::anthropic::client_toolset_history::validate(request, context)?;
    reject_raw_toolsets(profile)?;

    let has_declarations = !request.anthropic_client_toolsets().is_empty();
    let has_history = crate::providers::anthropic::client_toolset_history::has_metadata(request);
    if !has_declarations && !has_history {
        return Ok(());
    }

    if !supports_profile(profile) {
        return Err(unsupported(
            "Anthropic Browser and Computer toolsets require the first-party Anthropic Messages or Vertex Claude protocol",
        ));
    }
    if !SUPPORTED_MODELS.contains(&context.request_model()) {
        return Err(unsupported(format!(
            "Anthropic Browser and Computer toolsets are not documented for model {:?}",
            context.request_model()
        )));
    }
    if has_beta(profile, FINE_GRAINED_TOOL_STREAMING_BETA) {
        return Err(unsupported(
            "Anthropic Browser and Computer toolsets do not support the fine-grained-tool-streaming-2025-05-14 beta",
        ));
    }

    if !has_declarations {
        return Ok(());
    }

    request.validate_hosted_tools()?;

    let raw_named_collision = profile
        .extra
        .get("body")
        .and_then(|body| body.get("tools"))
        .and_then(Value::as_array)
        .is_some_and(|raw_tools| {
            raw_tools.iter().any(|raw| {
                raw.get("name").and_then(Value::as_str).is_some_and(|name| {
                    request
                        .anthropic_client_toolsets()
                        .iter()
                        .any(|toolset| toolset.kind_name() == name)
                })
            })
        });
    if raw_named_collision {
        return Err(invalid(
            "profile.extra.body.tools cannot declare a tool named browser or computer alongside the corresponding Anthropic client toolset",
        ));
    }

    let raw_tool_choice = profile
        .extra
        .get("body")
        .and_then(|body| body.get("tool_choice"))
        .and_then(|choice| {
            (choice.get("type").and_then(Value::as_str) == Some("tool"))
                .then(|| choice.get("name").and_then(Value::as_str))
                .flatten()
        });
    if raw_tool_choice.is_some_and(|name| {
        request
            .anthropic_client_toolsets()
            .iter()
            .any(|toolset| name == toolset.kind_name() || toolset.members().contains(&name))
    }) {
        return Err(invalid(
            "profile.extra.body.tool_choice cannot force an Anthropic client toolset or member",
        ));
    }

    let mut browser_seen = false;
    let mut computer_seen = false;
    let mut deferred = false;
    for toolset in request.anthropic_client_toolsets() {
        match toolset {
            AnthropicClientToolset::Browser(_) if browser_seen => {
                return Err(invalid(
                    "Anthropic Browser toolset may be declared only once per request",
                ));
            }
            AnthropicClientToolset::Computer(_) if computer_seen => {
                return Err(invalid(
                    "Anthropic Computer toolset may be declared only once per request",
                ));
            }
            AnthropicClientToolset::Browser(_) => browser_seen = true,
            AnthropicClientToolset::Computer(_) => computer_seen = true,
        }
        deferred |= validate_toolset(request, toolset)?;
    }

    if deferred && request.hosted_anthropic_tool_search().is_none() {
        return Err(invalid(
            "deferred Anthropic Browser and Computer toolsets require Anthropic hosted tool search on the same request",
        ));
    }

    let total_deferred = request
        .tools
        .iter()
        .filter(|tool| tool.defer_loading)
        .count()
        + request
            .hosted_anthropic_web_fetch()
            .is_some_and(|config| config.defer_loading) as usize
        + deferred_count(request);
    if total_deferred > 10_000 {
        return Err(invalid(
            "Anthropic tool search accepts at most 10,000 deferred tool definitions per request",
        ));
    }

    Ok(())
}

pub(crate) fn apply(request: &ChatRequest, body: &mut serde_json::Map<String, Value>) {
    if request.anthropic_client_toolsets().is_empty() {
        return;
    }
    let tools = body
        .entry("tools")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(tools) = tools.as_array_mut() else {
        return;
    };
    tools.extend(
        request
            .anthropic_client_toolsets()
            .iter()
            .map(AnthropicClientToolset::to_wire_value),
    );
}
