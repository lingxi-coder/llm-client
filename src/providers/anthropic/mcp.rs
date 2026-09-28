//! Anthropic Messages remote MCP connector on Claude API and Foundry.
//!
//! This module only encodes typed `mcp_servers`/`mcp_toolset` declarations
//! against Claude API (2026 beta) or Foundry (2025 beta). Native MCP history is
//! still replayed through the generic provider-content path for compatible
//! gateways.

use crate::codecs::CodecContext;
use crate::protocol::{
    CacheTtl, ChatRequest, ContentBlock, LlmError, ProtocolFamily, ProviderProfile,
};
use crate::providers::anthropic::types::AnthropicMcpCacheTtl;
use serde_json::{json, Map, Value};

const ANTHROPIC_API_BETA: &str = "mcp-client-2026-09-15";
const FOUNDRY_BETA: &str = "mcp-client-2025-11-20";

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

pub(crate) fn has_typed_servers(request: &ChatRequest) -> bool {
    request.anthropic_mcp_servers().next().is_some()
}

pub(crate) fn has_top_level_toolsets(request: &ChatRequest) -> bool {
    request
        .anthropic_mcp_servers()
        .any(|config| !config.inline_toolset())
}

/// Native MCP content is provider execution history. Retrying or failing over
/// after an uncertain outcome could execute the same remote action twice.
pub(crate) fn has_mcp(request: &ChatRequest) -> bool {
    has_typed_servers(request)
        || request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| {
                matches!(block, ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::AnthropicMessages,
                    value,
                } if value.get("type").and_then(Value::as_str).is_some_and(|kind| {
                    matches!(kind, "mcp_tool_use" | "mcp_tool_result" | "mcp_tool_listing")
                }))
            })
}

pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let profile = context.profile();
    validate_profile_extra(profile)?;
    let foundry = is_foundry_profile(profile);
    if foundry && has_native_tool_listing(request) {
        return Err(unsupported(
            "Foundry supports the mcp-client-2025-11-20 connector; mcp_tool_listing replay requires the Claude API-only mcp-client-2026-09-15 beta",
        ));
    }
    if foundry && has_native_inline_mcp_addition(request) {
        return Err(unsupported(
            "Anthropic inline MCP tool additions require the Claude API-only inline-tools-2026-09-15 beta",
        ));
    }
    let configs = request.anthropic_mcp_servers().collect::<Vec<_>>();
    if configs.is_empty() {
        return Ok(());
    }

    request.validate_hosted_tools()?;
    if !is_supported_profile(profile) {
        return Err(unsupported(
            "typed Anthropic MCP requires the first-party Messages API or Microsoft Foundry",
        ));
    }
    reject_raw_connector_fields(profile)?;

    if foundry
        && configs
            .iter()
            .any(|config| config.tools().is_some() || config.inline_toolset())
    {
        return Err(unsupported(
            "Foundry supports mcp-client-2025-11-20; pinned MCP tool lists and inline MCP additions require Claude API-only 2026 betas",
        ));
    }

    let deferred = configs.iter().any(|config| {
        !config.inline_toolset()
            && has_deferred_tools(config)
            && !crate::providers::anthropic::conversation::has_mcp_toolset_addition(
                request,
                config.name(),
            )
    });
    if deferred && request.hosted_anthropic_tool_search().is_none() {
        return Err(invalid(
            "defer_loading on an Anthropic MCP toolset requires Anthropic hosted tool search on the same request",
        ));
    }
    if request.remote_mcp_servers().next().is_some()
        || request.xai_remote_mcp_servers().next().is_some()
    {
        return Err(unsupported(
            "Anthropic MCP cannot be mixed with OpenAI or xAI remote MCP configurations",
        ));
    }
    if request.hosted_openai_tool_search().is_some() {
        return Err(unsupported(
            "OpenAI tool search cannot be combined with Anthropic MCP",
        ));
    }
    Ok(())
}

pub(crate) fn validate_profile_extra(profile: &ProviderProfile) -> Result<(), LlmError> {
    if is_supported_profile(profile) {
        reject_raw_connector_fields(profile)?;
    }
    if is_foundry_profile(profile) {
        reject_foundry_beta_injection(profile)?;
    }
    Ok(())
}

fn is_foundry_profile(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::FoundryClaude
}

fn is_supported_profile(profile: &ProviderProfile) -> bool {
    is_official_profile(profile) || is_foundry_profile(profile)
}

fn has_native_tool_listing(request: &ChatRequest) -> bool {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| {
            matches!(block, ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } if value.get("type").and_then(Value::as_str) == Some("mcp_tool_listing"))
        })
}

fn has_native_inline_mcp_addition(request: &ChatRequest) -> bool {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| {
            matches!(block, ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } if value.get("type").and_then(Value::as_str) == Some("tool_addition")
                && value.get("tool").and_then(|tool| tool.get("type")).and_then(Value::as_str)
                    == Some("tool_definition")
                && value.get("tool").and_then(|tool| tool.get("definition"))
                    .and_then(|definition| definition.get("type"))
                    .and_then(Value::as_str) == Some("mcp_toolset"))
        })
}

fn reject_foundry_beta_injection(profile: &ProviderProfile) -> Result<(), LlmError> {
    let mut values = Vec::new();
    if let Some(betas) = profile.extra.get("betas").and_then(Value::as_array) {
        values.extend(betas.iter().filter_map(Value::as_str));
    }
    if let Some(headers) = profile.extra.get("headers").and_then(Value::as_object) {
        values.extend(
            headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
                .filter_map(|(_, value)| value.as_str()),
        );
    }
    let incompatible = values
        .into_iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .find(|beta| {
            (beta.starts_with("mcp-client-") && *beta != FOUNDRY_BETA)
                || beta.starts_with("inline-tools-")
        });
    if let Some(beta) = incompatible {
        return Err(unsupported(format!(
            "Foundry MCP does not support profile beta {beta:?}; use mcp-client-2025-11-20 for the Foundry connector"
        )));
    }
    Ok(())
}

fn reject_raw_connector_fields(profile: &ProviderProfile) -> Result<(), LlmError> {
    let body = profile.extra.get("body").unwrap_or(&Value::Null);
    let raw_toolset = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool.get("type").and_then(Value::as_str) == Some("mcp_toolset"))
        });
    if body.get("mcp_servers").is_some() || raw_toolset || body.get("authorization_token").is_some()
    {
        return Err(invalid(
            "Anthropic MCP servers, toolsets, and authorization tokens must use typed request configuration",
        ));
    }
    Ok(())
}

fn has_deferred_tools(config: &crate::providers::anthropic::types::AnthropicMcpConfig) -> bool {
    let default = config.default_config().cloned().unwrap_or_default();
    let is_deferred = |name: &str| {
        let override_config = config.configs().get(name);
        let enabled = override_config
            .and_then(|settings| settings.enabled)
            .or(default.enabled)
            .unwrap_or(true);
        let deferred = override_config
            .and_then(|settings| settings.defer_loading)
            .or(default.defer_loading)
            .unwrap_or(false);
        enabled && deferred
    };
    match config.tools() {
        Some(tools) => tools.iter().any(|tool| is_deferred(&tool.name)),
        None => {
            (default.enabled.unwrap_or(true) && default.defer_loading.unwrap_or(false))
                || config.configs().iter().any(|(name, settings)| {
                    settings.enabled.or(default.enabled).unwrap_or(true)
                        && settings
                            .defer_loading
                            .or(default.defer_loading)
                            .unwrap_or(false)
                        && !name.trim().is_empty()
                })
        }
    }
}

fn is_official_profile(profile: &ProviderProfile) -> bool {
    crate::providers::anthropic::code_execution::is_official_profile(profile)
}

pub(crate) fn cache_ttls(request: &ChatRequest) -> Vec<CacheTtl> {
    request
        .anthropic_mcp_servers()
        .filter(|config| !config.inline_toolset())
        .filter_map(|config| config.cache_control())
        .map(|control| match control.ttl {
            Some(AnthropicMcpCacheTtl::OneHour) => CacheTtl::OneHour,
            None | Some(AnthropicMcpCacheTtl::FiveMinutes) => CacheTtl::FiveMinutes,
        })
        .collect()
}

pub(crate) fn apply(request: &ChatRequest, body: &mut Map<String, Value>) -> Result<(), LlmError> {
    let configs = request.anthropic_mcp_servers().collect::<Vec<_>>();
    if configs.is_empty() {
        return Ok(());
    }
    body.insert(
        "mcp_servers".into(),
        Value::Array(
            configs
                .iter()
                .map(|config| json!({"type":"url", "name":config.name(), "url":config.url()}))
                .collect(),
        ),
    );
    let top_level_toolsets = configs
        .iter()
        .copied()
        .filter(|config| !config.inline_toolset())
        .collect::<Vec<_>>();
    if top_level_toolsets.is_empty() {
        return Ok(());
    }
    let tools = body.entry("tools").or_insert_with(|| json!([]));
    let Some(tools) = tools.as_array_mut() else {
        return Err(invalid("Anthropic MCP toolsets require a tools array"));
    };
    for config in top_level_toolsets {
        tools.push(config.toolset_value());
    }
    Ok(())
}

pub(crate) fn apply_beta_header(
    request: &ChatRequest,
    profile: &ProviderProfile,
    headers: &mut Vec<(String, String)>,
) {
    if !is_supported_profile(profile) || !has_mcp(request) {
        return;
    }
    let foundry = is_foundry_profile(profile);
    let connector_beta = if foundry {
        FOUNDRY_BETA
    } else {
        ANTHROPIC_API_BETA
    };
    let mut candidates = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .flat_map(|(_, value)| value.split(','))
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if let Some(configured) = profile
        .extra
        .get("headers")
        .and_then(Value::as_object)
        .and_then(|headers| {
            headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        })
        .and_then(|(_, value)| value.as_str())
    {
        candidates.extend(configured.split(',').map(str::trim).map(str::to_owned));
    }
    let mut betas = Vec::new();
    for beta in candidates {
        if !beta.is_empty()
            && !beta.starts_with("mcp-client-")
            && (!foundry || !beta.starts_with("inline-tools-"))
            && !betas.iter().any(|existing| existing == &beta)
        {
            betas.push(beta);
        }
    }
    betas.push(connector_beta.into());
    let value = betas.join(",");
    headers.retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"));
    headers.push(("anthropic-beta".into(), value));
}
