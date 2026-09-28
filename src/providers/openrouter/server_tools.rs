//! Request-level validation and encoding for OpenRouter server tools.

use super::native::OpenRouterHostedTool;
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile};
use crate::providers::openrouter::types::{
    OpenRouterShellConfig, OpenRouterShellEnvironment, OpenRouterShellNetworkPolicy,
    OpenRouterToolSearchConfig,
};
use serde_json::{json, Map, Value};

pub(crate) fn has_tools(request: &ChatRequest) -> bool {
    request
        .hosted_tools
        .iter()
        .any(|tool| tool.native::<OpenRouterHostedTool>().is_some())
}

/// Validate the documented API/protocol combinations and any account-bound
/// container reference. Returns true when this request has a server tool, so
/// the executor can pin it to one attempt and avoid replaying a possible side
/// effect after a transport error.
pub(crate) fn validate(
    request: &ChatRequest,
    profile: &ProviderProfile,
    account_scope: Option<&str>,
    require_account_scope: bool,
) -> Result<bool, LlmError> {
    request.validate_hosted_tools()?;
    let tool_search = request.hosted_openrouter_tool_search();
    let shell = request.hosted_openrouter_shell();
    let has_tools = tool_search.is_some() || shell.is_some();
    if !has_tools {
        return Ok(false);
    }

    if shell.is_some()
        && tool_search.is_none()
        && request.tools.iter().any(|tool| tool.defer_loading)
    {
        return Err(unsupported(
            "OpenRouter shell does not discover deferred client tools; add OpenRouter tool search",
        ));
    }

    if !is_official_profile(profile) {
        return Err(unsupported(
            "OpenRouter server tools require the official global OpenRouter API profile",
        ));
    }
    if !matches!(
        profile.protocol,
        ProtocolFamily::OpenAiResponses | ProtocolFamily::AnthropicMessages
    ) {
        return Err(unsupported(
            "OpenRouter server tools are available only on Responses and Messages APIs",
        ));
    }
    if shell.is_some() && !is_global_profile(profile) {
        return Err(unsupported(
            "OpenRouter shell is unavailable on the regional OpenRouter endpoints",
        ));
    }

    if let Some(config) = tool_search {
        validate_tool_search(config)?;
        if request.tools.iter().any(|tool| tool.defer_loading)
            && !matches!(request.tool_choice, crate::protocol::ToolChoice::Auto)
        {
            return Err(unsupported(
                "OpenRouter tool search with deferred tools supports only automatic tool choice; allowed_tools is not represented by this client",
            ));
        }
    }

    if let Some(config) = shell {
        validate_shell(config)?;
        if let Some(OpenRouterShellEnvironment::ContainerReference { container, .. }) =
            &config.environment
        {
            validate_container_reference(container)?;
            let scope = container.scope();
            if scope.profile_name() != profile.profile_name.as_str()
                || scope.endpoint().trim_end_matches('/') != profile.base_url.trim_end_matches('/')
                || require_account_scope && account_scope.is_none()
                || account_scope.is_some_and(|current| current != scope.account_scope())
            {
                return Err(LlmError::InvalidRequest {
                    message: "OpenRouter container reference belongs to a different profile, endpoint, or account_scope".into(),
                });
            }
        }
    }
    Ok(true)
}

fn validate_tool_search(config: &OpenRouterToolSearchConfig) -> Result<(), LlmError> {
    if config.max_results.is_some_and(|value| value > 50) {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter tool-search max_results cannot exceed 50".into(),
        });
    }
    Ok(())
}

fn validate_shell(config: &OpenRouterShellConfig) -> Result<(), LlmError> {
    if config.timeout_ms.is_some_and(|value| value > 300_000)
        || config.max_output_length.is_some_and(|value| value > 65_536)
    {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter shell timeout_ms and max_output_length cannot exceed 300000 and 65536 respectively".into(),
        });
    }
    if let Some(environment) = &config.environment {
        let network_policy = match environment {
            OpenRouterShellEnvironment::ContainerAuto { network_policy }
            | OpenRouterShellEnvironment::ContainerReference { network_policy, .. } => {
                network_policy.as_ref()
            }
        };
        if let Some(OpenRouterShellNetworkPolicy::Allowlist { allowed_domains }) = network_policy {
            if allowed_domains.len() > 50
                || allowed_domains.iter().any(|domain| {
                    domain.is_empty()
                        || !domain.is_ascii()
                        || domain.split('.').any(str::is_empty)
                        || domain.bytes().any(|byte| {
                            byte.is_ascii_uppercase()
                                || !(byte.is_ascii_lowercase()
                                    || byte.is_ascii_digit()
                                    || matches!(byte, b'.' | b'-' | b'*'))
                        })
                })
            {
                return Err(LlmError::InvalidRequest {
                    message: "OpenRouter shell network policy requires at most 50 lowercase hostname/glob entries without schemes, paths, or ports".into(),
                });
            }
        }
    }
    Ok(())
}

fn validate_container_reference(
    container: &crate::providers::openrouter::types::OpenRouterContainerRef,
) -> Result<(), LlmError> {
    let scope = container.scope();
    if container.id().trim().is_empty()
        || container.id().chars().any(char::is_control)
        || scope.profile_name().trim().is_empty()
        || scope.endpoint().trim().is_empty()
        || scope.account_scope().trim().is_empty()
        || scope.profile_name().chars().any(char::is_control)
        || scope.endpoint().chars().any(char::is_control)
        || scope.account_scope().chars().any(char::is_control)
    {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter container reference has invalid ID or scope data".into(),
        });
    }
    Ok(())
}

pub(crate) fn is_official_profile(profile: &ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "openrouter"
        || !matches!(
            profile.protocol,
            ProtocolFamily::OpenAiResponses | ProtocolFamily::AnthropicMessages
        )
    {
        return false;
    }
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    url.scheme() == "https"
        && matches!(
            url.host_str(),
            Some("openrouter.ai" | "us.openrouter.ai" | "eu.openrouter.ai")
        )
        && url.path().trim_end_matches('/') == "/api/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

fn is_global_profile(profile: &ProviderProfile) -> bool {
    url::Url::parse(&profile.base_url)
        .ok()
        .is_some_and(|url| url.host_str() == Some("openrouter.ai"))
}

pub(crate) fn response_tool_values(request: &ChatRequest) -> Vec<Value> {
    request
        .hosted_tools
        .iter()
        .filter_map(|tool| match tool.native::<OpenRouterHostedTool>() {
            Some(OpenRouterHostedTool::ToolSearch(config)) => Some(tool_search_value(config)),
            Some(OpenRouterHostedTool::Shell(config)) => Some(shell_value(config)),
            _ => None,
        })
        .collect()
}

pub(crate) fn tool_search_value(config: &OpenRouterToolSearchConfig) -> Value {
    let mut tool = json!({"type":"openrouter:tool_search"});
    if let Some(max_results) = config.max_results {
        tool["parameters"] = json!({"max_results":max_results});
    }
    tool
}

pub(crate) fn shell_value(config: &OpenRouterShellConfig) -> Value {
    let mut parameters = Map::new();
    if let Some(engine) = config.engine {
        parameters.insert(
            "engine".into(),
            Value::String(
                match engine {
                    crate::providers::openrouter::types::OpenRouterShellEngine::Auto => "auto",
                    crate::providers::openrouter::types::OpenRouterShellEngine::OpenRouter => {
                        "openrouter"
                    }
                }
                .into(),
            ),
        );
    }
    if let Some(environment) = &config.environment {
        parameters.insert("environment".into(), environment_value(environment));
    }
    if let Some(timeout_ms) = config.timeout_ms {
        parameters.insert("timeout_ms".into(), json!(timeout_ms));
    }
    if let Some(max_output_length) = config.max_output_length {
        parameters.insert("max_output_length".into(), json!(max_output_length));
    }
    let mut tool = json!({"type":"openrouter:shell"});
    if !parameters.is_empty() {
        tool["parameters"] = Value::Object(parameters);
    }
    tool
}

fn environment_value(environment: &OpenRouterShellEnvironment) -> Value {
    match environment {
        OpenRouterShellEnvironment::ContainerAuto { network_policy } => {
            let mut value = json!({"type":"container_auto"});
            if let Some(policy) = network_policy {
                value["network_policy"] = network_policy_value(policy);
            }
            value
        }
        OpenRouterShellEnvironment::ContainerReference {
            container,
            network_policy,
        } => {
            let mut value = json!({
                "type":"container_reference",
                "container_id":container.id()
            });
            if let Some(policy) = network_policy {
                value["network_policy"] = network_policy_value(policy);
            }
            value
        }
    }
}

fn network_policy_value(policy: &OpenRouterShellNetworkPolicy) -> Value {
    match policy {
        OpenRouterShellNetworkPolicy::Disabled => json!({"type":"disabled"}),
        OpenRouterShellNetworkPolicy::Allowlist { allowed_domains } => {
            json!({"type":"allowlist","allowed_domains":allowed_domains})
        }
    }
}

fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}
