//! Anthropic Code Execution and account-bound container reuse on the first-party
//! API and typed Anthropic-hosted Foundry deployments.

use crate::codecs::CodecContext;
use crate::protocol::{
    ChatRequest, ContentBlock, FoundryHosting, HostedTool, LlmError, MessageRole, ProtocolFamily,
    ProviderProfile, ToolChoice, ToolSpec,
};
use crate::providers::anthropic::types::{AnthropicContainerScope, AnthropicToolCaller};
use serde_json::{json, Map, Value};

const TOOL_VERSION: &str = "code_execution_20260521";

pub(crate) fn inline_tool_value() -> Value {
    json!({"type": TOOL_VERSION, "name": "code_execution"})
}

// Exact documented first-party IDs; aliases are resolved to wire IDs by the
// high-level client. Do not infer support for older or future model families.
const SUPPORTED_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
    "claude-mythos-preview",
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
];

// Anthropic explicitly excludes Haiku 4.5 from programmatic tool calling even
// though it accepts the current Code Execution tool version.
const PROGRAMMATIC_TOOL_CALLING_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
];

pub(crate) fn is_official_profile(profile: &ProviderProfile) -> bool {
    profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages
        && AnthropicContainerScope::is_official_endpoint(&profile.base_url)
}

fn foundry_deployment(context: &CodecContext) -> Option<&crate::protocol::FoundryDeployment> {
    if context.profile().protocol != ProtocolFamily::FoundryClaude
        || AnthropicContainerScope::normalize_foundry_endpoint(&context.profile().base_url)
            .is_none()
    {
        return None;
    }
    crate::hosting::foundry::require_deployment(context)
        .ok()
        .filter(|deployment| deployment.hosting == FoundryHosting::Anthropic)
}

/// Whether the selected route and exact model identity support Code Execution.
/// Foundry's custom deployment name is not a model identity; its typed model
/// row must explicitly say Anthropic-hosted and name a documented model.
pub(crate) fn supports_execution(context: &CodecContext) -> bool {
    if is_official_profile(context.profile()) {
        return SUPPORTED_MODELS.contains(&context.request_model());
    }
    foundry_deployment(context)
        .is_some_and(|deployment| SUPPORTED_MODELS.contains(&deployment.model_id.as_str()))
}

/// Whether the selected route/model pair supports Programmatic Tool Calling.
/// Haiku 4.5 and the Foundry Mythos preview remain Code Execution-only.
pub(crate) fn supports_programmatic_tool_calling(context: &CodecContext) -> bool {
    if is_official_profile(context.profile()) {
        return PROGRAMMATIC_TOOL_CALLING_MODELS.contains(&context.request_model());
    }
    foundry_deployment(context).is_some_and(|deployment| {
        PROGRAMMATIC_TOOL_CALLING_MODELS.contains(&deployment.model_id.as_str())
    })
}

fn is_anthropic_messages_protocol(protocol: ProtocolFamily) -> bool {
    matches!(
        protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::FoundryClaude
    )
}

/// Native server execution can be resumed when its transcript is replayed.
/// Conservatively pin those requests even if the caller omitted tool settings.
pub(crate) fn has_execution(request: &ChatRequest) -> bool {
    request.hosted_anthropic_code_execution().is_some()
        || request.messages.iter().flat_map(|message| &message.content).any(|block| {
            matches!(block, ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } if value.get("type").and_then(Value::as_str).is_some_and(|kind| {
                kind == "container_upload" || kind.contains("code_execution")
            }) || value.get("type").and_then(Value::as_str) == Some("server_tool_use")
                && value.get("name").and_then(Value::as_str).is_some_and(|name| {
                    matches!(name, "code_execution" | "bash_code_execution" | "text_editor_code_execution")
                }))
        })
}

/// Validate before resolving attachments and authenticating, and again in the
/// encoder so direct codec calls cannot bypass account/route validation.
pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let inline_tools =
        crate::providers::anthropic::conversation::collect_inline_tool_specs(request)?;
    validate_with_inline_tools(request, context, &inline_tools)
}

/// Validate ordinary tools and Anthropic inline custom tools together. Inline
/// definitions stay in their native body shape; this view exists only for the
/// shared caller/model/schema/continuation checks.
pub(crate) fn validate_with_inline_tools(
    request: &ChatRequest,
    context: &CodecContext,
    inline_tools: &[ToolSpec],
) -> Result<(), LlmError> {
    let profile = context.profile();
    let config = request.hosted_anthropic_code_execution();
    if is_official_profile(profile) || profile.protocol == ProtocolFamily::FoundryClaude {
        let extra = &profile.extra["body"];
        let raw_execution = extra
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|kind| kind.starts_with("code_execution_"))
                })
            });
        if extra.get("container").is_some() || raw_execution {
            return Err(invalid("Anthropic execution tools and container IDs must use AnthropicCodeExecutionConfig, not extra.body"));
        }
    }

    let mut caller_metadata_present = false;
    let mut programmatic_caller_metadata_present = false;
    let mut server_tool_ids = std::collections::HashMap::<String, usize>::new();
    let mut completed_server_tools = std::collections::HashMap::<String, usize>::new();
    let mut programmatic_history_calls = Vec::new();
    let mut history_position = 0usize;
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::AnthropicMessages,
                    value,
                } if value.get("type").and_then(Value::as_str) == Some("server_tool_use")
                    && value
                        .get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| {
                            matches!(
                                name,
                                "code_execution"
                                    | "bash_code_execution"
                                    | "text_editor_code_execution"
                            )
                        }) =>
                {
                    if let Some(id) = value.get("id").and_then(Value::as_str) {
                        server_tool_ids.insert(id.to_owned(), history_position);
                    }
                }
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::AnthropicMessages,
                    value,
                } if matches!(
                    value.get("type").and_then(Value::as_str),
                    Some(
                        "code_execution_tool_result"
                            | "bash_code_execution_tool_result"
                            | "text_editor_code_execution_tool_result"
                    )
                ) =>
                {
                    if let Some(id) = value.get("tool_use_id").and_then(Value::as_str) {
                        completed_server_tools.insert(id.to_owned(), history_position);
                    }
                }
                ContentBlock::ToolUse {
                    name,
                    caller: Some(caller),
                    ..
                } => {
                    caller_metadata_present = true;
                    let Some(caller) = caller.as_object() else {
                        return Err(invalid(
                            "Anthropic tool caller metadata must remain a JSON object",
                        ));
                    };
                    let Some(caller_type) = caller.get("type").and_then(Value::as_str) else {
                        return Err(invalid(
                            "Anthropic tool caller metadata requires a string type",
                        ));
                    };
                    if matches!(
                        caller_type,
                        "code_execution_20260120" | "code_execution_20260521"
                    ) {
                        programmatic_caller_metadata_present = true;
                        let Some(tool_id) = caller.get("tool_id").and_then(Value::as_str) else {
                            return Err(invalid(
                                "Anthropic programmatic tool caller metadata requires a string tool_id",
                            ));
                        };
                        if tool_id.is_empty()
                            || server_tool_ids
                                .get(tool_id)
                                .is_none_or(|position| *position >= history_position)
                        {
                            return Err(invalid(
                                "Anthropic programmatic tool caller must reference a preceding server_tool_use block",
                            ));
                        }
                        programmatic_history_calls.push((name.as_str(), tool_id, history_position));
                    }
                }
                _ => {}
            }
            history_position = history_position.saturating_add(1);
        }
    }
    if caller_metadata_present && !is_anthropic_messages_protocol(profile.protocol) {
        return Err(unsupported(
            "Anthropic tool caller metadata can only be replayed through an Anthropic Messages-compatible codec",
        ));
    }
    if programmatic_caller_metadata_present && !supports_programmatic_tool_calling(context) {
        return Err(unsupported(
            "Anthropic Programmatic Tool Calling metadata requires a supported model on the first-party Anthropic API or an Anthropic-hosted Foundry deployment",
        ));
    }

    let tool_specs = request
        .tools
        .iter()
        .chain(inline_tools.iter())
        .collect::<Vec<_>>();
    let has_programmatic_callers = tool_specs.iter().any(|tool| {
        tool.anthropic_allowed_callers()
            .iter()
            .any(is_programmatic_caller)
    });
    let pending_programmatic_calls = programmatic_history_calls
        .into_iter()
        .filter(|(_, tool_id, call_position)| {
            completed_server_tools
                .get(*tool_id)
                .is_none_or(|result_position| *result_position <= *call_position)
        })
        .collect::<Vec<_>>();
    let has_programmatic_turn = has_programmatic_callers || !pending_programmatic_calls.is_empty();
    let has_allowed_callers = tool_specs
        .iter()
        .any(|tool| !tool.anthropic_allowed_callers().is_empty());
    if has_allowed_callers && !supports_execution(context) {
        return Err(unsupported(
            "Anthropic allowed_callers requires Code Execution on the first-party Anthropic API or an Anthropic-hosted Foundry deployment",
        ));
    }
    for tool in &tool_specs {
        let mut seen = std::collections::HashSet::new();
        for caller in tool.anthropic_allowed_callers() {
            if !seen.insert(*caller) {
                return Err(invalid(format!(
                    "tool {:?} repeats an allowed caller",
                    tool.name
                )));
            }
        }
        if tool
            .anthropic_allowed_callers()
            .contains(&AnthropicToolCaller::CodeExecution20260120)
            && tool
                .anthropic_allowed_callers()
                .contains(&AnthropicToolCaller::CodeExecution20260521)
        {
            return Err(invalid(format!(
                "tool {:?} lists two interchangeable Code Execution caller versions",
                tool.name
            )));
        }
    }
    if has_programmatic_turn {
        if config.is_none() {
            return Err(unsupported(
                "Anthropic programmatic tool calling requires the typed Code Execution hosted tool on the same request",
            ));
        }
        if !supports_programmatic_tool_calling(context) {
            return Err(unsupported(format!(
                "Anthropic programmatic tool calling is not documented for the selected model and hosting identity (wire model {:?})",
                context.request_model()
            )));
        }
        for tool in tool_specs.iter().filter(|tool| {
            tool.anthropic_allowed_callers()
                .iter()
                .any(is_programmatic_caller)
        }) {
            if tool.strict {
                return Err(unsupported(format!(
                    "Anthropic programmatic tool calling does not support strict tool {:?}",
                    tool.name
                )));
            }
            crate::codecs::structured::validate_no_recursive_schema_references(&tool.input_schema)?;
        }
        // The conversation validator has a chronological tool timeline when
        // inline definitions exist. Avoid matching a replayed call to a later
        // redefinition here; without inline history, preserve the static check.
        if inline_tools.is_empty() {
            for (name, _, _) in &pending_programmatic_calls {
                let Some(tool) = tool_specs.iter().find(|tool| tool.name == *name) else {
                    return Err(invalid(format!(
                        "programmatic tool call {name:?} must be replayed with its original tool definition"
                    )));
                };
                if !tool
                    .anthropic_allowed_callers()
                    .iter()
                    .any(is_programmatic_caller)
                {
                    return Err(invalid(format!(
                        "programmatic tool call {name:?} requires its original code_execution allowed caller"
                    )));
                }
                if tool.strict {
                    return Err(unsupported(format!(
                        "Anthropic programmatic tool calling does not support strict tool {name:?}"
                    )));
                }
            }
        }
        if !pending_programmatic_calls.is_empty()
            && config
                .and_then(|config| config.container.as_ref())
                .is_none()
        {
            return Err(invalid(
                "resuming an Anthropic programmatic tool call requires the response's scoped Code Execution container",
            ));
        }
        if let ToolChoice::Tool { name } = &request.tool_choice {
            let current_tool = inline_tools
                .iter()
                .rev()
                .find(|tool| tool.name == *name)
                .or_else(|| request.tools.iter().find(|tool| tool.name == *name));
            if current_tool.is_some_and(|tool| {
                tool.anthropic_allowed_callers()
                    .iter()
                    .any(is_programmatic_caller)
                    && !tool
                        .anthropic_allowed_callers()
                        .contains(&AnthropicToolCaller::Direct)
            }) {
                return Err(invalid(format!(
                    "tool_choice cannot force programmatic-only tool {name:?}"
                )));
            }
        }
        if profile
            .extra
            .get("body")
            .and_then(|body| body.get("tool_choice"))
            .and_then(|choice| choice.get("disable_parallel_tool_use"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            return Err(unsupported(
                "Anthropic programmatic tool calling cannot be combined with disable_parallel_tool_use",
            ));
        }
    }
    let Some(config) = config else {
        return Ok(());
    };
    request.validate_hosted_tools()?;
    if config.inline_definition && !is_official_profile(profile) {
        return Err(unsupported(
            "inline Anthropic Code Execution definitions require the first-party Messages API",
        ));
    }
    if !supports_execution(context) {
        if profile.protocol == ProtocolFamily::FoundryClaude {
            let deployment = crate::hosting::foundry::require_deployment(context)?;
            if deployment.hosting != FoundryHosting::Anthropic {
                return Err(unsupported(
                    "Foundry Code Execution requires a deployment hosted on Anthropic; Azure-hosted deployments do not support it",
                ));
            }
        } else if !is_official_profile(profile) {
            return Err(unsupported(
                "Anthropic Code Execution requires the first-party Anthropic Messages endpoint or a typed Anthropic-hosted Foundry deployment",
            ));
        }
        return Err(unsupported(format!(
            "Anthropic Code Execution is not documented for the selected model and hosting identity (wire model {:?})",
            context.request_model()
        )));
    }
    if request
        .tools
        .iter()
        .any(|tool| tool.name == "code_execution")
    {
        return Err(invalid(
            "client tool name code_execution conflicts with Anthropic's hosted execution tool",
        ));
    }
    if request.hosted_tools.iter().any(|tool| {
        !matches!(tool, HostedTool::WebSearch(_))
            && tool
                .native::<super::native::AnthropicHostedTool>()
                .is_none()
    }) {
        return Err(unsupported(
            "Anthropic Code Execution cannot be combined with another provider's hosted tool",
        ));
    }
    if config.skills.len() > 20 {
        return Err(invalid(
            "Anthropic Code Execution supports at most 20 Skills per request",
        ));
    }
    let foundry_skill_deployment = if profile.protocol == ProtocolFamily::FoundryClaude {
        Some(crate::hosting::foundry::require_deployment(context)?)
    } else {
        None
    };
    for skill in &config.skills {
        skill.validate_for(profile, context.account_scope(), foundry_skill_deployment)?;
    }
    if profile.extra["body"].get("tools").is_some() {
        return Err(invalid(
            "extra.body.tools cannot be combined with typed Anthropic Code Execution",
        ));
    }
    if !config.files.is_empty() {
        if request.messages.last().map(|message| message.role) != Some(MessageRole::User) {
            return Err(invalid(
                "Anthropic container uploads require the final message to have the user role",
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        let native_upload_ids = request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|block| match block {
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::AnthropicMessages,
                    value,
                } if value.get("type").and_then(Value::as_str) == Some("container_upload") => {
                    value.get("file_id").and_then(Value::as_str)
                }
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        for file in &config.files {
            crate::files::validate_provider_file_at(
                file,
                profile,
                context.file_scope(),
                context.file_validation_time(),
            )?;
            if file.file_id.chars().any(char::is_control)
                || !ids.insert(file.file_id.as_str())
                || native_upload_ids.contains(file.file_id.as_str())
            {
                return Err(invalid(
                    "Anthropic container uploads require distinct valid file IDs",
                ));
            }
        }
    }
    if let Some(reference) = &config.container {
        reference.validate()?;
        let scope = reference.scope();
        let scope_matches_route = if profile.protocol == ProtocolFamily::FoundryClaude {
            crate::hosting::foundry::require_deployment(context).is_ok_and(|deployment| {
                scope.foundry_deployment() == Some(deployment)
                    && AnthropicContainerScope::normalize_foundry_endpoint(scope.endpoint())
                        == AnthropicContainerScope::normalize_foundry_endpoint(&profile.base_url)
            })
        } else {
            scope.foundry_deployment().is_none()
                && is_official_profile(profile)
                && AnthropicContainerScope::is_official_endpoint(scope.endpoint())
        };
        if scope.profile_name() != profile.profile_name
            || !scope_matches_route
            || Some(scope.account_scope()) != context.account_scope()
            || scope.request_model() != context.request_model()
        {
            return Err(invalid("Anthropic container reference does not match the selected profile, endpoint, account_scope, or request model"));
        }
    }
    Ok(())
}

fn is_programmatic_caller(caller: &AnthropicToolCaller) -> bool {
    matches!(
        caller,
        AnthropicToolCaller::CodeExecution20260120 | AnthropicToolCaller::CodeExecution20260521
    )
}

pub(crate) fn apply(
    request: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    validate(request, context)?;
    let Some(config) = request.hosted_anthropic_code_execution() else {
        return Ok(());
    };
    if !config.inline_definition {
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| invalid("Anthropic execution requires tools to be an array"))?
            .push(inline_tool_value());
    }
    if !config.skills.is_empty() {
        let mut container = Map::new();
        if let Some(reference) = &config.container {
            container.insert("id".into(), json!(reference.id()));
        }
        container.insert(
            "skills".into(),
            Value::Array(
                config
                    .skills
                    .iter()
                    .map(|skill| skill.wire_value())
                    .collect(),
            ),
        );
        body.insert("container".into(), Value::Object(container));
    } else if let Some(reference) = &config.container {
        body.insert("container".into(), json!(reference.id()));
    }
    Ok(())
}

/// `None` means no update. An explicit JSON null clears previously observed
/// container metadata; it is different from an omitted field.
pub(crate) fn stream_container(frame: &Value) -> Option<&Value> {
    match frame.get("type").and_then(Value::as_str) {
        Some("message_start") => frame.get("message")?.get("container"),
        Some("message_delta") => frame.get("delta")?.get("container"),
        _ => None,
    }
}

pub(crate) fn stream_usage(frame: &Value) -> Option<&Value> {
    match frame.get("type").and_then(Value::as_str) {
        Some("message_start") => frame.get("message")?.get("usage"),
        Some("message_delta") => frame.get("usage"),
        _ => None,
    }
}

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
