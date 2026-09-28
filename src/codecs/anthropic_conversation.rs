//! Anthropic Messages mid-conversation system messages and inline tool changes.

use crate::codecs::CodecContext;
use crate::protocol::{
    AnthropicClearAt, AnthropicMcpConfig, AnthropicToolCaller, ChatRequest, ContentBlock,
    ConversationMessage, LlmError, MessageRole, ProtocolFamily, ProviderProfile, ToolSpec,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";
const TOOL_CHANGE_REFERENCES_BETA: &str = "mid-conversation-tool-changes-2026-07-01";
const SYSTEM_CLEAR_AT_BETA: &str = "mid-conversation-system-clear-at-2026-08-21";
const OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const INLINE_DEFINITION_BYTES_LIMIT: usize = 4 * 1024 * 1024;
const INLINE_TOOL_COUNT_LIMIT: usize = 10_000;

const SUPPORTED_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5-5",
    "claude-opus-4-8",
    "claude-opus-5",
];

const PER_MESSAGE_EFFORT_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
];

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

fn is_official_profile(profile: &ProviderProfile) -> bool {
    crate::codecs::anthropic_code_execution::is_official_profile(profile)
}

fn is_first_party_profile(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::AnthropicMessages && is_official_profile(profile)
}

/// Platforms that support Anthropic per-message controls and reference-based
/// mid-conversation tool changes. Model availability is checked separately
/// against the exact request model ID.
pub(crate) fn supports_profile(profile: &ProviderProfile) -> bool {
    is_first_party_profile(profile) || profile.protocol == ProtocolFamily::VertexClaude
}

fn contains_inline_tool_definition(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            matches!(block, ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } if system_change(value) == Some("tool_addition")
                && value
                    .get("tool")
                    .and_then(|tool| tool.get("type"))
                    .and_then(Value::as_str)
                    == Some("tool_definition"))
        })
    })
}

fn system_change(value: &Value) -> Option<&'static str> {
    match value.get("type").and_then(Value::as_str) {
        Some("tool_addition") => Some("tool_addition"),
        Some("tool_removal") => Some("tool_removal"),
        _ => None,
    }
}

pub(crate) fn has_tool_additions(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            matches!(block, ContentBlock::ProviderContent { value, .. }
                if system_change(value) == Some("tool_addition"))
        })
    })
}

pub(crate) fn has_tool_changes(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            matches!(block, ContentBlock::ProviderContent { value, .. }
                if system_change(value).is_some())
        })
    })
}

pub(crate) fn has_mcp_toolset_addition(request: &ChatRequest, server_name: &str) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| {
            let ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } = block
            else {
                return false;
            };
            if system_change(value) != Some("tool_addition") {
                return false;
            }
            let Some(tool) = value.get("tool") else {
                return false;
            };
            if tool.get("type").and_then(Value::as_str) == Some("mcp_toolset_reference") {
                return tool.get("server_name").and_then(Value::as_str) == Some(server_name);
            }
            if tool.get("type").and_then(Value::as_str) == Some("mcp_tool_reference") {
                return tool.get("server_name").and_then(Value::as_str) == Some(server_name);
            }
            tool.get("type").and_then(Value::as_str) == Some("tool_definition")
                && tool.get("definition").is_some_and(|definition| {
                    definition.get("type").and_then(Value::as_str) == Some("mcp_toolset")
                        && definition.get("mcp_server_name").and_then(Value::as_str)
                            == Some(server_name)
                })
        })
    })
}

pub(crate) fn apply_beta_header(
    request: &ChatRequest,
    profile: &ProviderProfile,
    headers: &mut Vec<(String, String)>,
) {
    let has_clear_at = request.messages.iter().any(|message| {
        message
            .anthropic
            .as_ref()
            .is_some_and(|options| options.clear_at.is_some())
    });
    let has_effort = request.messages.iter().any(|message| {
        message
            .anthropic
            .as_ref()
            .is_some_and(|options| options.effort.is_some())
    });
    let has_tool_changes = has_tool_changes(request);
    if !supports_profile(profile) || !(has_tool_changes || has_clear_at || has_effort) {
        return;
    }
    let values = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .flat_map(|(_, value)| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut deduplicated = Vec::new();
    for value in values {
        if !deduplicated.contains(&value) {
            deduplicated.push(value);
        }
    }
    let tool_change_beta = if is_first_party_profile(profile) {
        INLINE_TOOLS_BETA
    } else {
        TOOL_CHANGE_REFERENCES_BETA
    };
    for (needed, beta) in [
        (has_tool_changes, tool_change_beta),
        (has_clear_at, SYSTEM_CLEAR_AT_BETA),
        (has_effort, OUTPUT_CONFIG_BETA),
    ] {
        if needed && !deduplicated.iter().any(|value| value == beta) {
            deduplicated.push(beta.into());
        }
    }
    headers.retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"));
    headers.push(("anthropic-beta".into(), deduplicated.join(",")));
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Reference {
    Tool(String),
    McpTool { server: String, name: String },
    McpToolset(String),
}

fn valid_name(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn parse_reference(value: &Value) -> Result<Reference, LlmError> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Anthropic tool changes require a typed reference"))?;
    let name = |key: &str| -> Result<String, LlmError> {
        let value = value
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| valid_name(value))
            .ok_or_else(|| invalid(format!("Anthropic tool reference requires a valid {key}")))?;
        Ok(value.to_owned())
    };
    match kind {
        "tool_reference" => Ok(Reference::Tool(name("name")?)),
        "mcp_tool_reference" => Ok(Reference::McpTool {
            server: name("server_name")?,
            name: name("name")?,
        }),
        "mcp_toolset_reference" => Ok(Reference::McpToolset(name("server_name")?)),
        _ => Err(invalid(
            "Anthropic tool changes use an unsupported reference type",
        )),
    }
}

fn is_server_tool_result(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value,
        } if value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                kind.ends_with("_tool_result")
                    || matches!(kind, "mcp_tool_result" | "tool_search_tool_result_error")
            })
    )
}

fn system_group_after_valid_turn(
    request: &ChatRequest,
    start: usize,
    end: usize,
) -> Result<bool, LlmError> {
    let Some(previous) = start
        .checked_sub(1)
        .and_then(|index| request.messages.get(index))
    else {
        return Err(invalid(
            "Anthropic mid-conversation system messages cannot be the first message",
        ));
    };
    let paused_server_turn = previous.role == MessageRole::Assistant
        && previous.content.last().is_some_and(is_server_tool_result);
    if previous.role != MessageRole::User && !paused_server_turn {
        return Err(invalid(
            "Anthropic system messages must follow a user turn or an assistant server-tool result",
        ));
    }
    if let Some(next) = request.messages.get(end + 1) {
        if next.role != MessageRole::Assistant {
            return Err(invalid(
                "Anthropic system-message groups must end the history or precede an assistant turn",
            ));
        }
    }
    Ok(paused_server_turn)
}

fn has_message_options(message: &ConversationMessage) -> bool {
    message
        .anthropic
        .as_ref()
        .is_some_and(|options| options.clear_at.is_some() || options.effort.is_some())
}

fn is_effort_only_empty_system(message: &ConversationMessage) -> bool {
    message.role == MessageRole::System
        && message.content.is_empty()
        && message.anthropic.as_ref().is_some_and(|options| {
            options.effort.is_some() && options.clear_at != Some(AnthropicClearAt::NextUserMessage)
        })
}

fn is_anthropic_text_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { .. } => true,
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value,
        } => {
            value.get("type").and_then(Value::as_str) == Some("text")
                && value.get("text").is_some_and(Value::is_string)
                && value.get("cache_control").is_none()
        }
        _ => false,
    }
}

fn validate_message_options(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let has_options = request.messages.iter().any(has_message_options);
    if !has_options {
        return Ok(());
    }

    if !supports_profile(context.profile()) {
        return Err(unsupported(
            "Anthropic per-message system options require the first-party Messages API or Vertex Claude",
        ));
    }

    for message in &request.messages {
        let Some(options) = message
            .anthropic
            .as_ref()
            .filter(|options| options.clear_at.is_some() || options.effort.is_some())
        else {
            continue;
        };
        if message.role != MessageRole::System {
            return Err(invalid(
                "Anthropic per-message options are supported only on system messages",
            ));
        }
        if options.effort.is_some() && !PER_MESSAGE_EFFORT_MODELS.contains(&context.request_model())
        {
            return Err(unsupported(format!(
                "Anthropic per-message effort is not documented for model {:?}",
                context.request_model()
            )));
        }
        match options.clear_at {
            Some(AnthropicClearAt::NextUserMessage) => {
                if options.effort.is_some() {
                    return Err(invalid(
                        "Anthropic clear_at=next_user_message cannot be combined with per-message effort",
                    ));
                }
                if message.content.is_empty()
                    || !message.content.iter().all(is_anthropic_text_block)
                {
                    return Err(invalid(
                        "Anthropic clear_at=next_user_message requires one or more text blocks without cache markers",
                    ));
                }
            }
            Some(AnthropicClearAt::Never) | None => {}
        }
        if message.content.is_empty() && !is_effort_only_empty_system(message) {
            return Err(invalid(
                "an empty Anthropic system message requires per-message effort without turn-scoped clearing",
            ));
        }
    }
    Ok(())
}

fn collect_mcp_configs(request: &ChatRequest) -> BTreeMap<String, &AnthropicMcpConfig> {
    request
        .anthropic_mcp_servers()
        .map(|config| (config.name().to_owned(), config))
        .collect()
}

struct ToolTimeline<'a> {
    client_toolset_deferred_count: usize,
    direct_types: BTreeMap<String, String>,
    active_direct: BTreeSet<String>,
    deferred_direct: BTreeSet<String>,
    inline_definitions: BTreeMap<String, Value>,
    typed_inline_definitions: BTreeMap<String, Value>,
    inline_tool_history: Vec<ToolSpec>,
    active_inline: BTreeSet<String>,
    mcp_configs: BTreeMap<String, &'a AnthropicMcpConfig>,
    active_mcp: BTreeSet<String>,
    seen_inline_mcp: BTreeSet<String>,
    mcp_tool_overrides: BTreeMap<(String, String), bool>,
}

fn tool_type_family(kind: &str) -> String {
    if let Some((family, suffix)) = kind.rsplit_once('_') {
        if suffix.len() == 8 && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            return family.to_owned();
        }
    }
    kind.to_owned()
}

fn is_typed_inline_server_definition(definition: &Value) -> bool {
    definition
        .get("type")
        .and_then(Value::as_str)
        .map(tool_type_family)
        .is_some_and(|family| matches!(family.as_str(), "code_execution" | "web_fetch"))
}

fn parse_inline_callers(definition: &Value) -> Result<Vec<AnthropicToolCaller>, LlmError> {
    let Some(raw) = definition.get("allowed_callers") else {
        return Ok(Vec::new());
    };
    let raw = raw
        .as_array()
        .ok_or_else(|| invalid("Anthropic inline allowed_callers must be an array"))?;
    let mut callers = Vec::with_capacity(raw.len());
    for value in raw {
        let caller = match value.as_str() {
            Some("direct") => AnthropicToolCaller::Direct,
            Some("code_execution_20260120") => AnthropicToolCaller::CodeExecution20260120,
            Some("code_execution_20260521") => AnthropicToolCaller::CodeExecution20260521,
            Some(_) => {
                return Err(unsupported(
                    "inline allowed_callers supports direct and the typed Code Execution caller versions",
                ));
            }
            None => {
                return Err(invalid(
                    "Anthropic inline allowed_callers values must be strings",
                ))
            }
        };
        if callers.contains(&caller) {
            return Err(invalid("Anthropic inline allowed_callers repeats a value"));
        }
        callers.push(caller);
    }
    if callers.contains(&AnthropicToolCaller::CodeExecution20260120)
        && callers.contains(&AnthropicToolCaller::CodeExecution20260521)
    {
        return Err(invalid(
            "Anthropic inline allowed_callers cannot list both interchangeable Code Execution versions",
        ));
    }
    Ok(callers)
}

fn inline_tool_spec(definition: &Value) -> Result<ToolSpec, LlmError> {
    let name = definition
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| valid_name(name))
        .ok_or_else(|| invalid("Anthropic inline tool definitions require a valid name"))?
        .to_owned();
    let kind = match definition.get("type") {
        None => "custom",
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => return Err(invalid("Anthropic inline tool type must be a string")),
    };
    if kind != "custom" {
        return Err(unsupported(
            "Anthropic-defined server and client tools require their typed integration and cannot be added by generic inline definition",
        ));
    }
    let schema = match definition.get("input_schema") {
        Some(schema) if schema.is_object() => schema.clone(),
        Some(_) => return Err(invalid("Anthropic inline input_schema must be an object")),
        None if kind == "custom" => {
            return Err(invalid(
                "Anthropic inline custom tool definitions require an object input_schema",
            ));
        }
        None => json!({}),
    };
    if definition
        .get("description")
        .is_some_and(|description| !description.is_string() && !description.is_null())
    {
        return Err(invalid(
            "Anthropic inline tool description must be a string or null",
        ));
    }
    let strict = match definition.get("strict") {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid("Anthropic inline tool strict must be a boolean"))?,
        None => false,
    };
    let defer_loading = match definition.get("defer_loading") {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid("Anthropic inline tool defer_loading must be a boolean"))?,
        None => false,
    };
    Ok(ToolSpec {
        tool_type: None,
        extra: Value::Null,
        name,
        description: definition
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        input_schema: schema,
        strict,
        defer_loading,
        allowed_callers: parse_inline_callers(definition)?,
    })
}

fn add_tool_definition(definition: &Value, state: &mut ToolTimeline<'_>) -> Result<(), LlmError> {
    let ToolTimeline {
        direct_types,
        active_direct,
        deferred_direct,
        inline_definitions,
        typed_inline_definitions,
        inline_tool_history,
        active_inline,
        ..
    } = state;
    if let Some(name) = definition.get("name").and_then(Value::as_str) {
        if let Some(expected) = typed_inline_definitions.get(name) {
            if definition != expected {
                return Err(invalid(format!(
                    "inline Anthropic server tool {name:?} must exactly match its typed configuration"
                )));
            }
            let family = definition
                .get("type")
                .and_then(Value::as_str)
                .map(tool_type_family)
                .ok_or_else(|| invalid("inline Anthropic server tool requires a type"))?;
            if direct_types
                .get(name)
                .is_some_and(|existing| existing != &family)
            {
                return Err(invalid(
                    "Anthropic inline tool definition conflicts with an existing tool type",
                ));
            }
            direct_types.insert(name.to_owned(), family.clone());
            if family == "code_execution" {
                for execution_name in [
                    "code_execution",
                    "bash_code_execution",
                    "text_editor_code_execution",
                ] {
                    direct_types.insert(execution_name.to_owned(), family.clone());
                    active_direct.insert(execution_name.to_owned());
                }
            } else {
                active_direct.insert(name.to_owned());
            }
            if definition.get("defer_loading").and_then(Value::as_bool) == Some(true) {
                deferred_direct.insert(name.to_owned());
            } else {
                deferred_direct.remove(name);
            }
            inline_definitions.insert(name.to_owned(), definition.clone());
            active_inline.insert(name.to_owned());
            return Ok(());
        }
    }
    let spec = inline_tool_spec(definition)?;
    let name = spec.name.clone();
    let kind = definition
        .get("type")
        .and_then(Value::as_str)
        .map(tool_type_family)
        .unwrap_or_else(|| "custom".to_owned());
    if direct_types
        .get(&name)
        .is_some_and(|existing| existing != &kind)
    {
        return Err(invalid(
            "Anthropic inline tool definition conflicts with an existing tool type",
        ));
    }
    direct_types.insert(name.clone(), kind);
    active_direct.insert(name.clone());
    if spec.defer_loading {
        deferred_direct.insert(name.clone());
    } else {
        deferred_direct.remove(&name);
    }
    inline_definitions.insert(name.clone(), definition.clone());
    inline_tool_history.push(spec);
    active_inline.insert(name);
    Ok(())
}

fn encode_mcp_definition(
    definition: &Value,
    inline_configs: &BTreeMap<String, &AnthropicMcpConfig>,
    active_mcp: &mut BTreeSet<String>,
    seen_inline_mcp: &mut BTreeSet<String>,
    inline_definitions: &mut BTreeMap<String, Value>,
    active_inline: &mut BTreeSet<String>,
) -> Result<(), LlmError> {
    let server = definition
        .get("mcp_server_name")
        .and_then(Value::as_str)
        .filter(|name| valid_name(name))
        .ok_or_else(|| invalid("Anthropic MCP toolset definitions require mcp_server_name"))?
        .to_owned();
    let Some(config) = inline_configs.get(&server) else {
        return Err(invalid(
            "inline MCP toolsets require matching typed AnthropicMcpConfig with inline placement",
        ));
    };
    if &config.toolset_value() != definition {
        return Err(invalid(
            "inline MCP toolset definition must match its typed AnthropicMcpConfig",
        ));
    }
    let key = format!("\0mcp:{server}");
    if seen_inline_mcp.contains(&server) {
        if inline_definitions.get(&key) == Some(definition) {
            active_mcp.insert(server);
            active_inline.insert(key);
            return Ok(());
        }
        return Err(invalid(
            "an inline Anthropic MCP server cannot change its typed toolset definition",
        ));
    }
    seen_inline_mcp.insert(server.clone());
    active_mcp.insert(server.clone());
    inline_definitions.insert(key.clone(), definition.clone());
    active_inline.insert(key);
    Ok(())
}

fn process_system_change(
    value: &Value,
    state: &mut ToolTimeline<'_>,
    paused_server_turn: bool,
) -> Result<(), LlmError> {
    let ToolTimeline {
        direct_types,
        active_direct,
        inline_definitions,
        active_inline,
        mcp_configs,
        active_mcp,
        seen_inline_mcp,
        ..
    } = state;
    let kind = system_change(value).expect("caller checked type");
    if paused_server_turn {
        return Err(invalid(
            "Anthropic tool additions and removals cannot immediately follow a paused server-tool result",
        ));
    }
    let tool = value
        .get("tool")
        .ok_or_else(|| invalid("Anthropic tool changes require a tool reference or definition"))?;
    let tool_type = tool
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Anthropic tool changes require a typed tool field"))?;
    if kind == "tool_removal" && tool_type == "tool_definition" {
        return Err(invalid("Anthropic tool removal accepts references only"));
    }

    if tool_type == "tool_definition" {
        if kind != "tool_addition" {
            return Err(invalid("Anthropic tool definitions can only be added"));
        }
        let definition = tool
            .get("definition")
            .ok_or_else(|| invalid("Anthropic tool definitions require definition"))?;
        if definition.get("type").and_then(Value::as_str) == Some("mcp_toolset") {
            if let Some(server) = definition.get("mcp_server_name").and_then(Value::as_str) {
                // A whole-set addition supersedes prior member removals,
                // whether it uses a reference or repeats its definition.
                state
                    .mcp_tool_overrides
                    .retain(|(name, _), _| name != server);
            }
        }
        match definition.get("type").and_then(Value::as_str) {
            Some("mcp_toolset") => encode_mcp_definition(
                definition,
                &mcp_configs
                    .iter()
                    .filter(|(_, config)| config.inline_toolset())
                    .map(|(name, config)| (name.clone(), *config))
                    .collect(),
                active_mcp,
                seen_inline_mcp,
                inline_definitions,
                active_inline,
            ),
            _ => add_tool_definition(definition, state),
        }
    } else {
        let reference_value = parse_reference(tool)?;
        match reference_value {
            Reference::Tool(name) => {
                if !direct_types.contains_key(&name) {
                    return Err(invalid(format!(
                        "Anthropic tool reference {name:?} is unresolved"
                    )));
                }
                if kind == "tool_addition" {
                    active_direct.insert(name.clone());
                    if name == "code_execution" {
                        active_direct.insert("bash_code_execution".into());
                        active_direct.insert("text_editor_code_execution".into());
                    }
                    if inline_definitions.contains_key(&name) {
                        active_inline.insert(name);
                    }
                } else {
                    active_direct.remove(&name);
                    if name == "code_execution" {
                        active_direct.remove("bash_code_execution");
                        active_direct.remove("text_editor_code_execution");
                    }
                    active_inline.remove(&name);
                }
            }
            Reference::McpTool { server, name } => {
                let Some(config) = mcp_configs.get(&server) else {
                    return Err(invalid(format!(
                        "Anthropic MCP tool reference for server {server:?} is undeclared"
                    )));
                };
                if config.inline_toolset() && !seen_inline_mcp.contains(&server) {
                    return Err(invalid(
                        "inline MCP member reference precedes its toolset definition",
                    ));
                }
                if let Some(tools) = config.tools() {
                    if !tools.iter().any(|tool| tool.name == name)
                        || !mcp_tool_enabled(config, &name)
                    {
                        return Err(invalid(format!(
                            "Anthropic MCP tool reference {name:?} is absent or disabled in the pinned toolset"
                        )));
                    }
                }
                // Unpinned names are provider-resolved; pinned member state
                // updates local availability limits across removals and readds.
                state
                    .mcp_tool_overrides
                    .insert((server, name), kind == "tool_addition");
            }
            Reference::McpToolset(server) => {
                let Some(config) = mcp_configs.get(&server) else {
                    return Err(invalid(format!(
                        "Anthropic MCP toolset reference for server {server:?} is unresolved"
                    )));
                };
                if config.inline_toolset() && !seen_inline_mcp.contains(&server) {
                    return Err(invalid(format!(
                        "Anthropic inline MCP toolset reference for server {server:?} precedes its definition"
                    )));
                }
                state
                    .mcp_tool_overrides
                    .retain(|(name, _), _| name != &server);
                if kind == "tool_addition" {
                    active_mcp.insert(server.clone());
                    let key = format!("\0mcp:{server}");
                    if inline_definitions.contains_key(&key) {
                        active_inline.insert(key);
                    }
                } else {
                    active_mcp.remove(&server);
                    active_inline.remove(&format!("\0mcp:{server}"));
                }
            }
        }
        Ok(())
    }
}

fn deferred_direct_addition_name(value: &Value) -> Option<String> {
    if system_change(value) != Some("tool_addition") {
        return None;
    }
    let tool = value.get("tool")?;
    match tool.get("type")?.as_str()? {
        "tool_reference" => tool.get("name")?.as_str().map(str::to_owned),
        "tool_definition" => {
            let definition = tool.get("definition")?;
            (definition.get("type").and_then(Value::as_str) != Some("mcp_toolset"))
                .then(|| {
                    definition
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .flatten()
        }
        _ => None,
    }
}

impl ToolTimeline<'_> {
    fn enforce_inline_limits(&self) -> Result<(), LlmError> {
        let mut active_count = 0usize;
        let mut deferred_count = self
            .deferred_direct
            .intersection(&self.active_direct)
            .count()
            .saturating_add(self.client_toolset_deferred_count);
        let mut definition_bytes = 0usize;
        for name in &self.active_inline {
            let definition = self.inline_definitions.get(name).ok_or_else(|| {
                invalid("Anthropic inline-tool state lost a definition needed for replay")
            })?;
            if !name.starts_with("\0mcp:") {
                active_count = active_count.saturating_add(1);
                definition_bytes = definition_bytes.saturating_add(
                    serde_json::to_vec(definition)
                        .map_err(|_| invalid("Anthropic inline definition is not valid JSON"))?
                        .len(),
                );
            }
        }
        for (server, config) in &self.mcp_configs {
            if let Some(tools) = config.tools() {
                for tool in tools {
                    if !mcp_tool_enabled(config, &tool.name)
                        || !self.mcp_tool_is_active(server, &tool.name)
                    {
                        continue;
                    }
                    if config.inline_toolset() {
                        active_count = active_count.saturating_add(1);
                        definition_bytes = definition_bytes.saturating_add(
                            serde_json::to_vec(tool)
                                .map_err(|_| invalid("Anthropic pinned tool is not valid JSON"))?
                                .len(),
                        );
                    }
                    if mcp_tool_deferred(config, &tool.name) {
                        deferred_count = deferred_count.saturating_add(1);
                    }
                }
            }
        }
        if active_count > INLINE_TOOL_COUNT_LIMIT || deferred_count > INLINE_TOOL_COUNT_LIMIT {
            return Err(unsupported(
                "Anthropic mid-conversation tools exceed the 10,000-tool availability limit",
            ));
        }
        if definition_bytes > INLINE_DEFINITION_BYTES_LIMIT {
            return Err(unsupported(
                "Anthropic inline tool definitions exceed the 4 MiB request limit",
            ));
        }
        Ok(())
    }

    fn mcp_tool_is_active(&self, server: &str, name: &str) -> bool {
        self.mcp_tool_overrides
            .get(&(server.to_owned(), name.to_owned()))
            .copied()
            .unwrap_or_else(|| self.active_mcp.contains(server))
    }

    fn active_tool_spec(
        &self,
        request: &ChatRequest,
        name: &str,
    ) -> Result<Option<ToolSpec>, LlmError> {
        if !self.active_direct.contains(name) {
            return Ok(None);
        }
        if self.active_inline.contains(name) {
            if self
                .direct_types
                .get(name)
                .is_some_and(|family| family != "custom")
            {
                return Ok(None);
            }
            let definition = self.inline_definitions.get(name).ok_or_else(|| {
                invalid("Anthropic inline-tool state lost a definition needed for validation")
            })?;
            return inline_tool_spec(definition).map(Some);
        }
        Ok(request.tools.iter().find(|tool| tool.name == name).cloned())
    }
}

fn validate_programmatic_calls_at_timeline(
    message: &ConversationMessage,
    request: &ChatRequest,
    timeline: &ToolTimeline<'_>,
    surfaced_deferred: &BTreeSet<String>,
) -> Result<(), LlmError> {
    for block in &message.content {
        let ContentBlock::ToolUse {
            name,
            caller: Some(caller),
            ..
        } = block
        else {
            continue;
        };
        let Some(caller_type) = caller.get("type").and_then(Value::as_str) else {
            continue;
        };
        if !matches!(
            caller_type,
            "code_execution_20260120" | "code_execution_20260521"
        ) {
            continue;
        }
        if request
            .hosted_anthropic_code_execution()
            .is_some_and(|config| config.inline_definition)
            && !timeline.active_direct.contains("code_execution")
        {
            return Err(invalid(
                "programmatic tool calls require the inline Code Execution definition to be active at their position",
            ));
        }
        if !timeline.active_direct.contains(name)
            || (timeline.deferred_direct.contains(name) && !surfaced_deferred.contains(name))
        {
            return Err(invalid(format!(
                "programmatic tool call {name:?} was not available at its position in the conversation"
            )));
        }
        let Some(tool) = timeline.active_tool_spec(request, name)? else {
            return Err(invalid(format!(
                "programmatic tool call {name:?} has no matching tool definition at its position in the conversation"
            )));
        };
        validate_programmatic_tool_spec(&tool, name)?;
    }
    Ok(())
}

fn validate_inline_server_calls_at_timeline(
    message: &ConversationMessage,
    request: &ChatRequest,
    timeline: &ToolTimeline<'_>,
) -> Result<(), LlmError> {
    let inline_execution = request
        .hosted_anthropic_code_execution()
        .is_some_and(|config| config.inline_definition);
    let inline_fetch = request
        .hosted_anthropic_web_fetch()
        .is_some_and(|config| config.inline_definition);
    if !inline_execution && !inline_fetch {
        return Ok(());
    }
    for block in &message.content {
        let ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value,
        } = block
        else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) != Some("server_tool_use") {
            continue;
        }
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (enabled, definition_name) = match name {
            "code_execution" | "bash_code_execution" | "text_editor_code_execution" => {
                (inline_execution, "code_execution")
            }
            "web_fetch" => (inline_fetch, "web_fetch"),
            _ => (false, ""),
        };
        if enabled && !timeline.active_direct.contains(definition_name) {
            return Err(invalid(format!(
                "Anthropic server tool {name:?} was used before its inline definition became available"
            )));
        }
    }
    Ok(())
}

fn validate_programmatic_tool_spec(tool: &ToolSpec, name: &str) -> Result<(), LlmError> {
    if !tool.allowed_callers.iter().any(is_programmatic_caller) {
        return Err(invalid(format!(
            "programmatic tool call {name:?} requires its active definition to allow a Code Execution caller"
        )));
    }
    if tool.strict {
        return Err(unsupported(format!(
            "Anthropic programmatic tool calling does not support strict tool {name:?}"
        )));
    }
    crate::codecs::structured::validate_no_recursive_schema_references(&tool.input_schema)
}

fn validate_pending_programmatic_definitions(
    request: &ChatRequest,
    timeline: &ToolTimeline<'_>,
    surfaced_deferred: &BTreeSet<String>,
) -> Result<(), LlmError> {
    let mut completed_server_tools = BTreeMap::<String, usize>::new();
    let mut programmatic_calls = Vec::<(String, String, usize)>::new();
    let mut history_position = 0usize;
    for message in &request.messages {
        for block in &message.content {
            match block {
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
                } if caller
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|caller| {
                        matches!(
                            caller,
                            "code_execution_20260120" | "code_execution_20260521"
                        )
                    }) =>
                {
                    if let Some(tool_id) = caller.get("tool_id").and_then(Value::as_str) {
                        programmatic_calls.push((
                            name.clone(),
                            tool_id.to_owned(),
                            history_position,
                        ));
                    }
                }
                _ => {}
            }
            history_position = history_position.saturating_add(1);
        }
    }
    for (name, tool_id, call_position) in programmatic_calls {
        if completed_server_tools
            .get(&tool_id)
            .is_some_and(|result_position| *result_position > call_position)
        {
            continue;
        }
        if !timeline.active_direct.contains(&name)
            || (timeline.deferred_direct.contains(&name) && !surfaced_deferred.contains(&name))
        {
            return Err(invalid(format!(
                "pending programmatic tool call {name:?} has no active definition after the conversation history"
            )));
        }
        if request
            .hosted_anthropic_code_execution()
            .is_some_and(|config| config.inline_definition)
            && !timeline.active_direct.contains("code_execution")
        {
            return Err(invalid(
                "pending programmatic tool calls require the inline Code Execution definition to remain active",
            ));
        }
        let Some(tool) = timeline.active_tool_spec(request, &name)? else {
            return Err(invalid(format!(
                "pending programmatic tool call {name:?} has no active definition after the conversation history"
            )));
        };
        validate_programmatic_tool_spec(&tool, &name)?;
    }
    Ok(())
}

fn is_programmatic_caller(caller: &AnthropicToolCaller) -> bool {
    matches!(
        caller,
        AnthropicToolCaller::CodeExecution20260120 | AnthropicToolCaller::CodeExecution20260521
    )
}

/// Extract inline tool definitions for the Code Execution validator. Callers
/// must first run [`validate`], which checks placement and definition changes
/// against the chronological tool timeline.
pub(crate) fn collect_inline_tool_specs(request: &ChatRequest) -> Result<Vec<ToolSpec>, LlmError> {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } if system_change(value) == Some("tool_addition")
                && value.pointer("/tool/type").and_then(Value::as_str)
                    == Some("tool_definition") =>
            {
                value.pointer("/tool/definition")
            }
            _ => None,
        })
        .filter(|definition| {
            definition.get("type").and_then(Value::as_str) != Some("mcp_toolset")
                && !is_typed_inline_server_definition(definition)
        })
        .map(inline_tool_spec)
        .collect()
}

fn mcp_tool_enabled(config: &AnthropicMcpConfig, name: &str) -> bool {
    let default = config.default_config();
    let override_config = config.configs().get(name);
    override_config
        .and_then(|config| config.enabled)
        .or_else(|| default.and_then(|config| config.enabled))
        .unwrap_or(true)
}

fn mcp_tool_deferred(config: &AnthropicMcpConfig, name: &str) -> bool {
    let default = config.default_config();
    let override_config = config.configs().get(name);
    override_config
        .and_then(|config| config.defer_loading)
        .or_else(|| default.and_then(|config| config.defer_loading))
        .unwrap_or(false)
}

pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    validate_message_options(request, context)?;
    let contains_system = request
        .messages
        .iter()
        .any(|message| message.role == MessageRole::System);
    let contains_changes = has_tool_changes(request);
    let contains_mcp = request.anthropic_mcp_servers().next().is_some();
    let contains_inline_mcp = request
        .anthropic_mcp_servers()
        .any(|config| config.inline_toolset());
    let contains_deferred_fetch = request
        .hosted_anthropic_web_fetch()
        .is_some_and(|config| config.defer_loading);
    let contains_typed_inline = request
        .hosted_anthropic_code_execution()
        .is_some_and(|config| config.inline_definition)
        || request
            .hosted_anthropic_web_fetch()
            .is_some_and(|config| config.inline_definition);
    if !contains_system
        && !contains_changes
        && !contains_mcp
        && !contains_deferred_fetch
        && !contains_typed_inline
    {
        return Ok(());
    }

    let profile = context.profile();
    if !supports_profile(profile) {
        if !contains_system && !contains_changes && !contains_inline_mcp && !contains_typed_inline {
            return Ok(());
        }
        return Err(unsupported(
            "Anthropic mid-conversation system messages and tool changes require the first-party Messages API or Vertex Claude",
        ));
    }
    let vertex = profile.protocol == ProtocolFamily::VertexClaude;
    if vertex && (contains_inline_mcp || contains_inline_tool_definition(request)) {
        return Err(unsupported(
            "Vertex Claude supports reference-based mid-conversation tool changes only; inline tool definitions and MCP additions require the Claude API",
        ));
    }
    if (contains_system || contains_changes) && !SUPPORTED_MODELS.contains(&context.request_model())
    {
        return Err(unsupported(format!(
            "Anthropic mid-conversation system messages and tool changes are not documented for model {:?}",
            context.request_model()
        )));
    }
    if contains_changes
        && profile
            .extra
            .get("body")
            .and_then(Value::as_object)
            .is_some_and(|body| body.contains_key("tools"))
    {
        return Err(invalid(
            "mid-conversation Anthropic tool changes cannot use extra.body.tools; declare tools through ChatRequest.tools",
        ));
    }

    // Tool changes are native provider blocks and are meaningful only inside
    // Anthropic system messages. Reject attempts to smuggle them through normal
    // user/assistant content.
    for message in &request.messages {
        for block in &message.content {
            if let ContentBlock::ProviderContent { value, .. } = block {
                if system_change(value).is_some() && message.role != MessageRole::System {
                    return Err(invalid(
                        "Anthropic tool_addition and tool_removal blocks must be inside system messages",
                    ));
                }
            }
        }
    }

    let mut direct_types = request
        .tools
        .iter()
        .map(|tool| (tool.name.clone(), "custom".to_owned()))
        .collect::<BTreeMap<_, _>>();
    let mut typed_inline_definitions = BTreeMap::new();
    if let Some(search) = request.hosted_anthropic_tool_search() {
        let kind = match search.strategy {
            crate::protocol::AnthropicToolSearchStrategy::Regex => {
                "tool_search_tool_regex_20251119"
            }
            crate::protocol::AnthropicToolSearchStrategy::Bm25 => "tool_search_tool_bm25_20251119",
        };
        let name = if kind.starts_with("tool_search_tool_regex") {
            "tool_search_tool_regex"
        } else {
            "tool_search_tool_bm25"
        };
        direct_types.insert(name.into(), tool_type_family(kind));
    }
    if let Some(config) = request.hosted_anthropic_code_execution() {
        for name in [
            "code_execution",
            "bash_code_execution",
            "text_editor_code_execution",
        ] {
            direct_types.insert(name.into(), "code_execution".into());
        }
        if config.inline_definition {
            if !is_first_party_profile(profile) {
                return Err(unsupported(
                    "inline Anthropic Code Execution definitions require the first-party Messages API",
                ));
            }
            typed_inline_definitions.insert(
                "code_execution".into(),
                crate::codecs::anthropic_code_execution::inline_tool_value(),
            );
        }
    }
    if request.hosted_web_search().is_some() {
        direct_types.insert("web_search".into(), "web_search".into());
    }
    if let Some(config) = request.hosted_anthropic_web_fetch() {
        direct_types.insert("web_fetch".into(), "web_fetch".into());
        if config.inline_definition {
            if !is_first_party_profile(profile) {
                return Err(unsupported(
                    "inline Anthropic Web Fetch definitions require the first-party Messages API",
                ));
            }
            typed_inline_definitions.insert(
                "web_fetch".into(),
                crate::codecs::anthropic_web_fetch::tool_value(config)?,
            );
        }
    }
    let mut active_direct = request
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    if request
        .hosted_anthropic_code_execution()
        .is_some_and(|config| !config.inline_definition)
    {
        active_direct.extend([
            "code_execution".into(),
            "bash_code_execution".into(),
            "text_editor_code_execution".into(),
        ]);
    }
    if request
        .hosted_anthropic_web_fetch()
        .is_some_and(|config| !config.inline_definition && !config.defer_loading)
    {
        active_direct.insert("web_fetch".into());
    }
    active_direct.extend(
        direct_types
            .iter()
            .filter(|(name, kind)| {
                kind.as_str() != "custom"
                    && !matches!(
                        name.as_str(),
                        "code_execution"
                            | "bash_code_execution"
                            | "text_editor_code_execution"
                            | "web_fetch"
                    )
            })
            .map(|(name, _)| name.clone()),
    );
    let mut deferred_direct = request
        .tools
        .iter()
        .filter(|tool| tool.defer_loading)
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    if contains_deferred_fetch {
        deferred_direct.insert("web_fetch".into());
    }
    let active_mcp = request
        .anthropic_mcp_servers()
        .filter(|config| !config.inline_toolset())
        .map(|config| config.name().to_owned())
        .collect::<BTreeSet<_>>();
    let seen_inline_mcp = BTreeSet::new();
    let inline_definitions = BTreeMap::<String, Value>::new();
    let inline_tool_history = Vec::new();
    let active_inline = BTreeSet::<String>::new();
    let mut surfaced_deferred = BTreeSet::<String>::new();
    let mut timeline = ToolTimeline {
        client_toolset_deferred_count: super::anthropic_client_toolsets::deferred_count(request),
        direct_types,
        active_direct,
        deferred_direct,
        inline_definitions,
        typed_inline_definitions,
        inline_tool_history,
        active_inline,
        mcp_configs: collect_mcp_configs(request),
        active_mcp,
        seen_inline_mcp,
        mcp_tool_overrides: BTreeMap::new(),
    };

    if !contains_system
        && !contains_changes
        && !contains_inline_mcp
        && !contains_deferred_fetch
        && !contains_typed_inline
    {
        timeline.enforce_inline_limits()?;
        return Ok(());
    }

    let mut index = 0usize;
    while index < request.messages.len() {
        let message = &request.messages[index];
        if message.role != MessageRole::System {
            if message.content.iter().any(|block| {
                matches!(block, ContentBlock::ProviderContent { value, .. }
                    if system_change(value).is_some())
            }) {
                return Err(invalid(
                    "Anthropic tool changes must be inside system messages",
                ));
            }
            validate_programmatic_calls_at_timeline(
                message,
                request,
                &timeline,
                &surfaced_deferred,
            )?;
            validate_inline_server_calls_at_timeline(message, request, &timeline)?;
            timeline.enforce_inline_limits()?;
            index += 1;
            continue;
        }
        let start = index;
        while index + 1 < request.messages.len()
            && request.messages[index + 1].role == MessageRole::System
        {
            index += 1;
        }
        let end = index;
        let group_requires_placement = request.messages[start..=end]
            .iter()
            .any(|message| !is_effort_only_empty_system(message));
        let paused_server_turn = if group_requires_placement {
            system_group_after_valid_turn(request, start, end)?
        } else {
            false
        };
        for system_message in &request.messages[start..=end] {
            if system_message.content.is_empty() && !is_effort_only_empty_system(system_message) {
                return Err(invalid(
                    "an empty Anthropic system message requires per-message effort without turn-scoped clearing",
                ));
            }
            for block in &system_message.content {
                match block {
                    ContentBlock::Text { .. } => {}
                    ContentBlock::ProviderContent {
                        protocol: ProtocolFamily::AnthropicMessages,
                        value,
                    } if value.get("type").and_then(Value::as_str) == Some("text")
                        && value.get("text").is_some_and(Value::is_string) => {}
                    ContentBlock::ProviderContent { protocol, value }
                        if *protocol == ProtocolFamily::AnthropicMessages
                            && system_change(value).is_some() =>
                    {
                        let surface_name = deferred_direct_addition_name(value);
                        process_system_change(value, &mut timeline, paused_server_turn)?;
                        if let Some(name) = surface_name {
                            if timeline.deferred_direct.contains(&name)
                                && timeline.active_direct.contains(&name)
                            {
                                surfaced_deferred.insert(name);
                            }
                        }
                    }
                    _ => {
                        return Err(invalid(
                            "Anthropic mid-conversation system messages support text and tool-change blocks only",
                        ));
                    }
                }
            }
            timeline.enforce_inline_limits()?;
        }
        index += 1;
    }

    for config in request
        .anthropic_mcp_servers()
        .filter(|config| config.inline_toolset())
    {
        if !timeline.seen_inline_mcp.contains(config.name()) {
            return Err(invalid(format!(
                "inline Anthropic MCP server {:?} requires its tool_addition system message",
                config.name()
            )));
        }
    }
    for name in timeline.typed_inline_definitions.keys() {
        if !timeline.inline_definitions.contains_key(name) {
            return Err(invalid(format!(
                "inline Anthropic server tool {name:?} requires its matching tool_addition definition"
            )));
        }
    }
    if request.hosted_anthropic_tool_search().is_none() {
        for tool in request.tools.iter().filter(|tool| tool.defer_loading) {
            if !surfaced_deferred.contains(&tool.name) {
                return Err(invalid(format!(
                    "deferred Anthropic tool {:?} requires hosted tool search or a matching inline tool_addition",
                    tool.name
                )));
            }
        }
    }
    if let crate::protocol::ToolChoice::Tool { name } = &request.tool_choice {
        if !timeline.active_direct.contains(name) {
            return Err(invalid(format!(
                "Anthropic tool_choice references unavailable tool {name:?}"
            )));
        }
    }
    validate_pending_programmatic_definitions(request, &timeline, &surfaced_deferred)?;
    if !timeline.inline_tool_history.is_empty() {
        crate::codecs::structured::validate_messages_inline_tool_schemas(
            request,
            &timeline.inline_tool_history,
        )?;
        crate::codecs::anthropic_code_execution::validate_with_inline_tools(
            request,
            context,
            &timeline.inline_tool_history,
        )?;
    }
    Ok(())
}
