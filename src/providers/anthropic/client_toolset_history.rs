//! Validation for the Anthropic Browser and Computer client-toolset history.
//!
//! These calls remain host-executed [`ContentBlock::ToolUse`] values, but the
//! toolset namespace is required to distinguish members such as `screenshot`
//! and `key` from user-defined tools with the same names. Results echo that
//! namespace. This module validates only that provider-specific metadata and
//! the result shapes Anthropic interprets; it does not require a prior call to
//! still be declared in the current request's tool list.

use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, ContentBlock, LlmError, ToolUseId};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct ToolUseMetadata {
    name: String,
    toolset_name: Option<String>,
}

/// Whether the request contains Anthropic Browser/Computer namespace metadata.
/// The client uses this to reject metadata before another protocol can silently
/// discard it.
pub(crate) fn has_metadata(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::ToolUse { toolset_name, .. } => toolset_name.is_some(),
            ContentBlock::ToolResult {
                toolset_name,
                blocks,
                ..
            } => {
                toolset_name.is_some()
                    || blocks.as_ref().is_some_and(|blocks| {
                        blocks.iter().any(|block| {
                            block.get("type").and_then(Value::as_str) == Some("browser_state")
                        })
                    })
            }
            _ => false,
        })
    })
}

/// Validate the `toolset_name` round trip and the response content rules that
/// Anthropic applies to Browser/Computer members.
pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if !has_metadata(request) {
        return Ok(());
    }

    if !crate::providers::anthropic::client_toolsets::supports_profile(context.profile()) {
        return Err(unsupported(
            "Anthropic Browser/Computer toolset metadata requires the first-party Messages or Vertex Claude protocol",
        ));
    }

    let mut calls = HashMap::<ToolUseId, ToolUseMetadata>::new();
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse {
                    id,
                    name,
                    caller,
                    toolset_name,
                    ..
                } => {
                    if let Some(toolset_name) = toolset_name {
                        validate_member(toolset_name, name)?;
                        validate_direct_caller(caller.as_ref())?;
                    }
                    // Remember ordinary calls too: a result with toolset
                    // metadata cannot answer one of them.
                    calls.insert(
                        id.clone(),
                        ToolUseMetadata {
                            name: name.clone(),
                            toolset_name: toolset_name.clone(),
                        },
                    );
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    blocks,
                    toolset_name,
                    ..
                } => {
                    if let Some(toolset_name) = toolset_name {
                        validate_toolset_name(toolset_name)?;
                    }
                    let previous = calls.get(tool_use_id);
                    if let Some(previous) = previous {
                        if previous.toolset_name.as_deref() != toolset_name.as_deref() {
                            return Err(invalid(format!(
                                "Anthropic tool result for call {tool_use_id} must echo the call's toolset_name"
                            )));
                        }
                    }

                    let Some(toolset_name) = toolset_name.as_deref() else {
                        if blocks.as_ref().is_some_and(|blocks| {
                            blocks.iter().any(|block| {
                                block.get("type").and_then(Value::as_str) == Some("browser_state")
                            })
                        }) {
                            return Err(invalid(
                                "browser_state is valid only on a result for the Browser toolset",
                            ));
                        }
                        continue;
                    };
                    validate_result(
                        toolset_name,
                        previous.map(|call| call.name.as_str()),
                        *is_error,
                        blocks.as_deref(),
                    )?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_direct_caller(caller: Option<&Value>) -> Result<(), LlmError> {
    let Some(caller) = caller.filter(|caller| !caller.is_null()) else {
        return Ok(());
    };
    if caller.get("type").and_then(Value::as_str) != Some("direct") {
        return Err(invalid(
            "Anthropic Browser and Computer toolset calls support direct callers only",
        ));
    }
    Ok(())
}

fn validate_member(toolset_name: &str, member: &str) -> Result<(), LlmError> {
    validate_toolset_name(toolset_name)?;
    let members = match toolset_name {
        "browser" => crate::providers::anthropic::types::AnthropicBrowserMember::NAMES,
        "computer" => crate::providers::anthropic::types::AnthropicComputerMember::NAMES,
        _ => unreachable!("validate_toolset_name checked the toolset"),
    };
    if !members.contains(&member) {
        return Err(invalid(format!(
            "unknown Anthropic {toolset_name} toolset member {member:?}"
        )));
    }
    Ok(())
}

fn validate_toolset_name(toolset_name: &str) -> Result<(), LlmError> {
    if matches!(toolset_name, "browser" | "computer") {
        Ok(())
    } else {
        Err(invalid(format!(
            "unknown Anthropic client toolset name {toolset_name:?}"
        )))
    }
}

fn validate_result(
    toolset_name: &str,
    member: Option<&str>,
    is_error: bool,
    blocks: Option<&[Value]>,
) -> Result<(), LlmError> {
    let tab_management = toolset_name == "browser"
        && member.is_some_and(|member| {
            matches!(member, "new_tab" | "list_tabs" | "switch_tab" | "close_tab")
        });
    let mut browser_state = None;
    let mut has_text_representation = blocks.is_none();

    if let Some(blocks) = blocks {
        for (index, block) in blocks.iter().enumerate() {
            let kind = block
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Anthropic toolset result blocks require a string type"))?;
            let allowed = match toolset_name {
                "browser" => matches!(kind, "text" | "image" | "browser_state"),
                "computer" => matches!(kind, "text" | "image"),
                _ => unreachable!("validate_toolset_name checked the toolset"),
            };
            if !allowed {
                return Err(invalid(format!(
                    "Anthropic {toolset_name} tool results do not accept content block type {kind:?}"
                )));
            }
            if kind == "text" {
                if !block.get("text").is_some_and(Value::is_string) {
                    return Err(invalid(
                        "Anthropic toolset text result blocks require a string text field",
                    ));
                }
                has_text_representation = true;
            }
            if kind == "browser_state" {
                if is_error {
                    return Err(invalid(
                        "Anthropic Browser error results cannot include browser_state",
                    ));
                }
                if browser_state.is_some() {
                    return Err(invalid(
                        "Anthropic Browser results accept at most one browser_state block",
                    ));
                }
                browser_state = Some((index, validate_browser_state(block)?));
            }
        }
    }

    if is_error {
        if tab_management && !has_text_representation {
            return Err(invalid(
                "Anthropic Browser tab-management errors require text content and cannot include browser_state",
            ));
        }
        return Ok(());
    }

    if tab_management {
        let Some(blocks) = blocks else {
            return Err(invalid(
                "successful Anthropic Browser tab-management results require one browser_state block",
            ));
        };
        if blocks.len() != 1 || browser_state.is_none() {
            return Err(invalid(
                "successful Anthropic Browser tab-management results must contain exactly one browser_state block",
            ));
        }
        let (_, state) = browser_state.expect("checked above");
        if member == Some("new_tab") {
            let opened = state
                .tab_opened
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            if opened.len() != 1 || state.active_tab.as_deref() != opened.first().copied() {
                return Err(invalid(
                    "a successful browser new_tab result requires one tab_opened matching the active tab",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct BrowserStateSummary {
    active_tab: Option<String>,
    tab_opened: Vec<String>,
}

fn validate_browser_state(value: &Value) -> Result<BrowserStateSummary, LlmError> {
    if value.get("type").and_then(Value::as_str) != Some("browser_state") {
        return Err(invalid("expected an Anthropic browser_state block"));
    }
    let tabs = value
        .get("tabs")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Anthropic browser_state requires a tabs array"))?;
    if tabs.len() > 100 {
        return Err(invalid(
            "Anthropic browser_state can contain at most 100 tabs",
        ));
    }

    let mut summary = BrowserStateSummary::default();
    let mut tab_ids = HashSet::new();
    let mut active_count = 0usize;
    for tab in tabs {
        let object = tab
            .as_object()
            .ok_or_else(|| invalid("Anthropic browser_state tabs must be objects"))?;
        let tab_id = required_state_string(object, "tab_id", true)?;
        required_state_string(object, "title", false)?;
        required_state_string(object, "url", false)?;
        if !tab_ids.insert(tab_id.to_owned()) {
            return Err(invalid(
                "Anthropic browser_state contains duplicate tab_id values",
            ));
        }
        match object.get("active") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) => {
                active_count += 1;
                summary.active_tab = Some(tab_id.to_owned());
            }
            Some(_) => {
                return Err(invalid(
                    "Anthropic browser_state tab active fields must be booleans",
                ));
            }
        }
    }
    if (!tabs.is_empty() && active_count != 1) || active_count > 1 {
        return Err(invalid(
            "a non-empty Anthropic browser_state tabs array requires exactly one active tab",
        ));
    }

    if let Some(changes) = value
        .get("state_changes")
        .filter(|changes| !changes.is_null())
    {
        let changes = changes
            .as_array()
            .ok_or_else(|| invalid("Anthropic browser_state state_changes must be an array"))?;
        if changes.is_empty() {
            return Err(invalid(
                "omit Anthropic browser_state state_changes when there is nothing to report",
            ));
        }
        if changes.len() > 200 {
            return Err(invalid(
                "Anthropic browser_state can contain at most 200 state changes",
            ));
        }
        let mut download_ids = HashSet::new();
        for change in changes {
            let change = change
                .as_object()
                .ok_or_else(|| invalid("Anthropic browser_state changes must be objects"))?;
            let kind = change
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Anthropic browser_state changes require a string type"))?;
            match kind {
                "tab_opened" => {
                    only_fields(change, &["type", "tab_id"])?;
                    let tab_id = required_state_string(change, "tab_id", true)?;
                    if !tab_ids.contains(tab_id) {
                        return Err(invalid(
                            "Anthropic tab_opened state changes must reference a tab in tabs",
                        ));
                    }
                    summary.tab_opened.push(tab_id.to_owned());
                }
                "download_started" => {
                    only_fields(change, &["type", "download_id", "url"])?;
                    validate_download_fields(change, &mut download_ids, false, false)?;
                }
                "download_completed" => {
                    only_fields(
                        change,
                        &["type", "download_id", "url", "path", "size_bytes"],
                    )?;
                    validate_download_fields(change, &mut download_ids, true, false)?;
                }
                "download_failed" => {
                    only_fields(change, &["type", "download_id", "url", "error"])?;
                    validate_download_fields(change, &mut download_ids, false, true)?;
                }
                _ => {
                    return Err(invalid(format!(
                        "unknown Anthropic browser_state change type {kind:?}"
                    )));
                }
            }
        }
    }
    Ok(summary)
}

fn validate_download_fields(
    change: &serde_json::Map<String, Value>,
    download_ids: &mut HashSet<String>,
    allows_size: bool,
    allows_error: bool,
) -> Result<(), LlmError> {
    let download_id = required_state_string(change, "download_id", true)?;
    if !download_ids.insert(download_id.to_owned()) {
        return Err(invalid(
            "Anthropic browser_state may report a download_id only once per block",
        ));
    }
    required_state_string(change, "url", false)?;
    if change.contains_key("path") && allows_size {
        optional_nullable_state_string(change, "path")?;
    }
    if let Some(size) = change.get("size_bytes") {
        if !allows_size
            || (!size.is_null()
                && !size.as_f64().is_some_and(|size_bytes| {
                    size_bytes.is_finite() && size_bytes >= 0.0 && size_bytes.fract() == 0.0
                }))
        {
            return Err(invalid(
                "Anthropic download size_bytes must be null or a non-negative integer on download_completed",
            ));
        }
    }
    if change.contains_key("error") && allows_error {
        optional_nullable_state_string(change, "error")?;
    }
    Ok(())
}

fn required_state_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    non_empty: bool,
) -> Result<&'a str, LlmError> {
    let value = object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("Anthropic browser_state {key} must be a string")))?;
    validate_state_string(value, key, non_empty)?;
    Ok(value)
}

fn optional_nullable_state_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<(), LlmError> {
    if object.get(key).is_some_and(|value| !value.is_null()) {
        required_state_string(object, key, false)?;
    }
    Ok(())
}

fn validate_state_string(value: &str, field: &str, non_empty: bool) -> Result<(), LlmError> {
    if (non_empty && value.is_empty())
        || value.chars().count() > 4096
        || value
            .chars()
            .any(|ch| ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}'))
    {
        return Err(invalid(format!(
            "Anthropic browser_state {field} is empty or exceeds its allowed string format"
        )));
    }
    Ok(())
}

fn only_fields(object: &serde_json::Map<String, Value>, allowed: &[&str]) -> Result<(), LlmError> {
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            "Anthropic browser_state change contains a field not declared for its type",
        ));
    }
    Ok(())
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
