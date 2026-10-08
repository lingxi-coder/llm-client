//! Validation for the Anthropic Browser and Computer client-toolset history.
//!
//! These calls remain host-executed [`ContentBlock::ToolUse`] values, but the
//! toolset namespace is required to distinguish members such as `screenshot`
//! and `key` from user-defined tools with the same names. Results echo that
//! namespace. This module validates only that provider-specific metadata and
//! the result shapes Anthropic interprets; it does not require a prior call to
//! still be declared in the current request's tool list.

use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, ContentBlock, JsonPayload, LlmError, ToolUseId};
use crate::response_json::ResponseJsonScanner;
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
                        blocks.scanner().ok().is_some_and(|scanner| {
                            scanner.array_len("").is_some_and(|len| {
                                (0..len).any(|index| {
                                    scanner.string_at(&format!("/{index}/type"))
                                        == Some("browser_state")
                                })
                            })
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
                            blocks.scanner().ok().is_some_and(|scanner| {
                                scanner.array_len("").is_some_and(|len| {
                                    (0..len).any(|index| {
                                        scanner.string_at(&format!("/{index}/type"))
                                            == Some("browser_state")
                                    })
                                })
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
                        is_error.unwrap_or(false),
                        blocks.as_ref(),
                    )?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_direct_caller(caller: Option<&JsonPayload>) -> Result<(), LlmError> {
    let Some(caller) = caller.filter(|caller| !caller.is_root_null()) else {
        return Ok(());
    };
    if caller.string_at("/type").as_deref() != Some("direct") {
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
    blocks: Option<&JsonPayload>,
) -> Result<(), LlmError> {
    let tab_management = toolset_name == "browser"
        && member.is_some_and(|member| {
            matches!(member, "new_tab" | "list_tabs" | "switch_tab" | "close_tab")
        });
    let mut browser_state = None;
    let mut has_text_representation = blocks.is_none();

    if let Some(blocks) = blocks {
        blocks.validate_root_array("tool result blocks")?;
        let scanner = blocks.scanner()?;
        let length = scanner
            .array_len("")
            .ok_or_else(|| invalid("Anthropic toolset result blocks must be an array"))?;
        for index in 0..length {
            let block_pointer = format!("/{index}");
            let kind =
                required_state_string(&scanner, &format!("{block_pointer}/type"), "type", false)
                    .map_err(|_| {
                        invalid("Anthropic toolset result blocks require a string type")
                    })?;
            let kind = kind.as_str();
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
                if scanner.kind_at(&format!("{block_pointer}/text"))
                    != Some(crate::response_json::FlatJsonKind::String)
                {
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
                browser_state = Some((index, validate_browser_state(&scanner, &block_pointer)?));
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
        if blocks.array_len("") != Some(1) || browser_state.is_none() {
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

fn validate_browser_state(
    scanner: &ResponseJsonScanner,
    pointer: &str,
) -> Result<BrowserStateSummary, LlmError> {
    if scanner.string_at(&format!("{pointer}/type")) != Some("browser_state") {
        return Err(invalid("expected an Anthropic browser_state block"));
    }
    let tabs_pointer = format!("{pointer}/tabs");
    let tabs_len = scanner
        .array_len(&tabs_pointer)
        .ok_or_else(|| invalid("Anthropic browser_state requires a tabs array"))?;
    if tabs_len > 100 {
        return Err(invalid(
            "Anthropic browser_state can contain at most 100 tabs",
        ));
    }

    let mut summary = BrowserStateSummary::default();
    let mut tab_ids = HashSet::new();
    let mut active_count = 0usize;
    for index in 0..tabs_len {
        let tab_pointer = format!("{tabs_pointer}/{index}");
        if scanner.kind_at(&tab_pointer) != Some(crate::response_json::FlatJsonKind::Object) {
            return Err(invalid("Anthropic browser_state tabs must be objects"));
        }
        let tab_id =
            required_state_string(scanner, &format!("{tab_pointer}/tab_id"), "tab_id", true)?;
        required_state_string(scanner, &format!("{tab_pointer}/title"), "title", false)?;
        required_state_string(scanner, &format!("{tab_pointer}/url"), "url", false)?;
        if !tab_ids.insert(tab_id.clone()) {
            return Err(invalid(
                "Anthropic browser_state contains duplicate tab_id values",
            ));
        }
        match scanner.raw_source_at(&format!("{tab_pointer}/active")) {
            None | Some("false") => {}
            Some("true") => {
                active_count += 1;
                summary.active_tab = Some(tab_id);
            }
            Some(_) => {
                return Err(invalid(
                    "Anthropic browser_state tab active fields must be booleans",
                ));
            }
        }
    }
    if (tabs_len > 0 && active_count != 1) || active_count > 1 {
        return Err(invalid(
            "a non-empty Anthropic browser_state tabs array requires exactly one active tab",
        ));
    }

    let changes_pointer = format!("{pointer}/state_changes");
    if scanner.raw_source_at(&changes_pointer).is_some()
        && scanner.kind_at(&changes_pointer) != Some(crate::response_json::FlatJsonKind::Null)
    {
        let changes_len = scanner
            .array_len(&changes_pointer)
            .ok_or_else(|| invalid("Anthropic browser_state state_changes must be an array"))?;
        if changes_len == 0 {
            return Err(invalid(
                "omit Anthropic browser_state state_changes when there is nothing to report",
            ));
        }
        if changes_len > 200 {
            return Err(invalid(
                "Anthropic browser_state can contain at most 200 state changes",
            ));
        }
        let mut download_ids = HashSet::new();
        for index in 0..changes_len {
            let change_pointer = format!("{changes_pointer}/{index}");
            if scanner.kind_at(&change_pointer) != Some(crate::response_json::FlatJsonKind::Object)
            {
                return Err(invalid("Anthropic browser_state changes must be objects"));
            }
            let kind = required_state_string(
                scanner,
                &format!("{change_pointer}/type"),
                "change type",
                false,
            )?;
            match kind.as_str() {
                "tab_opened" => {
                    only_fields_at(scanner, &change_pointer, &["type", "tab_id"])?;
                    let tab_id = required_state_string(
                        scanner,
                        &format!("{change_pointer}/tab_id"),
                        "tab_id",
                        true,
                    )?;
                    if !tab_ids.contains(&tab_id) {
                        return Err(invalid(
                            "Anthropic tab_opened state changes must reference a tab in tabs",
                        ));
                    }
                    summary.tab_opened.push(tab_id);
                }
                "download_started" => {
                    only_fields_at(scanner, &change_pointer, &["type", "download_id", "url"])?;
                    validate_download_fields(
                        scanner,
                        &change_pointer,
                        &mut download_ids,
                        false,
                        false,
                    )?;
                }
                "download_completed" => {
                    only_fields_at(
                        scanner,
                        &change_pointer,
                        &["type", "download_id", "url", "path", "size_bytes"],
                    )?;
                    validate_download_fields(
                        scanner,
                        &change_pointer,
                        &mut download_ids,
                        true,
                        false,
                    )?;
                }
                "download_failed" => {
                    only_fields_at(
                        scanner,
                        &change_pointer,
                        &["type", "download_id", "url", "error"],
                    )?;
                    validate_download_fields(
                        scanner,
                        &change_pointer,
                        &mut download_ids,
                        false,
                        true,
                    )?;
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
    scanner: &ResponseJsonScanner,
    pointer: &str,
    download_ids: &mut HashSet<String>,
    allows_size: bool,
    allows_error: bool,
) -> Result<(), LlmError> {
    let download_id = required_state_string(
        scanner,
        &format!("{pointer}/download_id"),
        "download_id",
        true,
    )?;
    if !download_ids.insert(download_id) {
        return Err(invalid(
            "Anthropic browser_state may report a download_id only once per block",
        ));
    }
    required_state_string(scanner, &format!("{pointer}/url"), "url", false)?;
    if scanner.raw_source_at(&format!("{pointer}/path")).is_some() && allows_size {
        optional_nullable_state_string(scanner, &format!("{pointer}/path"), "path")?;
    }
    let size_pointer = format!("{pointer}/size_bytes");
    if scanner.raw_source_at(&size_pointer).is_some() {
        let valid_size = scanner.kind_at(&size_pointer)
            == Some(crate::response_json::FlatJsonKind::Null)
            || scanner
                .raw_source_at(&size_pointer)
                .and_then(|source| source.parse::<f64>().ok())
                .is_some_and(|size_bytes| {
                    scanner.kind_at(&size_pointer)
                        == Some(crate::response_json::FlatJsonKind::Number)
                        && size_bytes.is_finite()
                        && size_bytes >= 0.0
                        && size_bytes.fract() == 0.0
                });
        if !allows_size || !valid_size {
            return Err(invalid(
                "Anthropic download size_bytes must be null or a non-negative integer on download_completed",
            ));
        }
    }
    if scanner.raw_source_at(&format!("{pointer}/error")).is_some() && allows_error {
        optional_nullable_state_string(scanner, &format!("{pointer}/error"), "error")?;
    }
    Ok(())
}

fn required_state_string(
    scanner: &ResponseJsonScanner,
    pointer: &str,
    field: &str,
    non_empty: bool,
) -> Result<String, LlmError> {
    let units = scanner
        .string_units_at(pointer)
        .ok_or_else(|| invalid(format!("Anthropic browser_state {field} must be a string")))?;
    let value = String::from_utf16(&units).map_err(|_| {
        invalid(format!(
            "Anthropic browser_state {field} contains unpaired UTF-16 units"
        ))
    })?;
    validate_state_string(&value, field, non_empty)?;
    Ok(value)
}

fn optional_nullable_state_string(
    scanner: &ResponseJsonScanner,
    pointer: &str,
    field: &str,
) -> Result<(), LlmError> {
    if scanner.raw_source_at(pointer).is_some()
        && scanner.kind_at(pointer) != Some(crate::response_json::FlatJsonKind::Null)
    {
        required_state_string(scanner, pointer, field, false)?;
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

fn only_fields_at(
    scanner: &ResponseJsonScanner,
    pointer: &str,
    allowed: &[&str],
) -> Result<(), LlmError> {
    let fields = scanner
        .object_entries_at(pointer)
        .ok_or_else(|| invalid("Anthropic browser_state changes must be objects"))?;
    if fields.iter().any(|(key, units)| {
        !allowed
            .iter()
            .any(|allowed| key == allowed && units == &allowed.encode_utf16().collect::<Vec<_>>())
    }) {
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
