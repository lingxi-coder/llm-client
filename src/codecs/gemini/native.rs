//! Validation for the small set of Gemini `Part` variants that this codec
//! preserves as provider-native content.

use crate::{protocol::JsonPayload, response_json::FlatJsonKind};

pub(crate) fn is_media_part(value: &JsonPayload) -> bool {
    value.raw_source_at("/inlineData").is_some() || value.raw_source_at("/fileData").is_some()
}

/// Whether this is a replayable Gemini-native Part. Reads typed discriminants
/// from the flat source scanner and never constructs a recursive Value tree.
pub(super) fn is_replayable_native_part(value: &JsonPayload) -> bool {
    if value.kind_at("") != Some(FlatJsonKind::Object) {
        return false;
    }
    let data_keys = [
        "text",
        "inlineData",
        "functionCall",
        "functionResponse",
        "fileData",
        "executableCode",
        "codeExecutionResult",
        "toolCall",
        "toolResponse",
    ];
    let present = data_keys
        .iter()
        .filter(|key| value.raw_source_at(&format!("/{key}")).is_some())
        .copied()
        .collect::<Vec<_>>();
    if present.len() != 1
        || !matches!(
            present[0],
            "executableCode" | "codeExecutionResult" | "inlineData" | "toolCall" | "toolResponse"
        )
    {
        return false;
    }
    if value.raw_source_at("/thoughtSignature").is_some()
        && value.kind_at("/thoughtSignature") != Some(FlatJsonKind::String)
    {
        return false;
    }
    if value.raw_source_at("/thought").is_some()
        && value.kind_at("/thought") != Some(FlatJsonKind::Boolean)
    {
        return false;
    }

    if value.raw_source_at("/executableCode").is_some() {
        return value.kind_at("/executableCode") == Some(FlatJsonKind::Object)
            && value.kind_at("/executableCode/language") == Some(FlatJsonKind::String)
            && value.kind_at("/executableCode/code") == Some(FlatJsonKind::String)
            && optional_kind(value, "/executableCode/id", FlatJsonKind::String);
    }

    if value.raw_source_at("/inlineData").is_some() {
        return value.kind_at("/inlineData") == Some(FlatJsonKind::Object)
            && value
                .string_at("/inlineData/mimeType")
                .is_some_and(|mime| !mime.is_empty())
            && value.kind_at("/inlineData/data") == Some(FlatJsonKind::String);
    }

    if value.raw_source_at("/codeExecutionResult").is_some() {
        return value.kind_at("/codeExecutionResult") == Some(FlatJsonKind::Object)
            && value.kind_at("/codeExecutionResult/outcome") == Some(FlatJsonKind::String)
            && optional_kind(value, "/codeExecutionResult/output", FlatJsonKind::String)
            && optional_kind(value, "/codeExecutionResult/id", FlatJsonKind::String);
    }

    if value.raw_source_at("/toolCall").is_some() {
        return value.kind_at("/toolCall") == Some(FlatJsonKind::Object)
            && value
                .string_at("/toolCall/toolType")
                .is_some_and(|tool_type| !tool_type.is_empty())
            && optional_kind(value, "/toolCall/id", FlatJsonKind::String)
            && optional_kind(value, "/toolCall/toolName", FlatJsonKind::String)
            && optional_kind(value, "/toolCall/args", FlatJsonKind::Object);
    }

    value.kind_at("/toolResponse") == Some(FlatJsonKind::Object)
        && value
            .string_at("/toolResponse/toolType")
            .is_some_and(|tool_type| !tool_type.is_empty())
        && optional_kind(value, "/toolResponse/id", FlatJsonKind::String)
        && optional_kind(value, "/toolResponse/response", FlatJsonKind::Object)
}

fn optional_kind(value: &JsonPayload, pointer: &str, kind: FlatJsonKind) -> bool {
    value.raw_source_at(pointer).is_none() || value.kind_at(pointer) == Some(kind)
}
