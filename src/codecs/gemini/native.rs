//! Validation for the small set of Gemini `Part` variants that this codec
//! preserves as provider-native content.

use serde_json::Value;

pub(super) fn is_media_part(value: &Value) -> bool {
    value.get("inlineData").is_some() || value.get("fileData").is_some()
}

/// Whether `value` is a replayable Gemini-native Part. We keep its original
/// JSON shape on the wire after validating its discriminated payload; no
/// candidate-level metadata is accepted here.
pub(super) fn is_replayable_native_part(value: &Value) -> bool {
    let Some(part) = value.as_object() else {
        return false;
    };
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
        .filter(|key| part.contains_key(**key))
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
    if let Some(signature) = part.get("thoughtSignature") {
        if !signature.is_string() {
            return false;
        }
    }
    if let Some(thought) = part.get("thought") {
        if !thought.is_boolean() {
            return false;
        }
    }

    if let Some(code) = part.get("executableCode") {
        let Some(code) = code.as_object() else {
            return false;
        };
        return code.get("language").is_some_and(Value::is_string)
            && code.get("code").is_some_and(Value::is_string)
            && code.get("id").is_none_or(Value::is_string);
    }

    if let Some(blob) = part.get("inlineData").and_then(Value::as_object) {
        return blob
            .get("mimeType")
            .and_then(Value::as_str)
            .is_some_and(|mime| !mime.is_empty())
            && blob.get("data").is_some_and(Value::is_string);
    }

    if let Some(result) = part.get("codeExecutionResult") {
        let Some(result) = result.as_object() else {
            return false;
        };
        return result.get("outcome").is_some_and(Value::is_string)
            && result.get("output").is_none_or(Value::is_string)
            && result.get("id").is_none_or(Value::is_string);
    }

    if let Some(call) = part.get("toolCall") {
        let Some(call) = call.as_object() else {
            return false;
        };
        return call
            .get("toolType")
            .and_then(Value::as_str)
            .is_some_and(|tool_type| !tool_type.is_empty())
            && call.get("id").is_none_or(Value::is_string)
            && call.get("toolName").is_none_or(Value::is_string)
            && call.get("args").is_none_or(Value::is_object);
    }

    let Some(response) = part.get("toolResponse").and_then(Value::as_object) else {
        return false;
    };
    response
        .get("toolType")
        .and_then(Value::as_str)
        .is_some_and(|tool_type| !tool_type.is_empty())
        && response.get("id").is_none_or(Value::is_string)
        && response.get("response").is_none_or(Value::is_object)
}
