//! Message-level Chat reasoning metadata. Keep opaque details in their wire
//! order; display text cannot reconstruct encrypted or signed replay data.

use crate::protocol::{JsonPayload, LlmError, ProtocolFamily};
use serde_json::{Map, Value};

const ENVELOPE_TYPE: &str = "chat_reasoning";

fn fields(message: &Value) -> Result<Map<String, Value>, LlmError> {
    let mut fields = Map::new();
    for name in ["reasoning", "reasoning_details"] {
        if let Some(value) = message.get(name).filter(|value| !value.is_null()) {
            if (name == "reasoning" && !value.is_string())
                || (name == "reasoning_details" && !value.is_array())
            {
                return Err(LlmError::ProviderInternal {
                    message: format!("Chat response has invalid {name}"),
                });
            }
            fields.insert(name.to_owned(), value.clone());
        }
    }
    Ok(fields)
}

fn envelope(mut fields: Map<String, Value>) -> Option<Value> {
    if fields.is_empty() {
        return None;
    }
    fields.insert("type".into(), Value::String(ENVELOPE_TYPE.into()));
    Some(Value::Object(fields))
}

pub(super) fn from_message(message: &Value) -> Result<Option<Value>, LlmError> {
    Ok(envelope(fields(message)?))
}

pub(super) fn display_text(message: &Value) -> String {
    if let Some(text) = ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|name| {
            message
                .get(name)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
    {
        return text.to_owned();
    }
    message
        .get("reasoning_details")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|detail| {
            let field = match detail.get("type").and_then(Value::as_str) {
                Some("reasoning.text") => "text",
                Some("reasoning.summary") => "summary",
                _ => return None,
            };
            detail.get(field).and_then(Value::as_str)
        })
        .collect()
}

/// Only recognized message metadata may escape the content-block envelope.
/// Never allow arbitrary native fields to replace role, content or tool calls.
pub(super) fn replay_fields(
    protocol: ProtocolFamily,
    value: &JsonPayload,
) -> Result<Map<String, Value>, LlmError> {
    if protocol != ProtocolFamily::OpenAiChat
        || !value.string_is("/type", ENVELOPE_TYPE)
        || !value.has_only_fields_at("", &["type", "reasoning", "reasoning_details"])
    {
        return Err(LlmError::UnsupportedCapability {
            message: "native content is not Chat reasoning metadata".into(),
        });
    }
    let invalid = || {
        LlmError::InvalidRequest {
        message: "Chat reasoning metadata must contain a string reasoning or array reasoning_details and no other fields".into(),
    }
    };
    let mut result = Map::new();
    if value.raw_source_at("/reasoning").is_some() {
        let units = value.string_units_at("/reasoning").ok_or_else(invalid)?;
        let text = String::from_utf16(&units).map_err(|_| invalid())?;
        result.insert("reasoning".into(), Value::String(text));
    }
    if value.raw_source_at("/reasoning_details").is_some() {
        let source = value
            .raw_source_at("/reasoning_details")
            .ok_or_else(invalid)?;
        let details: Value = serde_json::from_str(&source).map_err(|_| invalid())?;
        if !details.is_array() {
            return Err(invalid());
        }
        result.insert("reasoning_details".into(), details);
    }
    if result.is_empty() {
        return Err(invalid());
    }
    Ok(result)
}

#[derive(Debug, Default)]
pub(super) struct ReasoningStream {
    fields: Map<String, Value>,
}

impl ReasoningStream {
    pub(super) fn observe(&mut self, delta: &Value) -> Result<bool, LlmError> {
        let incoming = fields(delta)?;
        let present = !incoming.is_empty();
        for (name, fragment) in incoming {
            match self.fields.get_mut(&name) {
                Some(Value::String(text)) => {
                    text.push_str(fragment.as_str().expect("validated text"))
                }
                Some(Value::Array(details)) => details.extend(
                    fragment
                        .as_array()
                        .expect("validated details")
                        .iter()
                        .cloned(),
                ),
                None => {
                    self.fields.insert(name, fragment);
                }
                _ => unreachable!("reasoning fields are validated before insertion"),
            }
        }
        Ok(present)
    }

    pub(super) fn take(&mut self) -> Option<Value> {
        envelope(std::mem::take(&mut self.fields))
    }
}
