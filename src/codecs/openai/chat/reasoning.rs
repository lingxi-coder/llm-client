//! Message-level Chat reasoning metadata. Keep opaque details in their wire
//! order; display text cannot reconstruct encrypted or signed replay data.

use crate::protocol::{LlmError, ProtocolFamily};
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
    value: &Value,
) -> Result<Map<String, Value>, LlmError> {
    if protocol != ProtocolFamily::OpenAiChat
        || value.get("type").and_then(Value::as_str) != Some(ENVELOPE_TYPE)
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
    let object = value.as_object().ok_or_else(invalid)?;
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "type" | "reasoning" | "reasoning_details"))
    {
        return Err(invalid());
    }
    if object
        .get("reasoning")
        .is_some_and(|value| !value.is_string())
        || object
            .get("reasoning_details")
            .is_some_and(|value| !value.is_array())
    {
        return Err(invalid());
    }
    let result = fields(value).map_err(|_| invalid())?;
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
