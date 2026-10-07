//! Typed facts from native Messages observations for host presentation.
//! The SDK owns wire interpretation; hosts choose their durable history format.

use crate::protocol::ProtocolFamily;
use serde_json::Value;

/// Text fields that can be represented canonically, or the complete opaque block.
#[derive(Debug, Clone, Copy)]
pub enum TextBlock<'a> {
    Known {
        text: &'a str,
        /// Absent, explicit null, and a JSON value remain distinct.
        citations: Option<Option<&'a Value>>,
    },
    Opaque(&'a Value),
}

pub fn text_block(value: &Value) -> Option<TextBlock<'_>> {
    if value.get("type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    if !crate::codecs::anthropic::decode::has_only_known_text_fields(value) {
        return Some(TextBlock::Opaque(value));
    }
    Some(TextBlock::Known {
        text: value.get("text")?.as_str()?,
        citations: value
            .get("citations")
            .map(|value| (!value.is_null()).then_some(value)),
    })
}

#[derive(Debug, Clone, Copy)]
pub struct StopDelta<'a> {
    pub reason: &'a str,
    pub details: Option<&'a Value>,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderEvent<'a> {
    pub block_index: Option<u64>,
    pub text_start: Option<TextBlock<'a>>,
    pub stop_delta: Option<StopDelta<'a>>,
}

/// Interpret only Messages observations, without inferring absent stop fields.
pub fn provider_event(protocol: ProtocolFamily, payload: &Value) -> Option<ProviderEvent<'_>> {
    if protocol != ProtocolFamily::AnthropicMessages {
        return None;
    }
    Some(ProviderEvent {
        block_index: payload.get("index").and_then(Value::as_u64),
        text_start: (payload.get("type").and_then(Value::as_str) == Some("content_block_start"))
            .then(|| payload.get("content_block").and_then(text_block))
            .flatten(),
        stop_delta: if payload.get("type").and_then(Value::as_str) == Some("message_delta") {
            payload.get("delta").and_then(|delta| {
                Some(StopDelta {
                    reason: delta.get("stop_reason")?.as_str()?,
                    details: delta.get("stop_details").filter(|value| !value.is_null()),
                })
            })
        } else {
            None
        },
    })
}

#[derive(Debug, Clone, Copy)]
pub enum NativeDelta<'a> {
    Citation(&'a Value),
    ConnectorText(&'a str),
}

pub fn native_delta(protocol: ProtocolFamily, delta: &Value) -> Option<NativeDelta<'_>> {
    if protocol != ProtocolFamily::AnthropicMessages {
        return None;
    }
    match delta.get("type").and_then(Value::as_str) {
        Some("citations_delta") => Some(NativeDelta::Citation(&delta["citation"])),
        Some("connector_text_delta") => Some(NativeDelta::ConnectorText(
            delta
                .get("connector_text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_start_preserves_citation_presence_and_opaque_fields() {
        for citations in [
            None,
            Some(Value::Null),
            Some(json!([])),
            Some(json!([{"url":"x"}])),
        ] {
            let mut block = json!({"type":"text","text":"seed"});
            if let Some(value) = &citations {
                block["citations"] = value.clone();
            }
            let payload = json!({"type":"content_block_start","index":7,"content_block":block});
            let event = provider_event(ProtocolFamily::AnthropicMessages, &payload).unwrap();
            assert_eq!(event.block_index, Some(7));
            let Some(TextBlock::Known {
                text,
                citations: actual,
            }) = event.text_start
            else {
                panic!("expected typed text start");
            };
            assert_eq!(text, "seed");
            assert_eq!(
                actual.map(|value| value.cloned()),
                citations.map(|value| (!value.is_null()).then_some(value))
            );
        }
        let block = json!({"type":"text","text":"seed","extra":{"keep":true}});
        assert!(matches!(text_block(&block), Some(TextBlock::Opaque(value)) if value == &block));
        assert!(text_block(&json!({"type":"text"})).is_none());
        assert!(text_block(&json!({"type":"thinking","text":"seed"})).is_none());
    }

    #[test]
    fn stop_facts_do_not_cross_protocols_or_infer_a_reason() {
        let payload = json!({"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","reason":"policy"}}});
        let event = provider_event(ProtocolFamily::AnthropicMessages, &payload).unwrap();
        let stop = event.stop_delta.unwrap();
        assert_eq!(stop.reason, "refusal");
        assert_eq!(stop.details, payload.pointer("/delta/stop_details"));
        assert!(provider_event(ProtocolFamily::OpenAiChat, &payload).is_none());
        for delta in [
            json!({}),
            json!({"stop_reason":null}),
            json!({"stop_details":{"reason":"policy"}}),
        ] {
            let payload = json!({"type":"message_delta","delta":delta});
            assert!(provider_event(ProtocolFamily::AnthropicMessages, &payload)
                .unwrap()
                .stop_delta
                .is_none());
        }
        let payload =
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_details":null}});
        assert!(provider_event(ProtocolFamily::AnthropicMessages, &payload)
            .unwrap()
            .stop_delta
            .unwrap()
            .details
            .is_none());
    }

    #[test]
    fn native_deltas_preserve_values_and_are_protocol_scoped() {
        let citation = json!({"type":"citations_delta","citation":{"url":"x","extra":null}});
        assert!(
            matches!(native_delta(ProtocolFamily::AnthropicMessages, &citation), Some(NativeDelta::Citation(value)) if value == &citation["citation"])
        );
        let connector = json!({"type":"connector_text_delta","connector_text":"partial"});
        assert!(matches!(
            native_delta(ProtocolFamily::AnthropicMessages, &connector),
            Some(NativeDelta::ConnectorText("partial"))
        ));
        assert!(native_delta(ProtocolFamily::OpenAiChat, &citation).is_none());
        assert!(native_delta(
            ProtocolFamily::AnthropicMessages,
            &json!({"type":"text_delta","text":"x"})
        )
        .is_none());
    }
}
