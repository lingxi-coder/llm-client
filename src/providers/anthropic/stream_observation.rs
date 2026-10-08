//! Typed facts from native Messages observations for host presentation.
//! The SDK owns wire interpretation; hosts choose their durable history format.

use crate::protocol::{JsonPayload, ProtocolFamily};

/// Text fields that can be represented canonically, or the complete opaque block.
#[derive(Debug, Clone)]
pub enum TextBlock {
    Known {
        text: String,
        /// Exact JavaScript UTF-16 units of the model-visible text.
        utf16_code_units: Vec<u16>,
        /// Absent, explicit null, and a JSON value remain distinct.
        citations: Option<Option<JsonPayload>>,
    },
    Opaque(JsonPayload),
}

pub fn text_block(value: &JsonPayload) -> Option<TextBlock> {
    if !value.string_is("/type", "text") {
        return None;
    }
    if !value.has_only_fields_at("", &["type", "text", "citations"]) {
        return Some(TextBlock::Opaque(value.clone()));
    }
    let citations = match value.raw_source_at("/citations") {
        None => None,
        Some(_)
            if value.kind_at("/citations") == Some(crate::response_json::FlatJsonKind::Null) =>
        {
            Some(None)
        }
        Some(source) => Some(Some(JsonPayload::parse(&source).ok()?)),
    };
    Some(TextBlock::Known {
        text: value.string_at("/text")?,
        utf16_code_units: value.string_units_at("/text")?,
        citations,
    })
}

#[derive(Debug, Clone)]
pub struct StopDelta {
    pub reason: String,
    pub details: Option<JsonPayload>,
}

#[derive(Debug, Clone)]
pub struct ProviderEvent {
    pub block_index: Option<u64>,
    pub text_start: Option<TextBlock>,
    pub stop_delta: Option<StopDelta>,
}

/// Interpret only Messages observations, without inferring absent stop fields.
pub fn provider_event(protocol: ProtocolFamily, payload: &JsonPayload) -> Option<ProviderEvent> {
    if protocol != ProtocolFamily::AnthropicMessages {
        return None;
    }
    Some(ProviderEvent {
        block_index: payload.unsigned_integer_at("/index"),
        text_start: payload
            .string_is("/type", "content_block_start")
            .then(|| {
                payload
                    .payload_at("/content_block")
                    .and_then(|value| text_block(&value))
            })
            .flatten(),
        stop_delta: if payload.string_is("/type", "message_delta") {
            payload.payload_at("/delta").and_then(|delta| {
                Some(StopDelta {
                    reason: delta.string_at("/stop_reason")?,
                    details: delta.payload_at("/stop_details").filter(|value| {
                        value.kind_at("") != Some(crate::response_json::FlatJsonKind::Null)
                    }),
                })
            })
        } else {
            None
        },
    })
}

#[derive(Debug, Clone)]
pub enum NativeDelta {
    Citation(JsonPayload),
    ConnectorText {
        text: String,
        utf16_code_units: Vec<u16>,
    },
}

pub fn native_delta(protocol: ProtocolFamily, delta: &JsonPayload) -> Option<NativeDelta> {
    if protocol != ProtocolFamily::AnthropicMessages {
        return None;
    }
    if delta.string_is("/type", "citations_delta") {
        return Some(NativeDelta::Citation(
            delta
                .payload_at("/citation")
                .unwrap_or_else(|| JsonPayload::parse("null").expect("null is valid JSON")),
        ));
    }
    if delta.string_is("/type", "connector_text_delta") {
        return Some(NativeDelta::ConnectorText {
            text: delta.string_at("/connector_text").unwrap_or_default(),
            utf16_code_units: delta.string_units_at("/connector_text").unwrap_or_default(),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn payload(value: serde_json::Value) -> JsonPayload {
        JsonPayload::from_generated_value(&value).unwrap()
    }

    fn payload_value(payload: &JsonPayload) -> serde_json::Value {
        serde_json::from_str(payload.as_json_source()).unwrap()
    }

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
            let event_payload =
                payload(json!({"type":"content_block_start","index":7,"content_block":block}));
            let event = provider_event(ProtocolFamily::AnthropicMessages, &event_payload).unwrap();
            assert_eq!(event.block_index, Some(7));
            let Some(TextBlock::Known {
                text,
                citations: actual,
                ..
            }) = event.text_start
            else {
                panic!("expected typed text start");
            };
            assert_eq!(text, "seed");
            assert_eq!(
                actual.map(|value| value.map(|payload| payload_value(&payload))),
                citations.map(|value| (!value.is_null()).then_some(value))
            );
        }
        let block = json!({"type":"text","text":"seed","extra":{"keep":true}});
        let block_payload = payload(block);
        assert!(
            matches!(text_block(&block_payload), Some(TextBlock::Opaque(value)) if value.as_json_source() == block_payload.as_json_source())
        );
        assert!(text_block(&payload(json!({"type":"text"}))).is_none());
        assert!(text_block(&payload(json!({"type":"thinking","text":"seed"}))).is_none());

        let exact = JsonPayload::parse(r#"{"type":"text","text":"\ud800"}"#).unwrap();
        let Some(TextBlock::Known {
            text,
            utf16_code_units,
            ..
        }) = text_block(&exact)
        else {
            panic!("expected a typed text block");
        };
        assert_eq!(text, "\u{fffd}");
        assert_eq!(utf16_code_units, vec![0xd800]);
    }

    #[test]
    fn stop_facts_do_not_cross_protocols_or_infer_a_reason() {
        let event_payload = payload(
            json!({"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","reason":"policy"}}}),
        );
        let event = provider_event(ProtocolFamily::AnthropicMessages, &event_payload).unwrap();
        let stop = event.stop_delta.unwrap();
        assert_eq!(stop.reason, "refusal");
        assert_eq!(
            stop.details,
            Some(payload(json!({"type":"refusal","reason":"policy"})))
        );
        assert!(provider_event(ProtocolFamily::OpenAiChat, &event_payload).is_none());
        for delta in [
            json!({}),
            json!({"stop_reason":null}),
            json!({"stop_details":{"reason":"policy"}}),
        ] {
            let event_payload = payload(json!({"type":"message_delta","delta":delta}));
            assert!(
                provider_event(ProtocolFamily::AnthropicMessages, &event_payload)
                    .unwrap()
                    .stop_delta
                    .is_none()
            );
        }
        let event_payload = payload(
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_details":null}}),
        );
        assert!(
            provider_event(ProtocolFamily::AnthropicMessages, &event_payload)
                .unwrap()
                .stop_delta
                .unwrap()
                .details
                .is_none()
        );
    }

    #[test]
    fn native_deltas_preserve_values_and_are_protocol_scoped() {
        let citation = json!({"type":"citations_delta","citation":{"url":"x","extra":null}});
        let citation_payload = payload(citation.clone());
        assert!(
            matches!(native_delta(ProtocolFamily::AnthropicMessages, &citation_payload), Some(NativeDelta::Citation(value)) if value == payload(citation["citation"].clone()))
        );
        let connector = payload(json!({"type":"connector_text_delta","connector_text":"partial"}));
        assert!(matches!(
            native_delta(ProtocolFamily::AnthropicMessages, &connector),
            Some(NativeDelta::ConnectorText { text, utf16_code_units })
                if text == "partial" && utf16_code_units == "partial".encode_utf16().collect::<Vec<_>>()
        ));
        assert!(native_delta(ProtocolFamily::OpenAiChat, &citation_payload).is_none());
        assert!(native_delta(
            ProtocolFamily::AnthropicMessages,
            &payload(json!({"type":"text_delta","text":"x"}))
        )
        .is_none());
    }
}
