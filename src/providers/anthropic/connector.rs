//! Incremental projection of legacy connector observations into native content.
use crate::protocol::{ContentBlock, JsonPayload, ProtocolFamily};
use std::collections::BTreeMap;

/// Assemble connector text observations while preserving provider metadata.
#[derive(Debug, Default)]
pub struct ConnectorTextAccumulator {
    blocks: BTreeMap<u64, JsonPayload>,
}
impl ConnectorTextAccumulator {
    /// Return a complete native block only after the provider closes it.
    pub fn push(&mut self, payload: &JsonPayload) -> Option<(u64, ContentBlock)> {
        let index = payload.raw_source_at("/index")?.parse().ok()?;
        match payload.string_at("/type").as_deref() {
            Some("content_block_start")
                if payload.string_is("/content_block/type", "connector_text") =>
            {
                let source = payload.raw_source_at("/content_block")?;
                self.blocks.insert(index, JsonPayload::parse(&source).ok()?);
            }
            Some("content_block_delta") => {
                if let Some(block) = self.blocks.get_mut(&index) {
                    let delta = if payload.string_is("/delta/type", "connector_text_delta") {
                        payload.string_units_at("/delta/connector_text")
                    } else {
                        None
                    };
                    if let Some(delta) = delta {
                        let mut units = block.string_units_at("/connector_text")?;
                        units.extend(delta);
                        let encoded = JsonPayload::string_json_source(&units);
                        match block.with_canonical_fields(
                            crate::exact_json::JsonEncoding::Serde,
                            &[("connector_text", &encoded)],
                        ) {
                            Ok(updated) => *block = updated,
                            Err(_) => {
                                self.blocks.remove(&index);
                            }
                        }
                    } else {
                        self.blocks.remove(&index);
                    }
                }
            }
            Some("content_block_stop") => {
                return self.blocks.remove(&index).map(|value| {
                    (
                        index,
                        ContentBlock::ProviderContent {
                            protocol: ProtocolFamily::AnthropicMessages,
                            value,
                        },
                    )
                });
            }
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: serde_json::Value) -> JsonPayload {
        JsonPayload::parse(&value.to_string()).unwrap()
    }

    #[test]
    fn connector_is_emitted_once_at_close_with_native_metadata() {
        let mut parser = ConnectorTextAccumulator::default();
        assert!(parser.push(&payload(json!({"type":"content_block_start","index":2,"content_block":{"type":"connector_text","connector_text":"a","source":"remote"}}))).is_none());
        assert!(parser.push(&payload(json!({"type":"content_block_delta","index":2,"delta":{"type":"connector_text_delta","connector_text":"b"}}))).is_none());
        let (index, ContentBlock::ProviderContent { value, .. }) = parser
            .push(&payload(json!({"type":"content_block_stop","index":2})))
            .unwrap()
        else {
            panic!("native block expected")
        };
        assert_eq!(index, 2);
        assert_eq!(value.string_at("/connector_text").as_deref(), Some("ab"));
        assert_eq!(value.string_at("/source").as_deref(), Some("remote"));
        assert!(parser
            .push(&payload(json!({"type":"content_block_stop","index":2})))
            .is_none());
    }
}
