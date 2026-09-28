//! Incremental projection of legacy connector observations into native content.
use crate::protocol::{ContentBlock, ProtocolFamily};
use serde_json::Value;
use std::collections::BTreeMap;

/// Assemble connector text observations while preserving provider metadata.
#[derive(Debug, Default)]
pub struct ConnectorTextAccumulator {
    blocks: BTreeMap<u64, Value>,
}
impl ConnectorTextAccumulator {
    /// Return a complete native block only after the provider closes it.
    pub fn push(&mut self, payload: &Value) -> Option<(u64, ContentBlock)> {
        let index = payload["index"].as_u64()?;
        match payload["type"].as_str() {
            Some("content_block_start") if payload["content_block"]["type"] == "connector_text" => {
                self.blocks.insert(index, payload["content_block"].clone());
            }
            Some("content_block_delta") => {
                if let Some(block) = self.blocks.get_mut(&index) {
                    let delta = &payload["delta"];
                    if delta["type"] == "connector_text_delta"
                        && delta["connector_text"].is_string()
                    {
                        block["connector_text"] = Value::String(format!(
                            "{}{}",
                            block["connector_text"].as_str().unwrap_or_default(),
                            delta["connector_text"].as_str().unwrap()
                        ));
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
    #[test]
    fn connector_is_emitted_once_at_close_with_native_metadata() {
        let mut parser = ConnectorTextAccumulator::default();
        assert!(parser.push(&json!({"type":"content_block_start","index":2,"content_block":{"type":"connector_text","connector_text":"a","source":"remote"}})).is_none());
        assert!(parser.push(&json!({"type":"content_block_delta","index":2,"delta":{"type":"connector_text_delta","connector_text":"b"}})).is_none());
        let (index, ContentBlock::ProviderContent { value, .. }) = parser
            .push(&json!({"type":"content_block_stop","index":2}))
            .unwrap()
        else {
            panic!("native block expected")
        };
        assert_eq!(index, 2);
        assert_eq!(value["connector_text"], "ab");
        assert_eq!(value["source"], "remote");
        assert!(parser
            .push(&json!({"type":"content_block_stop","index":2}))
            .is_none());
    }
}
