//! Explicit coordinates for content-preserving request transformations.
//!
//! This maps metadata, not text values. A caller registers every source block,
//! including deliberate removals. Expanded blocks choose their exact string
//! destinations explicitly; cache endpoints use the final output block.
use crate::protocol::{CachePosition, LlmError};
use std::collections::BTreeMap;

pub type BlockPosition = (usize, usize);

#[derive(Debug, Clone, Default)]
struct BlockMapping {
    outputs: Vec<BlockPosition>,
    fields: BTreeMap<String, Option<(BlockPosition, String)>>,
}

#[derive(Debug, Clone, Default)]
pub struct MessageProjectionMap {
    blocks: BTreeMap<BlockPosition, BlockMapping>,
    messages: BTreeMap<usize, Vec<usize>>,
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.to_owned(),
    }
}

impl MessageProjectionMap {
    /// Explicit message destinations also describe empty/native-options rows.
    pub fn insert_message(&mut self, source: usize, outputs: Vec<usize>) -> Result<(), LlmError> {
        if self.messages.contains_key(&source) {
            return Err(invalid("source message already has a projection"));
        }
        self.messages.insert(source, outputs);
        Ok(())
    }
    /// Empty outputs represent an intentional deletion, not an unknown source.
    pub fn insert(
        &mut self,
        source: BlockPosition,
        outputs: Vec<BlockPosition>,
    ) -> Result<(), LlmError> {
        if self.blocks.contains_key(&source) {
            return Err(invalid("source block already has a projection"));
        }
        self.blocks.insert(
            source,
            BlockMapping {
                outputs,
                fields: BTreeMap::new(),
            },
        );
        Ok(())
    }

    /// Field strings are relative JSON-pointer suffixes, without the first '/'.
    /// None explicitly consumes an exact string with its removed/replaced field.
    pub fn map_field(
        &mut self,
        source: BlockPosition,
        field: &str,
        target: Option<(BlockPosition, String)>,
    ) -> Result<(), LlmError> {
        if field.is_empty() {
            return Err(invalid("source field must not be empty"));
        }
        let mapping = self
            .blocks
            .get_mut(&source)
            .ok_or_else(|| invalid("unknown source block"))?;
        if let Some((position, field)) = &target {
            if !mapping.outputs.contains(position) || field.is_empty() {
                return Err(invalid(
                    "exact string target is outside its block projection",
                ));
            }
        }
        if mapping.fields.contains_key(field) {
            return Err(invalid("source field already has a projection"));
        }
        mapping.fields.insert(field.to_owned(), target);
        Ok(())
    }

    pub fn outputs(&self, source: BlockPosition) -> Result<&[BlockPosition], LlmError> {
        self.blocks
            .get(&source)
            .map(|mapping| mapping.outputs.as_slice())
            .ok_or_else(|| invalid("unknown source block"))
    }

    pub fn cache_position(&self, source: CachePosition) -> Result<Option<CachePosition>, LlmError> {
        match source {
            CachePosition::Message { index, block } => Ok(self
                .outputs((index, block))?
                .last()
                .map(|&(index, block)| CachePosition::Message { index, block })),
            other => Ok(Some(other)),
        }
    }

    /// A split source message has no unique whole-message destination.
    pub fn message_index(&self, source: usize) -> Option<usize> {
        if let Some(outputs) = self.messages.get(&source) {
            return (outputs.len() == 1).then(|| outputs[0]);
        }
        let mut target = None;
        for (_, mapping) in self.blocks.range((source, 0)..=(source, usize::MAX)) {
            for &(message, _) in &mapping.outputs {
                if target.is_some_and(|previous| previous != message) {
                    return None;
                }
                target = Some(message);
            }
        }
        target
    }

    pub fn string_overrides(
        &self,
        source: &BTreeMap<String, Vec<u16>>,
    ) -> Result<BTreeMap<String, Vec<u16>>, LlmError> {
        let mut output = BTreeMap::new();
        for (pointer, units) in source {
            let Some(path) = pointer.strip_prefix("/messages/") else {
                output.insert(pointer.clone(), units.clone());
                continue;
            };
            let (message, path) = path.split_once("/content/").ok_or_else(|| {
                invalid("exact message pointer does not identify a content block")
            })?;
            let (block, field) = path
                .split_once('/')
                .filter(|(_, field)| !field.is_empty())
                .ok_or_else(|| invalid("exact message pointer does not identify a field"))?;
            let index = |text: &str| {
                text.parse::<usize>()
                    .ok()
                    .filter(|index| index.to_string() == text)
                    .ok_or_else(|| invalid("exact message pointer has a noncanonical index"))
            };
            let mapping = self
                .blocks
                .get(&(index(message)?, index(block)?))
                .ok_or_else(|| invalid("exact message pointer has no block projection"))?;
            let destination = match mapping.fields.get(field) {
                Some(target) => target.clone(),
                None if mapping.outputs.is_empty() => None,
                None if mapping.outputs.len() == 1 => Some((mapping.outputs[0], field.to_owned())),
                None => {
                    return Err(invalid(
                        "expanded exact string needs an explicit field destination",
                    ))
                }
            };
            if let Some(((message, block), field)) = destination {
                let pointer = format!("/messages/{message}/content/{block}/{field}");
                if let Some(existing) = output.insert(pointer, units.clone()) {
                    if existing != *units {
                        return Err(invalid("conflicting exact strings map to one field"));
                    }
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_remaps_text_separately_from_cache_endpoint_and_source_identity() {
        let mut map = MessageProjectionMap::default();
        map.insert((0, 0), vec![(0, 0), (0, 1)]).unwrap();
        map.map_field((0, 0), "content", Some(((0, 1), "text".into())))
            .unwrap();
        map.insert((0, 1), vec![(0, 2)]).unwrap();
        let exact = BTreeMap::from([
            ("/messages/0/content/0/content".into(), vec![0xd800]),
            ("/messages/0/content/1/text".into(), vec![0xd801]),
        ]);
        assert_eq!(
            map.string_overrides(&exact).unwrap(),
            BTreeMap::from([
                ("/messages/0/content/1/text".into(), vec![0xd800]),
                ("/messages/0/content/2/text".into(), vec![0xd801]),
            ])
        );
        assert_eq!(
            map.cache_position(CachePosition::Message { index: 0, block: 0 })
                .unwrap(),
            Some(CachePosition::Message { index: 0, block: 1 })
        );
        assert_eq!(map.message_index(0), Some(0));
    }

    #[test]
    fn split_and_removed_blocks_do_not_leave_stale_coordinates() {
        let mut map = MessageProjectionMap::default();
        map.insert((0, 0), vec![(0, 0)]).unwrap();
        map.insert((0, 1), vec![(1, 0)]).unwrap();
        map.insert((1, 0), vec![]).unwrap();
        map.insert((2, 0), vec![(2, 0)]).unwrap();
        assert_eq!(map.message_index(0), None);
        assert_eq!(map.message_index(1), None);
        assert_eq!(map.message_index(2), Some(2));
        let exact = BTreeMap::from([("/messages/1/content/0/text".into(), vec![0xd800])]);
        assert!(map.string_overrides(&exact).unwrap().is_empty());
        assert_eq!(
            map.cache_position(CachePosition::Message { index: 1, block: 0 })
                .unwrap(),
            None
        );
        assert!(map
            .cache_position(CachePosition::Message { index: 3, block: 0 })
            .is_err());
        map.insert_message(3, vec![2]).unwrap();
        assert_eq!(map.message_index(3), Some(2), "empty native-options row");
    }

    #[test]
    fn ambiguous_colliding_and_invalid_overrides_fail_before_dispatch() {
        let mut map = MessageProjectionMap::default();
        map.insert((0, 0), vec![(0, 0), (0, 1)]).unwrap();
        let exact = |path: &str| BTreeMap::from([(path.into(), vec![0xd800])]);
        assert!(map
            .string_overrides(&exact("/messages/0/content/0/text"))
            .is_err());
        assert!(map
            .string_overrides(&exact("/messages/00/content/0/text"))
            .is_err());
        assert!(map
            .string_overrides(&exact("/messages/0/content/0"))
            .is_err());
        map.map_field((0, 0), "text", Some(((0, 0), "text".into())))
            .unwrap();
        map.insert((0, 1), vec![(0, 0)]).unwrap();
        let exact = BTreeMap::from([
            ("/messages/0/content/0/text".into(), vec![0xd800]),
            ("/messages/0/content/1/text".into(), vec![0xd801]),
        ]);
        assert!(map.string_overrides(&exact).is_err());
    }

    #[test]
    fn mapped_exact_fields_match_the_existing_wire_layout_in_both_modes() {
        use crate::codecs::{CodecContext, EncodeRequest, WireCodec};
        use crate::protocol::{
            ChatRequest, ContentBlock, ConversationMessage, JsonPayload, ProviderProfile,
        };
        for protocol in [
            crate::protocol::ProtocolFamily::OpenAiResponses,
            crate::protocol::ProtocolFamily::AnthropicMessages,
        ] {
            for mode in [crate::RequestMode::Stream, crate::RequestMode::Complete] {
                let (model, profile): (&str, ProviderProfile) = match protocol {
                    crate::protocol::ProtocolFamily::OpenAiResponses => ("gpt-6.1-sol", serde_json::from_value(serde_json::json!({
                        "provider_id":"openai", "profile_name":"fixture", "protocol":"open_ai_responses", "base_url":"https://api.openai.com/v1", "auth":"none"
                    })).unwrap()),
                    _ => ("claude-sonnet-4-6", serde_json::from_value(serde_json::json!({
                        "provider_id":"anthropic", "profile_name":"fixture", "protocol":"anthropic_messages", "base_url":"https://api.anthropic.com", "auth":"none"
                    })).unwrap()),
                };
                let mut request = ChatRequest::new(model);
                request.max_tokens = Some(128);
                request.messages = vec![
                    ConversationMessage::user_text("�"),
                    ConversationMessage::user_text("�"),
                ];
                let first = vec![0xd800];
                let second = vec![0xd801];
                // Identical displays force mapping to use source coordinates.
                let old = BTreeMap::from([
                    ("/messages/0/content/0/text".into(), first.clone()),
                    ("/messages/0/content/1/text".into(), second.clone()),
                ]);
                let expected = BTreeMap::from([
                    ("/messages/0/content/0/text".into(), first),
                    ("/messages/1/content/0/text".into(), second),
                ]);
                let mut map = MessageProjectionMap::default();
                map.insert((0, 0), vec![(0, 0)]).unwrap();
                map.insert((0, 1), vec![(1, 0)]).unwrap();
                let actual = map.string_overrides(&old).unwrap();
                let codec: &dyn WireCodec = match protocol {
                    crate::protocol::ProtocolFamily::OpenAiResponses => {
                        &crate::OpenAiResponsesCodec
                    }
                    _ => &crate::AnthropicMessagesCodec,
                };
                let prepared = codec
                    .encode_request(
                        EncodeRequest::new(&request),
                        &CodecContext::new(&profile, model, mode),
                    )
                    .unwrap();
                let body =
                    JsonPayload::parse(std::str::from_utf8(&prepared.body).unwrap()).unwrap();
                let encode = |exact| {
                    let mapped = crate::exact_json::map_message_text_overrides(
                        &request, protocol, &body, exact,
                    )
                    .unwrap();
                    crate::exact_json::serialize_json_payload_with_overrides(
                        &body,
                        &mapped,
                        crate::exact_json::JsonEncoding::for_protocol(protocol),
                    )
                    .unwrap()
                };
                assert_eq!(encode(&actual), encode(&expected));
                assert!(matches!(
                    request.messages[0].content[0],
                    ContentBlock::Text { .. }
                ));
            }
        }
    }
}
