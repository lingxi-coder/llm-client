//! Current Native model.complete normalization on the ordinary SDK path.

use crate::protocol::{
    CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, ContentBlock, ConversationMessage,
    LlmError, ModelCompleteInput, ModelCompleteTextBlock, PromptCachePolicy, SystemBlock,
    ThinkingConfig,
};

#[derive(Debug)]
struct NormalizedTextBlock {
    units: Vec<u16>,
    cache: bool,
}

/// Native d5t matches a contiguous, well-formed UTF-16 prefix. It returns the
/// aggregate alone when empty or when no supplied block matches.
fn normalize_text(
    aggregate: &[u16],
    blocks: Option<&[ModelCompleteTextBlock]>,
) -> Vec<NormalizedTextBlock> {
    let Some(blocks) = blocks.filter(|_| !aggregate.is_empty()) else {
        return vec![NormalizedTextBlock {
            units: aggregate.to_vec(),
            cache: false,
        }];
    };

    let mut offset = 0;
    let mut matched = Vec::new();
    for block in blocks {
        let Some(units) = block.text.root_string_code_units() else {
            break;
        };
        if !is_well_formed_utf16(&units)
            || offset > aggregate.len()
            || !aggregate[offset..].starts_with(&units)
        {
            break;
        }
        offset += units.len();
        matched.push(NormalizedTextBlock {
            units,
            cache: block.cache == Some(true),
        });
    }
    if matched.is_empty() {
        return vec![NormalizedTextBlock {
            units: aggregate.to_vec(),
            cache: false,
        }];
    }
    if offset < aggregate.len() {
        matched.push(NormalizedTextBlock {
            units: aggregate[offset..].to_vec(),
            cache: false,
        });
    }
    matched
}

fn is_well_formed_utf16(units: &[u16]) -> bool {
    let mut index = 0;
    while index < units.len() {
        match units[index] {
            0xD800..=0xDBFF => {
                if !units
                    .get(index + 1)
                    .is_some_and(|next| (0xDC00..=0xDFFF).contains(next))
                {
                    return false;
                }
                index += 2;
            }
            0xDC00..=0xDFFF => return false,
            _ => index += 1,
        }
    }
    true
}

fn request_text_block(units: Vec<u16>) -> ContentBlock {
    let display = String::from_utf16_lossy(&units);
    match String::from_utf16(&units) {
        Ok(text) => ContentBlock::Text {
            text,
            thought_signature: None,
            citations: None,
        },
        Err(_) => ContentBlock::TextJsUtf16 {
            text: display,
            utf16_code_units: units,
            thought_signature: None,
            citations: None,
        },
    }
}

impl ModelCompleteInput {
    /// Validate and lower to the shared current ChatRequest model. Cache
    /// markers remain positional until the selected provider codec encodes them.
    pub(crate) fn chat_request(&self) -> Result<ChatRequest, LlmError> {
        self.validate()?;
        let prompt_units = self
            .prompt
            .root_string_code_units()
            .ok_or_else(|| invalid("model.complete prompt must be a JSON string"))?;
        let prompt_blocks = normalize_text(&prompt_units, self.prompt_blocks.as_deref());

        let mut request = ChatRequest::new(self.model.effective_selector.clone());
        request.max_tokens = Some(self.max_tokens.unwrap_or(1024));
        request.thinking = self.effort.map(|effort| ThinkingConfig {
            effort: Some(effort),
            ..ThinkingConfig::default()
        });
        request.prompt_cache = PromptCachePolicy::default();

        if let Some(system) = &self.system {
            let units = system
                .root_string_code_units()
                .ok_or_else(|| invalid("model.complete system must be a JSON string"))?;
            if !units.is_empty() {
                let blocks = normalize_text(&units, self.system_blocks.as_deref());
                for (index, block) in blocks.into_iter().enumerate() {
                    request
                        .system
                        .push(SystemBlock::from_utf16_code_units(&block.units));
                    if block.cache {
                        request.prompt_cache.breakpoints.push(CacheBreakpoint {
                            scope: None,
                            position: CachePosition::System { index },
                            ttl: CacheTtl::FiveMinutes,
                        });
                    }
                }
            }
        }

        let message_index = request.messages.len();
        let mut content = Vec::with_capacity(prompt_blocks.len());
        for (block_index, block) in prompt_blocks.into_iter().enumerate() {
            content.push(request_text_block(block.units));
            if block.cache {
                request.prompt_cache.breakpoints.push(CacheBreakpoint {
                    scope: None,
                    position: CachePosition::Message {
                        index: message_index,
                        block: block_index,
                    },
                    ttl: CacheTtl::FiveMinutes,
                });
            }
        }
        request.messages.push(ConversationMessage {
            role: crate::protocol::MessageRole::User,
            content,
            native_options: Vec::new(),
        });
        Ok(request)
    }
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        NativeModelGateIdentity, NativeModelObservation, NativeModelSettingsSnapshot,
    };
    use std::collections::BTreeMap;

    fn payload(source: &str) -> crate::protocol::JsonPayload {
        crate::protocol::JsonPayload::parse(source).unwrap()
    }

    fn text(value: &str, cache: Option<bool>) -> ModelCompleteTextBlock {
        ModelCompleteTextBlock::new(payload(&serde_json::to_string(value).unwrap()), cache).unwrap()
    }

    fn input(prompt: &str, blocks: Option<Vec<ModelCompleteTextBlock>>) -> ModelCompleteInput {
        ModelCompleteInput {
            prompt: payload(&serde_json::to_string(prompt).unwrap()),
            prompt_blocks: blocks,
            native_settings: NativeModelSettingsSnapshot::new(
                "claude-sonnet-5-5",
                NativeModelObservation::Observed(Some(BTreeMap::new())),
            ),
            ..ModelCompleteInput::new(
                NativeModelGateIdentity::unchanged("claude-sonnet-5-5"),
                payload("\"placeholder\""),
                NativeModelSettingsSnapshot::new(
                    "claude-sonnet-5-5",
                    NativeModelObservation::Observed(None),
                ),
            )
        }
    }

    #[test]
    fn prefix_layout_keeps_empty_cache_block_and_uncached_suffix() {
        let input = input(
            "abc",
            Some(vec![
                text("", Some(true)),
                text("a", Some(false)),
                text("b", None),
            ]),
        );
        let request = input.chat_request().unwrap();
        // Native d5t keeps the matched empty/a/b blocks and appends the
        // unmatched prompt suffix as one uncached text block.
        let text_blocks = request.messages[0]
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text, .. } => text.clone(),
                other => panic!("text block expected, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(text_blocks, vec!["", "a", "b", "c"]);
        assert_eq!(request.prompt_cache.breakpoints.len(), 1);
        assert_eq!(
            request.prompt_cache.breakpoints[0].position,
            CachePosition::Message { index: 0, block: 0 }
        );
    }

    #[test]
    fn lone_block_stops_prefix_and_preserves_units_until_provider_egress() {
        let prompt = crate::protocol::JsonPayload::string_from_utf16_code_units(&[
            b'a' as u16,
            0xD800,
            b'z' as u16,
        ]);
        let lone = crate::protocol::JsonPayload::string_from_utf16_code_units(&[0xD800]);
        let mut input = input("", None);
        input.prompt = prompt;
        input.prompt_blocks = Some(vec![ModelCompleteTextBlock::new(lone, Some(true)).unwrap()]);
        let request = input.chat_request().unwrap();
        assert!(request.prompt_cache.breakpoints.is_empty());
        let ContentBlock::TextJsUtf16 {
            text,
            utf16_code_units,
            ..
        } = &request.messages[0].content[0]
        else {
            panic!("exact UTF-16 text block expected")
        };
        assert_eq!(text, "a\u{FFFD}z");
        assert_eq!(utf16_code_units, &[b'a' as u16, 0xD800, b'z' as u16]);
    }

    #[test]
    fn matched_prefix_keeps_lone_suffix_units_until_provider_egress() {
        let prompt = crate::protocol::JsonPayload::string_from_utf16_code_units(&[
            b'a' as u16,
            0xD800,
            b'z' as u16,
        ]);
        let mut input = input("", None);
        input.prompt = prompt;
        input.prompt_blocks = Some(vec![text("a", Some(true))]);
        let request = input.chat_request().unwrap();
        assert_eq!(request.prompt_cache.breakpoints.len(), 1);
        assert_eq!(
            request.prompt_cache.breakpoints[0].position,
            CachePosition::Message { index: 0, block: 0 }
        );
        let ContentBlock::TextJsUtf16 {
            text,
            utf16_code_units,
            ..
        } = &request.messages[0].content[1]
        else {
            panic!("exact UTF-16 suffix expected")
        };
        assert_eq!(text, "\u{FFFD}z");
        assert_eq!(utf16_code_units, &[0xD800, b'z' as u16]);
    }

    #[test]
    fn system_blocks_keep_lone_units_until_provider_egress() {
        let system =
            crate::protocol::JsonPayload::string_from_utf16_code_units(&[b's' as u16, 0xD800]);
        let mut input = input("", None);
        input.system = Some(system);
        input.system_blocks = Some(vec![text("s", Some(true))]);
        let request = input.chat_request().unwrap();
        assert_eq!(
            request.system[0].text_payload().root_string_code_units(),
            Some(vec![b's' as u16])
        );
        assert_eq!(
            request.system[1].text_payload().root_string_code_units(),
            Some(vec![0xD800])
        );
        assert_eq!(request.prompt_cache.breakpoints.len(), 1);
        assert_eq!(
            request.prompt_cache.breakpoints[0].position,
            CachePosition::System { index: 0 }
        );
    }

    #[test]
    fn empty_aggregate_does_not_mark_empty_blocks() {
        let request = input("", Some(vec![text("", Some(true))]))
            .chat_request()
            .unwrap();
        assert!(request.prompt_cache.breakpoints.is_empty());
        assert_eq!(request.messages[0].content.len(), 1);
    }

    #[test]
    fn empty_system_aggregate_emits_no_system_block() {
        let mut input = input("", None);
        input.system = Some(payload("\"\""));
        input.system_blocks = Some(vec![text("", Some(true))]);
        let request = input.chat_request().unwrap();
        assert!(request.system.is_empty());
        assert!(request.prompt_cache.breakpoints.is_empty());
    }
}
