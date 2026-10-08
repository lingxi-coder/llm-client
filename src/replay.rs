//! Explicit replay compatibility decisions for canonical message content.
//!
//! Native state and signatures belong to their wire family. Inspect a decision
//! first, or opt in to normalization; the default policy never drops content.
use crate::protocol::{ContentBlock, LlmError, ProtocolFamily};
use serde_json::Value;

/// Hosting adapters share their underlying wire's replay representation.
pub const fn native_family(family: ProtocolFamily) -> ProtocolFamily {
    match family {
        ProtocolFamily::BedrockClaude
        | ProtocolFamily::VertexClaude
        | ProtocolFamily::FoundryClaude => ProtocolFamily::AnthropicMessages,
        ProtocolFamily::AzureOpenAi => ProtocolFamily::OpenAiChat,
        ProtocolFamily::VertexGemini => ProtocolFamily::GeminiGenerateContent,
        family => family,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplayPolicy {
    #[default]
    Reject,
    /// Drop incompatible native state, retaining portable visible text and
    /// generic tool calls where possible. This is an explicit lossy policy.
    DropIncompatible,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayDecision {
    /// The block is compatible; hosting aliases may have been canonicalized.
    Compatible(ContentBlock),
    /// A caller must explicitly authorize the replacement or omission.
    Incompatible {
        replacement: Option<ContentBlock>,
        reason: &'static str,
    },
}
impl ReplayDecision {
    pub fn apply(self, policy: ReplayPolicy) -> Result<Option<ContentBlock>, LlmError> {
        match self {
            Self::Compatible(block) => Ok(Some(block)),
            Self::Incompatible {
                replacement,
                reason,
            } => match policy {
                ReplayPolicy::DropIncompatible => Ok(replacement),
                ReplayPolicy::Reject => Err(LlmError::InvalidRequest {
                    message: reason.into(),
                }),
            },
        }
    }
}

/// Message context identifies visible reasoning summaries paired with foreign
/// native reasoning. Such summaries are not signed Anthropic thinking blocks.
#[derive(Debug, Clone, Copy)]
pub struct ReplayContext {
    target: ProtocolFamily,
    foreign_reasoning: bool,
}
impl ReplayContext {
    pub fn for_message(blocks: &[ContentBlock], target: ProtocolFamily) -> Self {
        let target = native_family(target);
        let foreign_reasoning = blocks.iter().any(|block| {
            matches!(block, ContentBlock::ProviderContent { protocol, value }
                if native_family(*protocol) != target
                    && matches!(value["type"].as_str(), Some("reasoning" | "chat_reasoning")))
        });
        Self {
            target,
            foreign_reasoning,
        }
    }

    /// `source` supplies provenance for portable blocks whose optional native
    /// fields came from a particular response. ProviderContent has its own
    /// authoritative source tag and does not need this argument.
    pub fn decision(&self, block: &ContentBlock, source: Option<ProtocolFamily>) -> ReplayDecision {
        let source = source.map(native_family);
        let foreign = source.is_some_and(|source| source != self.target);
        let incompatible = |replacement, reason| ReplayDecision::Incompatible {
            replacement,
            reason,
        };
        match block {
            ContentBlock::Native { value } => {
                use crate::providers::openai::computer::{
                    OpenAiComputerCall, OpenAiComputerCallOutput,
                };
                if self.target == ProtocolFamily::GeminiInteractions
                    && !foreign
                    && matches!(
                        value.format(),
                        crate::providers::google::computer::CALL_FORMAT
                            | crate::providers::google::computer::RESULT_FORMAT
                    )
                {
                    ReplayDecision::Compatible(block.clone())
                } else if value.is::<OpenAiComputerCall>() {
                    incompatible(None, "computer calls continue through their response reference and paired output")
                } else if self.target == ProtocolFamily::OpenAiResponses
                    && value.is::<OpenAiComputerCallOutput>()
                {
                    ReplayDecision::Compatible(block.clone())
                } else {
                    incompatible(
                        None,
                        "typed native content is unsupported on the target protocol",
                    )
                }
            }
            ContentBlock::ProviderContent { protocol, value } => {
                if matches!(
                    value.get("type").and_then(serde_json::Value::as_str),
                    Some("computer_call" | "computer_call_output")
                ) {
                    incompatible(
                        None,
                        "computer calls and outputs require the validated typed continuation path",
                    )
                } else if native_family(*protocol) == self.target {
                    ReplayDecision::Compatible(ContentBlock::ProviderContent {
                        protocol: native_family(*protocol),
                        value: value.clone(),
                    })
                } else {
                    let replacement = if value["type"] == "text" {
                        value["text"].as_str().map(|text| ContentBlock::Text {
                            text: text.into(),
                            thought_signature: None,
                            citations: None,
                        })
                    } else if native_family(*protocol) == ProtocolFamily::AnthropicMessages
                        && value["type"] == "tool_result"
                    {
                        value["tool_use_id"]
                            .as_str()
                            .filter(|id| !id.is_empty())
                            .map(|id| {
                                let content = &value["content"];
                                ContentBlock::ToolResult {
                                    cache_reference: None,
                                    output_json: None,
                                    tool_use_id: id.into(),
                                    content: content
                                        .as_str()
                                        .map(str::to_owned)
                                        .unwrap_or_else(|| content.to_string()),
                                    blocks: content.as_array().cloned(),
                                    is_error: value.get("is_error").and_then(Value::as_bool),
                                    toolset_name: None,
                                }
                            })
                    } else {
                        None
                    };
                    incompatible(
                        replacement,
                        "native content belongs to a different protocol family",
                    )
                }
            }
            ContentBlock::Thinking { text, signature } => {
                if self.target == ProtocolFamily::AnthropicMessages
                    && (foreign || signature.is_none() && self.foreign_reasoning)
                {
                    return incompatible(
                        None,
                        "foreign reasoning is not signed Anthropic thinking",
                    );
                }
                if signature.is_some()
                    && (foreign
                        || !matches!(
                            self.target,
                            ProtocolFamily::AnthropicMessages
                                | ProtocolFamily::GeminiGenerateContent
                        ))
                {
                    return incompatible(
                        Some(ContentBlock::Thinking {
                            text: text.clone(),
                            signature: None,
                        }),
                        "reasoning signature belongs to a different protocol family",
                    );
                }
                ReplayDecision::Compatible(block.clone())
            }
            ContentBlock::RedactedThinking { .. }
                if self.target != ProtocolFamily::AnthropicMessages || foreign =>
            {
                incompatible(
                    None,
                    "redacted thinking requires its original Anthropic protocol family",
                )
            }
            ContentBlock::Text {
                text,
                citations: Some(_),
                ..
            } if foreign || self.target != ProtocolFamily::AnthropicMessages => incompatible(
                Some(ContentBlock::Text {
                    text: text.clone(),
                    thought_signature: None,
                    citations: None,
                }),
                "text citations field presence belongs to Anthropic Messages",
            ),
            ContentBlock::TextJsUtf16 {
                text,
                utf16_code_units,
                citations: Some(_),
                thought_signature,
            } if foreign || self.target != ProtocolFamily::AnthropicMessages => incompatible(
                Some(ContentBlock::TextJsUtf16 {
                    text: text.clone(),
                    utf16_code_units: utf16_code_units.clone(),
                    thought_signature: thought_signature.clone(),
                    citations: None,
                }),
                "text citations field presence belongs to Anthropic Messages",
            ),
            ContentBlock::Text {
                text,
                thought_signature: Some(_),
                citations,
            } if foreign || self.target != ProtocolFamily::GeminiGenerateContent => incompatible(
                Some(ContentBlock::Text {
                    text: text.clone(),
                    thought_signature: None,
                    citations: citations.clone(),
                }),
                "text thought signature belongs to a different protocol family",
            ),
            ContentBlock::TextJsUtf16 {
                text,
                utf16_code_units,
                thought_signature: Some(_),
                citations,
            } if foreign || self.target != ProtocolFamily::GeminiGenerateContent => incompatible(
                Some(ContentBlock::TextJsUtf16 {
                    text: text.clone(),
                    utf16_code_units: utf16_code_units.clone(),
                    thought_signature: None,
                    citations: citations.clone(),
                }),
                "text thought signature belongs to a different protocol family",
            ),
            ContentBlock::ToolUse {
                thought_signature,
                provider_id,
                caller,
                toolset_name,
                ..
            } => {
                let strip_signature = thought_signature.is_some()
                    && (foreign || self.target != ProtocolFamily::GeminiGenerateContent);
                let strip_caller = (caller.is_some() || toolset_name.is_some())
                    && (foreign || self.target != ProtocolFamily::AnthropicMessages);
                let strip_provider_id = provider_id.is_some() && foreign;
                if strip_signature || strip_caller || strip_provider_id {
                    let mut replacement = block.clone();
                    if let ContentBlock::ToolUse {
                        thought_signature,
                        provider_id,
                        caller,
                        toolset_name,
                        ..
                    } = &mut replacement
                    {
                        if strip_signature {
                            *thought_signature = None;
                        }
                        if strip_caller {
                            *caller = None;
                            *toolset_name = None;
                        }
                        if strip_provider_id {
                            *provider_id = None;
                        }
                    }
                    incompatible(
                        Some(replacement),
                        "tool replay metadata belongs to a different protocol family",
                    )
                } else {
                    ReplayDecision::Compatible(block.clone())
                }
            }
            ContentBlock::ToolResult {
                toolset_name: Some(_),
                ..
            } if foreign || self.target != ProtocolFamily::AnthropicMessages => {
                let mut replacement = block.clone();
                if let ContentBlock::ToolResult { toolset_name, .. } = &mut replacement {
                    *toolset_name = None;
                }
                incompatible(
                    Some(replacement),
                    "toolset result metadata belongs to a different protocol family",
                )
            }
            _ => ReplayDecision::Compatible(block.clone()),
        }
    }

    pub fn normalize(
        &self,
        block: &ContentBlock,
        source: Option<ProtocolFamily>,
        policy: ReplayPolicy,
    ) -> Result<Option<ContentBlock>, LlmError> {
        self.decision(block, source).apply(policy)
    }
}

/// A request adapted for another wire family. Positions map retained original
/// message/block indices to their output indices for caller-owned sidecars.
#[derive(Debug, Clone)]
pub struct RequestReplayProjection {
    pub request: crate::protocol::ChatRequest,
    pub block_positions: std::collections::BTreeMap<(usize, usize), (usize, usize)>,
}

/// Explicitly adapt portable history for a route change. This performs no send
/// and grants no permission to retry. Endpoint-scoped continuations cannot move.
/// Cache policies are family-specific and are dropped only with lossy permission.
pub fn adapt_request(
    request: &crate::protocol::ChatRequest,
    source: ProtocolFamily,
    target: ProtocolFamily,
    policy: ReplayPolicy,
) -> Result<RequestReplayProjection, LlmError> {
    let mut output = RequestReplayProjection {
        request: request.clone(),
        block_positions: Default::default(),
    };
    if native_family(source) == native_family(target) {
        for (mi, message) in request.messages.iter().enumerate() {
            for bi in 0..message.content.len() {
                output.block_positions.insert((mi, bi), (mi, bi));
            }
        }
        return Ok(output);
    }
    if request.continuation.is_some() || request.controls.responses.previous_response_id.is_some() {
        return Err(LlmError::InvalidRequest {
            message: "a provider continuation cannot change protocol family".into(),
        });
    }
    if request.prompt_cache != Default::default() {
        if policy == ReplayPolicy::Reject {
            return Err(LlmError::InvalidRequest {
                message: "prompt cache policy belongs to the original protocol family".into(),
            });
        }
        output.request.prompt_cache = Default::default();
    }
    output.request.messages.clear();
    for (mi, message) in request.messages.iter().enumerate() {
        let context = ReplayContext::for_message(&message.content, target);
        let mut projected = message.clone();
        projected.content.clear();
        for (bi, block) in message.content.iter().enumerate() {
            if let Some(block) = context.normalize(block, Some(source), policy)? {
                output.block_positions.insert(
                    (mi, bi),
                    (output.request.messages.len(), projected.content.len()),
                );
                projected.content.push(block);
            }
        }
        if !projected.content.is_empty() || !projected.native_options.is_empty() {
            output.request.messages.push(projected);
        }
    }
    Ok(output)
}

/// Visible text whose native citation locations need a replay representation.
pub fn native_cited_text(block: &ContentBlock) -> Option<&str> {
    let ContentBlock::ProviderContent { protocol, value } = block else {
        return None;
    };
    (native_family(*protocol) == ProtocolFamily::AnthropicMessages
        && value["type"] == "text"
        && value["citations"]
            .as_array()
            .is_some_and(|citations| !citations.is_empty()))
    .then(|| value["text"].as_str())
    .flatten()
}

/// Whether reducing a canonical block to generic display text/tool input would
/// omit replay metadata. This does not authorize that reduction.
pub fn has_replay_metadata(block: &ContentBlock) -> bool {
    if native_cited_text(block).is_some() {
        return true;
    }
    match block {
        ContentBlock::Text {
            thought_signature,
            citations,
            ..
        }
        | ContentBlock::TextJsUtf16 {
            thought_signature,
            citations,
            ..
        } => thought_signature.is_some() || citations.is_some(),
        ContentBlock::ToolUse {
            thought_signature,
            provider_id,
            caller,
            toolset_name,
            ..
        } => {
            thought_signature.is_some()
                || provider_id.is_some()
                || caller.is_some()
                || toolset_name.is_some()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    const ANTHROPIC: ProtocolFamily = ProtocolFamily::AnthropicMessages;
    const GEMINI: ProtocolFamily = ProtocolFamily::GeminiGenerateContent;

    #[test]
    fn hosted_families_share_native_state_without_lossy_permission() {
        for (source, target) in [
            (ANTHROPIC, ProtocolFamily::BedrockClaude),
            (ProtocolFamily::FoundryClaude, ProtocolFamily::VertexClaude),
            (GEMINI, ProtocolFamily::VertexGemini),
            (ProtocolFamily::VertexGemini, GEMINI),
            (ProtocolFamily::AzureOpenAi, ProtocolFamily::OpenAiChat),
        ] {
            let block = ContentBlock::ProviderContent {
                protocol: source,
                value: json!({"type":"future_native","opaque":true}),
            };
            let output = ReplayContext::for_message(&[], target)
                .normalize(&block, None, ReplayPolicy::default())
                .unwrap()
                .unwrap();
            assert!(
                matches!(output, ContentBlock::ProviderContent { protocol, value } if protocol == native_family(source) && value["opaque"] == true)
            );
        }
    }

    #[test]
    fn foreign_citations_require_explicit_policy_to_keep_only_visible_text() {
        let block = ContentBlock::ProviderContent {
            protocol: ANTHROPIC,
            value: json!({"type":"text","text":"answer","citations":[{"offset":3}]}),
        };
        assert_eq!(native_cited_text(&block), Some("answer"));
        assert!(has_replay_metadata(&block));
        let context = ReplayContext::for_message(&[], ProtocolFamily::OpenAiChat);
        assert!(context
            .normalize(&block, None, ReplayPolicy::default())
            .is_err());
        assert_eq!(
            context
                .normalize(&block, None, ReplayPolicy::DropIncompatible)
                .unwrap(),
            Some(ContentBlock::Text {
                text: "answer".into(),
                thought_signature: None,
                citations: None,
            })
        );
    }

    #[test]
    fn nullable_citations_survive_native_replay_without_nonempty_citation_sidecar() {
        for citations in [Some(None), Some(Some(json!([])))] {
            let block = ContentBlock::Text {
                text: "answer".into(),
                thought_signature: None,
                citations,
            };
            assert!(has_replay_metadata(&block));
            assert!(serde_json::to_value(&block)
                .unwrap()
                .get("citations")
                .is_some());
            let context = ReplayContext::for_message(
                std::slice::from_ref(&block),
                ProtocolFamily::AnthropicMessages,
            );
            assert!(matches!(
                context.decision(&block, None),
                ReplayDecision::Compatible(ref preserved) if preserved == &block
            ));
        }
    }

    #[test]
    fn cached_native_tool_result_keeps_generic_result_on_explicit_route_change() {
        let block = ContentBlock::ProviderContent {
            protocol: ANTHROPIC,
            value: json!({"type":"tool_result","tool_use_id":"id","content":[{"type":"text","text":"answer"}],"is_error":true,"cache_reference":"provider-cache","toolset_name":"browser"}),
        };
        let context = ReplayContext::for_message(&[], ProtocolFamily::OpenAiChat);
        assert!(context
            .normalize(&block, None, ReplayPolicy::Reject)
            .is_err());
        assert!(
            matches!(context.normalize(&block, None, ReplayPolicy::DropIncompatible).unwrap(), Some(ContentBlock::ToolResult { tool_use_id, blocks: Some(blocks), is_error:Some(true), toolset_name:None, .. }) if tool_use_id.as_str() == "id" && blocks == vec![json!({"type":"text","text":"answer"})])
        );
    }

    #[test]
    fn foreign_reasoning_summary_is_not_unsigned_anthropic_thinking() {
        let blocks = vec![
            ContentBlock::Thinking {
                text: "summary".into(),
                signature: None,
            },
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::OpenAiResponses,
                value: json!({"type":"reasoning","encrypted_content":"opaque"}),
            },
        ];
        let context = ReplayContext::for_message(&blocks, ANTHROPIC);
        for block in &blocks {
            assert!(context
                .normalize(block, None, ReplayPolicy::Reject)
                .is_err());
            assert!(context
                .normalize(block, None, ReplayPolicy::DropIncompatible)
                .unwrap()
                .is_none());
        }
        let signed = ContentBlock::Thinking {
            text: "reasoning".into(),
            signature: Some("signed".into()),
        };
        assert_eq!(
            context
                .normalize(&signed, Some(ANTHROPIC), ReplayPolicy::Reject)
                .unwrap(),
            Some(signed)
        );
        // An unsigned block with no foreign reasoning context remains subject
        // to the provider's own model/endpoint validation.
        assert!(ReplayContext::for_message(&[], ANTHROPIC)
            .normalize(&blocks[0], None, ReplayPolicy::Reject)
            .is_ok());
    }

    #[test]
    fn native_tool_metadata_and_gemini_signatures_cannot_cross_families() {
        let block = ContentBlock::ToolUse {
            input_json: None,
            id: "id".into(),
            name: "read".into(),
            input: json!({"x":1}),
            provider_id: Some("provider".into()),
            caller: Some(json!({"type":"direct"})),
            toolset_name: Some("browser".into()),
            thought_signature: Some("signature".into()),
        };
        let context = ReplayContext::for_message(&[], ProtocolFamily::OpenAiResponses);
        assert!(context
            .normalize(&block, Some(ANTHROPIC), ReplayPolicy::Reject)
            .is_err());
        let stripped = context
            .normalize(&block, Some(ANTHROPIC), ReplayPolicy::DropIncompatible)
            .unwrap()
            .unwrap();
        assert!(
            matches!(stripped, ContentBlock::ToolUse { id, name, input, provider_id:None, caller:None, toolset_name:None, thought_signature:None , .. } if id.as_str() == "id" && name == "read" && input == json!({"x":1}))
        );
        let text = ContentBlock::Text {
            text: "signed text".into(),
            thought_signature: Some("s".into()),
            citations: None,
        };
        assert_eq!(
            ReplayContext::for_message(&[], ProtocolFamily::VertexGemini)
                .normalize(&text, Some(GEMINI), ReplayPolicy::Reject)
                .unwrap(),
            Some(text)
        );
    }

    #[test]
    fn foreign_signed_thinking_is_never_replayed_with_its_signature() {
        let signed = ContentBlock::Thinking {
            text: "reason".into(),
            signature: Some("secret".into()),
        };
        assert!(ReplayContext::for_message(&[], ANTHROPIC)
            .normalize(&signed, Some(GEMINI), ReplayPolicy::DropIncompatible)
            .unwrap()
            .is_none());
        assert_eq!(
            ReplayContext::for_message(&[], GEMINI)
                .normalize(&signed, Some(ANTHROPIC), ReplayPolicy::DropIncompatible)
                .unwrap(),
            Some(ContentBlock::Thinking {
                text: "reason".into(),
                signature: None
            })
        );
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::protocol::{
        CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, ConversationMessage, MessageRole,
    };
    use serde_json::json;

    #[test]
    fn route_change_projects_history_and_cache_without_mutating_source() {
        let mut request = ChatRequest::new("model");
        request.messages = vec![
            ConversationMessage {
                role: MessageRole::Assistant,
                native_options: vec![],
                content: vec![ContentBlock::RedactedThinking {
                    data: "encrypted".into(),
                }],
            },
            ConversationMessage {
                role: MessageRole::Assistant,
                native_options: vec![],
                content: vec![
                    ContentBlock::Thinking {
                        text: "reason".into(),
                        signature: Some("signature".into()),
                    },
                    ContentBlock::ProviderContent {
                        protocol: ProtocolFamily::AnthropicMessages,
                        value: json!({"type":"text","text":"answer","citations":[{}]}),
                    },
                ],
            },
        ];
        request.prompt_cache.breakpoints.push(CacheBreakpoint {
            position: CachePosition::Message { index: 1, block: 1 },
            scope: None,
            ttl: CacheTtl::FiveMinutes,
        });
        let before = request.clone();
        let projected = adapt_request(
            &request,
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::OpenAiChat,
            ReplayPolicy::DropIncompatible,
        )
        .unwrap();
        assert_eq!(request, before);
        assert_eq!(projected.request.messages.len(), 1);
        assert!(matches!(
            &projected.request.messages[0].content[0],
            ContentBlock::Thinking {
                signature: None,
                ..
            }
        ));
        assert!(
            matches!(&projected.request.messages[0].content[1], ContentBlock::Text {text,..} if text == "answer")
        );
        assert_eq!(projected.block_positions.get(&(1, 1)), Some(&(0, 1)));
        assert_eq!(projected.request.prompt_cache, Default::default());
        assert!(adapt_request(
            &request,
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::OpenAiChat,
            ReplayPolicy::Reject
        )
        .is_err());
        let same = adapt_request(
            &request,
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::BedrockClaude,
            ReplayPolicy::Reject,
        )
        .unwrap();
        assert_eq!(same.request, request);
    }

    #[test]
    fn route_change_never_discards_explicit_continuation() {
        let mut request = ChatRequest::new("model");
        request.controls.responses.previous_response_id = Some("resp_1".into());
        assert!(adapt_request(
            &request,
            ProtocolFamily::OpenAiResponses,
            ProtocolFamily::OpenAiChat,
            ReplayPolicy::DropIncompatible
        )
        .is_err());
    }
}
