//! Assembly of canonical SDK events, without transport retries or host policy.
use crate::protocol::{
    ChatResponse, ContentBlock, ConversationMessage, FileSearchResult, InferenceReport,
    JsonPayload, JsonPayloadFragment, LlmError, ResponseId, StopReason, StreamEvent, ToolUseId,
    UsageReport, WebSearchResult,
};
use crate::{ModelStream, StreamBatch};
#[cfg(test)]
use serde_json::Value;
use std::collections::BTreeMap;

/// A response projection plus the lossless stream observations behind it.
/// `response` may be partial: consult the stop reason and incomplete-block
/// metadata as well as `terminal`. A length stop is terminal but truncated.
#[derive(Debug, Clone)]
pub struct StreamAssembly {
    pub response: ChatResponse,
    /// Primary content keyed by the original sparse block index. Phase metadata
    /// companions remain adjacent to their text in `response.message.content`.
    pub indexed_content: BTreeMap<usize, ContentBlock>,
    /// Original events preserve native annotations, raw provider frames and
    /// unfinished tool argument fragments, none of which are replayable blocks.
    pub events: Vec<StreamEvent>,
    pub terminal: bool,
    pub unfinished_blocks: Vec<usize>,
    pub incomplete_tools: Vec<IncompleteTool>,
}

/// Metadata for a tool excluded from the response. Argument bytes remain in
/// the original events, never in diagnostic text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteTool {
    pub block: usize,
    pub name: String,
    pub input_bytes: usize,
    pub ended: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("stream assembly failed: {source}")]
pub struct StreamAssemblyError {
    #[source]
    pub source: LlmError,
    pub partial: Box<StreamAssembly>,
}

#[derive(Debug, Clone)]
struct Tool {
    id: ToolUseId,
    provider_id: Option<String>,
    caller: Option<JsonPayload>,
    toolset_name: Option<String>,
    name: String,
    arguments: Vec<u16>,
}

#[derive(Debug, Clone, Default)]
struct Block {
    text: Option<String>,
    utf16_code_units: Option<Vec<u16>>,
    citations: Option<Option<JsonPayload>>,
    thinking: Option<String>,
    signature: Option<String>,
    tool: Option<Tool>,
    native: Option<ContentBlock>,
    // Phase metadata accompanies visible text instead of replacing it.
    replay_companion: Option<ContentBlock>,
    ended: bool,
}
impl Block {
    fn append_text(&mut self, text: &str) {
        if let Some(units) = &mut self.utf16_code_units {
            units.extend(text.encode_utf16());
        }
        // A Rust string cannot start with an unpaired low surrogate, so it
        // cannot complete a pending high surrogate from the previous delta.
        self.text.get_or_insert_default().push_str(text);
    }

    fn append_text_utf16(&mut self, fragment: &[u16]) {
        let text = self.text.get_or_insert_default();
        let units = self
            .utf16_code_units
            .get_or_insert_with(|| text.encode_utf16().collect());
        let trailing_high = units
            .last()
            .copied()
            .filter(|unit| (0xd800..=0xdbff).contains(unit));
        // The display already contains a replacement for a trailing high
        // surrogate. Decode only that boundary again with the new fragment,
        // allowing a split surrogate pair to replace the displayed U+FFFD.
        if trailing_high.is_some() {
            debug_assert!(text.ends_with(char::REPLACEMENT_CHARACTER));
            text.pop();
        }
        text.extend(
            char::decode_utf16(trailing_high.into_iter().chain(fragment.iter().copied()))
                .map(|decoded| decoded.unwrap_or(char::REPLACEMENT_CHARACTER)),
        );
        units.extend_from_slice(fragment);
    }

    fn content(&self, terminal: bool, native_terminal: bool) -> Option<ContentBlock> {
        if let Some(native) = &self.native {
            if matches!(native, ContentBlock::Native { .. }) && !native_terminal {
                return None;
            }
            return Some(native.clone());
        }
        if let Some(tool) = &self.tool {
            if tool.id.as_str().is_empty() || tool.name.is_empty() || !self.ended && !terminal {
                return None;
            }
            // Some providers omit fragments for an explicitly completed zero-argument call.
            let input = if tool.arguments.is_empty() {
                JsonPayload::parse("{}").ok()?
            } else {
                let fragment = JsonPayloadFragment::from_utf16_code_units(tool.arguments.clone());
                JsonPayload::parse_fragments([&fragment]).ok()?
            };
            return Some(ContentBlock::ToolUse {
                id: tool.id.clone(),
                name: tool.name.clone(),
                input,
                provider_id: tool.provider_id.clone(),
                caller: tool.caller.clone(),
                toolset_name: tool.toolset_name.clone(),
                thought_signature: self.signature.clone(),
            });
        }
        if let Some(text) = &self.thinking {
            return Some(ContentBlock::Thinking {
                text: text.clone(),
                signature: self.signature.clone(),
            });
        }
        self.text.as_ref().map(|text| match &self.utf16_code_units {
            Some(utf16_code_units) => ContentBlock::TextJsUtf16 {
                text: text.clone(),
                utf16_code_units: utf16_code_units.clone(),
                thought_signature: self.signature.clone(),
                citations: self.citations.clone(),
            },
            None => ContentBlock::Text {
                text: text.clone(),
                thought_signature: self.signature.clone(),
                citations: self.citations.clone(),
            },
        })
    }
}

/// Incrementally projects SDK events. Start sets identity only: it does not
/// clear already observed inference/accounting facts or completed blocks.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    blocks: BTreeMap<usize, Block>,
    completed_order: Vec<usize>,
    connectors: crate::providers::anthropic::ConnectorTextAccumulator,
    events: Vec<StreamEvent>,
    model: String,
    response_id: Option<ResponseId>,
    stop_reason: Option<StopReason>,
    usage: UsageReport,
    inference: InferenceReport,
    web_search: Option<WebSearchResult>,
    file_search: Option<FileSearchResult>,
    projection_error: Option<LlmError>,
}
impl StreamAccumulator {
    /// Completed content in block-completion order, which is the order used by
    /// the Anthropic server-fallback retention projection.
    pub fn completed_blocks(&self) -> Vec<(usize, ContentBlock)> {
        self.completed_order
            .iter()
            .filter_map(|&index| {
                self.blocks
                    .get(&index)?
                    .content(false, false)
                    .map(|content| (index, content))
            })
            .collect()
    }
    /// Set the selected response model after a host admits a native fallback
    /// start. SDK observation alone deliberately does not change this model.
    pub fn set_response_model(&mut self, model: &str) {
        self.model = model.to_owned();
    }

    /// Update the response identifier after a host admits a native fallback
    /// start. A missing identifier should not call this method, preserving the
    /// last observed response id.
    pub fn set_response_id(&mut self, response_id: &str) {
        self.response_id = Some(ResponseId::new(response_id));
    }

    /// Apply an already-admitted server-fallback content projection.
    pub fn apply_server_fallback(
        &mut self,
        event: &crate::providers::anthropic::fallback_response::ServerFallbackEvent,
    ) {
        self.set_response_model(&event.to_model);
        self.blocks
            .retain(|index, _| !event.discarded_blocks.contains(index));
        self.completed_order
            .retain(|index| !event.discarded_blocks.contains(index));
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Start { model, response_id } => {
                self.model.clone_from(model);
                self.response_id.clone_from(response_id);
            }
            StreamEvent::TextDelta { block, text } => {
                self.blocks.entry(*block).or_default().append_text(text);
            }
            StreamEvent::TextDeltaJsUtf16 {
                block,
                utf16_code_units,
                ..
            } => {
                self.blocks
                    .entry(*block)
                    .or_default()
                    .append_text_utf16(utf16_code_units);
            }
            StreamEvent::ReasoningDelta { block, text } => self
                .blocks
                .entry(*block)
                .or_default()
                .thinking
                .get_or_insert_default()
                .push_str(text),
            StreamEvent::ThoughtSignature { block, signature } => self
                .blocks
                .entry(*block)
                .or_default()
                .signature
                .get_or_insert_default()
                .push_str(signature),
            StreamEvent::RedactedThinking { block, data } => {
                self.blocks.entry(*block).or_default().native =
                    Some(ContentBlock::RedactedThinking { data: data.clone() })
            }
            StreamEvent::Native { block, value } => {
                let entry = self.blocks.entry(*block).or_default();
                entry.native = Some(ContentBlock::Native {
                    value: value.clone(),
                });
                entry.ended = true;
            }
            StreamEvent::ProviderContent {
                block,
                protocol,
                value,
            } => {
                let current = self.blocks.entry(*block).or_default();
                let text_projection = if *protocol
                    == crate::protocol::ProtocolFamily::AnthropicMessages
                    && value.string_is("/type", "text")
                    && value.has_only_fields_at("", &["type", "text", "citations"])
                {
                    let text = value.string_at("/text");
                    let units = value.string_units_at("/text");
                    let citations = match value.raw_source_at("/citations") {
                        None => Some(Ok(None)),
                        Some(_)
                            if value.kind_at("/citations")
                                == Some(crate::response_json::FlatJsonKind::Null) =>
                        {
                            Some(Ok(Some(None)))
                        }
                        Some(source) => {
                            Some(JsonPayload::parse(&source).map(|value| Some(Some(value))))
                        }
                    };
                    match (text, units, citations) {
                        (Some(text), Some(units), Some(Ok(citations))) => {
                            Some((text, units, citations))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                if *protocol == crate::protocol::ProtocolFamily::OpenAiResponses
                    && value.string_is("/type", crate::codecs::openai::responses::MESSAGE_PHASE)
                {
                    current.replay_companion = Some(ContentBlock::ProviderContent {
                        protocol: *protocol,
                        value: value.clone(),
                    });
                } else if let Some((text, units, citations)) = text_projection {
                    current.text = Some(text);
                    current.utf16_code_units = String::from_utf16(&units).is_err().then_some(units);
                    current.citations = citations;
                } else {
                    current.native = Some(ContentBlock::ProviderContent {
                        protocol: *protocol,
                        value: value.clone(),
                    });
                }
            }
            StreamEvent::ToolCallDelta {
                block,
                id,
                provider_id,
                caller,
                toolset_name,
                name,
                arguments_fragment,
            } => {
                let tool = self
                    .blocks
                    .entry(*block)
                    .or_default()
                    .tool
                    .get_or_insert_with(|| Tool {
                        id: id.clone(),
                        provider_id: None,
                        caller: None,
                        toolset_name: None,
                        name: name.clone(),
                        arguments: Vec::new(),
                    });
                if !id.as_str().is_empty() {
                    tool.id.clone_from(id);
                }
                if !name.is_empty() {
                    tool.name.clone_from(name);
                }
                if provider_id.is_some() {
                    tool.provider_id.clone_from(provider_id);
                }
                if caller.is_some() {
                    tool.caller.clone_from(caller);
                }
                if toolset_name.is_some() {
                    tool.toolset_name.clone_from(toolset_name);
                }
                tool.arguments
                    .extend_from_slice(arguments_fragment.utf16_code_units());
            }
            StreamEvent::BlockEnd { block } => {
                self.blocks.entry(*block).or_default().ended = true;
                if !self.completed_order.contains(block) {
                    self.completed_order.push(*block);
                }
            }
            StreamEvent::Inference { report } => self.inference.clone_from(report),
            StreamEvent::WebSearch { result } => {
                let current = self.web_search.get_or_insert_default();
                extend_unique(&mut current.citations, &result.citations);
                match current.metadata.merge_incremental(&result.metadata) {
                    Ok(merged) => current.metadata = merged,
                    Err(source) if self.projection_error.is_none() => {
                        self.projection_error = Some(source);
                    }
                    Err(_) => {}
                }
            }
            StreamEvent::FileSearch { result } => {
                let current = self.file_search.get_or_insert_default();
                extend_unique(&mut current.queries, &result.queries);
                extend_unique(&mut current.hits, &result.hits);
                match current.metadata.merge_incremental(&result.metadata) {
                    Ok(merged) => current.metadata = merged,
                    Err(source) if self.projection_error.is_none() => {
                        self.projection_error = Some(source);
                    }
                    Err(_) => {}
                }
            }
            StreamEvent::End {
                stop_reason,
                usage,
                inference,
            } => {
                self.stop_reason = Some(stop_reason.clone());
                self.usage.clone_from(usage);
                self.inference.clone_from(inference);
            }
            StreamEvent::ProviderEvent {
                protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
                payload,
            } => {
                if let Some((index, native)) = self.connectors.push(payload) {
                    if let Ok(index) = usize::try_from(index) {
                        let block = self.blocks.entry(index).or_default();
                        block.native = Some(native);
                        block.ended = true;
                        if !self.completed_order.contains(&index) {
                            self.completed_order.push(index);
                        }
                    }
                }
            }
            StreamEvent::NativeDelta { .. }
            | StreamEvent::ProviderEvent { .. }
            | StreamEvent::NativeControl { .. } => {}
        }
        self.events.push(event.clone());
    }

    /// Accounting can change in an empty batch or the batch containing an
    /// error. Observe every successful event and retain the first error.
    pub fn observe_batch(&mut self, batch: &StreamBatch) -> Result<(), LlmError> {
        let mut error = self.projection_error.clone();
        for event in &batch.events {
            match event {
                Ok(event) => {
                    self.observe(event);
                    if error.is_none() {
                        error.clone_from(&self.projection_error);
                    }
                }
                Err(source) if error.is_none() => error = Some(source.clone()),
                Err(_) => {}
            }
        }
        self.usage.clone_from(&batch.usage);
        self.inference.clone_from(&batch.inference);
        error.map_or(Ok(()), Err)
    }

    /// Project one block for incremental presentation without cloning the event log.
    pub fn content_at(&self, block: usize) -> Option<ContentBlock> {
        self.blocks
            .get(&block)?
            .content(self.is_terminal(), self.native_terminal())
    }

    fn native_terminal(&self) -> bool {
        match self.stop_reason.as_ref() {
            Some(StopReason::ToolUse | StopReason::EndTurn) => true,
            // A completed Responses turn may contain both a computer call and
            // another client action such as tool search or MCP approval.
            Some(StopReason::Other(reason)) if reason == "requires_action" => true,
            _ => false,
        }
    }

    /// Current canonical tool identity and all arguments received so far. This
    /// remains a delta observation, not an executable completed tool call.
    pub fn tool_progress(&self, block: usize) -> Option<StreamEvent> {
        let tool = self.blocks.get(&block)?.tool.as_ref()?;
        Some(StreamEvent::ToolCallDelta {
            block,
            id: tool.id.clone(),
            provider_id: tool.provider_id.clone(),
            caller: tool.caller.clone(),
            toolset_name: tool.toolset_name.clone(),
            name: tool.name.clone(),
            arguments_fragment: JsonPayloadFragment::from_utf16_code_units(tool.arguments.clone()),
        })
    }

    /// Whether the provider closed this block, including native blocks
    /// assembled from provider observations.
    pub fn block_finished(&self, block: usize) -> bool {
        self.blocks
            .get(&block)
            .is_some_and(|value| value.ended || value.native.is_some() || self.is_terminal())
    }

    pub fn is_terminal(&self) -> bool {
        self.stop_reason.is_some()
    }

    /// Includes partial text and reasoning, finalized native blocks and only
    /// syntactically complete tools whose block or response has ended.
    pub fn snapshot(&self) -> StreamAssembly {
        let terminal = self.stop_reason.is_some();
        let unfinished_blocks = self
            .blocks
            .iter()
            .filter_map(|(index, block)| {
                let unfinished = block.tool.is_some()
                    && block.content(terminal, self.native_terminal()).is_none()
                    || !terminal && !block.ended && block.native.is_none();
                unfinished.then_some(*index)
            })
            .collect();
        let incomplete_tools: Vec<IncompleteTool> = self
            .blocks
            .iter()
            .filter_map(|(index, block)| {
                let tool = block.tool.as_ref()?;
                if block.content(terminal, self.native_terminal()).is_some() {
                    return None;
                }
                Some(IncompleteTool {
                    block: *index,
                    name: tool.name.clone(),
                    input_bytes: String::from_utf16_lossy(&tool.arguments).len(),
                    ended: terminal || block.ended,
                })
            })
            .collect();
        let indexed_content: BTreeMap<_, _> = self
            .blocks
            .iter()
            .filter_map(|(index, block)| {
                block
                    .content(terminal, self.native_terminal())
                    .map(|content| (*index, content))
            })
            .collect();
        StreamAssembly {
            response: ChatResponse {
                message: ConversationMessage::assistant(
                    self.blocks
                        .values()
                        .flat_map(|block| {
                            block
                                .content(terminal, self.native_terminal())
                                .into_iter()
                                .chain(block.replay_companion.clone())
                        })
                        .collect(),
                ),
                inference: self.inference.clone(),
                response_cache: None,
                web_search: self.web_search.clone(),
                file_search: self.file_search.clone(),
                native_metadata: Vec::new(),
                tool_input_issues: incomplete_tools
                    .iter()
                    .map(|issue| crate::protocol::ToolInputIssue {
                        block_index: issue.block,
                        tool_name: issue.name.clone(),
                        input_bytes: issue.input_bytes,
                        tool_id: self
                            .blocks
                            .get(&issue.block)
                            .and_then(|b| b.tool.as_ref())
                            .map(|tool| tool.id.clone()),
                    })
                    .collect(),
                stop_reason: self
                    .stop_reason
                    .clone()
                    .unwrap_or_else(|| StopReason::Other("stream_interrupted".into())),
                usage: self.usage.clone(),
                native_resume_context_exhausted: None,
                model: self.model.clone(),
                response_id: self.response_id.clone(),
                continuation: None,
                executed_profile: None,
            },
            events: self.events.clone(),
            terminal,
            unfinished_blocks,
            incomplete_tools,
            indexed_content,
        }
    }

    pub fn finish(self) -> Result<StreamAssembly, StreamAssemblyError> {
        let result = self.snapshot();
        if let Some(source) = self.projection_error {
            return Err(StreamAssemblyError {
                source,
                partial: Box::new(result),
            });
        }
        // A length stop is a terminal provider outcome, not a broken transport.
        // Keep incomplete arguments in observations/metadata, never executable
        // content, so the host can apply its output-limit recovery policy.
        if !result.terminal
            || (!result.unfinished_blocks.is_empty()
                && result.response.stop_reason != StopReason::MaxTokens)
        {
            return Err(StreamAssemblyError {
                source: LlmError::StreamInterrupted {
                    message: if result.terminal {
                        "stream ended with an incomplete tool call".into()
                    } else {
                        "stream ended without a terminal event".into()
                    },
                },
                partial: Box::new(result),
            });
        }
        Ok(result)
    }
}

fn extend_unique<T: Clone + PartialEq>(current: &mut Vec<T>, incoming: &[T]) {
    for value in incoming {
        if !current.contains(value) {
            current.push(value.clone());
        }
    }
}

impl ModelStream {
    /// Collect a stream once. Errors carry salvageable output and all decoded
    /// observations; this never retries or decides whether a tool should run.
    pub async fn collect_response(mut self) -> Result<StreamAssembly, StreamAssemblyError> {
        let mut accumulator = StreamAccumulator::new();
        let mut failure = None;
        while let Some(batch) = self.next_batch().await {
            if let Err(source) = accumulator.observe_batch(&batch) {
                failure = Some(source);
                break;
            }
        }
        let mut result = match failure {
            Some(source) => Err(StreamAssemblyError {
                source,
                partial: Box::new(accumulator.snapshot()),
            }),
            None => accumulator.finish(),
        };
        let assembly = match &mut result {
            Ok(value) => value,
            Err(error) => &mut error.partial,
        };
        assembly.response.response_cache = self.response_cache().cloned();
        assembly.response.executed_profile = Some(self.executed_profile().to_owned());
        assembly.response.continuation = self.continuation().cloned();
        assembly.response.set_anthropic_metadata(
            self.anthropic_container().cloned(),
            self.anthropic_usage().cloned(),
        );
        assembly
            .response
            .set_anthropic_stop_details(self.anthropic_stop_details().cloned());
        assembly
            .response
            .set_anthropic_fallback(self.anthropic_fallback().cloned());
        assembly.response.native_resume_context_exhausted = self.native_resume_context_exhausted();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ProtocolFamily, Usage, UsageState, WebCitation};
    use serde_json::json;

    fn payload(value: Value) -> JsonPayload {
        JsonPayload::parse(&value.to_string()).unwrap()
    }

    fn tool(block: usize, fragment: &str) -> StreamEvent {
        StreamEvent::ToolCallDelta {
            block,
            id: format!("tool-{block}").into(),
            provider_id: Some("native-id".into()),
            caller: Some(payload(json!({"type":"code_execution", "future": true}))),
            toolset_name: Some("browser".into()),
            name: "read".into(),
            arguments_fragment: fragment.into(),
        }
    }
    fn end() -> StreamEvent {
        StreamEvent::End {
            stop_reason: StopReason::ToolUse,
            usage: UsageReport::measured(
                Usage {
                    input_tokens: 12,
                    output_tokens: 3,
                    ..Default::default()
                },
                UsageState::Complete,
            ),
            inference: Default::default(),
        }
    }

    fn exact_text(units: &[u16]) -> StreamEvent {
        StreamEvent::TextDeltaJsUtf16 {
            block: 0,
            text: String::from_utf16_lossy(units),
            utf16_code_units: units.to_vec(),
        }
    }

    fn assert_exact_text(block: &ContentBlock, expected: &[u16]) {
        let ContentBlock::TextJsUtf16 {
            text,
            utf16_code_units,
            ..
        } = block
        else {
            panic!("expected exact UTF-16 text, got {block:?}");
        };
        assert_eq!(utf16_code_units, expected);
        assert_eq!(text, &String::from_utf16_lossy(expected));
    }

    #[test]
    fn utf16_text_deltas_match_full_decode_at_every_split_boundary() {
        let cases: &[(&[u16], &str)] = &[
            (&[], ""),
            (&[0xd800], "\u{fffd}"),
            (&[0xdc00], "\u{fffd}"),
            (&[0xd83d, 0xde00], "😀"),
            (&[0xdbff, 0xdfff], "\u{10ffff}"),
            (
                &[0xd800, 0xd800, 0xdc00, 0xdc00],
                "\u{fffd}\u{10000}\u{fffd}",
            ),
            (
                &[0xd800, 0x61, 0xdc00, 0, 0xdbff],
                "\u{fffd}a\u{fffd}\0\u{fffd}",
            ),
        ];
        for &(units, expected) in cases {
            assert_eq!(String::from_utf16_lossy(units), expected);
            for first in 0..=units.len() {
                for second in first..=units.len() {
                    let mut accumulator = StreamAccumulator::new();
                    let mut observed = Vec::new();
                    for fragment in [&units[..first], &units[first..second], &units[second..]] {
                        accumulator.observe(&exact_text(fragment));
                        observed.extend_from_slice(fragment);
                        assert_exact_text(&accumulator.content_at(0).unwrap(), &observed);
                    }
                }
            }
        }
    }

    #[test]
    fn utf16_text_deltas_preserve_mixed_exact_and_scalar_chunks() {
        let mut accumulator = StreamAccumulator::new();
        let prefix = "prefix \u{fffd}";
        accumulator.observe(&StreamEvent::TextDelta {
            block: 0,
            text: prefix.into(),
        });
        let mut expected: Vec<_> = prefix.encode_utf16().collect();
        for event in [
            exact_text(&[0xd83d]),
            StreamEvent::TextDelta {
                block: 0,
                text: String::new(),
            },
            exact_text(&[0xde00]),
            exact_text(&[0xd800]),
            StreamEvent::TextDelta {
                block: 0,
                text: " 😀 ".into(),
            },
            exact_text(&[0xdc00, 0xe9, 0xd83d]),
            exact_text(&[]),
            exact_text(&[0xde00, 0xd800]),
            StreamEvent::TextDelta {
                block: 0,
                text: "suffix".into(),
            },
        ] {
            match &event {
                StreamEvent::TextDelta { text, .. } => expected.extend(text.encode_utf16()),
                StreamEvent::TextDeltaJsUtf16 {
                    utf16_code_units, ..
                } => expected.extend_from_slice(utf16_code_units),
                _ => unreachable!(),
            }
            accumulator.observe(&event);
            assert_exact_text(&accumulator.content_at(0).unwrap(), &expected);
        }
        accumulator.observe(&StreamEvent::BlockEnd { block: 0 });
        accumulator.observe(&end());
        let result = accumulator.finish().unwrap();
        assert_exact_text(&result.indexed_content[&0], &expected);
        assert_exact_text(&result.response.message.content[0], &expected);
    }

    #[test]
    fn utf16_text_deltas_repair_provider_projection_without_changing_snapshots() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&StreamEvent::ProviderContent {
            block: 0,
            protocol: ProtocolFamily::AnthropicMessages,
            value: JsonPayload::parse(
                r#"{"type":"text","text":"\ud83d","citations":[{"native":"citation"}]}"#,
            )
            .unwrap(),
        });
        accumulator.observe(&StreamEvent::ThoughtSignature {
            block: 0,
            signature: "signed".into(),
        });
        let partial = accumulator.snapshot();
        let cloned = partial.clone();
        accumulator.observe(&exact_text(&[0xde00]));
        accumulator.observe(&StreamEvent::TextDelta {
            block: 0,
            text: " done".into(),
        });
        accumulator.observe(&StreamEvent::BlockEnd { block: 0 });
        accumulator.observe(&end());
        let result = accumulator.finish().unwrap();
        let expected: Vec<_> = "😀 done".encode_utf16().collect();
        assert_exact_text(&result.indexed_content[&0], &expected);
        assert_exact_text(&result.response.message.content[0], &expected);
        assert_exact_text(&partial.indexed_content[&0], &[0xd83d]);
        assert_exact_text(&cloned.response.message.content[0], &[0xd83d]);
        assert!(matches!(
            &result.indexed_content[&0],
            ContentBlock::TextJsUtf16 {
                citations: Some(Some(citations)),
                thought_signature: Some(signature),
                ..
            } if citations.as_json_source() == r#"[{"native":"citation"}]"# && signature == "signed"
        ));
    }

    #[test]
    fn utf16_text_deltas_reuse_display_capacity_across_many_small_chunks() {
        const CHUNKS: usize = 4096;
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&exact_text(&[0xd800]));
        let display = accumulator
            .blocks
            .get_mut(&0)
            .unwrap()
            .text
            .as_mut()
            .unwrap();
        display.reserve(CHUNKS);
        let display_buffer = display.as_ptr();
        let scalar = StreamEvent::TextDelta {
            block: 0,
            text: "a".into(),
        };
        let exact = exact_text(&[0x61]);
        for chunk in 0..CHUNKS {
            accumulator.observe(if chunk % 2 == 0 { &scalar } else { &exact });
            assert_eq!(
                accumulator.blocks[&0].text.as_ref().unwrap().as_ptr(),
                display_buffer,
                "appending within reserved capacity must not replace the display buffer"
            );
        }
        let mut expected = vec![0xd800];
        expected.extend(std::iter::repeat_n(0x61, CHUNKS));
        assert_exact_text(&accumulator.content_at(0).unwrap(), &expected);
    }

    #[test]
    fn interrupted_arguments_preserve_complete_tools_without_fabricating_calls() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&tool(2, "{\"path\":"));
        accumulator.observe(&tool(2, "\"ok\"}"));
        accumulator.observe(&StreamEvent::ThoughtSignature {
            block: 2,
            signature: "signed".into(),
        });
        accumulator.observe(&StreamEvent::BlockEnd { block: 2 });
        accumulator.observe(&tool(8, "{\"path\":"));
        let error = accumulator.finish().unwrap_err();
        assert_eq!(error.partial.unfinished_blocks, vec![8]);
        assert_eq!(error.partial.response.message.content.len(), 1);
        assert!(
            matches!(&error.partial.indexed_content[&2], ContentBlock::ToolUse {
            input, caller: Some(caller), toolset_name: Some(toolset), provider_id: Some(id), thought_signature: Some(signature), ..
        } if input.as_json_source() == r#"{"path":"ok"}"# && caller.raw_source_at("/future").as_deref() == Some("true") && toolset == "browser" && id == "native-id" && signature == "signed")
        );
        assert!(
            matches!(error.partial.events.last(), Some(StreamEvent::ToolCallDelta { arguments_fragment, .. }) if arguments_fragment.utf16_code_units() == "{\"path\":".encode_utf16().collect::<Vec<_>>().as_slice())
        );
    }

    #[test]
    fn later_tool_identity_is_kept_when_opening_delta_omits_it() {
        let mut accumulator = StreamAccumulator::new();
        let mut opening = tool(0, "{");
        if let StreamEvent::ToolCallDelta { id, name, .. } = &mut opening {
            *id = "".into();
            name.clear();
        }
        accumulator.observe(&opening);
        accumulator.observe(&tool(0, "}"));
        assert!(
            matches!(accumulator.tool_progress(0), Some(StreamEvent::ToolCallDelta { id, name, arguments_fragment, .. }) if id.as_str() == "tool-0" && name == "read" && arguments_fragment.utf16_code_units() == "{}".encode_utf16().collect::<Vec<_>>().as_slice())
        );
        accumulator.observe(&end());
        assert!(
            matches!(&accumulator.finish().unwrap().indexed_content[&0], ContentBlock::ToolUse { id, name, .. } if id.as_str() == "tool-0" && name == "read")
        );
    }

    #[test]
    fn empty_arguments_require_explicit_completion() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&tool(1, ""));
        assert!(accumulator.snapshot().indexed_content.is_empty());
        accumulator.observe(&StreamEvent::BlockEnd { block: 1 });
        assert!(
            matches!(&accumulator.snapshot().indexed_content[&1], ContentBlock::ToolUse { input, .. } if input.as_json_source() == "{}")
        );
    }

    #[test]
    fn connector_native_replay_appears_only_after_close() {
        let mut accumulator = StreamAccumulator::new();
        for payload in [
            json!({"type":"content_block_start","index":7,"content_block":{"type":"connector_text","connector_text":"start","source":"remote"}}),
            json!({"type":"content_block_delta","index":7,"delta":{"type":"connector_text_delta","connector_text":" end"}}),
        ] {
            accumulator.observe(&StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::AnthropicMessages,
                payload: crate::protocol::JsonPayload::from_generated_value(&payload).unwrap(),
            });
        }
        assert!(accumulator.snapshot().indexed_content.is_empty());
        accumulator.observe(&StreamEvent::ProviderEvent {
            protocol: ProtocolFamily::AnthropicMessages,
            payload: payload(json!({"type":"content_block_stop","index":7})),
        });
        assert!(
            matches!(&accumulator.snapshot().indexed_content[&7], ContentBlock::ProviderContent { value, .. } if value.string_at("/connector_text").as_deref() == Some("start end") && value.string_at("/source").as_deref() == Some("remote"))
        );
    }

    #[test]
    fn valid_unfinished_json_is_still_not_a_completed_tool() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&tool(0, "{}"));
        assert!(accumulator.snapshot().response.message.content.is_empty());
        accumulator.observe(&end());
        assert_eq!(
            accumulator.finish().unwrap().response.message.content.len(),
            1
        );
    }

    #[test]
    fn terminal_malformed_tool_is_an_error_with_partial_content() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&tool(0, "{"));
        accumulator.observe(&StreamEvent::BlockEnd { block: 0 });
        accumulator.observe(&end());
        let error = accumulator.finish().unwrap_err();
        assert!(error.partial.terminal);
        assert!(error.partial.response.message.content.is_empty());
        assert_eq!(error.partial.response.usage.state, UsageState::Complete);
    }

    #[test]
    fn length_stop_preserves_incomplete_tool_metadata_without_executable_input() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&tool(0, "{\"content\":\"unfinished HTML"));
        accumulator.observe(&StreamEvent::BlockEnd { block: 0 });
        let mut terminal = end();
        if let StreamEvent::End { stop_reason, .. } = &mut terminal {
            *stop_reason = StopReason::MaxTokens;
        }
        accumulator.observe(&terminal);
        let result = accumulator.finish().unwrap();
        assert_eq!(result.response.stop_reason, StopReason::MaxTokens);
        assert!(result.response.message.content.is_empty());
        assert_eq!(result.incomplete_tools.len(), 1);
        assert_eq!(result.unfinished_blocks, vec![0]);
        assert_eq!(result.response.usage.state, UsageState::Complete);
    }

    #[test]
    fn citations_only_interruption_keeps_native_metadata_and_usage() {
        let mut accumulator = StreamAccumulator::new();
        let search = StreamEvent::WebSearch {
            result: WebSearchResult {
                citations: vec![WebCitation {
                    url: "https://example.com".into(),
                    title: None,
                }],
                metadata: payload(json!({"annotations":[{"offset":7}],"future":{"a":1}})),
            },
        };
        accumulator.observe(&search);
        accumulator.observe(&StreamEvent::WebSearch {
            result: WebSearchResult {
                citations: vec![],
                metadata: payload(json!({"annotations":[{"offset":9}],"future":{"b":2}})),
            },
        });
        accumulator.observe(&StreamEvent::NativeDelta {
            block: 1,
            protocol: ProtocolFamily::AnthropicMessages,
            delta: payload(json!({"citation":"native"})),
        });
        let usage = UsageReport::measured(
            Usage {
                input_tokens: 9,
                ..Default::default()
            },
            UsageState::Partial,
        );
        accumulator
            .observe_batch(&StreamBatch {
                events: vec![],
                usage: usage.clone(),
                inference: Default::default(),
                finished: false,
            })
            .unwrap();
        let partial = accumulator.finish().unwrap_err().partial;
        assert!(partial.response.message.content.is_empty());
        let search = partial.response.web_search.unwrap();
        assert_eq!(search.citations.len(), 1);
        let annotations = search.metadata.raw_source_at("/annotations").unwrap();
        let future = search.metadata.raw_source_at("/future").unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&annotations).unwrap(),
            json!([{"offset":7},{"offset":9}])
        );
        assert_eq!(
            serde_json::from_str::<Value>(&future).unwrap(),
            json!({"a":1,"b":2})
        );
        assert_eq!(partial.response.usage, usage);
        assert_eq!(partial.events.len(), 3);
    }

    #[test]
    fn native_reasoning_replaces_display_and_terminal_usage_remains_exact() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.observe(&StreamEvent::ReasoningDelta {
            block: 3,
            text: "display".into(),
        });
        accumulator.observe(&StreamEvent::ProviderContent {
            block: 3,
            protocol: ProtocolFamily::OpenAiResponses,
            value: payload(json!({"type":"reasoning","encrypted_content":"opaque"})),
        });
        accumulator.observe(&StreamEvent::Start {
            model: "observed".into(),
            response_id: Some("r".into()),
        });
        accumulator.observe(&end());
        let result = accumulator.finish().unwrap();
        assert_eq!(result.response.model, "observed");
        assert_eq!(result.response.usage.state, UsageState::Complete);
        assert_eq!(result.response.usage.usage.unwrap().input_tokens, 12);
        assert_eq!(result.response.message.content.len(), 1);
        assert!(matches!(
            result.response.message.content[0],
            ContentBlock::ProviderContent { .. }
        ));
    }

    #[test]
    fn error_batch_keeps_successful_observations_and_last_accounting() {
        let mut accumulator = StreamAccumulator::new();
        let usage = UsageReport::measured(
            Usage {
                output_tokens: 2,
                ..Default::default()
            },
            UsageState::Invalid,
        );
        let result = accumulator.observe_batch(&StreamBatch {
            events: vec![
                Ok(StreamEvent::TextDelta {
                    block: 0,
                    text: "partial".into(),
                }),
                Err(LlmError::StreamInterrupted {
                    message: "disconnected".into(),
                }),
            ],
            usage: usage.clone(),
            inference: Default::default(),
            finished: true,
        });
        assert!(result.is_err());
        assert_eq!(accumulator.snapshot().response.usage, usage);
        assert_eq!(accumulator.snapshot().indexed_content.len(), 1);
        assert!(!accumulator.snapshot().terminal);
    }
}
