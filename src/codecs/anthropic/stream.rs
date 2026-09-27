//! The Messages API stream decoder.
//!
//! This wire names its own block indices, so unlike the OpenAI decoder there is
//! nothing to synthesise: `content_block_start` carries the index and every
//! delta refers back to it.
//!
//! WebSearch metadata is a convenience projection. Native and unknown content
//! frames are also emitted as ProviderEvent observations, while complete opaque
//! blocks are delayed until block_stop before they become replayable content.

use super::decode;
use crate::codecs::usage;
use crate::codecs::web_search_decode;
use crate::codecs::EventDecoder;
use crate::protocol::{ContentBlock, LlmError, ProtocolFamily, StopReason, StreamEvent, ToolUseId};
use serde_json::Value;

/// Pending native blocks and cited text are held only until content_block_stop.
/// This aggregate cap also bounds a response that opens many blocks without
/// closing them; the SSE parser separately limits each individual frame.
const MAX_PENDING_NATIVE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
struct PendingNativeBlock {
    value: Value,
    assembles_input_json: bool,
    input_json: String,
    input_json_seen: bool,
    incomplete: bool,
    retained_bytes: usize,
}

#[derive(Debug)]
struct PendingCitedTextBlock {
    value: Value,
    text: String,
    citations: Vec<Value>,
    has_citations: bool,
    incomplete: bool,
    observing: bool,
    buffered_frames: Vec<Value>,
    buffered_frame_bytes: usize,
    retained_bytes: usize,
}

#[derive(Debug)]
struct PendingToolCall {
    index: usize,
    id: ToolUseId,
    name: String,
    caller: Option<Value>,
    toolset_name: Option<String>,
}

#[derive(Debug, Default)]
pub struct AnthropicStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    /// Block index → the tool call opened there, so a later `input_json_delta`
    /// can name the call it belongs to.
    tools: Vec<PendingToolCall>,
    /// Native output blocks are kept until block_stop so a later unknown delta
    /// cannot leave an already-published, falsely complete replay block.
    pending_native: Vec<(usize, PendingNativeBlock)>,
    pending_native_bytes: usize,
    /// A client tool call with an unknown argument delta must not be completed
    /// as a successful tool-use response.
    incomplete_tool_blocks: Vec<usize>,
    /// Text is always assembled while a block is open because citations may
    /// first appear in a later citations_delta.
    pending_cited_text: Vec<(usize, PendingCitedTextBlock)>,
    /// The provider's usage object, folded across the frames that report it.
    ///
    /// Kept raw rather than decoded-and-merged: this wire reports usage twice
    /// and the second report restates only some counters, so the merge has to
    /// know which fields a frame actually mentioned. Decoding first throws that
    /// away — an omitted counter and one stated as zero both arrive as `0`.
    usage_raw: Option<Value>,
    /// `message_start` usage is provisional even when it includes an output
    /// count. Only a numeric output count in a `message_delta` is final usage.
    final_output_count_reported: bool,
    retain_openrouter_container: bool,
    retain_anthropic_container: bool,
    stop: Option<StopReason>,
    done: bool,
    search_blocks: Vec<usize>,
}

impl EventDecoder for AnthropicStreamDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        let text = std::str::from_utf8(frame).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid UTF-8".to_owned(),
        })?;
        let data = text.trim();
        if data.is_empty() {
            return Ok(vec![]);
        }
        let root: Value = serde_json::from_str(data).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid JSON".to_owned(),
        })?;

        let mut out = Vec::new();
        self.inference.observe(&root, &mut out);
        match root.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let message = root.get("message").unwrap_or(&Value::Null);
                if self.retain_anthropic_container
                    && (message.get("container").is_some() || message.get("usage").is_some())
                    || self.retain_openrouter_container
                        && message
                            .get("container")
                            .is_some_and(|container| !container.is_null())
                {
                    self.push_provider_event(&root, &mut out);
                }
                if let Some(u) = message.get("usage") {
                    self.fold_usage(u);
                    if let Some(result) = web_search_decode::with_usage(None, Some(u)) {
                        out.push(StreamEvent::WebSearch { result });
                    }
                }
                out.push(StreamEvent::Start {
                    model: message
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    response_id: None,
                });
            }
            Some("content_block_start") => self.block_start(&root, data.as_bytes(), &mut out)?,
            Some("content_block_delta") => self.block_delta(&root, data.as_bytes(), &mut out)?,
            Some("content_block_stop") => self.block_stop(&root, data.as_bytes(), &mut out)?,
            Some("message_delta") => {
                if self.retain_anthropic_container
                    && (crate::codecs::anthropic_code_execution::stream_container(&root).is_some()
                        || crate::codecs::anthropic_code_execution::stream_usage(&root).is_some())
                {
                    self.push_provider_event(&root, &mut out);
                }
                if let Some(s) = root
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.stop = Some(decode::stop_reason(Some(s)));
                }
                // Usage becomes final here. Every counter this frame states
                // replaces the seed's, including a zero: a turn that read from
                // the cache without writing to it reports
                // `cache_creation_input_tokens: 0` at the end, and keeping the
                // seed's non-zero value would bill a write that did not happen.
                // Counters it does not mention keep what the seed said.
                if let Some(u) = root.get("usage") {
                    self.final_output_count_reported |=
                        usage::counter(u, "output_tokens").is_some();
                    self.fold_usage(u);
                    if let Some(result) = web_search_decode::with_usage(None, Some(u)) {
                        out.push(StreamEvent::WebSearch { result });
                    }
                }
            }
            Some("message_stop") => {
                if !self.incomplete_tool_blocks.is_empty() {
                    return Err(LlmError::StreamInterrupted {
                        message: "Anthropic tool input contained an unrecognized delta; refusing to complete the partial tool call".to_owned(),
                    });
                }
                if !self.pending_native.is_empty() || !self.pending_cited_text.is_empty() {
                    return Err(LlmError::StreamInterrupted {
                        message: "Anthropic stream ended with an open content block".to_owned(),
                    });
                }
                self.finish_into(&mut out)
            }
            // A keep-alive carries nothing.
            Some("ping") => {}
            Some("error") => return Err(decode::classify_error(500, &root, None)),
            // Tolerating an unknown type is part of this wire's contract: the
            // provider adds event types and a client must not break on them.
            _ => out.push(StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::AnthropicMessages,
                payload: root.clone(),
            }),
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if !self.pending_native.is_empty()
            || !self.pending_cited_text.is_empty()
            || !self.incomplete_tool_blocks.is_empty()
        {
            return Err(LlmError::StreamInterrupted {
                message: "provider stream ended with an incomplete content block or tool input"
                    .to_owned(),
            });
        }
        if !self.done && self.stop.is_none() {
            return Err(LlmError::StreamInterrupted {
                message: "provider stream ended before a terminal event".to_owned(),
            });
        }
        let mut out = Vec::new();
        self.finish_into(&mut out);
        Ok(out)
    }

    fn inference_report(&self) -> crate::protocol::InferenceReport {
        self.inference.report.clone()
    }
    fn set_response_headers(&mut self, headers: &[(String, String)]) {
        self.inference.headers(headers);
    }
    fn usage_report(&self) -> crate::protocol::UsageReport {
        usage::report(
            self.usage_raw.as_ref(),
            &usage::ANTHROPIC,
            decode::usage,
            self.done && self.final_output_count_reported,
        )
    }
}

impl AnthropicStreamDecoder {
    /// Fold one frame's usage object over what earlier frames reported.
    fn fold_usage(&mut self, reported: &Value) {
        match self.usage_raw.as_mut() {
            Some(seed) => usage::fold(seed, reported),
            None => self.usage_raw = Some(reported.clone()),
        }
    }

    fn start_cited_text_block(
        &mut self,
        index: usize,
        block: &Value,
        root: &Value,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        if self
            .pending_cited_text
            .iter()
            .any(|(pending_index, _)| *pending_index == index)
        {
            return Err(LlmError::StreamInterrupted {
                message: format!("Anthropic reused open text block index {index}"),
            });
        }

        let initial_text = block.get("text").and_then(Value::as_str);
        let text = initial_text.unwrap_or_default().to_owned();
        let citations_field = block.get("citations");
        let citations = citations_field
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let has_citations = !citations.is_empty();
        let citations_well_formed = match citations_field {
            None | Some(Value::Null) => true,
            Some(Value::Array(_)) => citations.iter().all(Value::is_object),
            Some(_) => false,
        };
        let incomplete = initial_text.is_none() || !citations_well_formed;
        let observing = has_citations || incomplete;
        let mut retained_bytes = serialized_value_size(block)?
            .checked_add(text.len())
            .ok_or_else(native_block_limit_error)?;
        for citation in &citations {
            retained_bytes = retained_bytes
                .checked_add(serialized_value_size(citation)?)
                .ok_or_else(native_block_limit_error)?;
        }
        let buffered_frame_bytes = if observing {
            0
        } else {
            let bytes = serialized_value_size(root)?;
            retained_bytes = retained_bytes
                .checked_add(bytes)
                .ok_or_else(native_block_limit_error)?;
            bytes
        };
        self.reserve_native_bytes(retained_bytes)?;
        self.pending_cited_text.push((
            index,
            PendingCitedTextBlock {
                value: block.clone(),
                text,
                citations,
                has_citations,
                incomplete,
                observing,
                buffered_frames: if observing {
                    Vec::new()
                } else {
                    vec![root.clone()]
                },
                buffered_frame_bytes,
                retained_bytes,
            },
        ));
        if observing {
            self.push_provider_event(root, out);
        }
        if let Some(text) = initial_text.filter(|text| !text.is_empty()) {
            out.push(StreamEvent::TextDelta {
                block: index,
                text: text.to_owned(),
            });
        }
        Ok(())
    }

    fn block_start(
        &mut self,
        root: &Value,
        _frame: &[u8],
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let block = root.get("content_block").unwrap_or(&Value::Null);
        let block_type = block.get("type").and_then(Value::as_str);
        if block_type == Some("text") {
            self.start_cited_text_block(index, block, root, out)?;
        } else if is_native_output_block(block_type) {
            if self
                .pending_native
                .iter()
                .any(|(pending_index, _)| *pending_index == index)
            {
                return Err(LlmError::StreamInterrupted {
                    message: format!("Anthropic reused open content block index {index}"),
                });
            }
            let retained_bytes = serde_json::to_vec(block)
                .map_err(|_| LlmError::StreamInterrupted {
                    message: "Anthropic native content block could not be retained".to_owned(),
                })?
                .len();
            self.reserve_native_bytes(retained_bytes)?;
            self.pending_native.push((
                index,
                PendingNativeBlock {
                    value: block.clone(),
                    assembles_input_json: matches!(
                        block_type,
                        Some("server_tool_use" | "mcp_tool_use")
                    ),
                    input_json: String::new(),
                    input_json_seen: false,
                    incomplete: false,
                    retained_bytes,
                },
            ));
            // Preserve this as an event frame immediately. If a future delta
            // makes the block incomplete, callers still retain the complete
            // event boundary without treating it as replayable content.
            self.push_provider_event(root, out);
        }

        if web_search_decode::is_anthropic_search_block(block) {
            self.search_blocks.push(index);
            if let Some(result) = web_search_decode::result(serde_json::json!({"events":[root]})) {
                out.push(StreamEvent::WebSearch { result });
            }
        } else if web_search_decode::anthropic(&serde_json::json!({"content":[block]})).is_some() {
            if let Some(result) = web_search_decode::result(serde_json::json!({"events":[root]})) {
                out.push(StreamEvent::WebSearch { result });
            }
        }
        if block_type == Some("redacted_thinking") {
            if let Some(ContentBlock::RedactedThinking { data }) = decode::decode_block(block) {
                out.push(StreamEvent::RedactedThinking { block: index, data });
            }
        }
        if block_type == Some("tool_use") {
            let id = ToolUseId::new(block.get("id").and_then(Value::as_str).unwrap_or_default());
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let caller = block.get("caller").cloned();
            let toolset_name = block
                .get("toolset_name")
                .and_then(Value::as_str)
                .map(str::to_owned);
            // The opening frame is the only one carrying the id and the name;
            // every argument fragment after it refers to this index alone.
            self.tools.push(PendingToolCall {
                index,
                id: id.clone(),
                name: name.clone(),
                caller: caller.clone(),
                toolset_name: toolset_name.clone(),
            });
            out.push(StreamEvent::ToolCallDelta {
                block: index,
                id,
                provider_id: None,
                caller,
                toolset_name,
                name,
                arguments_fragment: String::new(),
            });
        }
        Ok(())
    }

    fn block_delta(
        &mut self,
        root: &Value,
        _frame: &[u8],
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let delta = root.get("delta").unwrap_or(&Value::Null);
        if self.search_blocks.contains(&index)
            || (delta["type"].as_str() == Some("citations_delta")
                && delta["citation"]["type"].as_str() == Some("web_search_result_location"))
        {
            if let Some(result) = web_search_decode::result(serde_json::json!({"events":[root]})) {
                out.push(StreamEvent::WebSearch { result });
            }
        }

        if let Some(position) = self
            .pending_cited_text
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        {
            return self.cited_text_delta(position, index, root, delta, out);
        }

        if let Some(position) = self
            .pending_native
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        {
            self.push_provider_event(root, out);
            let delta_type = delta.get("type").and_then(Value::as_str);
            let partial_json = delta.get("partial_json").and_then(Value::as_str);
            let can_assemble_input = self.pending_native[position].1.assembles_input_json
                && delta_type == Some("input_json_delta");
            if can_assemble_input {
                if let Some(fragment) = partial_json {
                    self.reserve_native_bytes(fragment.len())?;
                    let pending = &mut self.pending_native[position].1;
                    pending.retained_bytes += fragment.len();
                    pending.input_json.push_str(fragment);
                    pending.input_json_seen = true;
                    return Ok(());
                }
            }
            self.pending_native[position].1.incomplete = true;
            return Ok(());
        }

        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => out.push(StreamEvent::TextDelta {
                block: index,
                text: delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            Some("thinking_delta") => out.push(StreamEvent::ReasoningDelta {
                block: index,
                text: delta
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            // The signature arrives after the thinking it signs, as its own
            // delta. It has to reach the transcript or the next turn cannot
            // replay the block (gate 18).
            Some("signature_delta") => out.push(StreamEvent::ThoughtSignature {
                block: index,
                signature: delta
                    .get("signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            Some("citations_delta") => {
                self.push_provider_event(root, out);
            }
            Some("input_json_delta") => {
                let Some(call) = self.tools.iter().find(|call| call.index == index) else {
                    self.push_provider_event(root, out);
                    return Ok(());
                };
                out.push(StreamEvent::ToolCallDelta {
                    block: index,
                    id: call.id.clone(),
                    provider_id: None,
                    caller: call.caller.clone(),
                    toolset_name: call.toolset_name.clone(),
                    name: call.name.clone(),
                    arguments_fragment: delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
                return Ok(());
            }
            // Unknown delta types are retained as raw events. A client tool
            // call cannot be completed after an unknown input delta.
            _ => {
                if self.tools.iter().any(|call| call.index == index)
                    && !self.incomplete_tool_blocks.contains(&index)
                {
                    self.incomplete_tool_blocks.push(index);
                }
                self.push_provider_event(root, out);
            }
        }
        Ok(())
    }

    fn cited_text_delta(
        &mut self,
        position: usize,
        index: usize,
        root: &Value,
        delta: &Value,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                let Some(text) = delta.get("text").and_then(Value::as_str) else {
                    return self.mark_cited_text_incomplete(position, root, out);
                };
                self.reserve_native_bytes(text.len())?;
                let pending = &mut self.pending_cited_text[position].1;
                pending.retained_bytes += text.len();
                pending.text.push_str(text);
                self.retain_cited_text_frame(position, root, out)?;
                out.push(StreamEvent::TextDelta {
                    block: index,
                    text: text.to_owned(),
                });
            }
            Some("citations_delta") => {
                let Some(citation) = delta.get("citation").filter(|value| value.is_object()) else {
                    return self.mark_cited_text_incomplete(position, root, out);
                };
                let citation_bytes = serialized_value_size(citation)?;
                self.reserve_native_bytes(citation_bytes)?;
                let was_observing = self.pending_cited_text[position].1.observing;
                let pending = &mut self.pending_cited_text[position].1;
                pending.retained_bytes += citation_bytes;
                pending.citations.push(citation.clone());
                pending.has_citations = true;
                self.retain_cited_text_frame(position, root, out)?;
                if !was_observing {
                    self.observe_cited_text_frames(position, out);
                }
            }
            _ => self.mark_cited_text_incomplete(position, root, out)?,
        }
        Ok(())
    }

    fn mark_cited_text_incomplete(
        &mut self,
        position: usize,
        root: &Value,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        self.pending_cited_text[position].1.incomplete = true;
        let was_observing = self.pending_cited_text[position].1.observing;
        self.retain_cited_text_frame(position, root, out)?;
        if !was_observing {
            self.observe_cited_text_frames(position, out);
        }
        Ok(())
    }

    fn retain_cited_text_frame(
        &mut self,
        position: usize,
        root: &Value,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        if self.pending_cited_text[position].1.observing {
            self.push_provider_event(root, out);
            return Ok(());
        }
        let bytes = serialized_value_size(root)?;
        self.reserve_native_bytes(bytes)?;
        let pending = &mut self.pending_cited_text[position].1;
        pending.retained_bytes += bytes;
        pending.buffered_frame_bytes += bytes;
        pending.buffered_frames.push(root.clone());
        Ok(())
    }

    fn observe_cited_text_frames(&mut self, position: usize, out: &mut Vec<StreamEvent>) {
        let (frames, buffered_bytes) = {
            let pending = &mut self.pending_cited_text[position].1;
            pending.observing = true;
            let buffered_bytes = pending.buffered_frame_bytes;
            pending.retained_bytes = pending.retained_bytes.saturating_sub(buffered_bytes);
            pending.buffered_frame_bytes = 0;
            (std::mem::take(&mut pending.buffered_frames), buffered_bytes)
        };
        self.pending_native_bytes = self.pending_native_bytes.saturating_sub(buffered_bytes);
        for frame in frames {
            self.push_provider_event(&frame, out);
        }
    }

    fn block_stop(
        &mut self,
        root: &Value,
        _frame: &[u8],
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        if self
            .pending_native
            .iter()
            .any(|(pending_index, _)| *pending_index == index)
        {
            self.push_provider_event(root, out);
            self.finish_native_block(index, out);
        } else {
            self.finish_cited_text_block(index, root, out);
        }
        if self.search_blocks.contains(&index) {
            if let Some(result) = web_search_decode::result(serde_json::json!({"events":[root]})) {
                out.push(StreamEvent::WebSearch { result });
            }
        }
        Ok(())
    }

    fn finish_native_block(&mut self, index: usize, out: &mut Vec<StreamEvent>) {
        let Some(position) = self
            .pending_native
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        else {
            return;
        };
        let (_, mut pending) = self.pending_native.remove(position);
        self.pending_native_bytes = self
            .pending_native_bytes
            .saturating_sub(pending.retained_bytes);
        if pending.incomplete {
            return;
        }
        if pending.assembles_input_json && pending.input_json_seen {
            let Ok(input) = serde_json::from_str::<Value>(&pending.input_json) else {
                return;
            };
            if !input.is_object() {
                return;
            }
            let Some(object) = pending.value.as_object_mut() else {
                return;
            };
            object.insert("input".to_owned(), input);
        }
        out.push(StreamEvent::ProviderContent {
            block: index,
            protocol: ProtocolFamily::AnthropicMessages,
            value: pending.value,
        });
    }

    fn finish_cited_text_block(&mut self, index: usize, root: &Value, out: &mut Vec<StreamEvent>) {
        let Some(position) = self
            .pending_cited_text
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        else {
            return;
        };
        if self.pending_cited_text[position].1.observing {
            self.push_provider_event(root, out);
        }
        let (_, mut pending) = self.pending_cited_text.remove(position);
        self.pending_native_bytes = self
            .pending_native_bytes
            .saturating_sub(pending.retained_bytes);
        if !pending.has_citations || pending.incomplete {
            return;
        }
        let Some(object) = pending.value.as_object_mut() else {
            return;
        };
        object.insert("text".to_owned(), Value::String(pending.text));
        object.insert("citations".to_owned(), Value::Array(pending.citations));
        out.push(StreamEvent::ProviderContent {
            block: index,
            protocol: ProtocolFamily::AnthropicMessages,
            value: pending.value,
        });
    }

    fn reserve_native_bytes(&mut self, bytes: usize) -> Result<(), LlmError> {
        let Some(next) = self.pending_native_bytes.checked_add(bytes) else {
            return Err(native_block_limit_error());
        };
        if next > MAX_PENDING_NATIVE_BYTES {
            return Err(native_block_limit_error());
        }
        self.pending_native_bytes = next;
        Ok(())
    }

    fn push_provider_event(&self, payload: &Value, out: &mut Vec<StreamEvent>) {
        out.push(StreamEvent::ProviderEvent {
            protocol: ProtocolFamily::AnthropicMessages,
            payload: payload.clone(),
        });
    }

    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        out.push(StreamEvent::End {
            stop_reason: self.stop.clone().unwrap_or(StopReason::EndTurn),
            usage: self.usage_report(),
            inference: self.inference.report.clone(),
        });
    }
}

impl AnthropicStreamDecoder {
    pub(crate) fn configured(context: &crate::codecs::CodecContext) -> Self {
        Self {
            inference: crate::codecs::inference::StreamInference::new(context),
            retain_openrouter_container:
                crate::codecs::openrouter_server_tools::is_official_profile(context.profile()),
            retain_anthropic_container: crate::codecs::anthropic_code_execution::is_official_profile(
                context.profile(),
            )
                || crate::codecs::anthropic_code_execution::supports_execution(context),
            ..Self::default()
        }
    }
}

fn is_native_output_block(block_type: Option<&str>) -> bool {
    !matches!(
        block_type,
        Some("text" | "thinking" | "redacted_thinking" | "tool_use")
    )
}

fn native_block_limit_error() -> LlmError {
    LlmError::StreamInterrupted {
        message: format!(
            "Anthropic pending native content exceeded the {MAX_PENDING_NATIVE_BYTES}-byte aggregate limit"
        ),
    }
}

fn serialized_value_size(value: &Value) -> Result<usize, LlmError> {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .map_err(|_| native_block_limit_error())
}
