//! The Messages API stream decoder.
//!
//! This wire names its own block indices, so unlike the OpenAI decoder there is
//! nothing to synthesise: `content_block_start` carries the index and every
//! delta refers back to it.
//!
//! WebSearch metadata `events` contains untouched start/delta/stop frames with
//! native indices. Append these records in order (including identical argument
//! fragments); combine text deltas and citation deltas by index to reconstruct
//! native ProviderContent blocks for replay.

use super::decode;
use crate::codecs::usage;
use crate::codecs::web_search_decode;
use crate::codecs::EventDecoder;
use crate::protocol::{ContentBlock, LlmError, StopReason, StreamEvent, ToolUseId};
use serde_json::Value;

#[derive(Debug, Default)]
pub struct AnthropicStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    /// Block index → the tool call opened there, so a later `input_json_delta`
    /// can name the call it belongs to.
    tools: Vec<(usize, ToolUseId, String)>,
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
    stop: Option<StopReason>,
    done: bool,
    search_blocks: Vec<usize>,
    native_blocks: std::collections::BTreeMap<usize, (Value, String)>,
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
                    response_id: message
                        .get("id")
                        .and_then(Value::as_str)
                        .map(crate::protocol::ResponseId::new),
                });
            }
            Some("content_block_start") => self.block_start(&root, &mut out),
            Some("content_block_delta") => self.block_delta(&root, &mut out),
            Some("content_block_stop") => {
                let index = root["index"].as_u64().unwrap_or_default() as usize;
                if let Some((mut value, arguments)) = self.native_blocks.remove(&index) {
                    if !arguments.is_empty() {
                        value["input"] = serde_json::from_str(&arguments).map_err(|_| {
                            LlmError::InvalidRequest {
                                message: "provider returned malformed native tool arguments".into(),
                            }
                        })?;
                    }
                    out.push(StreamEvent::ProviderContent {
                        block: index,
                        protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
                        value,
                    });
                }
                out.push(StreamEvent::BlockEnd { block: index });
                if self.search_blocks.contains(&index) {
                    if let Some(result) =
                        web_search_decode::result(serde_json::json!({"events":[root]}))
                    {
                        out.push(StreamEvent::WebSearch { result });
                    }
                }
            }
            Some("message_delta") => {
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
            Some("message_stop") => self.finish_into(&mut out),
            // A keep-alive carries nothing.
            Some("ping") => {}
            Some("error") => return Err(decode::classify_error(500, &root, None)),
            // Tolerating an unknown type is part of this wire's contract: the
            // provider adds event types and a client must not break on them.
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
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

    fn block_start(&mut self, root: &Value, out: &mut Vec<StreamEvent>) {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let block = root.get("content_block").unwrap_or(&Value::Null);
        if block
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                !matches!(kind, "text" | "thinking" | "redacted_thinking" | "tool_use")
            })
        {
            self.native_blocks
                .insert(index, (block.clone(), String::new()));
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
        if block.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
            if let Some(ContentBlock::RedactedThinking { data }) = decode::decode_block(block) {
                out.push(StreamEvent::RedactedThinking { block: index, data });
            }
        }
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            let id = ToolUseId::new(block.get("id").and_then(Value::as_str).unwrap_or_default());
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            // The opening frame is the only one carrying the id and the name;
            // every argument fragment after it refers to this index alone.
            self.tools.push((index, id.clone(), name.clone()));
            out.push(StreamEvent::ToolCallDelta {
                block: index,
                id,
                provider_id: None,
                name,
                arguments_fragment: String::new(),
            });
        }
    }

    fn block_delta(&mut self, root: &Value, out: &mut Vec<StreamEvent>) {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let delta = root.get("delta").unwrap_or(&Value::Null);
        if let Some((value, arguments)) = self.native_blocks.get_mut(&index) {
            match delta["type"].as_str() {
                Some("input_json_delta") => {
                    arguments.push_str(delta["partial_json"].as_str().unwrap_or_default())
                }
                Some("signature_delta") => value["signature"] = delta["signature"].clone(),
                Some("connector_text_delta") => {
                    let mut text = value["connector_text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                    text.push_str(delta["connector_text"].as_str().unwrap_or_default());
                    value["connector_text"] = Value::String(text);
                }
                _ => {}
            }
        }
        if self.search_blocks.contains(&index)
            || (delta["type"].as_str() == Some("citations_delta")
                && delta["citation"]["type"].as_str() == Some("web_search_result_location"))
        {
            if let Some(result) = web_search_decode::result(serde_json::json!({"events":[root]})) {
                out.push(StreamEvent::WebSearch { result });
            }
        }
        if matches!(
            delta["type"].as_str(),
            Some("citations_delta" | "connector_text_delta")
        ) {
            out.push(StreamEvent::NativeDelta {
                block: index,
                protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
                delta: delta.clone(),
            });
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
            Some("signature_delta") if !self.native_blocks.contains_key(&index) => {
                out.push(StreamEvent::ThoughtSignature {
                    block: index,
                    signature: delta
                        .get("signature")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                })
            }
            Some("input_json_delta") => {
                let Some((_, id, name)) = self.tools.iter().find(|(i, _, _)| *i == index) else {
                    return;
                };
                out.push(StreamEvent::ToolCallDelta {
                    block: index,
                    id: id.clone(),
                    provider_id: None,
                    name: name.clone(),
                    arguments_fragment: delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
            // Unknown delta types are ignored for the same reason as unknown
            // event types.
            _ => {}
        }
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
            ..Self::default()
        }
    }
}
