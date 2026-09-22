//! The SSE stream decoder.
//!
//! Ported from the previous project's `OpenAiStreamDecoder`. Block indices are
//! assigned here and only ever move forward, which is what §15 means by "a
//! provider with no native index gets a synthesised monotonic one": a consumer
//! can tell one text block from the next without the codec replaying anything.

use super::decode;
use crate::client::usage;
use crate::codecs::StreamDecoder;
use lingxi_agent_api::protocol::{LlmError, StopReason, StreamEvent, ToolUseId, Usage};
use serde_json::Value;

#[derive(Debug, Default)]
pub struct OpenAiStreamDecoder {
    started: bool,
    next_block: usize,
    text_block: Option<usize>,
    reasoning_block: Option<usize>,
    /// Wire index → (block, id, name). A provider streams a tool call's
    /// arguments in fragments keyed by its own index, not by the call id, and
    /// the id only arrives on the first fragment.
    tools: Vec<ToolStream>,
    /// The provider's usage object as sent. Kept raw so the report can be
    /// checked for self-consistency: the buckets we publish are derived by
    /// subtraction, and a saturated subtraction must not read as a measurement.
    ///
    /// This wire restates the whole object each time it reports, so the latest
    /// one wins outright; only the Anthropic wire splits its report across
    /// frames and needs a field-by-field fold.
    usage_raw: Option<Value>,
    stop: Option<StopReason>,
    done: bool,
}

#[derive(Debug)]
struct ToolStream {
    wire_index: u64,
    block: usize,
    id: ToolUseId,
    name: String,
}

impl StreamDecoder for OpenAiStreamDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        let text = std::str::from_utf8(frame).map_err(|_| LlmError::InvalidRequest {
            message: "OpenAI stream frame is not valid UTF-8".to_owned(),
        })?;
        let data = text.trim();
        let mut out = Vec::new();

        // The sentinel is not JSON, and arrives before the connection closes.
        if data == "[DONE]" {
            self.finish_into(&mut out);
            return Ok(out);
        }
        if data.is_empty() {
            return Ok(out);
        }

        let root: Value = serde_json::from_str(data).map_err(|_| LlmError::InvalidRequest {
            message: "OpenAI stream frame is not valid JSON".to_owned(),
        })?;

        if !self.started {
            self.started = true;
            out.push(StreamEvent::Start {
                model: root
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                response_id: None,
            });
        }

        // Usage may arrive on its own frame after the last choice, so it is
        // recorded whenever seen rather than read at the end.
        if let Some(u) = root.get("usage").filter(|v| !v.is_null()) {
            self.usage_raw = Some(u.clone());
        }

        let Some(choice) = root
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            return Ok(out);
        };

        if let Some(delta) = choice.get("delta") {
            if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
                let block = *self.reasoning_block.get_or_insert_with(|| {
                    let b = self.next_block;
                    self.next_block += 1;
                    b
                });
                out.push(StreamEvent::ReasoningDelta {
                    block,
                    text: r.to_owned(),
                });
            }
            if let Some(t) = delta.get("content").and_then(Value::as_str) {
                if !t.is_empty() {
                    let block = *self.text_block.get_or_insert_with(|| {
                        let b = self.next_block;
                        self.next_block += 1;
                        b
                    });
                    out.push(StreamEvent::TextDelta {
                        block,
                        text: t.to_owned(),
                    });
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.tool_fragment(call, &mut out);
                }
            }
        }

        if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
            self.stop = Some(decode::stop_reason(Some(f)));
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        let mut out = Vec::new();
        self.finish_into(&mut out);
        Ok(out)
    }

    fn observed_usage(&self) -> Option<Usage> {
        self.usage_raw.as_ref().map(decode::usage)
    }

    fn usage_is_complete(&self) -> bool {
        self.usage_raw
            .as_ref()
            .is_some_and(|raw| usage::is_complete(raw, &usage::OPENAI_CHAT))
    }

    fn set_provider_metadata(&mut self, _meta: Value) {}
}

impl OpenAiStreamDecoder {
    fn tool_fragment(&mut self, call: &Value, out: &mut Vec<StreamEvent>) {
        let wire_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function").unwrap_or(&Value::Null);
        let id = call.get("id").and_then(Value::as_str);
        let name = function.get("name").and_then(Value::as_str);

        if !self.tools.iter().any(|t| t.wire_index == wire_index) {
            let block = self.next_block;
            self.next_block += 1;
            self.tools.push(ToolStream {
                wire_index,
                block,
                id: ToolUseId::new(id.unwrap_or_default()),
                name: name.unwrap_or_default().to_owned(),
            });
        }
        let Some(slot) = self.tools.iter_mut().find(|t| t.wire_index == wire_index) else {
            return;
        };
        // Later fragments may still be the first to carry the id or the name.
        if let Some(id) = id {
            if slot.id.as_str().is_empty() {
                slot.id = ToolUseId::new(id);
            }
        }
        if let Some(name) = name {
            if slot.name.is_empty() {
                slot.name = name.to_owned();
            }
        }
        let fragment = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default();
        out.push(StreamEvent::ToolCallDelta {
            block: slot.block,
            id: slot.id.clone(),
            name: slot.name.clone(),
            arguments_fragment: fragment.to_owned(),
        });
    }

    /// `[DONE]` and the end of the byte stream are two ways to reach the same
    /// place, and a provider may do both — so the terminal event is emitted
    /// once.
    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        out.push(StreamEvent::End {
            stop_reason: self.stop.clone().unwrap_or(StopReason::EndTurn),
            usage: self
                .usage_raw
                .as_ref()
                .map(decode::usage)
                .unwrap_or_default(),
        });
    }
}
