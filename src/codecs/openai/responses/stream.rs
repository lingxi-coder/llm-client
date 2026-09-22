//! The Responses API stream decoder.
//!
//! Every event names the output index it belongs to, so block indices come from
//! the wire rather than being synthesised.

use super::decode;
use crate::client::usage;
use crate::codecs::StreamDecoder;
use lingxi_agent_api::protocol::{LlmError, ResponseId, StopReason, StreamEvent, ToolUseId, Usage};
use serde_json::Value;

#[derive(Debug, Default)]
pub struct ResponsesStreamDecoder {
    started: bool,
    /// output index → (call id, name), learned from `output_item.added`.
    calls: Vec<(usize, ToolUseId, String)>,
    /// The provider's usage object as sent. Kept raw so the report can be
    /// checked for self-consistency: the buckets we publish are derived by
    /// subtraction, and a saturated subtraction must not read as a measurement.
    ///
    /// This wire restates the whole object each time it reports, so the latest
    /// one wins outright; only the Anthropic wire splits its report across
    /// frames and needs a field-by-field fold.
    usage_raw: Option<Value>,
    stop: Option<StopReason>,
    saw_tool_call: bool,
    done: bool,
}

impl StreamDecoder for ResponsesStreamDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        let text = std::str::from_utf8(frame).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid UTF-8".to_owned(),
        })?;
        let data = text.trim();
        let mut out = Vec::new();

        // This wire ends at `response.completed` and sends no sentinel, but an
        // OpenAI-compatible gateway in front of it may append one.
        if data == "[DONE]" {
            self.finish_into(&mut out);
            return Ok(out);
        }
        if data.is_empty() {
            return Ok(out);
        }
        let root: Value = serde_json::from_str(data).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid JSON".to_owned(),
        })?;

        let index = |r: &Value| {
            r.get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize
        };
        let delta = |r: &Value| {
            r.get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };

        match root.get("type").and_then(Value::as_str) {
            Some("response.created") => {
                if !self.started {
                    self.started = true;
                    out.push(StreamEvent::Start {
                        model: root
                            .get("response")
                            .and_then(|r| r.get("model"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        response_id: root
                            .get("response")
                            .and_then(|r| r.get("id"))
                            .and_then(Value::as_str)
                            .map(ResponseId::new),
                    });
                }
            }
            Some("response.output_item.added") => {
                let item = root.get("item").unwrap_or(&Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    self.saw_tool_call = true;
                    let id = ToolUseId::new(
                        item.get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    );
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    // Only this event names the call; the argument deltas that
                    // follow carry the index alone.
                    self.calls.push((index(&root), id.clone(), name.clone()));
                    out.push(StreamEvent::ToolCallDelta {
                        block: index(&root),
                        id,
                        name,
                        arguments_fragment: String::new(),
                    });
                }
            }
            Some("response.output_text.delta") => out.push(StreamEvent::TextDelta {
                block: index(&root),
                text: delta(&root),
            }),
            Some("response.reasoning_text.delta" | "response.reasoning_summary_text.delta") => out
                .push(StreamEvent::ReasoningDelta {
                    block: index(&root),
                    text: delta(&root),
                }),
            Some("response.function_call_arguments.delta") => {
                let i = index(&root);
                if let Some((_, id, name)) = self.calls.iter().find(|(c, _, _)| *c == i) {
                    out.push(StreamEvent::ToolCallDelta {
                        block: i,
                        id: id.clone(),
                        name: name.clone(),
                        arguments_fragment: delta(&root),
                    });
                }
            }
            Some("response.completed" | "response.incomplete") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                if let Some(u) = response.get("usage") {
                    self.usage_raw = Some(u.clone());
                }
                self.stop = Some(decode::stop_reason(response));
                self.finish_into(&mut out);
            }
            Some("response.failed") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                return Err(decode::classify_error(
                    500,
                    response.get("error").map_or(&Value::Null, |e| {
                        // The failure payload is the Chat error envelope one
                        // level down.
                        e
                    }),
                    None,
                ));
            }
            Some("error") => return Err(decode::classify_error(500, &root, None)),
            // Unknown types (`response.in_progress`, `…output_text.done`, and
            // whatever is added next) are ignored.
            _ => {}
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
            .is_some_and(|raw| usage::is_complete(raw, &usage::OPENAI_RESPONSES))
    }

    fn set_provider_metadata(&mut self, _meta: Value) {}
}

impl ResponsesStreamDecoder {
    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        out.push(StreamEvent::End {
            stop_reason: if self.saw_tool_call {
                StopReason::ToolUse
            } else {
                self.stop.clone().unwrap_or(StopReason::EndTurn)
            },
            usage: self
                .usage_raw
                .as_ref()
                .map(decode::usage)
                .unwrap_or_default(),
        });
    }
}
