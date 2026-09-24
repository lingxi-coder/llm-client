//! The Responses API stream decoder.
//!
//! Every event names the output index it belongs to, so block indices come from
//! the wire rather than being synthesised.

use super::decode;
use crate::codecs::file_search_decode::FileSearchStream;
use crate::codecs::usage;
use crate::codecs::web_search_decode::{self, SearchStream};
use crate::codecs::EventDecoder;
use crate::protocol::{LlmError, ResponseId, StopReason, StreamEvent, ToolUseId};
use serde_json::Value;

#[derive(Debug, Default)]
pub struct ResponsesStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    started: bool,
    reasoning_items: std::collections::BTreeSet<usize>,
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
    saw_refusal: bool,
    done: bool,
    search: SearchStream,
    file_search: FileSearchStream,
}

impl EventDecoder for ResponsesStreamDecoder {
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

        self.inference.observe(&root, &mut out);
        match root.get("type").and_then(Value::as_str) {
            Some("response.output_text.annotation.added") => {
                if root["annotation"]["type"].as_str() == Some("url_citation") {
                    self.search.emit(web_search_decode::result(serde_json::json!({"annotations":[{
                        "output_index": index(&root), "content_index":root.get("content_index").and_then(Value::as_u64).unwrap_or(0), "annotation":root["annotation"]
                    }]})), &mut out);
                }
            }
            Some("response.output_item.done") => {
                let item = &root["item"];
                self.emit_reasoning(index(&root), item, &mut out);
                self.saw_refusal |= decode::has_refusal(item);
                if item.get("type").and_then(Value::as_str) == Some("file_search_call") {
                    self.file_search
                        .emit(&serde_json::json!({"output":[item]}), &mut out);
                }
                if item["type"].as_str() == Some("web_search_call") {
                    self.search.emit(
                        web_search_decode::result(serde_json::json!({"web_search_calls":[item]})),
                        &mut out,
                    );
                }
            }
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
                self.saw_refusal |= decode::has_refusal(item);
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
                        provider_id: None,
                    });
                }
            }
            Some("response.output_text.delta") => out.push(StreamEvent::TextDelta {
                block: index(&root),
                text: delta(&root),
            }),
            Some("response.refusal.delta" | "response.refusal.done") => {
                self.saw_refusal = true;
            }
            Some("response.content_part.added" | "response.content_part.done") => {
                self.saw_refusal |= decode::has_refusal(root.get("part").unwrap_or(&Value::Null));
            }
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
                        provider_id: None,
                    });
                }
            }
            Some("response.completed" | "response.incomplete") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                if let Some(items) = response.get("output").and_then(Value::as_array) {
                    for (index, item) in items.iter().enumerate() {
                        self.emit_reasoning(index, item, &mut out);
                    }
                }
                self.file_search.emit(response, &mut out);
                self.search.emit(
                    web_search_decode::with_usage(
                        web_search_decode::responses(response),
                        response.get("usage"),
                    ),
                    &mut out,
                );
                if let Some(u) = response.get("usage") {
                    self.usage_raw = Some(u.clone());
                }
                self.saw_refusal |= response
                    .get("output")
                    .and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(decode::has_refusal));
                self.stop = Some(decode::stop_reason(response));
                self.finish_into(&mut out);
            }
            Some("response.failed") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                return Err(decode::classify_error(500, response, None));
            }
            Some("error") => {
                let envelope = if root.get("error").is_some() {
                    root
                } else {
                    serde_json::json!({"error": root})
                };
                return Err(decode::classify_error(500, &envelope, None));
            }
            // Unknown types (`response.in_progress`, `…output_text.done`, and
            // whatever is added next) are ignored.
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if !self.done {
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
            &usage::OPENAI_RESPONSES,
            decode::usage,
            self.done,
        )
    }
}

impl ResponsesStreamDecoder {
    fn emit_reasoning(&mut self, block: usize, item: &Value, out: &mut Vec<StreamEvent>) {
        if item["type"].as_str() == Some("reasoning") && self.reasoning_items.insert(block) {
            out.push(StreamEvent::ProviderContent {
                block,
                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                value: item.clone(),
            });
        }
    }

    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        out.push(StreamEvent::End {
            stop_reason: if self
                .stop
                .as_ref()
                .is_some_and(|reason| *reason != StopReason::EndTurn)
            {
                self.stop.clone().unwrap()
            } else if self.saw_tool_call {
                StopReason::ToolUse
            } else if self.saw_refusal {
                StopReason::Refusal
            } else {
                self.stop.clone().unwrap_or(StopReason::EndTurn)
            },
            usage: self.usage_report(),
            inference: self.inference.report.clone(),
        });
    }
}

impl ResponsesStreamDecoder {
    pub(crate) fn configured(context: &crate::codecs::CodecContext) -> Self {
        Self {
            inference: crate::codecs::inference::StreamInference::new(context),
            ..Self::default()
        }
    }
}
