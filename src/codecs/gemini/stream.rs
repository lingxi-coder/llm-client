//! The `streamGenerateContent` SSE decoder.
//!
//! Each frame is a whole `GenerateContentResponse`, not a delta of one, so the
//! block indices are assigned here the way the OpenAI decoder does it.

use super::decode;
use crate::codecs::usage;
use crate::codecs::web_search_decode::{self, SearchStream};
use crate::codecs::EventDecoder;
use crate::protocol::{LlmError, StopReason, StreamEvent};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct GeminiStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    started: bool,
    next_block: usize,
    text_block: Option<usize>,
    reasoning_block: Option<usize>,
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
    prompt_blocked: bool,
    used_call_ids: HashSet<String>,
    done: bool,
    search: SearchStream,
}

impl EventDecoder for GeminiStreamDecoder {
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

        if let Some(u) = root.get("usageMetadata").filter(|v| !v.is_null()) {
            self.usage_raw = Some(u.clone());
        }

        if root.get("error").is_some_and(|error| !error.is_null()) {
            return Err(decode::classify_error(500, &root, None));
        }

        let mut out = Vec::new();
        self.inference.observe(&root, &mut out);
        if !self.started {
            self.started = true;
            out.push(StreamEvent::Start {
                model: root
                    .get("modelVersion")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                response_id: None,
            });
        }
        if let Some(u) = root.get("usageMetadata") {
            self.usage_raw = Some(u.clone());
        }

        if decode::prompt_feedback_is_blocking(&root) {
            self.prompt_blocked = true;
            self.stop = Some(StopReason::Refusal);
        }

        let Some(candidate) = root
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            return Ok(out);
        };

        self.search
            .emit(web_search_decode::gemini(candidate), &mut out);

        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        self.used_call_ids.extend(decode::provider_call_ids(parts));
        let mut saw_text_part_in_frame = false;
        let mut saw_reasoning_part_in_frame = false;
        for part in parts {
            let signature = part
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if let Some(call) = part.get("functionCall") {
                self.saw_tool_call = true;
                self.text_block = None;
                self.reasoning_block = None;
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let block = self.next_block;
                self.next_block += 1;
                let (id, provider_id) = decode::call_id(call, &mut self.used_call_ids);
                // Arguments arrive whole here rather than in fragments, so the
                // one delta carries the complete object.
                out.push(StreamEvent::ToolCallDelta {
                    block,
                    id,
                    provider_id,
                    name,
                    arguments_fragment: call
                        .get("args")
                        .cloned()
                        .unwrap_or(Value::Null)
                        .to_string(),
                });
                if let Some(signature) = signature {
                    out.push(StreamEvent::ThoughtSignature { block, signature });
                }
                continue;
            }
            let Some(text) = part.get("text").and_then(Value::as_str) else {
                continue;
            };
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                self.text_block = None;
                if saw_reasoning_part_in_frame {
                    self.reasoning_block = None;
                }
                saw_reasoning_part_in_frame = true;
                let block = *self.reasoning_block.get_or_insert_with(|| {
                    let b = self.next_block;
                    self.next_block += 1;
                    b
                });
                out.push(StreamEvent::ReasoningDelta {
                    block,
                    text: text.to_owned(),
                });
                if let Some(signature) = signature {
                    out.push(StreamEvent::ThoughtSignature { block, signature });
                    self.reasoning_block = None;
                }
            } else {
                self.reasoning_block = None;
                if saw_text_part_in_frame {
                    self.text_block = None;
                }
                saw_text_part_in_frame = true;
                let block = *self.text_block.get_or_insert_with(|| {
                    let b = self.next_block;
                    self.next_block += 1;
                    b
                });
                out.push(StreamEvent::TextDelta {
                    block,
                    text: text.to_owned(),
                });
                if let Some(signature) = signature {
                    out.push(StreamEvent::ThoughtSignature { block, signature });
                    self.text_block = None;
                }
            }
        }

        if let Some(f) = candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .filter(|_| !self.prompt_blocked)
        {
            self.stop = Some(decode::stop_reason(Some(f)));
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
        if !self.done {
            self.done = true;
            out.push(StreamEvent::End {
                stop_reason: decode::with_tool_stop_reason(
                    self.stop.clone().unwrap_or(StopReason::EndTurn),
                    self.saw_tool_call,
                ),
                usage: self.usage_report(),
                inference: self.inference.report.clone(),
            });
        }
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
            &usage::GEMINI,
            decode::usage,
            self.done,
        )
    }
}

impl GeminiStreamDecoder {
    pub(crate) fn configured(context: &crate::codecs::CodecContext) -> Self {
        Self {
            inference: crate::codecs::inference::StreamInference::new(context),
            ..Self::default()
        }
    }
}
