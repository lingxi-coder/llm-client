//! The `streamGenerateContent` SSE decoder.
//!
//! Each frame is a whole `GenerateContentResponse`, not a delta of one, so the
//! block indices are assigned here the way the OpenAI decoder does it.

use super::decode;
use crate::codecs::usage;
use crate::codecs::web_search_decode::{self, SearchStream};
use crate::codecs::EventDecoder;
use crate::protocol::{
    JsonPayload, JsonPayloadFragment, LlmError, ProtocolFamily, StopReason, StreamEvent,
};
use crate::response_json::{text_delta, FlatJsonKind, ResponseJsonScanner};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct GeminiStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    protocol: Option<ProtocolFamily>,
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
        let scanner =
            ResponseJsonScanner::parse(data.as_bytes(), "stream frame is not valid JSON")?;
        if scanner
            .kind_at("/error")
            .is_some_and(|kind| kind != FlatJsonKind::Null)
        {
            let selected_text = scanner.provider_error_text_at("/error/message", None)?;
            let provider_text = match &selected_text {
                Some(text) => text.prefixed("500 "),
                None => scanner.provider_error_text_or_source_at(
                    "/error/message",
                    Some(500),
                    "/error",
                )?,
            };
            let body = scanner.provider_error_classification_body(
                "/error",
                &["message", "status", "code", "type"],
                selected_text.as_ref(),
            );
            return Err(decode::classify_error(500, &body, None, false)
                .with_provider_error_text(provider_text));
        }
        let root_payload = JsonPayload::parse(data).map_err(|error| LlmError::InvalidRequest {
            message: format!("Gemini stream frame is invalid JSON: {error}"),
        })?;
        let mut opaque_paths = [
            "/candidates/0/groundingMetadata",
            "/candidates/0/urlContextMetadata",
        ]
        .into_iter()
        .filter(|pointer| scanner.raw_source_at(pointer).is_some())
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let mut native_parts = HashSet::new();
        for index in 0..scanner
            .array_len("/candidates/0/content/parts")
            .unwrap_or(0)
        {
            let part = format!("/candidates/0/content/parts/{index}");
            if [
                "executableCode",
                "codeExecutionResult",
                "inlineData",
                "toolCall",
                "toolResponse",
            ]
            .iter()
            .any(|field| scanner.raw_source_at(&format!("{part}/{field}")).is_some())
            {
                opaque_paths.push(part);
                native_parts.insert(index);
                continue;
            }
            let pointer = format!("{part}/functionCall/args");
            if scanner.raw_source_at(&pointer).is_some() {
                opaque_paths.push(pointer);
            }
        }
        let mut response_json = scanner.into_response_json_with_opaque_paths(&opaque_paths)?;
        for pointer in &opaque_paths {
            let _ = response_json.take_raw_json_source_at(pointer)?;
        }
        let root = response_json.value.clone();

        if let Some(u) = root.get("usageMetadata").filter(|v| !v.is_null()) {
            self.usage_raw = Some(u.clone());
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
            response_json.finish()?;
            return Ok(out);
        };

        let candidate_payload = root_payload
            .raw_source_at("/candidates/0")
            .and_then(|source| JsonPayload::parse(&source).ok());
        self.search.emit(
            candidate_payload
                .as_ref()
                .and_then(web_search_decode::gemini_payload),
            &mut out,
        );

        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        self.used_call_ids.extend(decode::provider_call_ids(parts));
        let mut saw_text_part_in_frame = false;
        let mut saw_reasoning_part_in_frame = false;
        for (part_index, part) in parts.iter().enumerate() {
            if native_parts.contains(&part_index) {
                let pointer = format!("/candidates/0/content/parts/{part_index}");
                if let Some(source) = root_payload.raw_source_at(&pointer) {
                    let value =
                        JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
                            message: format!("Gemini native Part is invalid JSON: {error}"),
                        })?;
                    self.text_block = None;
                    self.reasoning_block = None;
                    let block = self.next_block;
                    self.next_block += 1;
                    out.push(StreamEvent::ProviderContent {
                        block,
                        protocol: self
                            .protocol
                            .unwrap_or(ProtocolFamily::GeminiGenerateContent),
                        value,
                    });
                }
                continue;
            }
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
                let args_pointer =
                    format!("/candidates/0/content/parts/{part_index}/functionCall/args");
                let arguments_fragment = root_payload
                    .raw_source_at(&args_pointer)
                    .and_then(|source| JsonPayload::parse(&source).ok())
                    .and_then(|payload| {
                        payload
                            .canonical_json(crate::exact_json::JsonEncoding::Serde)
                            .ok()
                    })
                    .map(|source| JsonPayloadFragment::from_text(&source))
                    .unwrap_or_else(|| JsonPayloadFragment::from_text("null"));
                out.push(StreamEvent::ToolCallDelta {
                    block,
                    id,
                    provider_id,
                    caller: None,
                    toolset_name: None,
                    name,
                    arguments_fragment,
                });
                if let Some(signature) = signature {
                    out.push(StreamEvent::ThoughtSignature { block, signature });
                }
                out.push(StreamEvent::BlockEnd { block });
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
                let decoded = response_json.take_text(
                    &format!("/candidates/0/content/parts/{part_index}/text"),
                    text,
                )?;
                out.push(text_delta(block, decoded));
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
        response_json.finish()?;
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
            protocol: Some(context.profile().protocol),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod utf16_stream_tests {
    use super::*;

    #[test]
    fn stream_text_part_retains_lone_units() {
        let mut decoder = GeminiStreamDecoder::default();
        let events = decoder
            .decode_frame(
                br#"{"modelVersion":"gemini-test","candidates":[{"content":{"parts":[{"text":"A\ud800B"}]}}]}"#,
            )
            .unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDeltaJsUtf16 { text, utf16_code_units, .. }
                if text == "A�B" && utf16_code_units == &[0x41, 0xd800, 0x42]
        )));
    }
}

#[cfg(test)]
mod provider_error_utf16_tests {
    use super::*;

    #[test]
    fn stream_error_text_keeps_exact_units_and_classification() {
        let mut decoder = GeminiStreamDecoder::default();
        let error = decoder
            .decode_frame(br#"{"error":{"status":"INVALID_ARGUMENT","message":"bad\ud800"}}"#)
            .unwrap_err();
        assert!(matches!(
            error.classification_source(),
            LlmError::InvalidRequest { .. }
        ));
        let text = error.provider_error_text().unwrap();
        assert_eq!(text.display(), "500 bad�");
        assert_eq!(
            text.utf16_code_units(),
            &[53, 48, 48, 32, 98, 97, 100, 0xd800]
        );
    }
}
