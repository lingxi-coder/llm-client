//! The SSE stream decoder.
//!
//! Ported from the previous project's `OpenAiStreamDecoder`. Block indices are
//! assigned here and only ever move forward, which is what §15 means by "a
//! provider with no native index gets a synthesised monotonic one": a consumer
//! can tell one text block from the next without the codec replaying anything.

use super::decode;
use crate::codecs::usage;
use crate::codecs::web_search_decode::{self, SearchStream};
use crate::codecs::EventDecoder;
use crate::protocol::{
    JsonPayload, LlmError, ProtocolFamily, ResponseId, StopReason, StreamEvent, ToolUseId,
};
use crate::response_json::ResponseJson;
use crate::response_json::{text_delta, FlatJsonKind, ResponseJsonScanner};
use serde_json::Value;

#[derive(Debug, Default)]
pub struct OpenAiStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    started: bool,
    separate_reasoning: bool,
    qwen_cache: bool,
    next_block: usize,
    text_block: Option<usize>,
    reasoning_block: Option<usize>,
    audio_block: Option<usize>,
    native_reasoning_text_units: Vec<u16>,
    native_reasoning_text_seen: bool,
    native_reasoning_details_seen: bool,
    native_reasoning_detail_sources: Vec<String>,
    native_reasoning: super::reasoning::ReasoningStream,
    /// Wire index → (block, id, name). A provider streams a tool call's
    /// arguments in fragments keyed by its own index, not by the call id, and
    /// the id only arrives on the first fragment.
    tools: Vec<ToolStream>,
    /// The provider's usage object, with independently counted reasoning
    /// normalized into completion tokens when configured. The remaining raw
    /// fields allow checking consistency: published buckets are derived by
    /// subtraction, and a saturated subtraction must not read as a measurement.
    ///
    /// This wire restates the whole object each time it reports, so the latest
    /// one wins outright; only the Anthropic wire splits its report across
    /// frames and needs a field-by-field fold.
    usage_raw: Option<Value>,
    stop: Option<StopReason>,
    /// A refusal delta is text the caller needs to display and an outcome that
    /// refines an otherwise ordinary `finish_reason: "stop"`.
    saw_refusal: bool,
    done: bool,
    search: SearchStream,
}

#[derive(Debug)]
struct ToolStream {
    arguments_decoder: crate::response_json::ArgumentJsonDecoder,
    wire_index: u64,
    block: usize,
    id: ToolUseId,
    name: String,
}

impl EventDecoder for OpenAiStreamDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        let text = std::str::from_utf8(frame).map_err(|_| LlmError::InvalidRequest {
            message: "OpenAI stream frame is not valid UTF-8".to_owned(),
        })?;
        let data = text.trim();
        let mut out = Vec::new();

        // The sentinel is not JSON, and arrives before the connection closes.
        if data == "[DONE]" {
            self.finish_into(&mut out)?;
            return Ok(out);
        }
        if data.is_empty() {
            return Ok(out);
        }

        let scanner =
            ResponseJsonScanner::parse(data.as_bytes(), "OpenAI stream frame is not valid JSON")?;
        if scanner
            .kind_at("/error")
            .is_some_and(|kind| kind != FlatJsonKind::Null)
        {
            if let Some(usage) = usage_projection(&scanner, "/usage") {
                self.usage_raw = Some(decode::normalize_usage(
                    &usage,
                    self.separate_reasoning,
                    self.qwen_cache,
                ));
            }
        }
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
                &["message", "code", "type"],
                selected_text.as_ref(),
            );
            return Err(decode::classify_error(500, &body, None, false)
                .with_provider_error_text(provider_text));
        }
        let root_payload = JsonPayload::parse(data).map_err(|error| LlmError::InvalidRequest {
            message: format!("OpenAI stream frame is invalid JSON: {error}"),
        })?;
        let opaque_paths = [
            "/web_search",
            "/choices/0/delta/annotations",
            "/choices/0/delta/audio",
            "/choices/0/delta/reasoning_details",
        ]
        .into_iter()
        .filter(|pointer| scanner.raw_source_at(pointer).is_some())
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let mut response_json = scanner.into_response_json_with_opaque_paths(&opaque_paths)?;
        for pointer in &opaque_paths {
            let _ = response_json.take_raw_json_source_at(pointer)?;
        }
        let root = response_json.value.clone();

        if let Some(u) = root.get("usage").filter(|v| !v.is_null()) {
            self.usage_raw = Some(decode::normalize_usage(
                u,
                self.separate_reasoning,
                self.qwen_cache,
            ));
        }

        self.inference.observe(&root, &mut out);
        if !self.started {
            self.started = true;
            out.push(StreamEvent::Start {
                model: root
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                response_id: root
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ResponseId::new),
            });
        }

        // Usage may arrive on its own frame after the last choice, so it is
        // recorded whenever seen rather than read at the end.
        if let Some(u) = root.get("usage").filter(|v| !v.is_null()) {
            self.usage_raw = Some(decode::normalize_usage(
                u,
                self.separate_reasoning,
                self.qwen_cache,
            ));
            let usage_payload = root_payload
                .raw_source_at("/usage")
                .and_then(|source| JsonPayload::parse(&source).ok());
            self.search.emit(
                web_search_decode::with_usage_payload(None, usage_payload.as_ref()),
                &mut out,
            );
        }

        self.search.emit(
            web_search_decode::chat_response_payload(&root_payload, "/choices/0/delta"),
            &mut out,
        );
        let Some(choice) = root
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            response_json.finish()?;
            return Ok(out);
        };

        if let Some(delta) = choice.get("delta") {
            let reasoning_projection = self.native_reasoning.observe(delta)?;
            let native_reasoning = self
                .observe_native_reasoning(&root_payload, "/choices/0/delta")?
                || reasoning_projection;
            let mut text = super::reasoning::display_text(delta);
            if text.is_empty() {
                text =
                    display_reasoning_details(&root_payload, "/choices/0/delta/reasoning_details");
            }
            if native_reasoning || !text.is_empty() {
                let block = *self.reasoning_block.get_or_insert_with(|| {
                    let b = self.next_block;
                    self.next_block += 1;
                    b
                });
                if !text.is_empty() {
                    out.push(StreamEvent::ReasoningDelta { block, text });
                }
            }
            if let Some(t) = delta.get("content").and_then(Value::as_str) {
                let decoded = response_json.take_text("/choices/0/delta/content", t)?;
                if !decoded.text.is_empty() {
                    let block = *self.text_block.get_or_insert_with(|| {
                        let b = self.next_block;
                        self.next_block += 1;
                        b
                    });
                    out.push(text_delta(block, decoded));
                }
            }
            if let Some(refusal) = delta.get("refusal").and_then(Value::as_str) {
                self.saw_refusal = true;
                if !refusal.is_empty() {
                    let decoded = response_json.take_text("/choices/0/delta/refusal", refusal)?;
                    let block = *self.text_block.get_or_insert_with(|| {
                        let b = self.next_block;
                        self.next_block += 1;
                        b
                    });
                    out.push(text_delta(block, decoded));
                }
            }
            if root_payload
                .kind_at("/choices/0/delta/audio")
                .is_some_and(|kind| kind != FlatJsonKind::Null)
            {
                let block = *self.audio_block.get_or_insert_with(|| {
                    let block = self.next_block;
                    self.next_block += 1;
                    block
                });
                let mut native_fields = Vec::new();
                for (pointer, name) in [
                    ("/id", "response_id"),
                    ("/model", "model"),
                    ("/choices/0/index", "choice_index"),
                ] {
                    if let Some(source) = response_json
                        .take_raw_json_source_at(pointer)?
                        .map(str::to_owned)
                    {
                        native_fields.push((name.encode_utf16().collect(), source));
                    }
                }
                let audio_source = response_json
                    .take_raw_json_source_at("/choices/0/delta/audio")?
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "OpenAI audio source was not retained".into(),
                    })?;
                native_fields.push(("audio".encode_utf16().collect(), audio_source.to_owned()));
                let native = JsonPayload::object_from_fields(&native_fields).map_err(|error| {
                    LlmError::InvalidRequest {
                        message: format!("OpenAI audio metadata is invalid JSON: {error}"),
                    }
                })?;
                out.push(StreamEvent::ProviderContent {
                    block,
                    protocol: ProtocolFamily::OpenAiChat,
                    value: native,
                });
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (call_index, call) in calls.iter().enumerate() {
                    self.tool_fragment(call, call_index, &mut response_json, &mut out)?;
                }
            }
        }

        if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
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
        self.finish_into(&mut out)?;
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
            &usage::OPENAI_CHAT,
            decode::usage,
            self.done,
        )
    }
}

/// Preserve the current Chat usage counters from an error frame before its
/// terminal error is emitted. Only known usage scalars enter the accounting
/// DTO; arbitrary usage extensions remain in the original ProviderEvent source.
fn usage_projection(scanner: &ResponseJsonScanner, pointer: &str) -> Option<Value> {
    let mut root = serde_json::Map::new();
    for name in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cost",
    ] {
        let field = format!("{pointer}/{name}");
        if let Some(value) = scalar_projection(scanner, &field) {
            root.insert(name.to_owned(), value);
        }
    }
    for (group, fields) in [
        (
            "prompt_tokens_details",
            &["cached_tokens", "cache_write_tokens"][..],
        ),
        ("completion_tokens_details", &["reasoning_tokens"][..]),
    ] {
        let mut nested = serde_json::Map::new();
        for name in fields {
            let field = format!("{pointer}/{group}/{name}");
            if let Some(value) = scalar_projection(scanner, &field) {
                nested.insert((*name).to_owned(), value);
            }
        }
        if !nested.is_empty() {
            root.insert(group.to_owned(), Value::Object(nested));
        }
    }
    (!root.is_empty()).then_some(Value::Object(root))
}

fn scalar_projection(scanner: &ResponseJsonScanner, pointer: &str) -> Option<Value> {
    match scanner.kind_at(pointer)? {
        FlatJsonKind::Null => Some(Value::Null),
        FlatJsonKind::Boolean | FlatJsonKind::Number => {
            serde_json::from_str(scanner.raw_source_at(pointer)?).ok()
        }
        FlatJsonKind::String => scanner
            .string_at(pointer)
            .map(|value| Value::String(value.into())),
        FlatJsonKind::Array | FlatJsonKind::Object => None,
    }
}

impl OpenAiStreamDecoder {
    pub(crate) fn configured(context: &crate::codecs::CodecContext) -> Self {
        Self {
            inference: crate::codecs::inference::StreamInference::new(context),
            separate_reasoning: super::separate_reasoning(&context.profile.extra),
            qwen_cache: crate::providers::qwen::cache::applies(context),
            ..Default::default()
        }
    }
    fn tool_fragment(
        &mut self,
        call: &Value,
        call_index: usize,
        response_json: &mut ResponseJson,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let wire_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function").unwrap_or(&Value::Null);
        let id = call.get("id").and_then(Value::as_str);
        let name = function.get("name").and_then(Value::as_str);

        if !self.tools.iter().any(|t| t.wire_index == wire_index) {
            let block = self.next_block;
            self.next_block += 1;
            self.tools.push(ToolStream {
                arguments_decoder: Default::default(),
                wire_index,
                block,
                id: ToolUseId::new(id.unwrap_or_default()),
                name: name.unwrap_or_default().to_owned(),
            });
        }
        let Some(slot) = self.tools.iter_mut().find(|t| t.wire_index == wire_index) else {
            return Ok(());
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
        let decoded = response_json.take_text(
            &format!("/choices/0/delta/tool_calls/{call_index}/function/arguments"),
            fragment,
        )?;
        let arguments_fragment = slot.arguments_decoder.push(decoded)?;
        out.push(StreamEvent::ToolCallDelta {
            block: slot.block,
            id: slot.id.clone(),
            caller: None,
            toolset_name: None,
            name: slot.name.clone(),
            arguments_fragment,
            provider_id: None,
        });
        Ok(())
    }

    /// `[DONE]` and the end of the byte stream are two ways to reach the same
    /// place, and a provider may do both — so the terminal event is emitted
    /// once.
    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) -> Result<(), LlmError> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        let _reasoning_display_projection = self.native_reasoning.take();
        if let Some(value) = self.native_reasoning_payload()? {
            out.push(StreamEvent::ProviderContent {
                block: self
                    .reasoning_block
                    .expect("native reasoning allocated a block"),
                protocol: ProtocolFamily::OpenAiChat,
                value,
            });
        }
        out.push(StreamEvent::End {
            stop_reason: decode::with_refusal_stop_reason(
                self.stop.clone().unwrap_or(StopReason::EndTurn),
                self.saw_refusal,
            ),
            usage: self.usage_report(),
            inference: self.inference.report.clone(),
        });
        Ok(())
    }

    fn observe_native_reasoning(
        &mut self,
        payload: &JsonPayload,
        pointer: &str,
    ) -> Result<bool, LlmError> {
        let reasoning_pointer = format!("{pointer}/reasoning");
        let mut observed = false;
        if let Some(kind) = payload.kind_at(&reasoning_pointer) {
            if kind != FlatJsonKind::Null {
                if kind != FlatJsonKind::String {
                    return Err(LlmError::ProviderInternal {
                        message: "Chat response has invalid reasoning".into(),
                    });
                }
                self.native_reasoning_text_units.extend(
                    payload
                        .string_units_at(&reasoning_pointer)
                        .expect("a flat string has UTF-16 units"),
                );
                self.native_reasoning_text_seen = true;
                observed = true;
            }
        }

        let details_pointer = format!("{pointer}/reasoning_details");
        if let Some(kind) = payload.kind_at(&details_pointer) {
            if kind != FlatJsonKind::Null {
                if kind != FlatJsonKind::Array {
                    return Err(LlmError::ProviderInternal {
                        message: "Chat response has invalid reasoning_details".into(),
                    });
                }
                self.native_reasoning_details_seen = true;
                let count = payload
                    .array_len(&details_pointer)
                    .expect("an array kind has an array length");
                for index in 0..count {
                    let item_pointer = format!("{details_pointer}/{index}");
                    let source = payload.raw_source_at(&item_pointer).ok_or_else(|| {
                        LlmError::InvalidRequest {
                            message: "OpenAI reasoning detail source was not retained".into(),
                        }
                    })?;
                    self.native_reasoning_detail_sources.push(source);
                }
                observed = true;
            }
        }
        Ok(observed)
    }

    fn native_reasoning_payload(&self) -> Result<Option<JsonPayload>, LlmError> {
        if !self.native_reasoning_text_seen && !self.native_reasoning_details_seen {
            return Ok(None);
        }
        let mut fields = vec![(
            "type".encode_utf16().collect(),
            r#""chat_reasoning""#.to_owned(),
        )];
        if self.native_reasoning_text_seen {
            let text = JsonPayload::string_from_utf16_code_units(&self.native_reasoning_text_units);
            fields.push((
                "reasoning".encode_utf16().collect(),
                text.as_json_source().to_owned(),
            ));
        }
        if self.native_reasoning_details_seen {
            let details = format!("[{}]", self.native_reasoning_detail_sources.join(","));
            fields.push(("reasoning_details".encode_utf16().collect(), details));
        }
        JsonPayload::object_from_fields(&fields)
            .map(Some)
            .map_err(|error| LlmError::InvalidRequest {
                message: format!("OpenAI reasoning metadata is invalid JSON: {error}"),
            })
    }
}

fn display_reasoning_details(payload: &JsonPayload, pointer: &str) -> String {
    let mut text = String::new();
    for index in 0..payload.array_len(pointer).unwrap_or(0) {
        let item = format!("{pointer}/{index}");
        let field = match payload.string_at(&format!("{item}/type")).as_deref() {
            Some("reasoning.text") => "text",
            Some("reasoning.summary") => "summary",
            _ => continue,
        };
        if let Some(value) = payload.string_at(&format!("{item}/{field}")) {
            text.push_str(&value);
        }
    }
    text
}

#[cfg(test)]
mod utf16_stream_tests {
    use super::*;

    #[test]
    fn stream_error_text_keeps_exact_units_and_classification() {
        let mut decoder = OpenAiStreamDecoder::default();
        let error = decoder
            .decode_frame(br#"{"error":{"code":"invalid_prompt","message":"bad\ud800"}}"#)
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

    #[test]
    fn stream_text_delta_retains_lone_units() {
        let mut decoder = OpenAiStreamDecoder::default();
        let events = decoder
            .decode_frame(
                br#"{"model":"gpt-test","choices":[{"index":0,"delta":{"content":"A\ud800B"},"finish_reason":null}]}"#,
            )
            .unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDeltaJsUtf16 { text, utf16_code_units, .. }
                if text == "A�B" && utf16_code_units == &[0x41, 0xd800, 0x42]
        )));
    }

    #[test]
    fn adjacent_stream_frames_recombine_surrogate_halves() {
        let mut decoder = OpenAiStreamDecoder::default();
        let mut accumulator = crate::stream_assembly::StreamAccumulator::new();
        for frame in [
            br#"{"model":"gpt-test","choices":[{"index":0,"delta":{"content":"\ud83d"},"finish_reason":null}]}"#.as_slice(),
            br#"{"model":"gpt-test","choices":[{"index":0,"delta":{"content":"\ude00"},"finish_reason":"stop"}]}"#.as_slice(),
            b"[DONE]",
        ] {
            for event in decoder.decode_frame(frame).unwrap() {
                accumulator.observe(&event);
            }
        }
        let response = accumulator.finish().unwrap().response;
        assert!(matches!(
            response.message.content.as_slice(),
            [crate::protocol::ContentBlock::TextJsUtf16 { text, utf16_code_units, .. }]
                if text == "😀" && utf16_code_units == &[0xd83d, 0xde00]
        ));
    }
}
