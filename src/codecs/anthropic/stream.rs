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
use crate::protocol::{
    ContentBlock, JsonPayload, JsonPayloadFragment, LlmError, ProtocolFamily, StopReason,
    StreamEvent, ToolUseId,
};
use crate::providers::anthropic::fallback_response;
use crate::response_json::FlatJsonKind;
use crate::response_json::{DecodedText, ResponseJson, ResponseJsonScanner};
use serde_json::Value;

/// Pending native blocks and cited text are held only until content_block_stop.
/// This aggregate cap also bounds a response that opens many blocks without
/// closing them; the SSE parser separately limits each individual frame.
const MAX_PENDING_NATIVE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
struct PendingNativeBlock {
    value: JsonPayload,
    assembles_input_json: bool,
    input_json: Vec<u16>,
    input_json_seen: bool,
    incomplete: bool,
    retained_bytes: usize,
}

#[derive(Debug)]
struct PendingCitedTextBlock {
    value: JsonPayload,
    text: String,
    utf16_code_units: Option<Vec<u16>>,
    citations: Vec<JsonPayload>,
    has_citations: bool,
    citations_present: bool,
    opaque: bool,
    incomplete: bool,
    observing: bool,
    buffered_frames: Vec<JsonPayload>,
    buffered_frame_bytes: usize,
    retained_bytes: usize,
}

#[derive(Debug)]
struct PendingToolCall {
    index: usize,
    id: ToolUseId,
    name: String,
    caller: Option<JsonPayload>,
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
    /// Exact folded usage source for Native Rte/OP resumability selection.
    native_usage_raw: Option<JsonPayload>,
    native_context_window: Option<f64>,
    /// `message_start` usage is provisional even when it includes an output
    /// count. Only a numeric output count in a `message_delta` is final usage.
    final_output_count_reported: bool,
    retain_openrouter_container: bool,
    retain_anthropic_container: bool,
    stop: Option<StopReason>,
    done: bool,
    search_blocks: Vec<usize>,
    fallback_blocks: Vec<f64>,
    frame_payload: Option<JsonPayload>,
    native_anthropic_error_route: bool,
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
        self.frame_payload =
            Some(
                JsonPayload::parse(data).map_err(|error| LlmError::InvalidRequest {
                    message: format!("stream frame is not valid JSON: {error}"),
                })?,
            );
        let scanner =
            ResponseJsonScanner::parse(data.as_bytes(), "stream frame is not valid JSON")?;
        let native_usage_pointer = match scanner.string_at("/type") {
            Some("message_start") => "/message/usage",
            Some("message_delta") => "/usage",
            _ => "",
        };
        if !native_usage_pointer.is_empty() {
            if let Some(usage) = self
                .frame_payload
                .as_ref()
                .and_then(|payload| payload.payload_at(native_usage_pointer))
            {
                self.fold_native_usage(usage)?;
            }
        }
        if scanner.string_at("/type") == Some("error") {
            let error_source = scanner.raw_source_at("/error").unwrap_or(data);
            let error_scanner =
                ResponseJsonScanner::parse(error_source.as_bytes(), "stream error is not JSON")?;
            let selected_text = error_scanner.provider_error_text_at("/message", None)?;
            let body = error_scanner.provider_error_classification_body(
                "",
                &["type", "message"],
                selected_text.as_ref(),
            );
            let provider_text = if self.native_anthropic_error_route {
                scanner.native_anthropic_http_error_text(None)?
            } else {
                match &selected_text {
                    Some(text) => text.prefixed("500 "),
                    None => {
                        error_scanner.provider_error_text_or_source_at("/message", Some(500), "")?
                    }
                }
            };
            let error = if self.native_anthropic_error_route {
                decode::classify_stream_error(
                    &body,
                    decode::native_anthropic_retry_base_eligible(None, &scanner)?,
                )
            } else {
                let mut error = decode::classify_error(500, &body, None, false);
                if let LlmError::ProviderTimeout { status, .. } = error.classification_source_mut()
                {
                    *status = None;
                }
                error
            };
            return Err(error.with_provider_error_text(provider_text));
        }
        if !matches!(
            scanner.string_at("/type"),
            Some(
                "content_block_start"
                    | "content_block_delta"
                    | "content_block_stop"
                    | "message_start"
                    | "message_delta"
                    | "message_stop"
                    | "ping"
                    | "error"
            )
        ) {
            return Ok(vec![StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::AnthropicMessages,
                payload: self
                    .frame_payload
                    .clone()
                    .expect("stream frame source is retained"),
            }]);
        }
        let mut opaque_paths = Vec::new();
        for pointer in [
            "/message/usage",
            "/usage",
            "/delta/stop_details",
            "/message/container",
            "/delta/container",
        ] {
            if scanner.raw_source_at(pointer).is_some() {
                opaque_paths.push(pointer.to_owned());
            }
        }
        match scanner.string_at("/type") {
            Some("content_block_start") => match scanner.string_at("/content_block/type") {
                Some("web_search_tool_result") => {
                    if scanner.raw_source_at("/content_block/content").is_some() {
                        opaque_paths.push("/content_block/content".to_owned());
                    }
                }
                Some("server_tool_use" | "mcp_tool_use") => {
                    if scanner.raw_source_at("/content_block/input").is_some() {
                        opaque_paths.push("/content_block/input".to_owned());
                    }
                }
                Some("text") => {
                    if scanner.raw_source_at("/content_block/citations").is_some() {
                        opaque_paths.push("/content_block/citations".to_owned());
                    }
                }
                Some("tool_use") => {
                    if scanner.raw_source_at("/content_block/caller").is_some() {
                        opaque_paths.push("/content_block/caller".to_owned());
                    }
                }
                _ => {}
            },
            Some("content_block_delta")
                if scanner.string_at("/delta/type") == Some("citations_delta")
                    && scanner.raw_source_at("/delta/citation").is_some() =>
            {
                opaque_paths.push("/delta/citation".to_owned());
            }
            _ => {}
        }
        let mut response_json = scanner.into_response_json_with_opaque_paths(&opaque_paths)?;
        for pointer in &opaque_paths {
            let _ = response_json.take_raw_json_source_at(pointer)?;
        }
        let mut root = response_json.value.clone();
        for pointer in ["/message/usage", "/usage"] {
            if let Some(payload) = self
                .frame_payload
                .as_ref()
                .and_then(|payload| payload.payload_at(pointer))
            {
                if let Some(value) = root.pointer_mut(pointer) {
                    *value = decode::anthropic_usage_projection(&payload);
                }
            }
        }

        let mut out = Vec::new();
        self.inference.observe(&root, &mut out);
        if let Some(index) = self
            .frame_payload
            .as_ref()
            .and_then(fallback_response::control_index)
            .and_then(|index| index.as_f64())
        {
            if !self.fallback_blocks.contains(&index) {
                self.fallback_blocks.push(index);
            }
            // Controls (including malformed controls) remain observations and
            // never become replayable provider content or executable tool input.
            self.push_provider_event(&root, &mut out);
            if let Some(start) = self
                .frame_payload
                .as_ref()
                .and_then(fallback_response::stream_start)
            {
                out.push(StreamEvent::NativeControl {
                    protocol: ProtocolFamily::AnthropicMessages,
                    control: crate::protocol::NativeExtension::from_typed(
                        fallback_response::FallbackControl::Start { start },
                    )
                    .expect("fallback start contains JSON data"),
                });
            }
            response_json.finish()?;
            return Ok(out);
        }
        if matches!(
            root["type"].as_str(),
            Some("content_block_delta" | "content_block_stop")
        ) && root["index"]
            .as_f64()
            .is_some_and(|index| self.fallback_blocks.contains(&index))
        {
            response_json.finish()?;
            return Ok(out);
        }
        match root.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let message = root.get("message").unwrap_or(&Value::Null);
                let retain_message_metadata = self.frame_payload.as_ref().is_some_and(|payload| {
                    let container = payload.payload_at("/message/container");
                    (self.retain_anthropic_container
                        && (container.is_some()
                            || payload.raw_source_at("/message/usage").is_some()))
                        || (self.retain_openrouter_container
                            && container.is_some_and(|container| !container.is_root_null()))
                });
                if retain_message_metadata {
                    self.push_provider_event(&root, &mut out);
                }
                if let Some(u) = message.get("usage") {
                    self.fold_usage(u);
                    let usage_payload = self
                        .frame_payload
                        .as_ref()
                        .and_then(|payload| payload.raw_source_at("/message/usage"))
                        .and_then(|source| JsonPayload::parse(&source).ok());
                    if let Some(result) =
                        web_search_decode::with_usage_payload(None, usage_payload.as_ref())
                    {
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
            Some("content_block_start") => {
                self.block_start(&root, &mut response_json, data.as_bytes(), &mut out)?
            }
            Some("content_block_delta") => {
                self.block_delta(&root, &mut response_json, data.as_bytes(), &mut out)?
            }
            Some("content_block_stop") => {
                self.block_stop(&root, data.as_bytes(), &mut out)?;
                out.push(StreamEvent::BlockEnd {
                    block: root["index"].as_u64().unwrap_or_default() as usize,
                });
            }
            Some("message_delta") => {
                if !self.fallback_blocks.is_empty()
                    || self.frame_payload.as_ref().is_some_and(|payload| {
                        payload.payload_at("/usage").is_some_and(|usage| {
                            usage.kind_at("/iterations") == Some(FlatJsonKind::Array)
                        })
                    })
                {
                    out.push(StreamEvent::NativeControl {
                        protocol: ProtocolFamily::AnthropicMessages,
                        control: crate::protocol::NativeExtension::from_typed(
                            fallback_response::FallbackControl::Boundary {
                                stop_reason: root["delta"]["stop_reason"]
                                    .as_str()
                                    .map(str::to_owned),
                                iterations: self
                                    .frame_payload
                                    .as_ref()
                                    .and_then(|payload| payload.payload_at("/usage"))
                                    .map(|usage| fallback_response::usage_iterations(&usage))
                                    .unwrap_or_default(),
                                iterations_present: self
                                    .frame_payload
                                    .as_ref()
                                    .and_then(|payload| payload.payload_at("/usage"))
                                    .is_some_and(|usage| {
                                        usage.kind_at("/iterations") == Some(FlatJsonKind::Array)
                                    }),
                            },
                        )
                        .expect("fallback boundary contains JSON data"),
                    });
                }
                if root
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(Value::as_str)
                    .is_some()
                    || self
                        .frame_payload
                        .as_ref()
                        .and_then(crate::providers::anthropic::code_execution::stream_resumable)
                        .is_some()
                    || root
                        .get("delta")
                        .and_then(|delta| delta.get("stop_details"))
                        .is_some()
                    || self.retain_anthropic_container
                        && self.frame_payload.as_ref().is_some_and(|payload| {
                            crate::providers::anthropic::code_execution::stream_container(payload)
                                .is_some()
                                || crate::providers::anthropic::code_execution::stream_usage(
                                    payload,
                                )
                                .is_some()
                        })
                    || self.frame_payload.as_ref().is_some_and(|payload| {
                        payload.payload_at("/usage").is_some_and(|usage| {
                            usage.kind_at("/iterations") == Some(FlatJsonKind::Array)
                        })
                    })
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
                    let usage_payload = self
                        .frame_payload
                        .as_ref()
                        .and_then(|payload| payload.raw_source_at("/usage"))
                        .and_then(|source| JsonPayload::parse(&source).ok());
                    if let Some(result) =
                        web_search_decode::with_usage_payload(None, usage_payload.as_ref())
                    {
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
            // Tolerating an unknown type is part of this wire's contract: the
            // provider adds event types and a client must not break on them.
            _ => out.push(StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::AnthropicMessages,
                payload: self
                    .frame_payload
                    .clone()
                    .expect("stream frame source is retained"),
            }),
        }
        response_json.finish()?;
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
    fn native_resume_context_exhausted(&self) -> Option<bool> {
        decode::native_resume_context_exhausted(
            self.native_usage_raw.as_ref(),
            self.native_context_window,
        )
    }
}

impl AnthropicStreamDecoder {
    fn fold_native_usage(&mut self, incoming: JsonPayload) -> Result<(), LlmError> {
        self.native_usage_raw = Some(match self.native_usage_raw.take() {
            Some(current) => current.merge_overwriting_objects(&incoming)?,
            None => incoming,
        });
        Ok(())
    }

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
        response_json: &mut ResponseJson,
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
        let decoded_text = match initial_text {
            Some(text) => Some(response_json.take_text("/content_block/text", text)?),
            None => None,
        };
        let text = decoded_text
            .as_ref()
            .map(|decoded| decoded.text.clone())
            .unwrap_or_default();
        let utf16_code_units = decoded_text
            .as_ref()
            .and_then(|decoded| decoded.utf16_code_units.clone());
        let citations_payload = self
            .frame_payload
            .as_ref()
            .and_then(|payload| payload.raw_source_at("/content_block/citations"))
            .map(|source| JsonPayload::parse(&source))
            .transpose()
            .map_err(|_| LlmError::StreamInterrupted {
                message: "Anthropic text-block citations are invalid JSON".into(),
            })?;
        let citations = match citations_payload.as_ref() {
            Some(payload) if payload.is_root_array() => {
                payload
                    .array_items()
                    .ok_or_else(|| LlmError::StreamInterrupted {
                        message: "Anthropic text-block citations must be an array".into(),
                    })?
            }
            _ => Vec::new(),
        };
        let has_citations = !citations.is_empty();
        let citations_present = citations_payload.is_some();
        let block_source = self
            .frame_payload
            .as_ref()
            .and_then(|payload| payload.raw_source_at("/content_block"))
            .ok_or_else(|| LlmError::StreamInterrupted {
                message: "Anthropic text block source was not retained".into(),
            })?;
        let block_payload =
            JsonPayload::parse(&block_source).map_err(|_| LlmError::StreamInterrupted {
                message: "Anthropic text block is invalid JSON".into(),
            })?;
        let opaque = !decode::has_only_known_text_fields(&block_payload);
        if opaque
            && decoded_text
                .as_ref()
                .is_some_and(|text| text.utf16_code_units.is_some())
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic opaque text blocks with lone UTF-16 units cannot be replayed losslessly".into(),
            });
        }
        let citations_well_formed = match citations_payload.as_ref() {
            None => true,
            Some(value) if value.is_root_null() => true,
            Some(value) if value.is_root_array() => citations.iter().all(|citation| {
                citation.kind_at("") == Some(crate::response_json::FlatJsonKind::Object)
            }),
            Some(_) => false,
        };
        let incomplete = initial_text.is_none() || !citations_well_formed;
        let observing = has_citations || citations_present || opaque || incomplete;
        let mut retained_bytes = block_source
            .len()
            .checked_add(text.len())
            .ok_or_else(native_block_limit_error)?;
        for citation in &citations {
            retained_bytes = retained_bytes
                .checked_add(citation.as_json_source().len())
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
                value: block_payload,
                text,
                utf16_code_units,
                citations,
                has_citations,
                citations_present,
                opaque,
                incomplete,
                observing,
                buffered_frames: if observing {
                    Vec::new()
                } else {
                    vec![self
                        .frame_payload
                        .as_ref()
                        .expect("current frame is retained")
                        .clone()]
                },
                buffered_frame_bytes,
                retained_bytes,
            },
        ));
        if observing {
            self.push_provider_event(root, out);
        }
        if let Some(decoded) = decoded_text.filter(|decoded| !decoded.text.is_empty()) {
            out.push(text_delta(index, decoded));
        }
        Ok(())
    }

    fn block_start(
        &mut self,
        root: &Value,
        response_json: &mut ResponseJson,
        _frame: &[u8],
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let block = root.get("content_block").unwrap_or(&Value::Null);
        let block_type = block.get("type").and_then(Value::as_str);
        let search_block = self
            .frame_payload
            .as_ref()
            .and_then(|payload| payload.raw_source_at("/content_block"))
            .and_then(|source| JsonPayload::parse(&source).ok());
        if block_type == Some("text") {
            self.start_cited_text_block(index, block, root, response_json, out)?;
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
            let block_source = self
                .frame_payload
                .as_ref()
                .and_then(|payload| payload.raw_source_at("/content_block"))
                .ok_or_else(|| LlmError::StreamInterrupted {
                    message: "Anthropic native content block source was not retained".to_owned(),
                })?;
            let retained_bytes = block_source.len();
            let block_payload =
                JsonPayload::parse(&block_source).map_err(|_| LlmError::StreamInterrupted {
                    message: "Anthropic native content block is invalid JSON".to_owned(),
                })?;
            self.reserve_native_bytes(retained_bytes)?;
            self.pending_native.push((
                index,
                PendingNativeBlock {
                    value: block_payload,
                    assembles_input_json: matches!(
                        block_type,
                        Some("server_tool_use" | "mcp_tool_use")
                    ),
                    input_json: Vec::new(),
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

        if search_block
            .as_ref()
            .is_some_and(web_search_decode::anthropic_payload_is_search_block)
        {
            self.search_blocks.push(index);
            if let Some(result) = self
                .frame_payload
                .as_ref()
                .and_then(web_search_decode::anthropic_event_frame)
            {
                out.push(StreamEvent::WebSearch { result });
            }
        }
        if block_type == Some("redacted_thinking") {
            if let Some(payload) = self
                .frame_payload
                .as_ref()
                .and_then(|payload| payload.raw_source_at("/content_block"))
                .and_then(|source| JsonPayload::parse(&source).ok())
            {
                if let Some(ContentBlock::RedactedThinking { data }) =
                    decode::decode_block(&payload)
                {
                    out.push(StreamEvent::RedactedThinking { block: index, data });
                }
            }
        }
        if block_type == Some("tool_use") {
            let id = ToolUseId::new(block.get("id").and_then(Value::as_str).unwrap_or_default());
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let caller = self
                .frame_payload
                .as_ref()
                .and_then(|payload| payload.raw_source_at("/content_block/caller"))
                .map(|source| JsonPayload::parse(&source))
                .transpose()
                .map_err(|_| LlmError::StreamInterrupted {
                    message: "Anthropic tool caller metadata is invalid JSON".into(),
                })?;
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
                arguments_fragment: JsonPayloadFragment::from_text(""),
            });
        }
        Ok(())
    }

    fn block_delta(
        &mut self,
        root: &Value,
        response_json: &mut ResponseJson,
        _frame: &[u8],
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let index = root
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let delta = root.get("delta").unwrap_or(&Value::Null);
        if matches!(
            delta["type"].as_str(),
            Some("citations_delta" | "connector_text_delta")
        ) {
            out.push(StreamEvent::NativeDelta {
                block: index,
                protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
                delta: self
                    .frame_payload
                    .as_ref()
                    .and_then(|payload| payload.raw_source_at("/delta"))
                    .and_then(|source| JsonPayload::parse(&source).ok())
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "Anthropic native delta source was not retained".into(),
                    })?,
            });
        }

        let is_web_search_citation = self.frame_payload.as_ref().is_some_and(|payload| {
            payload.string_at("/delta/type").as_deref() == Some("citations_delta")
                && payload.string_at("/delta/citation/type").as_deref()
                    == Some("web_search_result_location")
        });
        if self.search_blocks.contains(&index) || is_web_search_citation {
            if let Some(result) = self
                .frame_payload
                .as_ref()
                .and_then(web_search_decode::anthropic_event_frame)
            {
                out.push(StreamEvent::WebSearch { result });
            }
        }

        if let Some(position) = self
            .pending_cited_text
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        {
            return self.cited_text_delta(position, index, root, delta, response_json, out);
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
                    let decoded = response_json.take_text("/delta/partial_json", fragment)?;
                    let units = decoded
                        .utf16_code_units
                        .unwrap_or_else(|| decoded.text.encode_utf16().collect());
                    let retained = units.len().saturating_mul(2);
                    self.reserve_native_bytes(retained)?;
                    let pending = &mut self.pending_native[position].1;
                    pending.retained_bytes = pending.retained_bytes.saturating_add(retained);
                    pending.input_json.extend(units);
                    pending.input_json_seen = true;
                    return Ok(());
                }
            }
            self.pending_native[position].1.incomplete = true;
            return Ok(());
        }

        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                let text = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let decoded = response_json.take_text("/delta/text", text)?;
                if !decoded.text.is_empty() {
                    out.push(text_delta(index, decoded));
                }
            }
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
                let partial_json = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let decoded = response_json.take_text("/delta/partial_json", partial_json)?;
                let arguments_fragment =
                    JsonPayloadFragment::from_decoded_text(&decoded.text, decoded.utf16_code_units)
                        .map_err(|message| LlmError::InvalidRequest {
                            message: message.into(),
                        })?;
                out.push(StreamEvent::ToolCallDelta {
                    block: index,
                    id: call.id.clone(),
                    provider_id: None,
                    caller: call.caller.clone(),
                    toolset_name: call.toolset_name.clone(),
                    name: call.name.clone(),
                    arguments_fragment,
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
        response_json: &mut ResponseJson,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                let Some(text) = delta.get("text").and_then(Value::as_str) else {
                    return self.mark_cited_text_incomplete(position, root, out);
                };
                self.reserve_native_bytes(text.len())?;
                let decoded = response_json.take_text("/delta/text", text)?;
                if self.pending_cited_text[position].1.opaque && decoded.utf16_code_units.is_some()
                {
                    return Err(LlmError::UnsupportedCapability {
                        message: "Anthropic opaque text blocks with lone UTF-16 units cannot be replayed losslessly".into(),
                    });
                }
                let pending = &mut self.pending_cited_text[position].1;
                pending.retained_bytes += text.len();
                append_text_units(pending, &decoded);
                self.retain_cited_text_frame(position, root, out)?;
                if !decoded.text.is_empty() {
                    out.push(text_delta(index, decoded));
                }
            }
            Some("citations_delta") => {
                let Some(citation_source) = self
                    .frame_payload
                    .as_ref()
                    .and_then(|payload| payload.raw_source_at("/delta/citation"))
                else {
                    return self.mark_cited_text_incomplete(position, root, out);
                };
                let citation = JsonPayload::parse(&citation_source).map_err(|_| {
                    LlmError::StreamInterrupted {
                        message: "Anthropic citation delta is invalid JSON".into(),
                    }
                })?;
                if citation.kind_at("") != Some(crate::response_json::FlatJsonKind::Object) {
                    return self.mark_cited_text_incomplete(position, root, out);
                }
                let citation_bytes = citation.as_json_source().len();
                self.reserve_native_bytes(citation_bytes)?;
                let was_observing = self.pending_cited_text[position].1.observing;
                let pending = &mut self.pending_cited_text[position].1;
                pending.retained_bytes += citation_bytes;
                pending.citations.push(citation);
                pending.has_citations = true;
                pending.citations_present = true;
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
        let bytes = self
            .frame_payload
            .as_ref()
            .map(|payload| payload.as_json_source().len())
            .ok_or_else(native_block_limit_error)?;
        self.reserve_native_bytes(bytes)?;
        let pending = &mut self.pending_cited_text[position].1;
        pending.retained_bytes += bytes;
        pending.buffered_frame_bytes += bytes;
        pending.buffered_frames.push(
            self.frame_payload
                .as_ref()
                .expect("current frame is retained")
                .clone(),
        );
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
            self.push_provider_payload(frame, out);
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
            self.finish_cited_text_block(index, root, out)?;
        }
        if self.search_blocks.contains(&index) {
            if let Some(result) = self
                .frame_payload
                .as_ref()
                .and_then(web_search_decode::anthropic_event_frame)
            {
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
            let fragment = JsonPayloadFragment::from_utf16_code_units(pending.input_json);
            let Ok(input) = JsonPayload::parse_fragments([&fragment]) else {
                return;
            };
            if input.kind_at("") != Some(crate::response_json::FlatJsonKind::Object) {
                return;
            }
            let Ok(updated) = pending.value.with_canonical_fields(
                crate::exact_json::JsonEncoding::Serde,
                &[("input", input.as_json_source())],
            ) else {
                return;
            };
            pending.value = updated;
        }
        out.push(StreamEvent::ProviderContent {
            block: index,
            protocol: ProtocolFamily::AnthropicMessages,
            value: pending.value,
        });
    }

    fn finish_cited_text_block(
        &mut self,
        index: usize,
        root: &Value,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), LlmError> {
        let Some(position) = self
            .pending_cited_text
            .iter()
            .position(|(pending_index, _)| *pending_index == index)
        else {
            return Ok(());
        };
        if self.pending_cited_text[position].1.observing {
            self.push_provider_event(root, out);
        }
        let (_, pending) = self.pending_cited_text.remove(position);
        self.pending_native_bytes = self
            .pending_native_bytes
            .saturating_sub(pending.retained_bytes);
        if pending.incomplete || (!pending.citations_present && !pending.opaque) {
            return Ok(());
        }
        let text_units = pending
            .utf16_code_units
            .unwrap_or_else(|| pending.text.encode_utf16().collect());
        let text_source = JsonPayload::string_json_source(&text_units);
        let mut fields = vec![("text", text_source)];
        if pending.citations_present
            && (pending.has_citations
                || pending.value.kind_at("/citations")
                    == Some(crate::response_json::FlatJsonKind::Array))
        {
            let citations = JsonPayload::array_from_items(
                &pending.citations,
                crate::exact_json::JsonEncoding::Serde,
            )?;
            fields.push(("citations", citations.as_json_source().to_owned()));
        }
        let fields = fields
            .iter()
            .map(|(key, source)| (*key, source.as_str()))
            .collect::<Vec<_>>();
        let Ok(updated) = pending
            .value
            .with_canonical_fields(crate::exact_json::JsonEncoding::Serde, &fields)
        else {
            return Ok(());
        };
        out.push(StreamEvent::ProviderContent {
            block: index,
            protocol: ProtocolFamily::AnthropicMessages,
            value: updated,
        });
        Ok(())
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

    fn push_provider_event(&self, _payload: &Value, out: &mut Vec<StreamEvent>) {
        if let Some(payload) = &self.frame_payload {
            self.push_provider_payload(payload.clone(), out);
        }
    }

    fn push_provider_payload(&self, payload: JsonPayload, out: &mut Vec<StreamEvent>) {
        out.push(StreamEvent::ProviderEvent {
            protocol: ProtocolFamily::AnthropicMessages,
            payload,
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
                crate::providers::openrouter::server_tools::is_official_profile(context.profile()),
            retain_anthropic_container:
                crate::providers::anthropic::code_execution::is_official_profile(context.profile())
                    || crate::providers::anthropic::code_execution::supports_execution(context),
            native_anthropic_error_route: context.native_anthropic_error_route().is_some(),
            native_context_window: context.native_anthropic_effective_context_window(),
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

fn text_delta(block: usize, text: DecodedText) -> StreamEvent {
    match text.utf16_code_units {
        Some(utf16_code_units) => StreamEvent::TextDeltaJsUtf16 {
            block,
            text: text.text,
            utf16_code_units,
        },
        None => StreamEvent::TextDelta {
            block,
            text: text.text,
        },
    }
}

fn append_text_units(pending: &mut PendingCitedTextBlock, fragment: &DecodedText) {
    if pending.utf16_code_units.is_none() && fragment.utf16_code_units.is_some() {
        pending.utf16_code_units = Some(pending.text.encode_utf16().collect());
    }
    if let Some(units) = &mut pending.utf16_code_units {
        match &fragment.utf16_code_units {
            Some(fragment_units) => units.extend(fragment_units),
            None => units.extend(fragment.text.encode_utf16()),
        }
        pending.text = String::from_utf16_lossy(units);
    } else {
        pending.text.push_str(&fragment.text);
    }
}

fn serialized_value_size(value: &Value) -> Result<usize, LlmError> {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .map_err(|_| native_block_limit_error())
}

#[cfg(test)]
mod stop_delta_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn stream_error_text_keeps_native_statusless_units_and_timeout_role() {
        for kind in ["api_error", "timeout_error"] {
            let mut decoder = AnthropicStreamDecoder::default();
            decoder.native_anthropic_error_route = true;
            let error = decoder
                .decode_frame(
                    format!(
                        r#"{{"type":"error","error":{{"type":"{kind}","message":"bad\ud800"}}}}"#
                    )
                    .as_bytes(),
                )
                .unwrap_err();
            let expected =
                format!(r#"{{"type":"error","error":{{"type":"{kind}","message":"bad\ud800"}}}}"#);
            assert_eq!(error.provider_error_text().unwrap().display(), expected);
            assert_eq!(
                error.provider_error_text().unwrap().utf16_code_units(),
                expected.encode_utf16().collect::<Vec<_>>().as_slice()
            );
            let parsed = JsonPayload::parse(&expected).unwrap();
            assert_eq!(
                parsed.string_units_at("/error/message").unwrap(),
                vec![b'b' as u16, b'a' as u16, b'd' as u16, 0xD800]
            );
            if kind == "timeout_error" {
                assert!(matches!(
                    error.classification_source(),
                    LlmError::ProviderTimeout { status: None, .. }
                ));
            }
        }
    }

    #[test]
    fn stop_delta_is_observable_before_stop_or_a_subsequent_error() {
        for reason in ["end_turn", "tool_use", "max_tokens", "refusal"] {
            for with_usage in [false, true] {
                let mut frame = json!({"type":"message_delta","delta":{"stop_reason":reason}});
                if with_usage {
                    frame["usage"] = json!({"output_tokens":2});
                }
                let expected_payload = JsonPayload::parse(&frame.to_string()).unwrap();
                for error in ["api_error", "timeout_error"] {
                    let mut decoder = AnthropicStreamDecoder::default();
                    let events = decoder
                        .decode_frame(&serde_json::to_vec(&frame).unwrap())
                        .unwrap();
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| matches!(
                                event,
                                StreamEvent::ProviderEvent { payload, .. }
                                    if payload == &expected_payload
                            ))
                            .count(),
                        1
                    );
                    assert!(!events.iter().any(|e| matches!(e, StreamEvent::End { .. })));
                    let failure = decoder
                        .decode_frame(
                            &serde_json::to_vec(
                                &json!({"type":"error","error":{"type":error,"message":"fixture"}}),
                            )
                            .unwrap(),
                        )
                        .unwrap_err();
                    assert!(matches!(
                        failure.classification_source(),
                        LlmError::ProviderInternal { .. } | LlmError::ProviderTimeout { .. }
                    ));
                }
                let mut decoder = AnthropicStreamDecoder::default();
                decoder
                    .decode_frame(&serde_json::to_vec(&frame).unwrap())
                    .unwrap();
                let events = decoder.decode_frame(br#"{"type":"message_stop"}"#).unwrap();
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| matches!(e, StreamEvent::End { .. }))
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn resumable_only_message_delta_is_retained_as_provider_event() {
        let source = r#"{"type":"message_delta","delta":{"resumable":false}}"#;
        let mut decoder = AnthropicStreamDecoder::default();
        let events = decoder.decode_frame(source.as_bytes()).unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ProviderEvent { payload, .. } if payload.as_json_source() == source
        )));
    }
}
