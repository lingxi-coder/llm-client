//! Response and error decoding for the Messages API.

use crate::protocol::{
    ChatResponse, ContentBlock, ConversationMessage, JsonPayload, LlmError, MessageRole,
    ProtocolFamily, StopReason, ToolUseId, Usage,
};
use crate::response_json::{DecodedText, FlatJsonKind, ResponseJson, ResponseJsonScanner};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub fn response(
    resp: &HttpResponse,
    context: Option<&crate::codecs::CodecContext>,
    retain_openrouter_container: bool,
    retain_anthropic_container: bool,
) -> Result<ChatResponse, LlmError> {
    let native_route = context.and_then(crate::codecs::CodecContext::native_anthropic_error_route);
    let is_http_error = !(200..300).contains(&resp.status);
    let response_text =
        (is_http_error && native_route.is_some()).then(|| ResponseJson::response_text(&resp.body));
    let response_bytes = response_text
        .as_deref()
        .map(str::as_bytes)
        .unwrap_or(&resp.body);
    let scanner = match ResponseJsonScanner::parse(
        response_bytes,
        "Anthropic response is not valid JSON",
    ) {
        Ok(scanner) => scanner,
        Err(_error) if is_http_error => {
            let (body, provider_text) = if native_route.is_some() {
                ResponseJson::native_anthropic_raw_http_error(resp.status, &resp.body)
            } else {
                ResponseJson::raw_http_error(resp.status, &resp.body)
            };
            let error = classify_error(
                resp.status,
                &body,
                retry_after(resp),
                native_route.is_some() && resp.status == 529,
            );
            return Err(match native_route {
                    Some((provider_id, protocol)) => error.with_native_anthropic_error(
                        provider_text,
                        native_error_facts(
                            provider_id,
                            protocol,
                            resp.status,
                            context.and_then(
                                crate::codecs::CodecContext::native_anthropic_thinking_resumption_beta_requested,
                            ).or(Some(false)),
                            None,
                            None,
                            resp,
                        ),
                    ),
                    None => error.with_provider_error_text(provider_text),
                });
        }
        Err(error) => return Err(error),
    };
    if is_http_error {
        let selected_text = scanner.provider_error_text_at("/error/message", None)?;
        let native_text = native_route
            .is_some()
            .then(|| scanner.native_anthropic_http_error_text(Some(resp.status)))
            .transpose()?;
        let provider_text = match &native_text {
            Some(text) => text.clone(),
            None => match &selected_text {
                Some(text) => text.prefixed(&format!("{} ", resp.status)),
                None => {
                    scanner.provider_error_text_or_body_at("/error/message", Some(resp.status))?
                }
            },
        };
        let body = scanner.provider_error_classification_body(
            "/error",
            &["type", "message"],
            selected_text.as_ref(),
        );
        let error = classify_error(
            resp.status,
            &body,
            retry_after(resp),
            native_route.is_some()
                && native_anthropic_retry_base_eligible(Some(resp.status), &scanner)?,
        );
        return Err(match native_route {
            Some((provider_id, protocol)) => error.with_native_anthropic_error(
                provider_text,
                native_error_facts(
                    provider_id,
                    protocol,
                    resp.status,
                    context.and_then(
                        crate::codecs::CodecContext::native_anthropic_thinking_resumption_beta_requested,
                    ).or(Some(false)),
                    native_text.as_ref(),
                    Some(&scanner),
                    resp,
                ),
            ),
            None => error.with_provider_error_text(provider_text),
        });
    }
    let body_payload = JsonPayload::parse(
        scanner
            .raw_source_at("")
            .expect("a parsed provider response has a root source"),
    )
    .map_err(|error| LlmError::InvalidRequest {
        message: format!("Anthropic response source is invalid JSON: {error}"),
    })?;
    let content_count =
        scanner
            .array_len("/content")
            .ok_or_else(|| LlmError::ProviderInternal {
                message: "provider response has no valid content array".to_owned(),
            })?;
    let mut opaque_paths = Vec::new();
    let mut raw_blocks = Vec::with_capacity(content_count);
    for index in 0..content_count {
        let pointer = format!("/content/{index}");
        let source = scanner
            .raw_source_at(&pointer)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Anthropic content block source was not retained".into(),
            })?;
        let fallback = scanner.string_at(&format!("{pointer}/type")) == Some("fallback");
        raw_blocks.push((fallback, source.to_owned()));
        if !fallback {
            opaque_paths.push(pointer);
        }
    }
    for pointer in ["/usage", "/stop_details", "/container", "/resumable"] {
        if scanner.raw_source_at(pointer).is_some() {
            opaque_paths.push(pointer.to_owned());
        }
    }
    let search_blocks = raw_blocks
        .iter()
        .filter(|(fallback, _)| !fallback)
        .filter_map(|(_, source)| JsonPayload::parse(source).ok())
        .collect::<Vec<_>>();
    let web_search = crate::codecs::web_search_decode::anthropic_payload_blocks(&search_blocks);
    let usage_payload = scanner
        .raw_source_at("/usage")
        .map(JsonPayload::parse)
        .transpose()
        .map_err(|error| LlmError::InvalidRequest {
            message: format!("Anthropic usage source is invalid JSON: {error}"),
        })?;
    let stop_details_payload = scanner
        .raw_source_at("/stop_details")
        .map(JsonPayload::parse)
        .transpose()
        .map_err(|error| LlmError::InvalidRequest {
            message: format!("Anthropic stop details source is invalid JSON: {error}"),
        })?;
    let container_payload = scanner
        .raw_source_at("/container")
        .map(JsonPayload::parse)
        .transpose()
        .map_err(|error| LlmError::InvalidRequest {
            message: format!("Anthropic container source is invalid JSON: {error}"),
        })?;
    let resumable_payload = scanner
        .raw_source_at("/resumable")
        .map(JsonPayload::parse)
        .transpose()
        .map_err(|error| LlmError::InvalidRequest {
            message: format!("Anthropic resumable source is invalid JSON: {error}"),
        })?;
    let mut response_json = scanner.into_response_json_with_opaque_paths(&opaque_paths)?;
    for pointer in ["/usage", "/stop_details", "/container", "/resumable"] {
        if response_json.take_raw_json_source_at(pointer)?.is_some() {
            // Their exact source is carried by the JsonPayload fields above.
        }
    }
    let body = response_json.value.clone();
    let has_valid_fallback = raw_blocks
        .iter()
        .filter(|(fallback, _)| *fallback)
        .filter_map(|(_, source)| JsonPayload::parse(source).ok())
        .any(|block| {
            crate::providers::anthropic::fallback_response::complete_block(&block).is_some()
        });
    let mut content = Vec::new();
    for (index, (fallback, source)) in raw_blocks.into_iter().enumerate() {
        if fallback {
            continue;
        }
        response_json
            .take_raw_json_source_at(&format!("/content/{index}"))?
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Anthropic content block source was not handed to its typed carrier"
                    .into(),
            })?;
        let payload = JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
            message: format!("Anthropic content block is invalid JSON: {error}"),
        })?;
        let kind = payload.string_at("/type");
        let valid = match kind.as_deref() {
            Some("text") => payload.kind_at("/text") == Some(FlatJsonKind::String),
            Some("redacted_thinking") => payload.kind_at("/data") == Some(FlatJsonKind::String),
            Some("tool_use") => {
                payload.kind_at("/id") == Some(FlatJsonKind::String)
                    && payload.kind_at("/name") == Some(FlatJsonKind::String)
            }
            Some(kind) => !kind.is_empty(),
            None => false,
        };
        if !valid {
            return Err(LlmError::ProviderInternal {
                message: format!(
                    "provider response contains malformed content block at index {index}"
                ),
            });
        }
        if kind.as_deref() == Some("text")
            && !has_only_known_text_fields(&payload)
            && payload
                .string_units_at("/text")
                .is_some_and(|units| String::from_utf16(&units).is_err())
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic opaque text blocks with lone UTF-16 units cannot be replayed losslessly".into(),
            });
        }
        let block = decode_source_block(payload).ok_or_else(|| LlmError::ProviderInternal {
            message: format!("provider response contains malformed content block at index {index}"),
        })?;
        content.push(block);
    }
    // Native Messages materializes a valid fallback block before removing its
    // control representation from the SDK replay. A malformed-only control
    // response has neither that materialized block nor ordinary content, so
    // preserve the native empty-response placeholder in the SDK projection.
    if content.is_empty() && content_count > 0 && !has_valid_fallback {
        content.push(ContentBlock::Text {
            text: "(no content)".into(),
            thought_signature: None,
            citations: Some(Some(JsonPayload::parse("[]").expect("empty array is JSON"))),
        });
    }
    response_json.finish()?;
    let typed_usage = usage_payload.as_ref().map(anthropic_usage_projection);
    let mut response = ChatResponse {
        inference: Default::default(),
        response_cache: None,
        web_search: crate::codecs::web_search_decode::with_usage_payload(
            web_search,
            usage_payload.as_ref(),
        ),
        file_search: None,
        native_metadata: Vec::new(),
        tool_input_issues: Vec::new(),
        message: ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: stop_reason(body.get("stop_reason").and_then(Value::as_str)),
        usage: crate::codecs::usage::report(
            typed_usage.as_ref(),
            &crate::codecs::usage::ANTHROPIC,
            usage,
            true,
        ),
        native_resume_context_exhausted: context.and_then(|context| {
            native_resume_context_exhausted(
                usage_payload.as_ref(),
                context.native_anthropic_effective_context_window(),
            )
        }),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        response_id: body
            .get("id")
            .and_then(Value::as_str)
            .map(crate::protocol::ResponseId::new),
        continuation: None,
        executed_profile: None,
    };
    response
        .set_anthropic_stop_details(stop_details_payload.filter(|payload| !payload.is_root_null()));
    let container = if retain_anthropic_container {
        container_payload
            .clone()
            .filter(|container| !container.is_root_null())
            .map(
                |envelope| crate::providers::anthropic::types::AnthropicContainerMetadata {
                    envelope,
                },
            )
    } else {
        None
    };
    response.set_anthropic_metadata(container, usage_payload.clone());
    if retain_openrouter_container {
        response.set_openrouter_container(
            container_payload
                .filter(|container| !container.is_root_null())
                .map(
                    |envelope| crate::providers::openrouter::types::OpenRouterContainerMetadata {
                        envelope,
                    },
                ),
        );
    }
    response.set_anthropic_fallback(
        crate::providers::anthropic::fallback_response::from_nonstream_body(&body_payload),
    );
    response.set_anthropic_resumable(resumable_payload);
    Ok(response)
}

#[cfg(test)]
mod non_json_error_tests {
    use super::*;
    #[test]
    fn non_json_rate_limit_retains_http_status_and_retry_after() {
        for body in ["", "gateway unavailable", "<html>rate limited</html>"] {
            let response = HttpResponse {
                status: 429,
                headers: vec![("retry-after".into(), "7".into())],
                body: body.as_bytes().to_vec().into(),
            };
            assert!(
                super::response(&response, None, false, false).is_err_and(|error| matches!(error.classification_source(), LlmError::RateLimited {retry_after:Some(delay),..} if *delay == Duration::from_secs(7)))
            );
        }
    }
    #[test]
    fn non_json_success_cannot_become_model_content() {
        let response = HttpResponse {
            status: 200,
            headers: vec![],
            body: br#"not json"#.as_slice().into(),
        };
        assert!(super::response(&response, None, false, false).is_err());
    }
}

/// Project only the counters consumed by the typed Usage/UsageReport API.
/// The complete provider object remains in `AnthropicResponseMetadata` as a
/// JsonPayload; this shallow view avoids recursive Value materialization.
pub(crate) fn anthropic_usage_projection(payload: &JsonPayload) -> Value {
    use crate::response_json::FlatJsonKind;

    fn scalar(payload: &JsonPayload, pointer: &str) -> Option<Value> {
        let kind = payload.kind_at(pointer)?;
        if matches!(kind, FlatJsonKind::Object | FlatJsonKind::Array) {
            return Some(if kind == FlatJsonKind::Object {
                Value::Object(Default::default())
            } else {
                Value::Array(Vec::new())
            });
        }
        if kind == FlatJsonKind::String {
            return Some(Value::String(
                payload.string_at(pointer).unwrap_or_default(),
            ));
        }
        serde_json::from_str(&payload.raw_source_at(pointer)?).ok()
    }

    fn insert_pointer(root: &mut Value, pointer: &str, value: Value) {
        let mut parts = pointer.trim_start_matches('/').split('/').peekable();
        let Some(mut current) = root.as_object_mut() else {
            return;
        };
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                current.insert(part.to_owned(), value);
                return;
            }
            let child = current
                .entry(part.to_owned())
                .or_insert_with(|| Value::Object(Default::default()));
            let Some(object) = child.as_object_mut() else {
                return;
            };
            current = object;
        }
    }

    if payload.is_root_null() {
        return Value::Null;
    }
    if !payload.is_root_object() {
        return scalar(payload, "").unwrap_or(Value::Null);
    }

    let mut result = Value::Object(Default::default());
    for pointer in [
        "/input_tokens",
        "/output_tokens",
        "/cache_read_input_tokens",
        "/cache_creation_input_tokens",
        "/total_tokens",
        "/reasoning_output_tokens",
        "/cache_creation",
        "/output_tokens_details",
        "/server_tool_use",
        "/speed",
        "/cache_creation/ephemeral_5m_input_tokens",
        "/cache_creation/ephemeral_1h_input_tokens",
        "/output_tokens_details/thinking_tokens",
        "/server_tool_use/web_search_requests",
        "/server_tool_use/web_fetch_requests",
        "/server_tool_use/file_search_requests",
        "/server_tool_use/code_execution_requests",
    ] {
        if let Some(value) = scalar(payload, pointer) {
            insert_pointer(&mut result, pointer, value);
        }
    }
    result
}

/// Native `Rte`/`OP` usage selection followed by the `Oqo` 2048-token gate.
/// The final context window is supplied only from a trusted Native source;
/// absent context is not replaced with the SDK's generic advertised window.
pub(crate) fn native_resume_context_exhausted(
    payload: Option<&JsonPayload>,
    context_window: Option<f64>,
) -> Option<bool> {
    use FlatJsonKind::{Null, Number, Object};

    let payload = payload?;
    if payload.kind_at("") != Some(Object) {
        return None;
    }

    fn top_bucket(payload: &JsonPayload, pointer: &str) -> Option<f64> {
        match payload.kind_at(pointer) {
            None | Some(Null) => Some(0.0),
            Some(Number) => payload.number_at(pointer),
            _ => None,
        }
    }

    fn valid_iteration_buckets(payload: &JsonPayload, pointer: &str) -> Option<[f64; 4]> {
        let kind = payload.string_at(&format!("{pointer}/type"))?;
        if !matches!(kind.as_str(), "message" | "fallback_message") {
            return None;
        }
        let mut buckets = [0.0; 4];
        for (index, name) in [
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
        .into_iter()
        .enumerate()
        {
            let value = payload.number_at(&format!("{pointer}/{name}"))?;
            if !value.is_finite() || value < 0.0 {
                return None;
            }
            buckets[index] = value;
        }
        ((buckets[0] + buckets[2] + buckets[3]) > 0.0).then_some(buckets)
    }

    let names = [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ];
    let mut top = [0.0; 4];
    for (index, name) in names.into_iter().enumerate() {
        top[index] = top_bucket(payload, &format!("/{name}"))?;
    }

    // Native uses `findLast(!advisor_message && !compaction)`, then validates
    // only that selected item; an invalid last candidate falls back to top-level.
    let top_prompt = top[0] + top[2] + top[3];
    let selected = if top_prompt == 0.0 {
        None
    } else if let Some(length) = payload.array_len("/iterations") {
        let mut candidate = None;
        for index in (0..length).rev() {
            let pointer = format!("/iterations/{index}");
            if matches!(
                payload.string_at(&format!("{pointer}/type")).as_deref(),
                Some("advisor_message" | "compaction")
            ) {
                continue;
            }
            candidate = Some(pointer);
            break;
        }
        candidate.and_then(|pointer| valid_iteration_buckets(payload, &pointer))
    } else {
        None
    };
    let buckets = selected.unwrap_or(top);
    let total = buckets[0] + buckets[2] + buckets[3] + buckets[1];
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Some(false);
    }
    let context_window = context_window.filter(|window| window.is_finite() && *window > 0.0)?;
    Some(total + 2048.0 >= context_window)
}

fn decode_source_block(value: JsonPayload) -> Option<ContentBlock> {
    let provider_content = || ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: value.clone(),
    };
    let kind = value.string_at("/type")?;
    match kind.as_str() {
        "text" if value.has_only_fields_at("", &["type", "text", "citations"]) => {
            let text = value.string_at("/text")?;
            let utf16_code_units = value.string_units_at("/text")?;
            let citations = payload_citations(&value).ok()?;
            let decoded = DecodedText {
                text,
                utf16_code_units: String::from_utf16(&utf16_code_units)
                    .is_err()
                    .then_some(utf16_code_units),
            };
            Some(text_block(decoded, citations))
        }
        "tool_use" => {
            let id = payload_scalar_string(&value, "/id")?;
            let name = payload_scalar_string(&value, "/name")?;
            let input = value
                .raw_source_at("/input")
                .map(|source| JsonPayload::parse(&source))
                .transpose()
                .ok()?
                .unwrap_or(JsonPayload::parse("null").ok()?);
            let caller = value
                .raw_source_at("/caller")
                .map(|source| JsonPayload::parse(&source))
                .transpose()
                .ok()?;
            let toolset_name = payload_optional_scalar_string(&value, "/toolset_name")?;
            Some(ContentBlock::ToolUse {
                id: ToolUseId::new(id),
                name,
                input,
                provider_id: None,
                caller,
                toolset_name,
                thought_signature: None,
            })
        }
        "thinking" if value.has_only_fields_at("", &["type", "thinking", "signature"]) => {
            Some(ContentBlock::Thinking {
                text: payload_scalar_string(&value, "/thinking")?,
                signature: payload_optional_scalar_string(&value, "/signature")?,
            })
        }
        "redacted_thinking" if value.has_only_fields_at("", &["type", "data"]) => {
            Some(ContentBlock::RedactedThinking {
                data: payload_scalar_string(&value, "/data")?,
            })
        }
        _ => Some(provider_content()),
    }
}

fn payload_scalar_string(value: &JsonPayload, pointer: &str) -> Option<String> {
    let units = value.string_units_at(pointer)?;
    String::from_utf16(&units).ok()
}

fn payload_optional_scalar_string(value: &JsonPayload, pointer: &str) -> Option<Option<String>> {
    if value.raw_source_at(pointer).is_none() {
        Some(None)
    } else {
        payload_scalar_string(value, pointer).map(Some)
    }
}

fn payload_citations(value: &JsonPayload) -> Result<Option<Option<JsonPayload>>, ()> {
    match value.raw_source_at("/citations") {
        None => Ok(None),
        Some(_) if value.kind_at("/citations") == Some(FlatJsonKind::Null) => Ok(Some(None)),
        Some(source) => JsonPayload::parse(&source)
            .map(|citations| Some(Some(citations)))
            .map_err(|_| ()),
    }
}

fn text_block(text: DecodedText, citations: Option<Option<JsonPayload>>) -> ContentBlock {
    match text.utf16_code_units {
        Some(utf16_code_units) => ContentBlock::TextJsUtf16 {
            text: text.text,
            utf16_code_units,
            thought_signature: None,
            citations,
        },
        None => ContentBlock::Text {
            text: text.text,
            thought_signature: None,
            citations,
        },
    }
}

/// Decode one source-backed Messages block without materializing arbitrary provider fields.
pub fn decode_block(value: &JsonPayload) -> Option<ContentBlock> {
    decode_source_block(value.clone())
}

/// Only these text-block fields have a typed carrier in the current SDK.
pub(crate) fn has_only_known_text_fields(block: &JsonPayload) -> bool {
    block.has_only_fields_at("", &["type", "text", "citations"])
}

/// Mirrors 2.1.292 `vU(error)` only for the current Anthropic HTTP error
/// route, whose provider response is constructed as Native's `Rt` error.
///
/// Native checks status 529 first. Otherwise `Rt.makeMessage` uses a truthy
/// top-level string message verbatim, JSON-stringifies a truthy non-string
/// message value, and JSON-stringifies the full body when the message is falsy.
/// The compact marker is searched in that final message, not in arbitrary raw
/// bytes or a nested provider error field.
pub(super) fn native_anthropic_retry_base_eligible(
    status: impl Into<Option<u16>>,
    scanner: &ResponseJsonScanner,
) -> Result<bool, LlmError> {
    const MARKER: &str = "\"type\":\"overloaded_error\"";
    let status = status.into();
    if status == Some(529) {
        return Ok(true);
    }
    let message = scanner.native_anthropic_http_error_text(None)?;
    Ok(message.display().contains(MARKER))
}

fn native_error_facts(
    provider_id: crate::protocol::ProviderId,
    protocol: ProtocolFamily,
    status: u16,
    beta_requested: Option<bool>,
    native_text: Option<&crate::protocol::ProviderErrorText>,
    scanner: Option<&ResponseJsonScanner>,
    response: &HttpResponse,
) -> crate::protocol::NativeAnthropicErrorFacts {
    // `gVr` consumes apiErrorStatus and the normalized `apiError` value. The
    // current jio path maps these two exact nested error codes to its excluded
    // semantic reasons; `apiErrorCode` itself is a different field and is not
    // consumed by Hqo.
    let normalized_error_code =
        scanner.and_then(|scanner| scanner.string_at("/error/error/details/error_code"));
    let thinking_resumption_refused = (400..500).contains(&status)
        && !matches!(status, 401 | 408 | 409 | 429 | 451)
        && !matches!(
            normalized_error_code,
            Some("dlp_request_denied" | "claude_code_version_too_old")
        );
    crate::protocol::NativeAnthropicErrorFacts {
        provider_id,
        protocol,
        request_authority: None,
        thinking_resumption_refused,
        thinking_resumption_beta_requested: beta_requested,
        thinking_resumption_beta_refused: status == 400
            && beta_requested == Some(true)
            && native_text.is_some_and(|text| {
                let message = text.display();
                (message.contains("thinking-resumption-2026-07-17")
                    && (message.contains("anthropic-beta") || message.contains("anthropic_beta")))
                    || crate::providers::anthropic::beta_repair::capability_rejected(
                        message,
                        "beta_header:thinking-resumption-2026-07-17",
                    )
            }),
        response: Some(crate::protocol::NativeAnthropicErrorResponse {
            body: scanner
                .and_then(|scanner| scanner.raw_source_at(""))
                .and_then(|source| JsonPayload::parse(source).ok()),
            unified_representative_claim: response
                .header("anthropic-ratelimit-unified-representative-claim")
                .map(str::to_owned),
            unified_overage_disabled_reason: response
                .header("anthropic-ratelimit-unified-overage-disabled-reason")
                .map(str::to_owned),
        }),
    }
}

/// Map a non-success response onto the taxonomy.
///
/// Two shapes mean "the transcript no longer fits" and both must become
/// `ContextOverflow`, because that and only that drives reactive compaction
/// (gate 31): a `request_too_large` that mentions the context window, and an
/// `invalid_request_error` whose message says the prompt is too long.
pub fn classify_error(
    status: u16,
    body: &Value,
    retry_after: Option<Duration>,
    native_retry_base_eligible: bool,
) -> LlmError {
    classify_error_for_status(Some(status), body, retry_after, native_retry_base_eligible)
}

/// Classify an SSE-created Native `Rt` error, which has no HTTP status. The
/// provider error text remains the separate exact Native message carrier.
pub(crate) fn classify_stream_error(body: &Value, native_retry_base_eligible: bool) -> LlmError {
    classify_error_for_status(None, body, None, native_retry_base_eligible)
}

fn classify_error_for_status(
    status: Option<u16>,
    body: &Value,
    retry_after: Option<Duration>,
    native_retry_base_eligible: bool,
) -> LlmError {
    let error = body.get("error");
    let kind = error
        .and_then(|e| e.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let message = error
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let display = if message.is_empty() {
        status.map_or_else(|| body.to_string(), |status| format!("{status} {body}"))
    } else {
        status.map_or_else(|| message.clone(), |status| format!("{status} {message}"))
    };
    let lower = message.to_ascii_lowercase();

    // Native uB classifies every HTTP 529 as overload, even when a gateway
    // supplies a different error type. Keep its diagnostic message intact.
    if status == Some(529) {
        return LlmError::Overloaded {
            message: display,
            native_retry_base_eligible,
        };
    }

    match kind {
        "authentication_error" => LlmError::Authentication { message: display },
        "permission_error" => LlmError::PermissionDenied { message: display },
        "not_found_error" => LlmError::ModelUnavailable { message: display },
        "rate_limit_error" => LlmError::RateLimited {
            message: display,
            retry_after,
        },
        // The same split the other wire makes on 413: the context window is a
        // token overflow compaction can fix; anything else is accumulated
        // attachment bytes, which it cannot.
        "request_too_large" if lower.contains("context window") => LlmError::ContextOverflow {
            message: display,
            limit: None,
            actual: None,
        },
        "request_too_large" => LlmError::RequestTooLarge { message: display },
        "invalid_request_error" if lower.contains("prompt is too long") => {
            let (actual, limit) = prompt_too_long_counts(&message);
            LlmError::ContextOverflow {
                message: display,
                limit,
                actual,
            }
        }
        "invalid_request_error" => LlmError::InvalidRequest { message: display },
        "overloaded_error" => LlmError::Overloaded {
            message: display,
            native_retry_base_eligible,
        },
        "api_error" => LlmError::ProviderInternal { message: display },
        "timeout_error" => LlmError::ProviderTimeout {
            message: display,
            status,
        },
        _ => match status {
            Some(401) => LlmError::Authentication { message: display },
            Some(403) => LlmError::PermissionDenied { message: display },
            Some(404) => LlmError::ModelUnavailable { message: display },
            Some(413) if lower.contains("context window") => LlmError::ContextOverflow {
                message: display,
                limit: None,
                actual: None,
            },
            Some(413) => LlmError::RequestTooLarge { message: display },
            Some(429) => LlmError::RateLimited {
                message: display,
                retry_after,
            },
            Some(400 | 422) => LlmError::InvalidRequest { message: display },
            Some(529) => LlmError::Overloaded {
                message: display,
                native_retry_base_eligible,
            },
            _ => LlmError::ProviderInternal { message: display },
        },
    }
}

/// Pull `actual` and `limit` out of "prompt is too long: 260085 tokens > 256000".
///
/// Parsed from the RAW provider message, never from the display string: the
/// display carries a leading status, and that would be the first digit run
/// found. Hand-rolled rather than a regex so this crate keeps its dependencies.
fn prompt_too_long_counts(message: &str) -> (Option<u64>, Option<u64>) {
    let lower = message.to_ascii_lowercase();
    let Some(i) = lower.find("prompt is too long") else {
        return (None, None);
    };
    let rest = &message[i + "prompt is too long".len()..];
    let mut nums = Vec::new();
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() && nums.len() < 2 {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if let Ok(n) = rest[start..i].parse::<u64>() {
                nums.push(n);
            }
        } else {
            i += 1;
        }
    }
    match nums.len() {
        2 => (Some(nums[0]), Some(nums[1])),
        1 => (Some(nums[0]), None),
        _ => (None, None),
    }
}

pub fn stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("refusal") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::EndTurn,
    }
}

/// This wire reports cache reads and writes separately, which is the whole
/// reason `Usage` has both fields.
/// The one wire whose buckets are already disjoint: `input_tokens` counts only
/// what follows the last cache breakpoint, so the four counters sum to the bill
/// with nothing to subtract. Newer models report a thinking-token subset of output; absent breakdowns
/// stay zero rather than being inferred from visible thinking text.
pub fn usage(u: &Value) -> Usage {
    let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
    let web_search_requests = u
        .pointer("/server_tool_use/web_search_requests")
        .and_then(Value::as_u64);
    let web_fetch_requests = u
        .pointer("/server_tool_use/web_fetch_requests")
        .and_then(Value::as_u64);
    let file_search_requests = u
        .pointer("/server_tool_use/file_search_requests")
        .and_then(Value::as_u64);
    let code_execution_requests = u
        .pointer("/server_tool_use/code_execution_requests")
        .and_then(Value::as_u64);
    Usage {
        input_tokens: n("input_tokens"),
        output_tokens: n("output_tokens"),
        cache_read_tokens: n("cache_read_input_tokens"),
        cache_write_tokens: n("cache_creation_input_tokens"),
        cache_write_1h_tokens: u
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        reasoning_tokens: u
            .pointer("/output_tokens_details/thinking_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cost: None,
        server_tool_usage: (web_search_requests.is_some()
            || web_fetch_requests.is_some()
            || file_search_requests.is_some()
            || code_execution_requests.is_some())
        .then_some(crate::protocol::ServerToolUsage {
            web_search_requests,
            web_fetch_requests,
            web_extractor_requests: None,
            file_search_requests,
            code_interpreter_requests: code_execution_requests,
        }),
    }
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .and_then(|v| Duration::try_from_secs_f64(v / 1000.0).ok())
        .or_else(|| {
            resp.header("retry-after")
                .and_then(|v| v.trim().parse::<f64>().ok())
                .and_then(|v| Duration::try_from_secs_f64(v).ok())
        })
}

#[cfg(test)]
mod complete_response_validation_tests {
    use super::*;
    use serde_json::json;

    fn native_context(beta_requested: Option<bool>) -> crate::codecs::CodecContext {
        let profile: crate::protocol::ProviderProfile = serde_json::from_value(json!({
            "profile_name": "native",
            "provider_id": "anthropic",
            "base_url": "https://api.anthropic.com",
            "protocol": "anthropic_messages",
            "auth": "none",
            "models": [{
                "display_model": "Claude",
                "request_model": "claude-opus-5",
                "billing_model": "claude-opus-5"
            }]
        }))
        .unwrap();
        crate::codecs::CodecContext::new(
            &profile,
            "claude-opus-5",
            crate::codecs::RequestMode::Complete,
        )
        .with_native_anthropic_context(crate::client::NativeAnthropicContextInput {
            thinking_resumption_beta_requested: beta_requested,
            ..Default::default()
        })
    }

    fn response(
        resp: &HttpResponse,
        retain_openrouter_container: bool,
        retain_anthropic_container: bool,
    ) -> Result<ChatResponse, LlmError> {
        super::response(
            resp,
            Some(&native_context(None)),
            retain_openrouter_container,
            retain_anthropic_container,
        )
    }

    #[test]
    fn native_retry_base_gate_uses_rt_message_before_serialized_body() {
        let parse = |source: &str| {
            ResponseJsonScanner::parse(source.as_bytes(), "invalid fixture").unwrap()
        };
        assert!(
            native_anthropic_retry_base_eligible(529, &parse(r#"{"message":"busy"}"#),).unwrap()
        );
        assert!(!native_anthropic_retry_base_eligible(
            503,
            &parse(r#"{"message":"busy","error":{"type":"overloaded_error"}}"#),
        )
        .unwrap());
        assert!(native_anthropic_retry_base_eligible(
            503,
            &parse(r#"{ "error": { "type": "overloaded_error" } }"#),
        )
        .unwrap());
        assert!(native_anthropic_retry_base_eligible(
            503,
            &parse(r#"{"message":{"type":"overloaded_error"}}"#),
        )
        .unwrap());
        assert!(native_anthropic_retry_base_eligible(
            503,
            &parse(r#"{"message":[{"type":"overloaded_error"}]}"#),
        )
        .unwrap());
        assert!(!native_anthropic_retry_base_eligible(
            503,
            &parse(r#"{"message":["\"type\":\"overloaded_error\""]}"#),
        )
        .unwrap());
    }

    #[test]
    fn decoded_anthropic_overload_carries_the_native_rt_gate() {
        let classify = |status, body: &str| {
            response(
                &HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: body.as_bytes().to_vec().into(),
                },
                false,
                false,
            )
            .unwrap_err()
        };
        let status_529 = classify(529, r#"{"message":"busy","error":{"type":"api_error"}}"#);
        assert!(matches!(
            status_529.classification_source(),
            LlmError::Overloaded {
                native_retry_base_eligible: true,
                ..
            }
        ));
        let truthy_message = classify(
            503,
            r#"{"message":"busy","error":{"type":"overloaded_error"}}"#,
        );
        assert!(matches!(
            truthy_message.classification_source(),
            LlmError::Overloaded {
                native_retry_base_eligible: false,
                ..
            }
        ));
        let serialized_body = classify(503, r#"{ "error": { "type": "overloaded_error" } }"#);
        assert!(matches!(
            serialized_body.classification_source(),
            LlmError::Overloaded {
                native_retry_base_eligible: true,
                ..
            }
        ));
        let object_message = classify(
            503,
            r#"{"message":{"type":"overloaded_error"},"error":{"type":"overloaded_error","message":"busy"}}"#,
        );
        assert!(matches!(
            object_message.classification_source(),
            LlmError::Overloaded {
                native_retry_base_eligible: true,
                ..
            }
        ));
        let array_message = classify(
            503,
            r#"{"message":[{"type":"overloaded_error"}],"error":{"type":"overloaded_error","message":"busy"}}"#,
        );
        assert!(matches!(
            array_message.classification_source(),
            LlmError::Overloaded {
                native_retry_base_eligible: true,
                ..
            }
        ));
    }

    #[test]
    fn native_resume_usage_gate_keeps_unknown_context_and_zero_distinct() {
        let parse = |source: &str| JsonPayload::parse(source).unwrap();
        let zero = parse(
            r#"{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}"#,
        );
        assert_eq!(
            native_resume_context_exhausted(Some(&zero), None),
            Some(false)
        );

        let positive = parse(
            r#"{"input_tokens":99,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}"#,
        );
        assert_eq!(native_resume_context_exhausted(Some(&positive), None), None);
        assert_eq!(
            native_resume_context_exhausted(Some(&positive), Some(2_148.0)),
            Some(true)
        );
        assert_eq!(
            native_resume_context_exhausted(Some(&positive), Some(2_149.0)),
            Some(false)
        );
    }

    #[test]
    fn native_resume_usage_gate_uses_last_valid_non_advisor_iteration() {
        let selected = JsonPayload::parse(
            r#"{"input_tokens":50,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"iterations":[{"type":"advisor_message","input_tokens":900,"output_tokens":900,"cache_creation_input_tokens":0,"cache_read_input_tokens":0},{"type":"message","input_tokens":20,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":1}]}"#,
        )
        .unwrap();
        assert_eq!(
            native_resume_context_exhausted(Some(&selected), Some(2_080.0)),
            Some(false)
        );

        let invalid_last = JsonPayload::parse(
            r#"{"input_tokens":50,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"iterations":[{"type":"message","input_tokens":2}]}"#,
        )
        .unwrap();
        assert_eq!(
            native_resume_context_exhausted(Some(&invalid_last), Some(2_080.0)),
            Some(true)
        );
    }

    #[test]
    fn native_api_error_facts_separate_hqo_refusal_from_beta_latch() {
        let decode = |status, body: &str, beta_requested: Option<bool>| {
            super::response(
                &HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: body.as_bytes().to_vec().into(),
                },
                Some(&native_context(beta_requested)),
                false,
                false,
            )
            .unwrap_err()
        };

        let refused = decode(
            422,
            r#"{"message":"prompt rejected","error":{"type":"invalid_request_error","message":"prompt rejected"}}"#,
            None,
        );
        let facts = refused.native_anthropic_error_facts().unwrap();
        assert!(facts.thinking_resumption_refused);
        assert_eq!(facts.thinking_resumption_beta_requested, Some(false));
        assert!(!facts.thinking_resumption_beta_refused);

        let semantic_exclusion = decode(
            400,
            r#"{"message":"blocked","error":{"type":"invalid_request_error","message":"blocked","error":{"details":{"error_code":"dlp_request_denied"}}}}"#,
            None,
        );
        assert!(
            !semantic_exclusion
                .native_anthropic_error_facts()
                .unwrap()
                .thinking_resumption_refused
        );

        let beta_rejection = decode(
            400,
            r#"{"message":"anthropic-beta rejected thinking-resumption-2026-07-17","error":{"type":"invalid_request_error","message":"rejected"}}"#,
            Some(true),
        );
        let facts = beta_rejection.native_anthropic_error_facts().unwrap();
        assert!(facts.thinking_resumption_refused);
        assert_eq!(facts.thinking_resumption_beta_requested, Some(true));
        assert!(facts.thinking_resumption_beta_refused);
    }

    #[test]
    fn native_error_facts_retain_full_json_body_and_only_model_validator_headers() {
        let body = r#"{"message":"billing limit","error":{"type":"rate_limit_error","message":"billing limit","details":{"error_code":"credits_required","disabled_reason":"out_of_credits"}}}"#;
        let error = super::response(
            &HttpResponse {
                status: 429,
                headers: vec![
                    (
                        "anthropic-ratelimit-unified-representative-claim".into(),
                        "claimed".into(),
                    ),
                    (
                        "anthropic-ratelimit-unified-overage-disabled-reason".into(),
                        "out_of_credits".into(),
                    ),
                    ("unrelated-header".into(), "must not be retained".into()),
                ],
                body: body.as_bytes().to_vec().into(),
            },
            Some(&native_context(None)),
            false,
            false,
        )
        .unwrap_err();

        let facts = error.native_anthropic_error_facts().unwrap();
        let response = facts.response.as_ref().unwrap();
        assert_eq!(
            response.body.as_ref().map(JsonPayload::as_json_source),
            Some(body)
        );
        assert_eq!(
            response.unified_representative_claim.as_deref(),
            Some("claimed")
        );
        assert_eq!(
            response.unified_overage_disabled_reason.as_deref(),
            Some("out_of_credits")
        );
        assert!(!format!("{facts:?}").contains("billing limit"));
        assert!(!format!("{facts:?}").contains("must not be retained"));
    }

    fn decode_block_fixture(value: Value) -> Option<ContentBlock> {
        let payload = JsonPayload::parse(&value.to_string()).unwrap();
        decode_block(&payload)
    }

    fn decode(content: Value) -> Result<ChatResponse, LlmError> {
        response(
            &HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: serde_json::to_vec(&json!({
                    "model": "claude-opus-5", "stop_reason": "end_turn",
                    "content": content, "usage": {"input_tokens": 3, "output_tokens": 5}
                }))
                .unwrap()
                .into(),
            },
            false,
            false,
        )
    }

    #[test]
    fn http_error_uses_native_top_message_or_complete_json_message() {
        let error = response(
            &HttpResponse {
                status: 400,
                headers: Vec::new(),
                body: br#"{"error":{"type":"invalid_request_error","message":"bad\ud800"}}"#
                    .as_slice()
                    .into(),
            },
            false,
            false,
        )
        .unwrap_err();
        let LlmError::InvalidRequest { message } = error.classification_source() else {
            panic!("HTTP 400 must retain InvalidRequest classification");
        };
        assert_eq!(message, "400 bad�");
        let text = error.provider_error_text().unwrap();
        let expected = r#"400 {"error":{"type":"invalid_request_error","message":"bad\ud800"}}"#;
        assert_eq!(text.display(), expected);
        assert_eq!(
            text.utf16_code_units(),
            expected.encode_utf16().collect::<Vec<_>>()
        );
    }

    #[test]
    fn malformed_blocks_fail_the_whole_response_instead_of_being_omitted() {
        for malformed in [
            json!({"type":"text"}),
            json!({"type":"text", "text":17}),
            json!({"type":"redacted_thinking"}),
            json!({"type":"tool_use", "name":"read", "input":{}}),
            json!({"type":"tool_use", "id":"tool_1", "name":false}),
            json!({"type":"text", "citations":[{"type":"future_citation"}]}),
            json!(null),
            json!("text"),
            json!({}),
            json!({"type":7}),
            json!({"type":""}),
        ] {
            let result = decode(json!([
                {"type":"text", "text":"valid prefix"},
                {"type":"server_tool_use", "id":"srv_1", "name":"web_search", "input":{"query":"x"}},
                malformed,
                {"type":"text", "text":"valid suffix"}
            ]));
            assert!(
                matches!(result, Err(LlmError::ProviderInternal { ref message }) if message.contains("index 2")),
                "{malformed}: {result:?}"
            );
        }
    }

    #[test]
    fn explicit_null_citations_remain_typed_text_metadata() {
        let block =
            decode_block_fixture(json!({"type":"text","text":"x","citations":null})).unwrap();
        assert!(matches!(
            block,
            ContentBlock::Text {
                citations: Some(None),
                ..
            }
        ));
        let empty = decode_block_fixture(json!({"type":"text","text":"x","citations":[]})).unwrap();
        assert!(
            matches!(empty, ContentBlock::Text { citations: Some(Some(value)), .. } if value.as_json_source() == "[]")
        );
        let cited = decode_block_fixture(json!({
            "type":"text",
            "text":"x",
            "citations":[{"type":"web_search_result_location","url":"https://example.test"}]
        }))
        .unwrap();
        assert!(matches!(
            cited,
            ContentBlock::Text {
                citations: Some(Some(value)),
                ..
            } if value.as_json_source() == r#"[{"type":"web_search_result_location","url":"https://example.test"}]"#
        ));
    }

    #[test]
    fn unknown_text_fields_remain_one_raw_native_block() {
        for raw in [
            json!({
                "type":"text",
                "text":"plain",
                "future_annotation":{"opaque":[1,2,3]}
            }),
            json!({
            "type":"text",
            "text":"cited",
            "citations":[{"type":"future_citation"}],
            "future_annotation":{"opaque":[1,2,3]}
            }),
        ] {
            let raw_source = raw.to_string();
            let decoded = decode_block_fixture(raw.clone()).unwrap();
            assert!(matches!(
                decoded,
                ContentBlock::ProviderContent { value, .. } if value.as_json_source() == raw_source
            ));
        }
    }

    #[test]
    fn normal_and_hosted_native_blocks_keep_their_order_and_payload() {
        let hosted = vec![
            json!({"type":"server_tool_use", "id":"srv_1", "name":"web_search", "input":{"query":"x"}}),
            json!({"type":"web_search_tool_result", "tool_use_id":"srv_1", "content":[]}),
            json!({"type":"mcp_tool_use", "id":"mcp_1", "name":"lookup", "server_name":"docs", "input":{}}),
            json!({"type":"future_provider_block", "opaque":{"retain":[1,2,3]}}),
            json!({"type":"text", "text":"cited", "citations":[{"type":"future_citation", "opaque":true}], "provider_metadata":{"keep":"this"}}),
        ];
        let mut content = vec![
            json!({"type":"text", "text":"answer"}),
            json!({"type":"tool_use", "id":"tool_1", "name":"read", "input":{"path":"a"}}),
        ];
        content.extend(hosted.clone());
        let decoded = decode(json!(content)).unwrap();
        assert!(
            matches!(&decoded.message.content[0], ContentBlock::Text { text, .. } if text == "answer")
        );
        assert!(
            matches!(&decoded.message.content[1], ContentBlock::ToolUse { name, input, .. } if name == "read" && input.as_json_source() == r#"{"path":"a"}"#)
        );
        assert_eq!(decoded.message.content.len(), content.len());
        for (actual, expected) in decoded.message.content[2..].iter().zip(hosted) {
            let expected = JsonPayload::parse(&expected.to_string()).unwrap();
            assert!(
                matches!(actual, ContentBlock::ProviderContent { value, .. } if value == &expected)
            );
        }
    }
}

#[cfg(test)]
mod stop_details_tests {
    use super::*;
    use serde_json::json;

    fn response(
        resp: &HttpResponse,
        retain_openrouter_container: bool,
        retain_anthropic_container: bool,
    ) -> Result<ChatResponse, LlmError> {
        super::response(
            resp,
            None,
            retain_openrouter_container,
            retain_anthropic_container,
        )
    }

    #[test]
    fn http_529_preserves_overload_category_with_nonstandard_payloads() {
        for kind in [
            "api_error",
            "timeout_error",
            "invalid_request_error",
            "unknown",
        ] {
            let error = classify_error(
                529,
                &json!({"error":{"type":kind,"message":"fixture"}}),
                None,
                true,
            );
            assert!(matches!(
                error,
                LlmError::Overloaded {
                    message,
                    native_retry_base_eligible: true
                } if message == "529 fixture"
            ));
        }
    }

    #[test]
    fn stop_details_survive_decode_and_other_metadata_updates() {
        let details_value = json!({"type":"refusal","future":{"reason":"blocked"}});
        let details = JsonPayload::from_generated_value(&details_value).unwrap();
        let wire = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: serde_json::to_vec(
                &json!({"model":"m","content":[],"stop_reason":"refusal","stop_details":details}),
            )
            .unwrap()
            .into(),
        };
        let mut result = response(&wire, false, false).unwrap();
        assert_eq!(result.anthropic_stop_details(), Some(&details));
        let resumable = JsonPayload::parse(r#"{"future":null}"#).unwrap();
        result.set_anthropic_resumable(Some(resumable.clone()));
        let usage = JsonPayload::from_generated_value(&json!({"output_tokens":1})).unwrap();
        result.set_anthropic_metadata(None, Some(usage.clone()));
        assert_eq!(result.anthropic_stop_details(), Some(&details));
        assert_eq!(result.anthropic_resumable(), Some(&resumable));
        result.set_anthropic_stop_details(Some(
            JsonPayload::from_generated_value(&json!({"type":"other"})).unwrap(),
        ));
        assert_eq!(result.anthropic_usage(), Some(&usage));
    }

    #[test]
    fn deep_usage_and_stop_details_stay_source_backed_during_decode() {
        let nested = format!("{}0{}", "[".repeat(512), "]".repeat(512));
        let source = format!(
            r#"{{"model":"m","content":[],"stop_reason":"end_turn","usage":{{"input_tokens":3,"output_tokens":1,"future":"\ud800","deep":{nested}}},"stop_details":{{"future":{nested}}}}}"#
        );
        let wire = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: source.into(),
        };
        let result = response(&wire, false, false).unwrap();
        let usage = result.anthropic_usage().unwrap().as_json_source();
        assert!(usage.contains(&nested));
        assert!(usage.contains(r#""future":"\ud800""#));
        assert!(result
            .anthropic_stop_details()
            .unwrap()
            .as_json_source()
            .contains(&nested));
        assert_eq!(result.usage.usage.as_ref().unwrap().input_tokens, 3);
    }

    #[test]
    fn deep_lone_surrogate_container_metadata_stays_source_backed() {
        let nested = format!("{}0{}", "[".repeat(16_384), "]".repeat(16_384));
        let container = format!(r#"{{"id":"container-deep","future":"\ud800","deep":{nested}}}"#);
        let source = format!(
            r#"{{"model":"m","content":[],"stop_reason":"end_turn","container":{container}}}"#
        );
        let expected = JsonPayload::parse(&container).unwrap();
        let wire = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: source.into(),
        };

        let mut result = response(&wire, true, true).unwrap();
        assert_eq!(&result.anthropic_container().unwrap().envelope, &expected);
        assert_eq!(&result.openrouter_container().unwrap().envelope, &expected);
        assert!(result
            .anthropic_container()
            .unwrap()
            .envelope
            .as_json_source()
            .contains(r#""future":"\ud800""#));

        let wire = serde_json::to_string(&result).unwrap();
        result = serde_json::from_str(&wire).unwrap();
        assert_eq!(&result.anthropic_container().unwrap().envelope, &expected);
        assert_eq!(&result.openrouter_container().unwrap().envelope, &expected);
    }

    #[test]
    fn resumable_response_member_preserves_presence_and_arbitrary_json_source() {
        let nested = format!("{}0{}", "[".repeat(16_384), "]".repeat(16_384));
        for raw_value in [
            "null".to_owned(),
            "false".to_owned(),
            "true".to_owned(),
            "17".to_owned(),
            r#""\ud800""#.to_owned(),
            format!(r#"{{"deep":{nested},"future":"\ud800"}}"#),
        ] {
            let source = format!(
                r#"{{"model":"m","content":[],"stop_reason":"max_tokens","resumable":{raw_value}}}"#
            );
            let wire = HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: source.into(),
            };
            let result = response(&wire, false, false).unwrap();
            assert_eq!(
                result
                    .anthropic_resumable()
                    .map(JsonPayload::as_json_source),
                Some(raw_value.as_str())
            );
        }

        let wire = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: r#"{"model":"m","content":[],"stop_reason":"max_tokens"}"#.into(),
        };
        assert!(response(&wire, false, false)
            .unwrap()
            .anthropic_resumable()
            .is_none());
    }
}
