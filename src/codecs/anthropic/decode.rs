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
    let scanner =
        match ResponseJsonScanner::parse(response_bytes, "Anthropic response is not valid JSON") {
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
                    Some((provider_id, protocol)) => {
                        let facts = native_error_facts(
                            (provider_id, protocol),
                            context,
                            Some(&provider_text),
                            None,
                            resp,
                        );
                        error.with_native_anthropic_error(provider_text, facts)
                    }
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
                    (provider_id, protocol),
                    context,
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

fn native_hq_error_body_truthy(scanner: &ResponseJsonScanner) -> bool {
    match scanner.kind_at("") {
        Some(FlatJsonKind::Null) | None => false,
        Some(FlatJsonKind::Boolean) => scanner.raw_source_at("") == Some("true"),
        Some(FlatJsonKind::Number) => scanner
            .number_source_at("")
            .and_then(|source| source.parse::<f64>().ok())
            .is_some_and(|number| number != 0.0 && !number.is_nan()),
        Some(FlatJsonKind::String) => scanner.string_at("").is_some_and(|text| !text.is_empty()),
        Some(FlatJsonKind::Array | FlatJsonKind::Object) => true,
    }
}

/// Reconstruct the exact `new xt(400, Kn.error, Kn.message, ...)` message that
/// HQ passes to its `gB` classifier. A truthy parsed body is re-rendered at
/// status 400; a falsy or unavailable body falls back to the already-built
/// Native message, preserving its original 422 prefix before adding 400.
fn native_hq_wrapped_error_text(
    status: u16,
    original: Option<&crate::protocol::ProviderErrorText>,
    scanner: Option<&ResponseJsonScanner>,
) -> Option<crate::protocol::ProviderErrorText> {
    match status {
        400 => original.cloned(),
        422 => {
            let original = original?;
            match scanner.filter(|scanner| native_hq_error_body_truthy(scanner)) {
                Some(scanner) => scanner.native_anthropic_http_error_text(Some(400)).ok(),
                None => Some(original.prefixed("400 ")),
            }
        }
        _ => None,
    }
}

fn native_error_facts(
    (provider_id, protocol): (crate::protocol::ProviderId, ProtocolFamily),
    context: Option<&crate::codecs::CodecContext>,
    native_text: Option<&crate::protocol::ProviderErrorText>,
    scanner: Option<&ResponseJsonScanner>,
    response: &HttpResponse,
) -> crate::protocol::NativeAnthropicErrorFacts {
    let status = response.status;
    let beta_requested = context
        .and_then(crate::codecs::CodecContext::native_anthropic_thinking_resumption_beta_requested)
        .or(Some(false));
    let thinking_binding_controls_beta_requested = context
        .and_then(crate::codecs::CodecContext::native_thinking_binding_controls_beta_requested);
    let native_hq_model_source =
        context.and_then(crate::codecs::CodecContext::native_hq_model_source);
    let native_hq_model_provider_family =
        context.and_then(crate::codecs::CodecContext::native_hq_model_provider_family);
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
    let native_thinking_binding_controls_error_evidence = if status == 400
        && thinking_binding_controls_beta_requested == Some(true)
    {
        native_text.map(|text| {
                let message = text.display();
                let has_protected_error =
                    crate::providers::anthropic::beta_repair::prefix_rejected(status, message)
                        || crate::providers::anthropic::beta_repair::signature_rejected(
                            status, message,
                        );
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: !has_protected_error
                        && crate::providers::anthropic::beta_repair::thinking_binding_controls_explicit_refusal(
                            message,
                        ),
                    token_refusal: !has_protected_error
                        && message.contains(
                            crate::providers::anthropic::beta_repair::Beta::ThinkingBindingControls
                                .header(),
                        ),
                    invalid_beta_flag: message.to_lowercase().contains("invalid beta flag"),
                }
            })
    } else {
        None
    };
    let native_hq_wrapped_text = native_hq_wrapped_error_text(status, native_text, scanner);
    let native_hq_known_error_evidence =
        crate::providers::anthropic::error_recognition::native_hq_known_error_evidence(
            status,
            native_hq_wrapped_text
                .as_ref()
                .map(crate::protocol::ProviderErrorText::display),
            native_hq_model_source,
            native_hq_model_provider_family,
            scanner,
            response.header("x-amzn-errortype"),
        );
    let native_hq_hrn15_evidence =
        crate::providers::anthropic::error_recognition::native_hq_hrn15_error_evidence(
            status,
            native_hq_wrapped_text
                .as_ref()
                .map(crate::protocol::ProviderErrorText::display),
            scanner,
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
        native_thinking_binding_controls_beta_requested: thinking_binding_controls_beta_requested,
        native_thinking_binding_controls_error_evidence,
        native_hq_model_source: native_hq_model_source.cloned(),
        native_hq_model_provider_family,
        native_hq_known_error_evidence,
        native_hq_hrn15_evidence,
        response: Some(crate::protocol::NativeAnthropicErrorResponse {
            body: scanner
                .and_then(|scanner| scanner.raw_source_at(""))
                .and_then(|source| JsonPayload::parse(source).ok()),
            thinking_prefix_mismatch:
                crate::providers::anthropic::thinking_prefix_mismatch::parse_response_headers(
                    &response.headers,
                ),
            api_error_name: Some("Error".into()),
            amzn_error_type: response.header("x-amzn-errortype").map(str::to_owned),
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
        native_context_with_hq(beta_requested, None, None, None)
    }

    fn native_context_with_hq(
        beta_requested: Option<bool>,
        model_source: Option<JsonPayload>,
        provider_family: Option<crate::protocol::NativeAnthropicModelProviderFamily>,
        tbc_requested: Option<bool>,
    ) -> crate::codecs::CodecContext {
        let (provider_id, base_url, protocol, request_model) = match provider_family {
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Bedrock) => (
                "bedrock",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "bedrock_claude",
                "wire-model",
            ),
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Vertex) => (
                "vertex",
                "https://us-central1-aiplatform.googleapis.com",
                "vertex_claude",
                "wire-model",
            ),
            _ => (
                "anthropic",
                "https://api.anthropic.com",
                "anthropic_messages",
                "claude-opus-5",
            ),
        };
        let profile: crate::protocol::ProviderProfile = serde_json::from_value(json!({
            "profile_name": "native-hq-test",
            "provider_id": provider_id,
            "base_url": base_url,
            "protocol": protocol,
            "auth": "none",
            "models": [{
                "display_model": "Claude",
                "request_model": request_model,
                "billing_model": request_model
            }]
        }))
        .unwrap();
        crate::codecs::CodecContext::new(
            &profile,
            request_model,
            crate::codecs::RequestMode::Complete,
        )
        .with_native_anthropic_context(crate::client::NativeAnthropicContextInput {
            thinking_resumption_beta_requested: beta_requested,
            native_thinking_binding_controls_beta_requested: tbc_requested,
            native_hq_model_source: model_source,
            native_hq_model_provider_family: provider_family,
            ..Default::default()
        })
    }

    #[test]
    fn native_hq_claims_use_real_http_decode_and_preserve_tri_state() {
        let decode = |status: u16,
                      message: &str,
                      model_source: Option<JsonPayload>,
                      provider_family: Option<
            crate::protocol::NativeAnthropicModelProviderFamily,
        >,
                      tbc_requested: Option<bool>,
                      amzn_error_type: Option<&str>,
                      nested_error: Option<serde_json::Value>| {
            let context =
                native_context_with_hq(None, model_source, provider_family, tbc_requested);
            let mut error = json!({
                "type": "invalid_request_error",
                "message": message,
            });
            if let Some(nested_error) = nested_error {
                error["error"] = nested_error;
            }
            let body = serde_json::to_vec(&json!({"message": message, "error": error})).unwrap();
            let headers = amzn_error_type
                .map(|value| vec![("x-amzn-errortype".into(), value.into())])
                .unwrap_or_default();
            super::response(
                &HttpResponse {
                    status,
                    headers,
                    body: body.into(),
                },
                Some(&context),
                false,
                false,
            )
            .unwrap_err()
        };
        let model = || Some(JsonPayload::parse(r#""claude-opus-5""#).unwrap());
        let other = Some(crate::protocol::NativeAnthropicModelProviderFamily::Other);

        let no_claim = decode(
            400,
            "billing limit reached",
            model(),
            other,
            None,
            None,
            None,
        );
        let no_claim_facts = no_claim.native_anthropic_error_facts().unwrap();
        let no_claim_evidence = no_claim_facts
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(no_claim_evidence.normalized_status, 400);
        assert_eq!(no_claim_evidence.aggregate_claim(), Some(false));
        assert_eq!(no_claim_evidence.model_refused, Some(false));
        assert!(no_claim_facts
            .native_hq_model_source
            .as_ref()
            .is_some_and(|source| source.root_string_code_units().is_some()));

        let positive = decode(
            400,
            "anthropic-beta rejected afk-mode-2026-01-31",
            model(),
            other,
            None,
            None,
            None,
        );
        let positive_evidence = positive
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(positive_evidence.afk_mode_beta_rejected, Some(true));
        assert_eq!(positive_evidence.aggregate_claim(), Some(true));

        let no_model_context = native_context_with_hq(None, None, None, None);
        let positive_with_unknown_claims = super::response(
            &HttpResponse {
                status: 400,
                headers: Vec::new(),
                body: b"anthropic-beta rejected afk-mode-2026-01-31"
                    .to_vec()
                    .into(),
            },
            Some(&no_model_context),
            false,
            false,
        )
        .unwrap_err();
        let partial_evidence = positive_with_unknown_claims
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(partial_evidence.model_refused, None);
        assert_eq!(partial_evidence.named_beta_rejected, None);
        assert_eq!(partial_evidence.afk_mode_beta_rejected, Some(true));
        assert_eq!(partial_evidence.aggregate_claim(), Some(true));

        let model_refusal = decode(
            400,
            "The provided model identifier is invalid",
            model(),
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Bedrock),
            None,
            None,
            None,
        );
        let refusal_facts = model_refusal.native_anthropic_error_facts().unwrap();
        assert_eq!(
            refusal_facts
                .native_hq_known_error_evidence
                .as_ref()
                .unwrap()
                .model_refused,
            Some(true)
        );

        let credential_precedence = decode(
            400,
            "The provided model identifier is invalid",
            model(),
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Bedrock),
            None,
            Some("UnrecognizedClientException"),
            None,
        );
        let credential_facts = credential_precedence
            .native_anthropic_error_facts()
            .unwrap();
        assert_eq!(
            credential_facts
                .native_hq_known_error_evidence
                .as_ref()
                .unwrap()
                .model_refused,
            Some(false)
        );
        let response = credential_facts.response.as_ref().unwrap();
        assert_eq!(response.api_error_name.as_deref(), Some("Error"));
        assert_eq!(
            response.amzn_error_type.as_deref(),
            Some("UnrecognizedClientException")
        );

        let outdated = decode(
            422,
            "The provided model identifier is invalid",
            model(),
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Bedrock),
            Some(true),
            None,
            Some(json!({
                "details": {"error_code": "claude_code_version_too_old"},
                "message": "Claude Code 2.1 is older than the minimum version required by your organization's policy"
            })),
        );
        let outdated_facts = outdated.native_anthropic_error_facts().unwrap();
        let outdated_evidence = outdated_facts
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(outdated_evidence.normalized_status, 400);
        assert_eq!(outdated_evidence.model_refused, Some(false));
        assert_eq!(
            outdated_facts.native_thinking_binding_controls_error_evidence,
            None
        );

        let tbc_422 = decode(
            422,
            "anthropic-beta rejected thinking-binding-controls-2026-08-01",
            model(),
            other,
            Some(true),
            None,
            None,
        );
        let tbc_facts = tbc_422.native_anthropic_error_facts().unwrap();
        let tbc_evidence = tbc_facts.native_hq_known_error_evidence.as_ref().unwrap();
        assert_eq!(tbc_evidence.thinking_binding_controls_rejected, Some(true));
        assert_eq!(tbc_evidence.aggregate_claim(), Some(true));
        assert_eq!(
            tbc_facts.native_thinking_binding_controls_error_evidence,
            None
        );

        let unobserved_model = decode(400, "billing limit reached", None, other, None, None, None);
        let unobserved = unobserved_model
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(unobserved.model_refused, None);
        assert_eq!(unobserved.aggregate_claim(), None);

        let non_string_model = decode(
            400,
            "billing limit reached",
            Some(JsonPayload::parse(r#"{"model":"claude-opus-5"}"#).unwrap()),
            Some(crate::protocol::NativeAnthropicModelProviderFamily::Bedrock),
            None,
            None,
            None,
        );
        let non_string_facts = non_string_model.native_anthropic_error_facts().unwrap();
        assert_eq!(non_string_facts.native_hq_model_source, None);
        assert_eq!(
            non_string_facts
                .native_hq_known_error_evidence
                .as_ref()
                .unwrap()
                .model_refused,
            None
        );

        let lone_surrogate = JsonPayload::parse(r#""claude-\ud800""#).unwrap();
        let lone_surrogate_error = decode(
            400,
            "billing limit reached",
            Some(lone_surrogate),
            other,
            None,
            None,
            None,
        );
        let source_units = lone_surrogate_error
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_model_source
            .as_ref()
            .unwrap()
            .root_string_code_units()
            .unwrap();
        assert_eq!(
            source_units,
            "claude-".encode_utf16().chain([0xd800]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn native_hq_422_wrapper_message_matches_xt_make_message_truthiness() {
        let native_original = |status: u16, body: &[u8]| {
            let response_text = ResponseJson::response_text(body);
            match ResponseJsonScanner::parse(response_text.as_bytes(), "fixture response") {
                Ok(scanner) => {
                    let text = scanner
                        .native_anthropic_http_error_text(Some(status))
                        .unwrap();
                    (Some(scanner), text)
                }
                Err(_) => {
                    let (_, text) = ResponseJson::native_anthropic_raw_http_error(status, body);
                    (None, text)
                }
            }
        };
        let assert_message = |status: u16, body: &[u8], expected: &str| {
            let (scanner, original) = native_original(status, body);
            let wrapped =
                super::native_hq_wrapped_error_text(status, Some(&original), scanner.as_ref())
                    .unwrap();
            assert_eq!(wrapped.display(), expected, "body={body:?}");
        };

        assert_message(
            422,
            br#"{"message":"afk-mode-2026-01-31 rejected"}"#,
            "400 afk-mode-2026-01-31 rejected",
        );
        assert_message(
            422,
            br#"{"message":{"detail":"nested"}}"#,
            r#"400 {"detail":"nested"}"#,
        );
        assert_message(
            422,
            br#"{"message":"","other":1}"#,
            r#"400 {"message":"","other":1}"#,
        );
        assert_message(422, br#"{"message":false}"#, r#"400 {"message":false}"#);
        assert_message(422, b"true", "400 true");
        assert_message(422, b"1", "400 1");
        assert_message(422, b"[]", "400 []");
        assert_message(422, b"raw Native response", "400 422 raw Native response");
        assert_message(422, b"null", "400 422 null");
        assert_message(422, b"false", "400 422 false");
        assert_message(422, b"0", "400 422 0");
        assert_message(422, b"-0", "400 422 -0");
        assert_message(422, b"\"\"", "400 422 \"\"");
        assert_message(422, b"", "400 422 status code (no body)");
        assert_message(
            400,
            br#"{"message":"afk-mode-2026-01-31 rejected"}"#,
            "400 afk-mode-2026-01-31 rejected",
        );

        let body = br#"{"message":"claim \ud800"}"#;
        let (scanner, original) = native_original(422, body);
        let wrapped =
            super::native_hq_wrapped_error_text(422, Some(&original), scanner.as_ref()).unwrap();
        assert_eq!(wrapped.display(), "400 claim �");
        assert_eq!(
            wrapped.utf16_code_units(),
            &[52, 48, 48, 32, 99, 108, 97, 105, 109, 32, 0xd800]
        );
    }

    #[test]
    fn native_hq_422_truthiness_cases_flow_through_http_decoder() {
        let source = Some(JsonPayload::parse(r#""claude-opus-5""#).unwrap());
        let family = Some(crate::protocol::NativeAnthropicModelProviderFamily::Other);
        let decode = |status: u16, body: &[u8]| {
            let context = native_context_with_hq(None, source.clone(), family, Some(true));
            super::response(
                &HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: body.to_vec().into(),
                },
                Some(&context),
                false,
                false,
            )
            .unwrap_err()
        };

        let falsy_bodies: [&[u8]; 5] = [b"null", b"false", b"0", b"-0", b"\"\""];
        for body in falsy_bodies {
            let error = decode(422, body);
            let facts = error.native_anthropic_error_facts().unwrap();
            let evidence = facts.native_hq_known_error_evidence.as_ref().unwrap();
            assert_eq!(evidence.normalized_status, 400);
            assert_eq!(evidence.aggregate_claim(), Some(false), "body={body:?}");
            assert_eq!(facts.native_thinking_binding_controls_error_evidence, None);
        }

        let empty_raw = decode(422, b"");
        let empty_facts = empty_raw.native_anthropic_error_facts().unwrap();
        let empty_evidence = empty_facts.native_hq_known_error_evidence.as_ref().unwrap();
        assert_eq!(empty_evidence.normalized_status, 400);
        assert_eq!(empty_evidence.aggregate_claim(), None);

        let body = br#"{"message":"anthropic-beta rejected afk-mode-2026-01-31"}"#;
        for status in [400, 422] {
            let error = decode(status, body);
            let evidence = error
                .native_anthropic_error_facts()
                .unwrap()
                .native_hq_known_error_evidence
                .as_ref()
                .unwrap();
            assert_eq!(evidence.afk_mode_beta_rejected, Some(true));
            assert_eq!(evidence.aggregate_claim(), Some(true));
        }

        let raw = decode(422, b"anthropic-beta rejected afk-mode-2026-01-31");
        let raw_evidence = raw
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_known_error_evidence
            .as_ref()
            .unwrap();
        assert_eq!(raw_evidence.afk_mode_beta_rejected, Some(true));
        assert_eq!(raw_evidence.aggregate_claim(), Some(true));
    }

    #[test]
    fn native_prefix_mismatch_header_is_bounded_and_route_scoped() {
        use crate::protocol::{NativeThinkingPrefixBlock, NativeThinkingPrefixMismatch};

        let native = native_context_with_hq(None, None, None, None);
        let compatible_profile: crate::protocol::ProviderProfile = serde_json::from_value(json!({
            "profile_name": "compatible-proxy",
            "provider_id": "custom-prefix-proxy",
            "base_url": "https://proxy.invalid",
            "protocol": "anthropic_messages",
            "auth": "none",
            "models": [{
                "display_model": "Claude",
                "request_model": "wire-model",
                "billing_model": "wire-model"
            }]
        }))
        .unwrap();
        let compatible = crate::codecs::CodecContext::new(
            &compatible_profile,
            "wire-model",
            crate::codecs::RequestMode::Complete,
        );
        let decode_headers = |context: &crate::codecs::CodecContext,
                              headers: Vec<(String, String)>| {
            let body = serde_json::to_vec(&json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Not created in this conversation"
                }
            }))
            .unwrap();
            super::response(
                &HttpResponse {
                    status: 400,
                    headers,
                    body: body.into(),
                },
                Some(context),
                false,
                false,
            )
            .unwrap_err()
        };
        let decode = |context: &crate::codecs::CodecContext, header: String| {
            decode_headers(
                context,
                vec![("anthropic-thinking-prefix-mismatch".into(), header)],
            )
        };
        let header =
            "block=messages.1.content.2;block=messages.3.content.4;kind=bad_kind;kind=UPPER";
        let error = decode(&native, header.into());
        let mismatch = error
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .as_ref()
            .unwrap();
        assert_eq!(
            mismatch,
            &NativeThinkingPrefixMismatch {
                block: Some(NativeThinkingPrefixBlock {
                    message_index: 1,
                    block_index: 2,
                }),
                kind: Some("bad_kind".into()),
            }
        );

        let mut exactly_2048_units = "block=messages.0.content.0;pad=".to_owned();
        while exactly_2048_units.encode_utf16().count() + 2 <= 2048 {
            exactly_2048_units.push('😀');
        }
        while exactly_2048_units.encode_utf16().count() < 2048 {
            exactly_2048_units.push('x');
        }
        assert_eq!(exactly_2048_units.encode_utf16().count(), 2048);
        let exact_limit = decode(&native, exactly_2048_units.clone());
        assert!(exact_limit
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_some());
        exactly_2048_units.push('x');
        let over_limit = decode(&native, exactly_2048_units);
        assert!(over_limit
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_none());

        let duplicate_values = decode_headers(
            &native,
            vec![
                (
                    "anthropic-thinking-prefix-mismatch".into(),
                    "block=messages.1.content.2".into(),
                ),
                (
                    "ANTHROPIC-THINKING-PREFIX-MISMATCH".into(),
                    "kind=thinking".into(),
                ),
            ],
        );
        assert!(duplicate_values
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_none());

        let mut first_duplicate_value = "block=messages.1.content.2;pad=".to_owned();
        while first_duplicate_value.encode_utf16().count() + 2 <= 2046 {
            first_duplicate_value.push('😀');
        }
        while first_duplicate_value.encode_utf16().count() < 2046 {
            first_duplicate_value.push('x');
        }
        assert_eq!(first_duplicate_value.encode_utf16().count(), 2046);
        assert!(decode(&native, first_duplicate_value.clone())
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_some());
        let combined_over_limit = decode_headers(
            &native,
            vec![
                (
                    "ANTHROPIC-THINKING-PREFIX-MISMATCH".into(),
                    first_duplicate_value,
                ),
                ("anthropic-thinking-prefix-mismatch".into(), "x".into()),
            ],
        );
        assert!(combined_over_limit
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_none());

        let malformed = decode(
            &native,
            "block=messages.1234567.content.0;kind=NotValid".into(),
        );
        assert!(malformed
            .native_anthropic_error_facts()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .thinking_prefix_mismatch
            .is_none());

        let compatible_error = decode(
            &compatible,
            "block=messages.1.content.2;kind=thinking".into(),
        );
        assert!(compatible_error.native_anthropic_error_facts().is_none());
    }

    #[test]
    fn native_hq_hrn15_claims_use_http_decoder_and_keep_prefix_marker_unknown() {
        let decode_body = |status: u16, body: &[u8]| {
            let context = native_context_with_hq(None, None, None, None);
            super::response(
                &HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: body.to_vec().into(),
                },
                Some(&context),
                false,
                false,
            )
            .unwrap_err()
            .native_anthropic_error_facts()
            .unwrap()
            .native_hq_hrn15_evidence
            .unwrap()
        };
        let decode_message = |status: u16, message: &str| {
            let body = serde_json::to_vec(&json!({"message": message})).unwrap();
            decode_body(status, &body)
        };
        let assert_positive = |evidence: crate::protocol::NativeHrn15ErrorEvidence, claim: bool| {
            assert!(claim);
            assert_eq!(evidence.aggregate_claim(None), Some(true));
        };

        let baseline = decode_message(400, "billing limit reached");
        assert_eq!(baseline.original_status, 400);
        assert_eq!(baseline.aggregate_claim(None), Some(false));
        assert!(!baseline.native_hie_claim);
        assert!(!baseline.native_su_claim);

        let prefix_400 = decode_message(400, "Not created in this conversation");
        assert!(prefix_400.prefix_lock_message_match);
        assert_eq!(prefix_400.aggregate_claim(None), None);
        assert_eq!(prefix_400.aggregate_claim(Some(true)), Some(false));
        assert_eq!(prefix_400.aggregate_claim(Some(false)), Some(true));
        let prefix_422 = decode_message(422, "Not created in this conversation");
        assert!(prefix_422.prefix_lock_message_match);
        assert_eq!(prefix_422.original_status, 422);
        assert_eq!(prefix_422.aggregate_claim(None), Some(true));
        assert_eq!(prefix_422.aggregate_claim(Some(true)), Some(true));

        let prompt = decode_message(400, "Input is too long for requested model");
        assert_positive(prompt, prompt.prompt_too_long);
        let prompt_marker = decode_message(400, "capability_rejected: prompt_too_long");
        assert_positive(prompt_marker, prompt_marker.prompt_too_long);
        let context = decode_message(400, "input length and `max_tokens` exceed context limit");
        assert_positive(context, context.context_overflow);
        let context_marker =
            decode_message(400, "capability_rejected: max_tokens_context_overflow");
        assert_positive(context_marker, context_marker.context_overflow);
        let credit = decode_message(400, "Credit balance is too low");
        assert_positive(credit, credit.credit_balance_too_low);
        let organization = decode_message(400, "Organization has been disabled");
        assert_positive(organization, organization.organization_disabled);
        let fast_mode = decode_message(400, "Fast mode is not enabled");
        assert_positive(fast_mode, fast_mode.fast_mode_not_enabled);
        let advisor = decode_message(
            400,
            "tool cannot be used as an advisor when the request model is incompatible",
        );
        assert_positive(advisor, advisor.advisor_model_unavailable);

        let dlp = decode_body(
            400,
            br#"{"error":{"details":{"error_code":"dlp_request_denied"}}}"#,
        );
        assert_positive(dlp, dlp.dlp_request_denied);
        let policy = decode_body(400, br#"{"error":{"type":"policy_blocked"}}"#);
        assert_positive(policy, policy.policy_blocked);
        let content_filter = decode_message(400, "Output blocked by content filtering policy");
        assert_positive(
            content_filter,
            content_filter.output_blocked_by_content_filter,
        );
        let mid_system = decode_message(400, "capability_rejected: mid_conv_system");
        assert_positive(mid_system, mid_system.mid_conversation_system_rejected);
        let mid_system_copy = decode_message(400, "role system is not supported in this message");
        assert_positive(
            mid_system_copy,
            mid_system_copy.mid_conversation_system_rejected,
        );
        let encrypted_search =
            decode_message(400, "Invalid `encrypted_content` in `search_result` block");
        assert_positive(
            encrypted_search,
            encrypted_search.encrypted_search_payload_rejected,
        );
        let encrypted_code = decode_message(
            400,
            "Invalid `encrypted_stdout` in encrypted_code_execution_result block",
        );
        assert_positive(
            encrypted_code,
            encrypted_code.encrypted_code_execution_payload_rejected,
        );

        let raw = decode_body(400, b"dlp_request_denied");
        assert!(!raw.dlp_request_denied);
        assert!(!raw.policy_blocked);
        assert_eq!(raw.aggregate_claim(None), Some(false));
        let falsy_root = decode_body(400, b"false");
        assert!(!falsy_root.dlp_request_denied);
        assert!(!falsy_root.policy_blocked);
        assert_eq!(falsy_root.aggregate_claim(None), Some(false));
        let array_root = decode_body(400, b"[]");
        assert!(!array_root.dlp_request_denied);
        assert!(!array_root.policy_blocked);
        assert_eq!(array_root.aggregate_claim(None), Some(false));
        let duplicate_error = decode_body(
            400,
            br#"{"error":{"type":"policy_blocked"},"error":{"type":"other"}}"#,
        );
        assert!(!duplicate_error.policy_blocked);
        assert_eq!(duplicate_error.aggregate_claim(None), Some(false));

        assert!(super::response(
            &HttpResponse {
                status: 503,
                headers: Vec::new(),
                body: br#"{"message":"busy"}"#.to_vec().into(),
            },
            Some(&native_context_with_hq(None, None, None, None)),
            false,
            false,
        )
        .unwrap_err()
        .native_anthropic_error_facts()
        .unwrap()
        .native_hq_hrn15_evidence
        .is_none());
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
    fn native_tbc_error_evidence_matches_qee_classes_and_guards() {
        let profile: crate::protocol::ProviderProfile = serde_json::from_value(json!({
            "profile_name": "bedrock",
            "provider_id": "bedrock",
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "protocol": "bedrock_claude",
            "auth": "none",
            "models": [{
                "display_model": "wire-model",
                "request_model": "wire-model",
                "billing_model": "wire-model"
            }]
        }))
        .unwrap();
        let decode = |status: u16, message: &str, requested: Option<bool>| {
            let mut context = crate::codecs::CodecContext::new(
                &profile,
                "wire-model",
                crate::codecs::RequestMode::Complete,
            )
            .with_native_anthropic_context(Default::default());
            context.set_native_thinking_binding_controls_beta_requested(requested);
            let body = format!(
                r#"{{"message":{},"error":{{"type":"invalid_request_error","message":{}}}}}"#,
                serde_json::to_string(message).unwrap(),
                serde_json::to_string(message).unwrap()
            );
            let error = super::response(
                &HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: body.into_bytes().into(),
                },
                Some(&context),
                false,
                false,
            )
            .unwrap_err();
            let facts = error.native_anthropic_error_facts().unwrap();
            (
                facts.native_thinking_binding_controls_beta_requested,
                facts.native_thinking_binding_controls_error_evidence,
            )
        };
        let token = "thinking-binding-controls-2026-08-01";
        for (message, expected) in [
            (
                format!("anthropic-beta rejected {token}"),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: true,
                    token_refusal: true,
                    invalid_beta_flag: false,
                },
            ),
            (
                format!("server rejected token {token}"),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: false,
                    token_refusal: true,
                    invalid_beta_flag: false,
                },
            ),
            (
                "thinking.adaptive.block_binding.prefix_mismatch_behavior: Extra inputs are not permitted".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: true,
                    token_refusal: false,
                    invalid_beta_flag: false,
                },
            ),
            (
                "The backend reported INVALID BETA FLAG".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: false,
                    token_refusal: false,
                    invalid_beta_flag: true,
                },
            ),
            (
                "billing limit reached".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence::default(),
            ),
            (
                format!("thinking block is not created in this conversation; anthropic-beta rejected {token}"),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence::default(),
            ),
            (
                format!("thinking.signature field required; anthropic-beta rejected {token}"),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence::default(),
            ),
            (
                "thinking.adaptive.block_binding: Extra inputs are not permitted; thinking block is not created in this conversation".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence::default(),
            ),
            (
                "thinking.enabled.block_binding.prefix_mismatch_behavior: Extra inputs are not permitted; thinking.signature field required".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence::default(),
            ),
            (
                "thinking.signature field required; invalid beta flag".into(),
                crate::protocol::NativeThinkingBindingControlsErrorEvidence {
                    explicit_refusal: false,
                    token_refusal: false,
                    invalid_beta_flag: true,
                },
            ),
        ] {
            let (requested, evidence) = decode(400, &message, Some(true));
            assert_eq!(requested, Some(true));
            assert_eq!(evidence, Some(expected), "message={message}");
        }

        for requested in [None, Some(false)] {
            let (_, evidence) = decode(
                400,
                "anthropic-beta rejected thinking-binding-controls-2026-08-01",
                requested,
            );
            assert_eq!(evidence, None, "requested={requested:?}");
        }
        let (_, evidence) = decode(
            422,
            "anthropic-beta rejected thinking-binding-controls-2026-08-01",
            Some(true),
        );
        assert_eq!(evidence, None);
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
