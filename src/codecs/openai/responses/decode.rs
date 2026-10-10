//! Response and error decoding for the Responses API.

use crate::codecs::openai::chat::classify_error as chat_classify;
use crate::protocol::{
    ChatResponse, ContentBlock, ConversationMessage, JsonPayload, JsonPayloadFragment, LlmError,
    MessageRole, ResponseId, StopReason, ToolUseId, Usage,
};
use crate::providers::openai::computer::OpenAiComputerCall;
use crate::response_json::{content_block, ResponseJson, ResponseJsonScanner};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub(crate) fn response_with_approval_support(
    resp: &HttpResponse,
    openai_approval_semantics: bool,
    openai_tool_search_semantics: bool,
    chatgpt_plan: bool,
) -> Result<ChatResponse, LlmError> {
    if chatgpt_plan {
        let body: Value = serde_json::from_slice(&resp.body)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&resp.body).into_owned()));
        let status = if !(200..300).contains(&resp.status) {
            Some(resp.status)
        } else if body.get("status").and_then(Value::as_str) == Some("failed") {
            Some(500)
        } else {
            None
        };
        if let Some(status) = status {
            let retry_after = retry_after(resp);
            let classified = classify_error(status, &body, retry_after, false);
            return Err(plan_provider_error(
                resp,
                body,
                classified.kind(),
                retry_after,
            ));
        }
    }
    let scanner =
        match ResponseJsonScanner::parse(&resp.body, "OpenAI Responses body is not valid JSON") {
            Ok(scanner) => scanner,
            Err(_error) if !(200..300).contains(&resp.status) => {
                let (body, provider_text) = ResponseJson::raw_http_error(resp.status, &resp.body);
                return Err(classify_error(resp.status, &body, retry_after(resp), false)
                    .with_provider_error_text(provider_text));
            }
            Err(error) => return Err(error),
        };
    if !(200..300).contains(&resp.status) {
        let selected_text = scanner.provider_error_text_at("/error/message", Some(resp.status))?;
        let provider_text = match &selected_text {
            Some(text) => text.clone(),
            None => scanner.provider_error_text_or_body_at("/error/message", Some(resp.status))?,
        };
        let body = scanner.provider_error_classification_body(
            "/error",
            &["message", "code", "type"],
            selected_text.as_ref(),
        );
        return Err(classify_error(resp.status, &body, retry_after(resp), false)
            .with_provider_error_text(provider_text));
    }
    if scanner.string_at("/status") == Some("failed") {
        let selected_text = scanner.provider_error_text_at("/error/message", Some(500))?;
        let provider_text = match &selected_text {
            Some(text) => text.clone(),
            None => scanner.provider_error_text_or_body_at("/error/message", Some(500))?,
        };
        let body = scanner.provider_error_classification_body(
            "/error",
            &["message", "code", "type"],
            selected_text.as_ref(),
        );
        return Err(classify_error(500, &body, None, false).with_provider_error_text(provider_text));
    }
    let provider_source = scanner
        .raw_source_at("")
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "OpenAI Responses source was not retained".into(),
        })?;
    let provider_payload =
        JsonPayload::parse(provider_source).map_err(|error| LlmError::InvalidRequest {
            message: format!("OpenAI Responses source is invalid JSON: {error}"),
        })?;
    let mut opaque_paths = Vec::new();
    for index in 0..scanner.array_len("/output").unwrap_or(0) {
        let item = format!("/output/{index}");
        match scanner.string_at(&format!("{item}/type")) {
            Some("web_search_call" | "file_search_call") => opaque_paths.push(item),
            Some("message") => {
                for content_index in 0..scanner.array_len(&format!("{item}/content")).unwrap_or(0) {
                    let annotations = format!("{item}/content/{content_index}/annotations");
                    if scanner.raw_source_at(&annotations).is_some() {
                        opaque_paths.push(annotations);
                    }
                }
            }
            _ => {}
        }
    }
    let mut response_json = scanner.into_response_json_with_opaque_paths(&opaque_paths)?;
    for pointer in &opaque_paths {
        let _ = response_json.take_raw_json_source_at(pointer)?;
    }
    let body = response_json.value.clone();
    if !body.get("output").is_some_and(Value::is_array) {
        return Err(LlmError::ProviderInternal {
            message: "provider response has no valid output array".to_owned(),
        });
    }
    let mut content = Vec::new();
    let mut saw_tool_call = false;
    let mut saw_refusal = false;
    let output = body
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    validate_output_call_ids(output)?;
    let response_status = body.get("status").and_then(Value::as_str);
    let has_computer_calls = output
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("computer_call"));
    if has_computer_calls {
        if !matches!(response_status, Some("completed" | "incomplete")) {
            return Err(LlmError::InvalidRequest {
                message: "Responses computer calls require a completed or incomplete terminal response status".into(),
            });
        }
        if response_status == Some("completed")
            && body
                .get("id")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(LlmError::InvalidRequest {
                message: "completed Responses computer calls require a nonempty response ID".into(),
            });
        }
    }
    for (item_index, item) in output.iter().enumerate() {
        saw_refusal |= has_refusal(item);
        if item.get("type").and_then(Value::as_str) == Some("computer_call") {
            if !openai_tool_search_semantics {
                return Err(LlmError::UnsupportedCapability {
                    message: "OpenAI computer calls require the official OpenAI Responses profile"
                        .into(),
                });
            }
            if response_status == Some("completed") {
                let call = OpenAiComputerCall::from_response_item(item)?;
                call.validate_completed_generation()?;
                content.push(call.into_content_block()?);
                saw_tool_call = true;
            } else {
                // Preserve unfinished provider output for inspection, but do
                // not expose it as a typed call that a caller might dispatch.
                content.push(ContentBlock::ProviderContent {
                    protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                    value: JsonPayload::from_generated_value(item).map_err(|error| {
                        LlmError::InvalidRequest {
                            message: format!("cannot retain provider item: {error}"),
                        }
                    })?,
                });
            }
            continue;
        }
        if item.is_null() {
            let source = response_json
                .take_raw_json_source_at(&format!("/output/{item_index}"))?
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "opaque Responses output item source was not retained".into(),
                })?;
            content.push(ContentBlock::ProviderContent {
                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                value: JsonPayload::parse(source).map_err(|error| LlmError::InvalidRequest {
                    message: format!("opaque Responses output item is invalid JSON: {error}"),
                })?,
            });
            continue;
        }
        if item["type"].as_str() == Some("function_call") {
            for key in ["call_id", "name", "arguments"] {
                let pointer = format!("/output/{item_index}/{key}");
                if provider_payload.kind_at(&pointer)
                    != Some(crate::response_json::FlatJsonKind::String)
                    || provider_payload
                        .string_units_at(&pointer)
                        .and_then(|units| String::from_utf16(&units).ok())
                        .is_none()
                {
                    return Err(LlmError::InvalidRequest {
                        message: format!("provider function call has no {key} string"),
                    });
                }
            }
            let arguments = JsonPayloadFragment::from_utf16_code_units(
                provider_payload
                    .string_units_at(&format!("/output/{item_index}/arguments"))
                    .expect("function-call argument string was checked"),
            );
            if JsonPayload::parse_fragments([&arguments]).is_err() {
                if body["status"].as_str() == Some("incomplete") {
                    let source = response_json
                        .take_raw_json_source_at(&format!("/output/{item_index}"))?
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "incomplete Responses function call source was not retained"
                                .into(),
                        })?;
                    content.push(ContentBlock::ProviderContent {
                        protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                        value: JsonPayload::parse(source).map_err(|error| {
                            LlmError::InvalidRequest {
                                message: format!(
                                    "incomplete function call source is invalid JSON: {error}"
                                ),
                            }
                        })?,
                    });
                    continue;
                }
                return Err(LlmError::InvalidRequest {
                    message: "provider returned malformed tool arguments".into(),
                });
            }
        }
        decode_item_with_json(
            item,
            item_index,
            &mut response_json,
            &mut content,
            &mut saw_tool_call,
        )?;
    }
    response_json.finish()?;
    Ok(ChatResponse {
        inference: Default::default(),
        response_cache: None,
        web_search: crate::codecs::web_search_decode::with_usage_payload(
            crate::codecs::web_search_decode::responses_payload(&provider_payload),
            provider_payload
                .raw_source_at("/usage")
                .and_then(|source| JsonPayload::parse(&source).ok())
                .as_ref(),
        ),
        file_search: crate::codecs::file_search_decode::responses_payload(&provider_payload),
        native_metadata: Vec::new(),
        message: ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: if has_pending_client_action(
            &body,
            openai_approval_semantics,
            openai_tool_search_semantics,
        ) {
            StopReason::Other("requires_action".into())
        } else if body.get("status").and_then(Value::as_str) == Some("incomplete") {
            stop_reason_with_approval_support(
                &body,
                openai_approval_semantics,
                openai_tool_search_semantics,
            )
        } else if saw_tool_call {
            StopReason::ToolUse
        } else if saw_refusal {
            StopReason::Refusal
        } else {
            stop_reason_with_approval_support(
                &body,
                openai_approval_semantics,
                openai_tool_search_semantics,
            )
        },
        usage: crate::codecs::usage::report(
            body.get("usage"),
            &crate::codecs::usage::OPENAI_RESPONSES,
            usage,
            true,
        ),
        native_resume_context_exhausted: None,
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        response_id: body.get("id").and_then(Value::as_str).map(ResponseId::new),
        continuation: None,
        executed_profile: None,
    })
}

fn plan_provider_error(
    resp: &HttpResponse,
    body: Value,
    classification: crate::protocol::LlmErrorKind,
    retry_after: Option<Duration>,
) -> LlmError {
    LlmError::ProviderResponse {
        status: resp.status,
        request_id: resp
            .header("x-request-id")
            .or_else(|| resp.header("openai-request-id"))
            .map(str::to_owned),
        body: Box::new(body),
        classification,
        retry_after,
    }
}

fn validate_output_call_ids(items: &[Value]) -> Result<(), LlmError> {
    let mut ids = std::collections::BTreeSet::new();
    for item in items {
        if !matches!(
            item.get("type").and_then(Value::as_str),
            Some("function_call" | "computer_call")
        ) {
            continue;
        }
        if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
            if !ids.insert(call_id) {
                return Err(LlmError::InvalidRequest {
                    message: "Responses output contains duplicate call_id values".into(),
                });
            }
        }
    }
    Ok(())
}

fn decode_item_with_json(
    item: &Value,
    item_index: usize,
    response_json: &mut ResponseJson,
    out: &mut Vec<ContentBlock>,
    saw_tool_call: &mut bool,
) -> Result<(), LlmError> {
    if item.get("type").and_then(Value::as_str) == Some("message") {
        for (part_index, part) in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    let decoded = response_json.take_text(
                        &format!("/output/{item_index}/content/{part_index}/text"),
                        text,
                    )?;
                    out.push(content_block(decoded, None, None));
                }
            }
        }
        return Ok(());
    }
    if item.get("type").and_then(Value::as_str) == Some("function_call") {
        let arguments = item["arguments"].as_str().unwrap_or_default();
        let arguments =
            response_json.take_text(&format!("/output/{item_index}/arguments"), arguments)?;
        let arguments_fragment =
            JsonPayloadFragment::from_decoded_text(&arguments.text, arguments.utf16_code_units)
                .map_err(|message| LlmError::InvalidRequest {
                    message: message.into(),
                })?;
        let input = JsonPayload::parse_fragments([&arguments_fragment]).map_err(|_| {
            LlmError::InvalidRequest {
                message: "provider returned malformed tool arguments".into(),
            }
        })?;
        let call_id = response_json.take_text(
            &format!("/output/{item_index}/call_id"),
            item["call_id"].as_str().unwrap_or_default(),
        )?;
        let call_id_units = call_id
            .utf16_code_units
            .unwrap_or_else(|| call_id.text.encode_utf16().collect());
        let call_id = String::from_utf16(&call_id_units).map_err(|_| LlmError::InvalidRequest {
            message: "provider function call id contains an unpaired UTF-16 unit".into(),
        })?;
        let name = response_json.take_text(
            &format!("/output/{item_index}/name"),
            item["name"].as_str().unwrap_or_default(),
        )?;
        let name_units = name
            .utf16_code_units
            .unwrap_or_else(|| name.text.encode_utf16().collect());
        let name = String::from_utf16(&name_units).map_err(|_| LlmError::InvalidRequest {
            message: "provider function name contains an unpaired UTF-16 unit".into(),
        })?;
        *saw_tool_call = true;
        out.push(ContentBlock::ToolUse {
            id: ToolUseId::new(call_id),
            name,
            input,
            provider_id: None,
            caller: None,
            toolset_name: None,
            thought_signature: None,
        });
    } else if item.get("type").and_then(Value::as_str) == Some("reasoning") {
        if let Some(summary) = item.get("summary").and_then(Value::as_array) {
            for (part_index, part) in summary.iter().enumerate() {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    let decoded = response_json.take_text(
                        &format!("/output/{item_index}/summary/{part_index}/text"),
                        text,
                    )?;
                    out.push(ContentBlock::Thinking {
                        text: decoded.text,
                        signature: None,
                    });
                }
            }
        }
        let source = response_json
            .take_raw_json_source_at(&format!("/output/{item_index}"))?
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Responses reasoning source was not retained".into(),
            })?;
        out.push(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
            value: JsonPayload::parse(source).map_err(|error| LlmError::InvalidRequest {
                message: format!("Responses reasoning source is invalid JSON: {error}"),
            })?,
        });
    } else if item.get("type").and_then(Value::as_str).is_some() {
        let source = response_json
            .take_raw_json_source_at(&format!("/output/{item_index}"))?
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Responses native item source was not retained".into(),
            })?;
        out.push(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
            value: JsonPayload::parse(source).map_err(|error| LlmError::InvalidRequest {
                message: format!("Responses native item is invalid JSON: {error}"),
            })?,
        });
    }
    Ok(())
}

/// Refusal text remains excluded from answer content, but it still determines
/// the result of a completed response.
pub(super) fn has_refusal(item: &Value) -> bool {
    match item.get("type").and_then(Value::as_str) {
        Some("refusal") => true,
        Some("message") => item
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|part| part.get("type").and_then(Value::as_str) == Some("refusal"))
            }),
        _ => false,
    }
}

/// The error envelope is the Chat wire's, so the classification is shared —
/// including the 413 context-window split (gate 31).
pub fn classify_error(
    status: u16,
    body: &Value,
    retry_after: Option<Duration>,
    native_retry_base_eligible: bool,
) -> LlmError {
    chat_classify(status, body, retry_after, native_retry_base_eligible)
}

/// `incomplete` carries the reason in its own object; `completed` is an end of
/// turn.
pub(crate) fn stop_reason_with_approval_support(
    response: &Value,
    openai_approval_semantics: bool,
    openai_tool_search_semantics: bool,
) -> StopReason {
    if has_pending_client_action(
        response,
        openai_approval_semantics,
        openai_tool_search_semantics,
    ) {
        return StopReason::Other("requires_action".into());
    }
    match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => match response
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::Refusal,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("incomplete".to_owned()),
        },
        Some("completed") | None => StopReason::EndTurn,
        Some(status) => StopReason::Other(status.to_owned()),
    }
}

fn has_pending_client_action(
    response: &Value,
    openai_approval_semantics: bool,
    openai_tool_search_semantics: bool,
) -> bool {
    (openai_tool_search_semantics && has_client_tool_search_call(response))
        || (openai_approval_semantics
            && response["output"].as_array().is_some_and(|items| {
                items
                    .iter()
                    .any(|item| item["type"] == "mcp_approval_request")
            }))
}

pub(super) fn has_client_tool_search_call(response: &Value) -> bool {
    response
        .get("output")
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(is_client_tool_search_call))
}

pub(super) fn is_client_tool_search_call(item: &Value) -> bool {
    item["type"] == "tool_search_call" && item["execution"] == "client"
}

/// Same folding as the chat wire, different key names: `input_tokens` already
/// contains the cached tokens, so they are subtracted back out to keep the
/// buckets disjoint. `reasoning_tokens` stays a subset of `output_tokens`.
pub fn usage(u: &Value) -> Usage {
    let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
    let detail = |group: &str, key: &str| {
        u.get(group)
            .and_then(|d: &Value| d.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let cache_read = detail("input_tokens_details", "cached_tokens");
    let cache_write = detail("input_tokens_details", "cache_write_tokens");
    Usage {
        input_tokens: n("input_tokens")
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output_tokens: n("output_tokens"),
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        cache_write_1h_tokens: 0,
        reasoning_tokens: detail("output_tokens_details", "reasoning_tokens"),
        cost: None,
        server_tool_usage: server_tool_usage(u),
    }
}

fn server_tool_usage(u: &Value) -> Option<crate::protocol::ServerToolUsage> {
    let web_search_requests = u
        .pointer("/x_tools/web_search/count")
        .and_then(Value::as_u64);
    let web_extractor_requests = u
        .pointer("/x_tools/web_extractor/count")
        .and_then(Value::as_u64);
    let file_search_requests = u
        .pointer("/x_tools/file_search/count")
        .and_then(Value::as_u64);
    let code_interpreter_requests = u
        .pointer("/x_tools/code_interpreter/count")
        .and_then(Value::as_u64);
    (web_search_requests.is_some()
        || web_extractor_requests.is_some()
        || file_search_requests.is_some()
        || code_interpreter_requests.is_some())
    .then_some(crate::protocol::ServerToolUsage {
        web_search_requests,
        web_fetch_requests: None,
        web_extractor_requests,
        file_search_requests,
        code_interpreter_requests,
    })
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

#[cfg(test)]
mod utf16_response_tests {
    use super::*;

    #[test]
    fn html_http_error_body_reaches_the_current_error_carrier() {
        let body = b"<!DOCTYPE html><html><body>proxy</body></html>";
        let error = response_with_approval_support(
            &crate::transport::HttpResponse {
                status: 400,
                headers: Vec::new(),
                body: body.as_slice().into(),
            },
            false,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            error.classification_source(),
            LlmError::InvalidRequest { .. }
        ));
        let text = error.provider_error_text().unwrap();
        assert_eq!(
            text.display(),
            format!("400 {}", String::from_utf8_lossy(body))
        );
        assert_eq!(
            text.utf16_code_units(),
            text.display().encode_utf16().collect::<Vec<_>>()
        );
    }

    #[test]
    fn failed_response_status_keeps_exact_provider_units() {
        let error = response_with_approval_support(
            &crate::transport::HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: br#"{"status":"failed","error":{"message":"bad\ud800","code":"invalid_prompt"}}"#.as_slice().into(),
            },
            false,
            false,
            false,
        )
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
    fn completed_response_text_retains_lone_units() {
        let response = response_with_approval_support(
            &crate::transport::HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: br#"{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"A\ud800B"}]}]}"#.as_slice().into(),
            },
            false,
            false,
            false,
        )
        .unwrap();
        assert!(matches!(
            response.message.content.as_slice(),
            [ContentBlock::TextJsUtf16 { text, utf16_code_units, .. }]
                if text == "A�B" && utf16_code_units == &[0x41, 0xd800, 0x42]
        ));
    }
}
