//! Response and error decoding for the Responses API.

use crate::codecs::openai::chat::classify_error as chat_classify;
use crate::protocol::{
    ChatResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, ResponseId, StopReason,
    ToolUseId, Usage,
};
use crate::providers::openai::computer::OpenAiComputerCall;
use crate::response_json::{content_block, ResponseJson};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub(crate) fn response_with_approval_support(
    resp: &HttpResponse,
    openai_approval_semantics: bool,
    openai_tool_search_semantics: bool,
    chatgpt_plan: bool,
) -> Result<ChatResponse, LlmError> {
    if !(200..300).contains(&resp.status) {
        let body: Value = serde_json::from_slice(&resp.body).unwrap_or_else(|_| {
            if chatgpt_plan {
                Value::String(String::from_utf8_lossy(&resp.body).into_owned())
            } else {
                Value::Null
            }
        });
        let retry_after = retry_after(resp);
        let classified = classify_error(resp.status, &body, retry_after);
        return Err(if chatgpt_plan {
            plan_provider_error(resp, body, classified.kind(), retry_after)
        } else {
            classified
        });
    }
    let mut response_json =
        ResponseJson::parse(&resp.body, "OpenAI Responses body is not valid JSON")?;
    let body = response_json.value.clone();
    if body.get("status").and_then(Value::as_str) == Some("failed") {
        response_json.finish()?;
        let classified = classify_error(500, &body, None);
        return Err(if chatgpt_plan {
            plan_provider_error(resp, body, classified.kind(), None)
        } else {
            classified
        });
    }
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
                    value: item.clone(),
                });
            }
            continue;
        }
        if item["type"].as_str() == Some("function_call") {
            for key in ["call_id", "name", "arguments"] {
                if item.get(key).and_then(Value::as_str).is_none() {
                    return Err(LlmError::InvalidRequest {
                        message: format!("provider function call has no {key} string"),
                    });
                }
            }
            if serde_json::from_str::<Value>(item["arguments"].as_str().unwrap()).is_err() {
                if body["status"].as_str() == Some("incomplete") {
                    content.push(ContentBlock::ProviderContent {
                        protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                        value: item.clone(),
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
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::responses(&body),
            body.get("usage"),
        ),
        file_search: crate::codecs::file_search_decode::responses(&body),
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
        body,
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
    decode_item(item, out, saw_tool_call);
    Ok(())
}

/// An output item is a message, a function call, or something this codec does
/// not model; only `output_text` parts carry model text, so a refusal part is
/// skipped rather than shown as the answer.
pub fn decode_item(item: &Value, out: &mut Vec<ContentBlock>, saw_tool_call: &mut bool) {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            for part in item
                .get("content")
                .and_then(Value::as_array)
                .unwrap_or(&Vec::new())
            {
                if part.get("type").and_then(Value::as_str) == Some("output_text") {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        out.push(ContentBlock::Text {
                            text: text.to_owned(),
                            thought_signature: None,
                            citations: None,
                        });
                    }
                }
            }
        }
        Some("function_call") => {
            *saw_tool_call = true;
            let arguments = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            out.push(ContentBlock::ToolUse {
                id: ToolUseId::new(
                    item.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                input: serde_json::from_str(arguments).unwrap_or(Value::Null),
                provider_id: None,
                caller: None,
                toolset_name: None,
                thought_signature: None,
            });
        }
        Some("reasoning") => {
            out.push(ContentBlock::ProviderContent {
                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                value: item.clone(),
            });
            if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                for part in summary {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        out.push(ContentBlock::Thinking {
                            text: text.to_owned(),
                            signature: None,
                        });
                    }
                }
            }
        }
        Some(_) => out.push(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
            value: item.clone(),
        }),
        None => {}
    }
}

#[cfg(test)]
mod utf16_response_tests {
    use super::*;

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
pub fn classify_error(status: u16, body: &Value, retry_after: Option<Duration>) -> LlmError {
    chat_classify(status, body, retry_after)
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
