//! Response and error decoding for the Responses API.

use crate::codecs::openai::chat::classify_error as chat_classify;
use crate::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, ResponseId,
    StopReason, ToolUseId, Usage,
};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub fn response(resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
    let body: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
    if !(200..300).contains(&resp.status) {
        return Err(classify_error(resp.status, &body, retry_after(resp)));
    }
    if body.get("status").and_then(Value::as_str) == Some("failed") {
        return Err(classify_error(500, &body, None));
    }
    if !body.get("output").is_some_and(Value::is_array) {
        return Err(LlmError::ProviderInternal {
            message: "provider response has no valid output array".to_owned(),
        });
    }
    let mut content = Vec::new();
    let mut saw_tool_call = false;
    let mut saw_refusal = false;
    for item in body
        .get("output")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        saw_refusal |= has_refusal(item);
        decode_item(item, &mut content, &mut saw_tool_call);
    }
    Ok(CompletionResponse {
        inference: Default::default(),
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::responses(&body),
            body.get("usage"),
        ),
        file_search: crate::codecs::file_search_decode::responses(&body),
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: if body.get("status").and_then(Value::as_str) == Some("incomplete") {
            stop_reason(&body)
        } else if saw_tool_call {
            StopReason::ToolUse
        } else if saw_refusal {
            StopReason::Refusal
        } else {
            stop_reason(&body)
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
        executed_profile: None,
    })
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
        _ => {}
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
pub fn stop_reason(response: &Value) -> StopReason {
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
        _ => StopReason::EndTurn,
    }
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
    let file_search_requests = u
        .pointer("/x_tools/file_search/count")
        .and_then(Value::as_u64);
    (web_search_requests.is_some() || file_search_requests.is_some()).then_some(
        crate::protocol::ServerToolUsage {
            web_search_requests,
            file_search_requests,
        },
    )
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
