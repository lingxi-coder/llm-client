//! Response and error decoding for the Responses API.

use crate::codecs::openai::chat::classify_error as chat_classify;
use crate::transport::HttpResponse;
use lingxi_agent_api::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, ResponseId,
    StopReason, ToolUseId, Usage,
};
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
    for item in body
        .get("output")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        decode_item(item, &mut content, &mut saw_tool_call);
    }
    Ok(CompletionResponse {
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::responses(&body),
            body.get("usage"),
        ),
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: if body.get("status").and_then(Value::as_str) == Some("incomplete") {
            stop_reason(&body)
        } else if saw_tool_call {
            StopReason::ToolUse
        } else {
            stop_reason(&body)
        },
        usage: body.get("usage").map(usage).unwrap_or_default(),
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
            if let Some(text) = item
                .get("summary")
                .and_then(Value::as_array)
                .and_then(|s| s.first())
                .and_then(|s| s.get("text"))
                .and_then(Value::as_str)
            {
                out.push(ContentBlock::Thinking {
                    text: text.to_owned(),
                    signature: None,
                });
            }
        }
        _ => {}
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
    Usage {
        input_tokens: n("input_tokens").saturating_sub(cache_read),
        output_tokens: n("output_tokens"),
        cache_read_tokens: cache_read,
        cache_write_tokens: 0,
        reasoning_tokens: detail("output_tokens_details", "reasoning_tokens"),
        cost: None,
    }
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
