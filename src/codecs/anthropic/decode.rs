//! Response and error decoding for the Messages API.

use crate::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, StopReason,
    ToolUseId, Usage,
};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub fn response(resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
    let body: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
    if !(200..300).contains(&resp.status) {
        return Err(classify_error(resp.status, &body, retry_after(resp)));
    }
    if !body.get("content").is_some_and(Value::is_array) {
        return Err(LlmError::ProviderInternal {
            message: "provider response has no valid content array".to_owned(),
        });
    }
    let mut content = Vec::new();
    for b in body
        .get("content")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        if let Some(block) = decode_block(b) {
            content.push(block);
        }
    }
    Ok(CompletionResponse {
        inference: Default::default(),
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::anthropic(&body),
            body.get("usage"),
        ),
        file_search: None,
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: stop_reason(body.get("stop_reason").and_then(Value::as_str)),
        usage: crate::codecs::usage::report(
            body.get("usage"),
            &crate::codecs::usage::ANTHROPIC,
            usage,
            true,
        ),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        response_id: None,
        executed_profile: None,
    })
}

/// An unknown block type is dropped, not an error: this provider adds block
/// types over time and a client is expected to tolerate them.
pub fn decode_block(v: &Value) -> Option<ContentBlock> {
    if crate::codecs::web_search_decode::is_anthropic_search_block(v)
        || v.get("citations")
            .and_then(Value::as_array)
            .is_some_and(|cs| {
                cs.iter()
                    .any(|c| c["type"].as_str() == Some("web_search_result_location"))
            })
    {
        return Some(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: v.clone(),
        });
    }

    match v.get("type").and_then(Value::as_str) {
        Some("text") => Some(ContentBlock::Text {
            text: v.get("text").and_then(Value::as_str)?.to_owned(),
            thought_signature: None,
        }),
        Some("thinking") => Some(ContentBlock::Thinking {
            text: v
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            // Kept so the next turn can replay it; without it the replay is
            // refused at encode time (gate 18).
            signature: v
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }),
        Some("redacted_thinking") => Some(ContentBlock::RedactedThinking {
            data: v.get("data").and_then(Value::as_str)?.to_owned(),
        }),
        Some("tool_use") => Some(ContentBlock::ToolUse {
            id: ToolUseId::new(v.get("id").and_then(Value::as_str)?),
            name: v.get("name").and_then(Value::as_str)?.to_owned(),
            input: v.get("input").cloned().unwrap_or(Value::Null),
            provider_id: None,
            thought_signature: None,
        }),
        _ => None,
    }
}

/// Map a non-success response onto the taxonomy.
///
/// Two shapes mean "the transcript no longer fits" and both must become
/// `ContextOverflow`, because that and only that drives reactive compaction
/// (gate 31): a `request_too_large` that mentions the context window, and an
/// `invalid_request_error` whose message says the prompt is too long.
pub fn classify_error(status: u16, body: &Value, retry_after: Option<Duration>) -> LlmError {
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
        format!("{status} {body}")
    } else {
        format!("{status} {message}")
    };
    let lower = message.to_ascii_lowercase();

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
        "overloaded_error" => LlmError::Overloaded { message: display },
        "api_error" => LlmError::ProviderInternal { message: display },
        _ => match status {
            401 => LlmError::Authentication { message: display },
            403 => LlmError::PermissionDenied { message: display },
            404 => LlmError::ModelUnavailable { message: display },
            413 if lower.contains("context window") => LlmError::ContextOverflow {
                message: display,
                limit: None,
                actual: None,
            },
            413 => LlmError::RequestTooLarge { message: display },
            429 => LlmError::RateLimited {
                message: display,
                retry_after,
            },
            400 | 422 => LlmError::InvalidRequest { message: display },
            529 => LlmError::Overloaded { message: display },
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
    let file_search_requests = u
        .pointer("/server_tool_use/file_search_requests")
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
        server_tool_usage: (web_search_requests.is_some() || file_search_requests.is_some())
            .then_some(crate::protocol::ServerToolUsage {
                web_search_requests,
                file_search_requests,
            }),
    }
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
