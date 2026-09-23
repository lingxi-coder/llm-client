//! Response and error decoding for `generateContent`.

use crate::transport::HttpResponse;
use lingxi_agent_api::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, StopReason,
    ToolUseId, Usage,
};
use serde_json::Value;
use std::time::Duration;

pub fn response(resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
    let body: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
    if !(200..300).contains(&resp.status) {
        return Err(classify_error(resp.status, &body, retry_after(resp)));
    }
    if !body.get("candidates").is_some_and(Value::is_array)
        && !body.get("promptFeedback").is_some_and(Value::is_object)
    {
        return Err(LlmError::ProviderInternal {
            message: "provider response has no valid candidates array".to_owned(),
        });
    }
    let candidate = body
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .cloned()
        .unwrap_or(Value::Null);

    let mut content = Vec::new();
    let mut saw_tool_call = false;
    for part in candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        if let Some(block) = decode_part(part, &mut saw_tool_call) {
            content.push(block);
        }
    }

    Ok(CompletionResponse {
        web_search: crate::codecs::web_search_decode::gemini(&candidate),
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        // A candidate that produced a function call is a tool turn whatever
        // `finishReason` says: this wire reports STOP for both.
        stop_reason: if saw_tool_call {
            StopReason::ToolUse
        } else {
            stop_reason(candidate.get("finishReason").and_then(Value::as_str))
        },
        usage: body.get("usageMetadata").map(usage).unwrap_or_default(),
        model: body
            .get("modelVersion")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        response_id: None,
    })
}

pub fn decode_part(part: &Value, saw_tool_call: &mut bool) -> Option<ContentBlock> {
    if let Some(call) = part.get("functionCall") {
        *saw_tool_call = true;
        let name = call.get("name").and_then(Value::as_str)?.to_owned();
        return Some(ContentBlock::ToolUse {
            // This wire issues no call id, so the function name is the only
            // stable handle; the encoder's name map is keyed off what it puts
            // here.
            id: ToolUseId::new(&name),
            name,
            input: call.get("args").cloned().unwrap_or(Value::Null),
        });
    }
    let text = part.get("text").and_then(Value::as_str)?.to_owned();
    if part.get("thought").and_then(Value::as_bool) == Some(true) {
        return Some(ContentBlock::Thinking {
            text,
            signature: None,
        });
    }
    Some(ContentBlock::Text { text })
}

/// Map a non-success response onto the taxonomy.
///
/// The one that matters: this provider reports a prompt over the context window
/// as HTTP 400 `INVALID_ARGUMENT`, not 413. Left as `InvalidRequest` the turn
/// ends terminally instead of compacting and retrying (gate 31).
pub fn classify_error(status: u16, body: &Value, retry_after: Option<Duration>) -> LlmError {
    let error = body.get("error");
    let message = error
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let google_status = error
        .and_then(|e| e.get("status"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let display = if message.is_empty() {
        format!("{status} {body}")
    } else {
        format!("{status} {message}")
    };

    match google_status {
        "UNAUTHENTICATED" => LlmError::Authentication { message: display },
        "PERMISSION_DENIED" => LlmError::PermissionDenied { message: display },
        "NOT_FOUND" => LlmError::ModelUnavailable { message: display },
        "RESOURCE_EXHAUSTED" => LlmError::RateLimited {
            message: display,
            retry_after,
        },
        "UNAVAILABLE" => LlmError::Overloaded { message: display },
        "INTERNAL" => LlmError::ProviderInternal { message: display },
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" => {
            if is_context_overflow(&message) {
                LlmError::ContextOverflow {
                    message: display,
                    limit: None,
                    actual: None,
                }
            } else {
                LlmError::InvalidRequest { message: display }
            }
        }
        _ => match status {
            401 => LlmError::Authentication { message: display },
            403 => LlmError::PermissionDenied { message: display },
            404 => LlmError::ModelUnavailable { message: display },
            429 => LlmError::RateLimited {
                message: display,
                retry_after,
            },
            400 | 422 => LlmError::InvalidRequest { message: display },
            503 => LlmError::Overloaded { message: display },
            _ => LlmError::ProviderInternal { message: display },
        },
    }
}

/// Deliberately tight — both `token` and `exceed` — so an unrelated argument
/// error is not routed into the overflow-recovery loop, where it would compact
/// and retry against a request that was never too long.
fn is_context_overflow(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("token") && m.contains("exceed")
}

pub fn stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("STOP") => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some("SAFETY") | Some("RECITATION") | Some("PROHIBITED_CONTENT") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::EndTurn,
    }
}

/// Cached tokens are reported as a *subset* of the prompt count here, so they
/// are subtracted out: every bucket has to be independently billable or the
/// same tokens are paid for twice.
/// This wire is the other convention: `candidatesTokenCount` does **not**
/// contain the thinking tokens, so they are added to reach `output_tokens`,
/// and then reported again as the `reasoning_tokens` subset of it. The cached
/// tokens are folded into `promptTokenCount` as elsewhere and come back out.
pub fn usage(u: &Value) -> Usage {
    let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
    let prompt = n("promptTokenCount");
    let cached = n("cachedContentTokenCount");
    let thoughts = n("thoughtsTokenCount");
    Usage {
        input_tokens: prompt.saturating_sub(cached),
        output_tokens: n("candidatesTokenCount").saturating_add(thoughts),
        cache_read_tokens: cached,
        cache_write_tokens: 0,
        reasoning_tokens: thoughts,
        cost: None,
    }
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
