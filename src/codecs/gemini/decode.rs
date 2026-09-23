//! Response and error decoding for `generateContent`.

use crate::transport::HttpResponse;
use lingxi_agent_api::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, StopReason,
    ToolUseId, Usage,
};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static NEXT_LOCAL_CALL_ID: AtomicU64 = AtomicU64::new(0);

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

    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut content = Vec::new();
    let mut saw_tool_call = false;
    let mut used_ids = provider_call_ids(parts);
    for part in parts {
        if let Some(block) = decode_part(part, &mut saw_tool_call, &mut used_ids) {
            content.push(block);
        }
    }

    Ok(CompletionResponse {
        web_search: crate::codecs::web_search_decode::gemini(&candidate),
        file_search: None,
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        // A candidate that produced a function call is a tool turn whatever
        // `finishReason` says: this wire reports STOP for both.
        stop_reason: if prompt_feedback_is_blocking(&body) {
            StopReason::Refusal
        } else if saw_tool_call {
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
        executed_profile: None,
    })
}

pub(super) fn provider_call_ids(parts: &[Value]) -> HashSet<String> {
    parts
        .iter()
        .filter_map(|part| part.get("functionCall")?.get("id")?.as_str())
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(super) fn call_id(call: &Value, used_ids: &mut HashSet<String>) -> (ToolUseId, Option<String>) {
    if let Some(id) = call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        return (ToolUseId::new(id), Some(id.to_owned()));
    }
    loop {
        let id = format!(
            "__gemini_local_call_{}",
            NEXT_LOCAL_CALL_ID.fetch_add(1, Ordering::Relaxed)
        );
        if used_ids.insert(id.clone()) {
            return (ToolUseId::new(id), None);
        }
    }
}

fn decode_part(
    part: &Value,
    saw_tool_call: &mut bool,
    used_ids: &mut HashSet<String>,
) -> Option<ContentBlock> {
    let thought_signature = part
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(call) = part.get("functionCall") {
        *saw_tool_call = true;
        let name = call.get("name").and_then(Value::as_str)?.to_owned();
        let (id, provider_id) = call_id(call, used_ids);
        return Some(ContentBlock::ToolUse {
            id,
            name,
            input: call.get("args").cloned().unwrap_or(Value::Null),
            provider_id,
            thought_signature,
        });
    }
    let text = part.get("text").and_then(Value::as_str)?.to_owned();
    if part.get("thought").and_then(Value::as_bool) == Some(true) {
        return Some(ContentBlock::Thinking {
            text,
            signature: thought_signature,
        });
    }
    Some(ContentBlock::Text {
        text,
        thought_signature,
    })
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

/// Prompt-level blocking applies even when the provider returns no candidate.
/// Keep the buffered and streaming decoders on the same interpretation.
pub(super) fn prompt_feedback_is_blocking(body: &Value) -> bool {
    body.get("promptFeedback")
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(Value::as_str)
        .is_some_and(|reason| !reason.is_empty() && reason != "BLOCK_REASON_UNSPECIFIED")
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
        input_tokens: prompt
            .saturating_sub(cached)
            .saturating_add(n("toolUsePromptTokenCount")),
        output_tokens: n("candidatesTokenCount").saturating_add(thoughts),
        cache_read_tokens: cached,
        cache_write_tokens: 0,
        cache_write_1h_tokens: 0,
        reasoning_tokens: thoughts,
        cost: None,
        server_tool_usage: None,
    }
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
