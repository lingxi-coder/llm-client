//! Response and error decoding.
//!
//! Provider context-limit errors are normalized to `ContextOverflow` so the
//! caller can choose how to reduce input or otherwise recover.

use crate::protocol::{
    CompletionResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, ProtocolFamily,
    ReportedCost, StopReason, ToolUseId, Usage,
};
use serde_json::Value;

pub(super) fn response_with_usage_mode(
    resp: &HttpResponse,
    separate_reasoning: bool,
) -> Result<CompletionResponse, LlmError> {
    let body: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
    if !(200..300).contains(&resp.status) {
        return Err(classify_error(resp.status, &body, retry_after(resp)));
    }
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "OpenAI response has no choices".to_owned(),
        })?;
    let message = choice.get("message").unwrap_or(&Value::Null);
    let refusal = message.get("refusal").and_then(Value::as_str);

    let mut content = Vec::new();
    let reasoning = super::reasoning::display_text(message);
    if !reasoning.is_empty() {
        content.push(ContentBlock::Thinking {
            text: reasoning,
            signature: None,
        });
    }
    if let Some(value) = super::reasoning::from_message(message)? {
        content.push(ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiChat,
            value,
        });
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: text.to_owned(),
                thought_signature: None,
            });
        }
    }
    if let Some(text) = refusal.filter(|text| !text.is_empty()) {
        content.push(ContentBlock::Text {
            text: text.to_owned(),
            thought_signature: None,
        });
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let function = call.get("function").unwrap_or(&Value::Null);
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        content.push(ContentBlock::ToolUse {
            id: ToolUseId::new(call.get("id").and_then(Value::as_str).unwrap_or_default()),
            name: function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            // Arguments arrive as a JSON *string*; a provider that streamed a
            // truncated fragment leaves it unparseable, and the tool's own
            // schema validation is where that should surface.
            input: serde_json::from_str(arguments).unwrap_or(Value::Null),
            provider_id: None,
            thought_signature: None,
        });
    }

    Ok(CompletionResponse {
        inference: Default::default(),
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::chat_with_search(message, &body),
            body.get("usage"),
        ),
        file_search: None,
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
        },
        stop_reason: with_refusal_stop_reason(
            stop_reason(choice.get("finish_reason").and_then(Value::as_str)),
            refusal.is_some(),
        ),
        usage: crate::codecs::usage::report(
            body.get("usage")
                .map(|raw| normalize_usage(raw, separate_reasoning))
                .as_ref(),
            &crate::codecs::usage::OPENAI_CHAT,
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

/// Map a non-success response onto the error taxonomy.
///
/// The 413 split is the part that matters and the part that is not obvious:
/// "context window" in the message means a token overflow, which compaction can
/// fix; any other 413 is an oversized request body (accumulated images or
/// attachments), which compaction cannot. Collapsing them would send the turn
/// into a compaction loop that never shrinks the thing that is too big.
pub fn classify_error(status: u16, body: &Value, retry_after: Option<Duration>) -> LlmError {
    let error = body.get("error");
    let message = error
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let codes = ["code", "type"]
        .into_iter()
        .filter_map(|key| error.and_then(|e| e.get(key)).and_then(Value::as_str));

    // The provider's own code is more precise than the status, so it wins.
    for code in codes {
        match code {
            "insufficient_quota" | "credit_balance_exhausted" => {
                return LlmError::QuotaExceeded {
                    message: display(status, body, &message),
                }
            }
            "context_length_exceeded" => {
                return LlmError::ContextOverflow {
                    message: display(status, body, &message),
                    limit: None,
                    actual: None,
                }
            }
            "invalid_api_key" | "invalid_authentication" => {
                return LlmError::Authentication {
                    message: display(status, body, &message),
                }
            }
            "model_not_found" => {
                return LlmError::ModelUnavailable {
                    message: display(status, body, &message),
                }
            }
            _ => {}
        }
    }

    match status {
        // The provider's text is kept, not just classified on: an auth message
        // downstream is what tells a user which credential to fix.
        401 => LlmError::Authentication {
            message: display(status, body, &message),
        },
        403 => LlmError::PermissionDenied {
            message: display(status, body, &message),
        },
        404 => LlmError::ModelUnavailable {
            message: display(status, body, &message),
        },
        413 if message.to_ascii_lowercase().contains("context window") => {
            LlmError::ContextOverflow {
                message: display(status, body, &message),
                limit: None,
                actual: None,
            }
        }
        413 => LlmError::RequestTooLarge {
            message: display(status, body, &message),
        },
        429 => LlmError::RateLimited {
            message: display(status, body, &message),
            retry_after,
        },
        400 | 422 => LlmError::InvalidRequest {
            message: display(status, body, &message),
        },
        529 => LlmError::Overloaded {
            message: display(status, body, &message),
        },
        _ => LlmError::ProviderInternal {
            message: display(status, body, &message),
        },
    }
}

/// The status is prefixed and the whole body is kept when there is no usable
/// `message`: a caller that has to diagnose a provider needs what it actually
/// said, not a summary of it.
fn display(status: u16, body: &Value, message: &str) -> String {
    if message.is_empty() {
        format!("{status} {body}")
    } else {
        format!("{status} {message}")
    }
}

pub fn stop_reason(finish: Option<&str>) -> StopReason {
    match finish {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::EndTurn,
    }
}

/// Refusal replaces an ordinary completed turn, while a more specific finish
/// reason such as token exhaustion or a tool call remains authoritative.
pub(super) fn with_refusal_stop_reason(stop: StopReason, saw_refusal: bool) -> StopReason {
    if stop == StopReason::EndTurn && saw_refusal {
        StopReason::Refusal
    } else {
        stop
    }
}

/// OpenAI counts prompt/completion; cached reads live under
/// `prompt_tokens_details`. A provider that omits them reports zero rather than
/// failing: usage is telemetry, not correctness.
/// This wire folds its cached tokens into `prompt_tokens`, so they must be
/// subtracted back out: `Usage`'s four buckets partition the bill, and leaving
/// them in would charge every cached token twice. `completion_tokens` likewise
/// already contains the reasoning tokens, but there `reasoning_tokens` is
/// declared a subset of `output_tokens`, so nothing is subtracted.
///
/// `cache_write_tokens` is absent from OpenAI proper and present on some
/// compatible providers (OpenRouter sends it); read it when it is there.
pub fn usage(u: &Value) -> Usage {
    let n = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0);
    let detail = |group: &str, key: &str| n(u.get(group).and_then(|d: &Value| d.get(key)));
    let cache_read = detail("prompt_tokens_details", "cached_tokens");
    let cache_write = detail("prompt_tokens_details", "cache_write_tokens");
    Usage {
        input_tokens: n(u.get("prompt_tokens"))
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output_tokens: n(u.get("completion_tokens")),
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        cache_write_1h_tokens: 0,
        reasoning_tokens: detail("completion_tokens_details", "reasoning_tokens"),
        // Only an aggregator knows what a call actually cost, because only it
        // knows which upstream served it. OpenAI itself sends no such field, so
        // this is absent on most providers speaking this wire — which is the
        // honest answer, not zero.
        cost: u
            .get("cost")
            .and_then(Value::as_f64)
            .and_then(ReportedCost::from_usd),
        server_tool_usage: None,
    }
}

use crate::transport::HttpResponse;
use std::time::Duration;

/// `Retry-After` is seconds or an HTTP date; only the seconds form is useful
/// to a backoff, and a provider that sends a date gets the default.
fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Convert independently reported reasoning to the shared output-subset contract.
pub(super) fn normalize_usage(raw: &Value, separate_reasoning: bool) -> Value {
    let mut normalized = raw.clone();
    if separate_reasoning {
        if let Some(output) = raw.get("completion_tokens").and_then(Value::as_u64) {
            let reasoning = raw
                .pointer("/completion_tokens_details/reasoning_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            // An overflow must remain an invalid report, not look like a
            // valid subtotal after silently dropping the extra tokens.
            normalized["completion_tokens"] = output
                .checked_add(reasoning)
                .map_or(Value::Null, |total| serde_json::json!(total));
        }
    }
    normalized
}
