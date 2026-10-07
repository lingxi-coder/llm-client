//! Response and error decoding for the Messages API.

use crate::protocol::{
    ChatResponse, ContentBlock, ConversationMessage, LlmError, MessageRole, StopReason, ToolUseId,
    Usage,
};
use crate::response_json::{DecodedText, ResponseJson};
use crate::transport::HttpResponse;
use serde_json::Value;
use std::time::Duration;

pub fn response(
    resp: &HttpResponse,
    retain_openrouter_container: bool,
    retain_anthropic_container: bool,
) -> Result<ChatResponse, LlmError> {
    // Status and Retry-After remain authoritative when an error response has
    // an empty, HTML or otherwise non-JSON body. Strict JSON decoding applies
    // to successful model responses, whose content can become executable.
    if !(200..300).contains(&resp.status) {
        return Err(classify_error(
            resp.status,
            &resp.payload_value(),
            retry_after(resp),
        ));
    }
    let mut response_json =
        ResponseJson::parse(&resp.body, "Anthropic response is not valid JSON")?;
    let body = response_json.value.clone();
    if !body.get("content").is_some_and(Value::is_array) {
        return Err(LlmError::ProviderInternal {
            message: "provider response has no valid content array".to_owned(),
        });
    }
    let raw_content = body["content"]
        .as_array()
        .expect("content array checked above");
    let mut content = raw_content
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            !crate::providers::anthropic::fallback_response::is_fallback_block(block)
        })
        .map(|(index, block)| {
            // A complete Messages block must identify its wire type. Unknown
            // types stay native, but malformed known content must never be
            // silently dropped and mistaken for a successfully completed turn.
            let kind = block.get("type").and_then(Value::as_str);
            let valid = match kind {
                Some("text") => block.get("text").is_some_and(Value::is_string),
                Some("redacted_thinking") => block.get("data").is_some_and(Value::is_string),
                Some("tool_use") => {
                    block.get("id").is_some_and(Value::is_string)
                        && block.get("name").is_some_and(Value::is_string)
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
            decode_response_block(&mut response_json, block, &format!("/content/{index}"))?
                .ok_or_else(|| LlmError::ProviderInternal {
                    message: format!(
                        "provider response contains malformed content block at index {index}"
                    ),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Native Messages materializes a valid fallback block before removing its
    // control representation from the SDK replay. A malformed-only control
    // response has neither that materialized block nor ordinary content, so
    // preserve the native empty-response placeholder in the SDK projection.
    if content.is_empty()
        && !raw_content.is_empty()
        && !raw_content.iter().any(|block| {
            crate::providers::anthropic::fallback_response::complete_block(block).is_some()
        })
    {
        content.push(ContentBlock::Text {
            text: "(no content)".into(),
            thought_signature: None,
            citations: Some(Some(serde_json::json!([]))),
        });
    }
    response_json.finish()?;
    let mut response = ChatResponse {
        inference: Default::default(),
        response_cache: None,
        web_search: crate::codecs::web_search_decode::with_usage(
            crate::codecs::web_search_decode::anthropic(&body),
            body.get("usage"),
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
        response_id: body
            .get("id")
            .and_then(Value::as_str)
            .map(crate::protocol::ResponseId::new),
        continuation: None,
        executed_profile: None,
    };
    response.set_anthropic_stop_details(
        body.get("stop_details")
            .filter(|value| !value.is_null())
            .cloned(),
    );
    if retain_anthropic_container {
        response.set_anthropic_metadata(
            body.get("container")
                .filter(|container| !container.is_null())
                .cloned()
                .map(
                    |envelope| crate::providers::anthropic::types::AnthropicContainerMetadata {
                        envelope,
                    },
                ),
            body.get("usage").cloned(),
        );
    }
    if retain_openrouter_container {
        response.set_openrouter_container(
            body.get("container")
                .filter(|container| !container.is_null())
                .cloned()
                .map(
                    |envelope| crate::providers::openrouter::types::OpenRouterContainerMetadata {
                        envelope,
                    },
                ),
        );
    }
    response.set_anthropic_fallback(
        crate::providers::anthropic::fallback_response::from_nonstream_body(&body),
    );
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
                matches!(super::response(&response,false,false),Err(LlmError::RateLimited{retry_after:Some(delay),..}) if delay==Duration::from_secs(7))
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
        assert!(super::response(&response, false, false).is_err());
    }
}

fn decode_response_block(
    response_json: &mut ResponseJson,
    value: &Value,
    pointer: &str,
) -> Result<Option<ContentBlock>, LlmError> {
    if value.get("type").and_then(Value::as_str) != Some("text")
        || !has_only_known_text_fields(value)
    {
        return Ok(decode_block(value));
    }
    let Some(display) = value.get("text").and_then(Value::as_str) else {
        return Ok(None);
    };
    let decoded = response_json.take_text(&format!("{pointer}/text"), display)?;
    let citations = value.get("citations").map(|citations| {
        if citations.is_null() {
            None
        } else {
            Some(citations.clone())
        }
    });
    Ok(Some(text_block(decoded, citations)))
}

fn text_block(text: DecodedText, citations: Option<Option<Value>>) -> ContentBlock {
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

/// An unknown complete content block is retained as native provider content.
/// Anthropic adds block types over time; keeping the block under this protocol
/// identity allows exact replay without interpreting it as a client tool.
pub fn decode_block(v: &Value) -> Option<ContentBlock> {
    if crate::codecs::web_search_decode::is_anthropic_search_block(v)
        || is_anthropic_tool_search_block(v)
    {
        return Some(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: v.clone(),
        });
    }

    if v.get("type").and_then(Value::as_str) == Some("text") && !has_only_known_text_fields(v) {
        return Some(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: v.clone(),
        });
    }

    match v.get("type").and_then(Value::as_str) {
        // MCP blocks carry server-owned tool identity and input. Keep the
        // entire block native so replay retains its server name, tool ID,
        // listing schema, and any fields added by Anthropic.
        Some("mcp_tool_use" | "mcp_tool_result" | "mcp_tool_listing") => {
            Some(ContentBlock::ProviderContent {
                protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
                value: v.clone(),
            })
        }
        Some("text") => Some(ContentBlock::Text {
            text: v.get("text").and_then(Value::as_str)?.to_owned(),
            thought_signature: None,
            citations: v.get("citations").map(|value| {
                if value.is_null() {
                    None
                } else {
                    Some(value.clone())
                }
            }),
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
            caller: v.get("caller").cloned(),
            toolset_name: v
                .get("toolset_name")
                .and_then(Value::as_str)
                .map(str::to_owned),
            thought_signature: None,
        }),
        Some(_) => Some(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: v.clone(),
        }),
        // Keep malformed or future block objects intact as well. Their type is
        // not interpreted, but silently removing the provider value would make
        // an otherwise replayable transcript incomplete.
        None => Some(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: v.clone(),
        }),
    }
}

/// Only these text-block fields have a typed carrier in the current SDK.
/// Keeping any additional provider fields native avoids silently dropping new
/// Anthropic response metadata while still making citations a first-class Text
/// field when the block has the shape this codec understands.
pub(crate) fn has_only_known_text_fields(block: &Value) -> bool {
    block.as_object().is_some_and(|object| {
        object
            .keys()
            .all(|key| matches!(key.as_str(), "type" | "text" | "citations"))
    })
}

pub(crate) fn is_anthropic_tool_search_block(block: &Value) -> bool {
    match block.get("type").and_then(Value::as_str) {
        Some("tool_search_tool_result") => true,
        Some("server_tool_use") => block
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| matches!(name, "tool_search_tool_regex" | "tool_search_tool_bm25")),
        _ => false,
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

    // Native uB classifies every HTTP 529 as overload, even when a gateway
    // supplies a different error type. Keep its diagnostic message intact.
    if status == 529 {
        return LlmError::Overloaded { message: display };
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
        "overloaded_error" => LlmError::Overloaded { message: display },
        "api_error" => LlmError::ProviderInternal { message: display },
        "timeout_error" => LlmError::ProviderTimeout {
            message: display,
            status: Some(status),
        },
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
        let block = decode_block(&json!({"type":"text","text":"x","citations":null})).unwrap();
        assert!(matches!(
            block,
            ContentBlock::Text {
                citations: Some(None),
                ..
            }
        ));
        let empty = decode_block(&json!({"type":"text","text":"x","citations":[]})).unwrap();
        assert!(
            matches!(empty, ContentBlock::Text { citations: Some(Some(value)), .. } if value == json!([]))
        );
        let cited = decode_block(&json!({
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
            } if value == json!([{"type":"web_search_result_location","url":"https://example.test"}])
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
            let decoded = decode_block(&raw).unwrap();
            assert!(matches!(
                decoded,
                ContentBlock::ProviderContent { value, .. } if value == raw
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
            matches!(&decoded.message.content[1], ContentBlock::ToolUse { name, input, .. } if name == "read" && input == &json!({"path":"a"}))
        );
        assert_eq!(decoded.message.content.len(), content.len());
        for (actual, expected) in decoded.message.content[2..].iter().zip(hosted) {
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
            );
            assert!(matches!(error, LlmError::Overloaded { message } if message == "529 fixture"));
        }
    }

    #[test]
    fn stop_details_survive_decode_and_other_metadata_updates() {
        let details = json!({"type":"refusal","future":{"reason":"blocked"}});
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
        result.set_anthropic_metadata(None, Some(json!({"output_tokens":1})));
        assert_eq!(result.anthropic_stop_details(), Some(&details));
        result.set_anthropic_stop_details(Some(json!({"type":"other"})));
        assert_eq!(result.anthropic_usage(), Some(&json!({"output_tokens":1})));
    }
}
