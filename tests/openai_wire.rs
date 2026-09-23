//! Gate 31: every way this provider family says "the transcript no longer
//! fits" must arrive as `ContextOverflow`, because that and only that sends
//! the turn into reactive compaction (§13 step 3). Everything else in the
//! taxonomy is here for the same reason: a misclassified error either loops
//! compaction forever or reports the wrong thing to the user.

use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, FailoverTriggers, LlmError, MessageRole,
    ModelCapabilities, ProviderId, ProviderProfile, StopReason, StreamEvent, ToolChoice, ToolSpec,
    Usage,
};
use lingxi_llm_client::codecs::openai::chat::{classify_error, OpenAiChatCodec};
use lingxi_llm_client::{HttpResponse, PricingModelRef, RequestOptions, ResolvedRoute, WireCodec};
use serde_json::{json, Value};

fn profile(extra: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": "https://api.acme.test/v1",
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": [{"display_model": "m", "request_model": "wire-m", "billing_model": "wire-m"}],
        "extra": extra,
    }))
    .unwrap()
}

fn route() -> ResolvedRoute {
    ResolvedRoute {
        provider_id: ProviderId::new("acme"),
        profile_name: "acme".to_owned(),
        request_model: "wire-m".to_owned(),
        display_model: "m".to_owned(),
        pricing_model: PricingModelRef {
            pricing_provider_id: ProviderId::new("acme"),
            billing_model: "wire-m".to_owned(),
            request_model: "wire-m".to_owned(),
            display_model: "m".to_owned(),
        },
        capabilities: ModelCapabilities::default(),
        connection_chain: vec![],
        failover: FailoverTriggers::default(),
    }
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: "m".to_owned(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hello".to_owned(),
                thought_signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

fn body_of(r: &lingxi_llm_client::HttpRequest) -> Value {
    serde_json::from_slice(&r.body).unwrap()
}

// --- gate 31 ---------------------------------------------------------------

#[test]
fn every_overflow_shape_becomes_context_overflow() {
    let by_code = classify_error(
        400,
        &json!({"error": {"code": "context_length_exceeded", "message": "too long"}}),
        None,
    );
    assert!(
        matches!(by_code, LlmError::ContextOverflow { .. }),
        "the provider's own code is the clearest signal: {by_code:?}"
    );

    let by_413 = classify_error(
        413,
        &json!({"error": {"message": "Request exceeds the model's context window"}}),
        None,
    );
    assert!(
        matches!(by_413, LlmError::ContextOverflow { .. }),
        "a 413 that names the context window is a token overflow: {by_413:?}"
    );
}

#[test]
fn a_413_that_is_not_about_context_is_an_oversized_body() {
    let err = classify_error(
        413,
        &json!({"error": {"message": "payload too large"}}),
        None,
    );
    assert!(
        matches!(err, LlmError::RequestTooLarge { .. }),
        "compaction cannot shrink accumulated attachments, so calling this an \
         overflow would loop compaction against something it never shrinks: {err:?}"
    );
}

#[test]
fn the_rest_of_the_taxonomy_lands_where_failover_expects_it() {
    type Case = (u16, Value, fn(&LlmError) -> bool);
    let cases: Vec<Case> = vec![
        (401, json!({}), |e| {
            matches!(e, LlmError::Authentication { .. })
        }),
        (403, json!({}), |e| {
            matches!(e, LlmError::PermissionDenied { .. })
        }),
        (404, json!({}), |e| {
            matches!(e, LlmError::ModelUnavailable { .. })
        }),
        (429, json!({}), |e| {
            matches!(e, LlmError::RateLimited { .. })
        }),
        (400, json!({}), |e| {
            matches!(e, LlmError::InvalidRequest { .. })
        }),
        (529, json!({}), |e| matches!(e, LlmError::Overloaded { .. })),
        (500, json!({}), |e| {
            matches!(e, LlmError::ProviderInternal { .. })
        }),
        (402, json!({"error": {"code": "insufficient_quota"}}), |e| {
            matches!(e, LlmError::QuotaExceeded { .. })
        }),
        (401, json!({"error": {"code": "invalid_api_key"}}), |e| {
            matches!(e, LlmError::Authentication { .. })
        }),
    ];
    for (status, body, want) in cases {
        let got = classify_error(status, &body, None);
        assert!(want(&got), "status {status} mapped to {got:?}");
    }
}

#[test]
fn the_providers_own_text_survives_classification() {
    let err = classify_error(
        401,
        &json!({"error": {"message": "Incorrect API key provided: sk-***"}}),
        None,
    );
    assert!(
        format!("{err}").contains("Incorrect API key provided"),
        "a user has to know which credential to fix: {err}"
    );
}

// --- encoding --------------------------------------------------------------

#[test]
fn the_wire_model_is_the_routes_not_the_requests() {
    let http = OpenAiChatCodec
        .encode_request(
            &request(),
            &profile(Value::Null),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();

    assert_eq!(http.url, "https://api.acme.test/v1/chat/completions");
    assert_eq!(
        body_of(&http)["model"],
        "wire-m",
        "the request carries the display ref; the provider only accepts its own id"
    );
}

#[test]
fn a_tool_result_becomes_its_own_message() {
    let mut req = request();
    req.messages.push(ConversationMessage {
        role: MessageRole::Assistant,
        content: vec![ContentBlock::ToolUse {
            id: lingxi_agent_api::protocol::ToolUseId::new("call-1"),
            name: "read".to_owned(),
            input: json!({"path": "a"}),
            provider_id: None,
            thought_signature: None,
        }],
    });
    req.messages.push(ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: lingxi_agent_api::protocol::ToolUseId::new("call-1"),
            content: "contents".to_owned(),
            is_error: false,
            blocks: None,
        }],
    });

    let http = OpenAiChatCodec
        .encode_request(
            &req,
            &profile(Value::Null),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();
    let messages = body_of(&http)["messages"].as_array().unwrap().clone();

    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool"],
        "a tool result cannot share a message with anything else on this wire"
    );
    assert_eq!(messages[2]["tool_call_id"], "call-1");
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"a\"}",
        "arguments go on the wire as a JSON string, not as an object"
    );
}

#[test]
fn assistant_text_and_tool_calls_replay_in_the_same_chat_message() {
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": "I will read the file.",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "read", "arguments": "{\"path\":\"a\"}"
                }}]
            }}]
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiChatCodec.decode_response(&response).unwrap();
    let messages = encoded(&request_with_assistant(decoded.message), Value::Null)["messages"]
        .as_array()
        .unwrap()
        .clone();

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "I will read the file.");
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call-1");
}

#[test]
fn nonstream_reasoning_content_replays_when_profile_requires_it() {
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "choices": [{"finish_reason": "stop", "message": {
                "reasoning_content": "check the path",
                "content": "The file exists."
            }}]
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiChatCodec.decode_response(&response).unwrap();
    let messages = encoded(
        &request_with_assistant(decoded.message),
        json!({"preserve_reasoning_content": true}),
    )["messages"]
        .as_array()
        .unwrap()
        .clone();

    assert_eq!(messages[1]["reasoning_content"], "check the path");
    assert_eq!(messages[1]["content"], "The file exists.");
}

fn request_with_assistant(assistant: ConversationMessage) -> CompletionRequest {
    let mut req = request();
    req.messages.push(assistant);
    req
}

#[test]
fn a_profile_flag_turns_on_the_usage_opt_in_without_naming_a_provider() {
    let opts = RequestOptions {
        stream: true,
        ..RequestOptions::default()
    };

    let plain = OpenAiChatCodec
        .encode_request(&request(), &profile(Value::Null), &route(), &opts)
        .unwrap();
    assert!(body_of(&plain).get("stream_options").is_none());

    let opted = OpenAiChatCodec
        .encode_request(
            &request(),
            &profile(json!({"stream_usage_opt_in": true})),
            &route(),
            &opts,
        )
        .unwrap();
    assert_eq!(
        body_of(&opted)["stream_options"],
        json!({"include_usage": true}),
        "without this the transcript and every token counter downstream see zero"
    );
}

/// The endpoint answers 400 to `required` and to a named function while
/// thinking is on, but accepts `none`, `auto` and the tools themselves. So a
/// forced choice relaxes to `auto` and everything else passes through — and
/// the tools are never dropped.
fn with_tools(choice: ToolChoice) -> CompletionRequest {
    let mut req = request();
    req.tool_choice = choice;
    req.tools.push(ToolSpec {
        name: "read".to_owned(),
        description: "read a file".to_owned(),
        input_schema: json!({"type": "object"}),
        strict: false,
    });
    req
}

fn encoded(req: &CompletionRequest, extra: Value) -> Value {
    body_of(
        &OpenAiChatCodec
            .encode_request(req, &profile(extra), &route(), &RequestOptions::default())
            .unwrap(),
    )
}

#[test]
fn a_forced_tool_choice_relaxes_to_auto_where_thinking_would_reject_it() {
    let quirk = json!({"thinking_rejects_forced_tool_choice": true});

    for forced in [
        ToolChoice::Any,
        ToolChoice::Tool {
            name: "read".to_owned(),
        },
    ] {
        let body = encoded(&with_tools(forced.clone()), quirk.clone());
        assert_eq!(
            body["tool_choice"], "auto",
            "{forced:?} comes back 400 from this endpoint while thinking is on"
        );
        assert!(
            body["tools"].as_array().is_some_and(|t| t.len() == 1),
            "the tools themselves are accepted; dropping them would leave the \
             model unable to act at all"
        );
    }
}

#[test]
fn none_survives_the_relaxation_because_it_is_supported() {
    let body = encoded(
        &with_tools(ToolChoice::None),
        json!({"thinking_rejects_forced_tool_choice": true}),
    );
    assert_eq!(
        body["tool_choice"], "none",
        "`none` means do not call a tool, and it is accepted in thinking mode; \
         dropping the key would let the model call one"
    );
}

#[test]
fn a_profile_without_the_quirk_sends_the_forced_choice_unchanged() {
    let body = encoded(
        &with_tools(ToolChoice::Tool {
            name: "read".to_owned(),
        }),
        Value::Null,
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "read"}})
    );
}

// --- streaming -------------------------------------------------------------

fn decode(frames: &[&str]) -> Vec<StreamEvent> {
    let mut d = OpenAiChatCodec.stream_decoder();
    let mut out = Vec::new();
    for f in frames {
        out.extend(d.decode_frame(f.as_bytes()).unwrap());
    }
    out.extend(d.finish().unwrap());
    out
}

#[test]
fn text_and_reasoning_get_separate_block_indices() {
    let events = decode(&[
        r#"{"model":"wire-m","choices":[{"delta":{"reasoning_content":"think"}}]}"#,
        r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
        r#"{"choices":[{"delta":{"content":" world"}}]}"#,
        "[DONE]",
    ]);

    let blocks: Vec<usize> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { block, .. } | StreamEvent::ReasoningDelta { block, .. } => {
                Some(*block)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        blocks,
        vec![0, 1, 1],
        "the two text deltas are one block; reasoning is another"
    );
    assert!(matches!(events.first(), Some(StreamEvent::Start { .. })));
}

#[test]
fn a_tool_call_streamed_in_fragments_keeps_the_id_from_its_first_frame() {
    let events = decode(&[
        r#"{"model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]}}]}"#,
        r#"{"choices":[{"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);

    let fragments: Vec<(&str, &str)> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta {
                id,
                arguments_fragment,
                ..
            } => Some((id.as_str(), arguments_fragment.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        fragments,
        vec![("call-1", "{\"pa"), ("call-1", "th\":\"a\"}")],
        "later fragments carry no id, so the decoder has to remember it"
    );
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::ToolUse,
            ..
        })
    ));
}

#[test]
fn usage_arriving_on_its_own_frame_still_reaches_the_end_event() {
    let events = decode(&[
        r#"{"model":"m","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":7}}}"#,
        "[DONE]",
    ]);

    match events.last() {
        Some(StreamEvent::End { usage, stop_reason }) => {
            assert_eq!(
                *usage,
                Usage {
                    // 11 prompt tokens of which 7 were cached: this wire folds
                    // them in, so only 4 are uncached input. Asserting 11 here
                    // is what billed the cached tokens twice.
                    input_tokens: 4,
                    output_tokens: 5,
                    cache_read_tokens: 7,
                    cache_write_tokens: 0,
                    reasoning_tokens: 0,
                    cost: None,
                }
            );
            assert_eq!(*stop_reason, StopReason::EndTurn);
        }
        other => panic!("expected a terminal End, got {other:?}"),
    }
}

#[test]
fn the_end_event_is_emitted_once_even_when_done_and_eof_both_arrive() {
    let events = decode(&[
        r#"{"model":"m","choices":[{"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::End { .. }))
            .count(),
        1,
        "a provider may send [DONE] and then close; the turn must not see two ends"
    );
}

#[test]
fn a_non_success_body_comes_back_as_an_error_not_as_a_response() {
    let resp = HttpResponse {
        status: 429,
        headers: vec![("retry-after".to_owned(), "30".to_owned())],
        body: serde_json::to_vec(&json!({"error": {"message": "slow down"}}))
            .unwrap()
            .into(),
    };

    match OpenAiChatCodec.decode_response(&resp) {
        Err(LlmError::RateLimited { retry_after, .. }) => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
        }
        other => panic!("expected a rate limit carrying its retry-after, got {other:?}"),
    }
}

// --- per-profile wire extras -----------------------------------------------

/// An OpenAI-compatible endpoint is rarely only OpenAI-compatible. An
/// aggregator takes a `provider` object choosing which upstream serves the
/// call, an end-user id, attribution headers — none of which belong in the
/// neutral request, and none of which this crate may special-case by provider
/// name (gate 30). They arrive as profile data and are merged in.
#[test]
fn a_profile_can_add_what_its_endpoint_understands() {
    let extra = json!({
        "body": {
            "provider": {"sort": "throughput", "data_collection": "deny", "only": ["upstream-a"]},
            "user": "stable-id",
        },
        "headers": {"HTTP-Referer": "https://app.test", "X-Title": "App"},
    });
    let req = OpenAiChatCodec
        .encode_request(
            &request(),
            &profile(extra),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();

    let body = body_of(&req);
    assert_eq!(
        body.get("provider"),
        Some(&json!({"sort": "throughput", "data_collection": "deny", "only": ["upstream-a"]})),
        "a nested structure this wire does not define, passed through whole"
    );
    assert_eq!(body.get("user"), Some(&json!("stable-id")));
    assert_eq!(body["model"], json!("wire-m"), "the request is unchanged");

    let header = |name: &str| {
        req.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    assert_eq!(header("http-referer"), Some("https://app.test"));
    assert_eq!(header("x-title"), Some("App"));
    assert_eq!(header("content-type"), Some("application/json"));
}

#[test]
fn a_profile_cannot_redirect_the_request_or_smuggle_a_key() {
    let req = OpenAiChatCodec
        .encode_request(
            &request(),
            &profile(json!({
                "body": {"model": "someone-elses-model", "messages": []},
                "headers": {"Authorization": "Bearer sk-not-yours"},
            })),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();

    let body = body_of(&req);
    assert_eq!(
        body["model"],
        json!("wire-m"),
        "the model was resolved, priced and recorded; sending another would \
         make every one of those records wrong"
    );
    assert!(
        body["messages"].as_array().is_some_and(|m| !m.is_empty()),
        "the conversation is not replaceable from config either"
    );
    assert!(
        !req.headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("authorization")),
        "the authenticator owns that header and runs after this"
    );
}
