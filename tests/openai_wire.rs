//! Wire encoding, response replay, and provider error classification.
//!
//! Context-limit failures must remain distinguishable from authentication,
//! transport, and other failures so callers can select their own recovery.

#[path = "support/wire_api.rs"]
mod wire_api;

use lingxi_llm_client::codecs::openai::chat::{classify_error, OpenAiChatCodec};
use lingxi_llm_client::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, FailoverTriggers,
    LlmError, MessageRole, ModelCapabilitySupport, ProtocolFamily, ProviderFileSource, ProviderId,
    ProviderProfile, StopReason, StreamEvent, ToolChoice, ToolSpec, Usage,
};
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
        capability_support: ModelCapabilitySupport::default(),
        connection_chain: vec![],
        failover: FailoverTriggers::default(),
    }
}

fn request() -> CompletionRequest {
    CompletionRequest {
        controls: Default::default(),
        service_tier: None,
        model: "m".to_owned(),
        web_search: None,
        file_search: None,
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
            lingxi_llm_client::EncodeRequest::new(&request()),
            &wire_api::context(
                &profile(Value::Null),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
            id: lingxi_llm_client::protocol::ToolUseId::new("call-1"),
            name: "read".to_owned(),
            input: json!({"path": "a"}),
            provider_id: None,
            thought_signature: None,
        }],
    });
    req.messages.push(ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: lingxi_llm_client::protocol::ToolUseId::new("call-1"),
            content: "contents".to_owned(),
            is_error: false,
            blocks: None,
        }],
    });

    let http = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile(Value::Null),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
    let decoded = OpenAiChatCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
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
    let decoded = OpenAiChatCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
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

#[test]
fn a_chat_refusal_keeps_its_text_and_refusal_stop_reason() {
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "choices": [{"finish_reason": "stop", "message": {
                "refusal": "I can't help with that request."
            }}]
        }))
        .unwrap()
        .into(),
    };

    let decoded = OpenAiChatCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
    assert_eq!(decoded.stop_reason, StopReason::Refusal);
    assert!(matches!(
        decoded.message.content.as_slice(),
        [ContentBlock::Text { text, .. }] if text == "I can't help with that request."
    ));
}

#[test]
fn chat_refusals_do_not_replace_a_more_specific_finish_reason() {
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "choices": [{"finish_reason": "length", "message": {
                "refusal": "Partial refusal"
            }}]
        }))
        .unwrap()
        .into(),
    };

    let decoded = OpenAiChatCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
    assert_eq!(decoded.stop_reason, StopReason::MaxTokens);
    assert!(matches!(
        decoded.message.content.as_slice(),
        [ContentBlock::Text { text, .. }] if text == "Partial refusal"
    ));
}

#[test]
fn chat_document_urls_fail_before_the_request_is_sent() {
    let mut req = request();
    req.messages[0].content = vec![ContentBlock::Document {
        source: DocumentSource::Url {
            url: "https://example.test/document.pdf".to_owned(),
        },
        title: None,
    }];

    let error = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile(Value::Null),
                &route().request_model,
                &RequestOptions::default(),
            ),
        )
        .unwrap_err();
    assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
}

#[test]
fn provider_file_ids_without_a_bound_account_scope_are_rejected() {
    let mut req = request();
    req.messages[0].content = vec![ContentBlock::Document {
        source: DocumentSource::ProviderFile {
            file: ProviderFileSource {
                protocol: ProtocolFamily::OpenAiChat,
                provider_id: ProviderId::new("acme"),
                profile_name: "acme".into(),
                endpoint_fingerprint: String::new(),
                account_scope: None,
                file_id: "file_from_another_login".into(),
                uri: None,
                media_type: Some("application/pdf".into()),
                purpose: None,
            },
        },
        title: None,
    }];

    let error = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile(json!({"chat_pdf_only": true})),
                &route().request_model,
                &RequestOptions::default(),
            ),
        )
        .unwrap_err();

    assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
}

#[test]
fn direct_openai_chat_rejects_non_pdf_file_payloads() {
    let mut req = request();
    let mut openai = profile(Value::Null);
    openai.base_url = "https://api.openai.com/v1".into();
    for source in [
        DocumentSource::Text {
            media_type: "text/plain".into(),
            data: "plain text".into(),
        },
        DocumentSource::Base64 {
            media_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                .into(),
            data: "dGVzdA==".into(),
        },
    ] {
        req.messages[0].content = vec![ContentBlock::Document {
            source,
            title: None,
        }];
        let error = OpenAiChatCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(&openai, &route().request_model, &RequestOptions::default()),
            )
            .unwrap_err();
        assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
    }

    req.messages[0].content = vec![ContentBlock::Document {
        source: DocumentSource::Base64 {
            media_type: "application/pdf".into(),
            data: "JVBERi0=".into(),
        },
        title: None,
    }];
    assert!(OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(&openai, &route().request_model, &RequestOptions::default())
        )
        .is_ok());
}

#[test]
fn chat_output_limit_field_defaults_to_max_tokens_and_can_be_selected() {
    let mut req = request();
    req.max_tokens = Some(73);

    let default = encoded(&req, Value::Null);
    assert_eq!(default["max_tokens"], 73);
    assert!(default.get("max_completion_tokens").is_none());

    let explicit_default = encoded(&req, json!({"max_tokens_field": "max_tokens"}));
    assert_eq!(explicit_default["max_tokens"], 73);
    assert!(explicit_default.get("max_completion_tokens").is_none());

    let o_series = encoded(
        &req,
        json!({
            "max_tokens_field": "max_completion_tokens",
            "body": {"max_tokens": 999}
        }),
    );
    assert_eq!(o_series["max_completion_tokens"], 73);
    assert!(o_series.get("max_tokens").is_none());

    let default_with_extra = encoded(&req, json!({"body": {"max_completion_tokens": 999}}));
    assert_eq!(default_with_extra["max_tokens"], 73);
    assert!(default_with_extra.get("max_completion_tokens").is_none());
}

#[test]
fn chat_rejects_an_invalid_output_limit_field_setting() {
    for invalid in [json!("max_output_tokens"), json!(73), Value::Null] {
        let error = OpenAiChatCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&request()),
                &wire_api::context(
                    &profile(json!({"max_tokens_field": invalid})),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
            )
            .unwrap_err();
        assert!(
            matches!(error, LlmError::InvalidRequest { .. }),
            "invalid profile setting {invalid} should fail before sending"
        );
    }
}

fn request_with_assistant(assistant: ConversationMessage) -> CompletionRequest {
    let mut req = request();
    req.messages.push(assistant);
    req
}

#[test]
fn a_profile_flag_turns_on_the_usage_opt_in_without_naming_a_provider() {
    let opts = wire_api::EncodingOptions {
        stream: true,
        ..wire_api::EncodingOptions::default()
    };

    let plain = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&request()),
            &wire_api::context(&profile(Value::Null), &route().request_model, &opts),
        )
        .unwrap();
    assert!(body_of(&plain).get("stream_options").is_none());

    let opted = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&request()),
            &wire_api::context(
                &profile(json!({"stream_usage_opt_in": true})),
                &route().request_model,
                &opts,
            ),
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
/// forced choice is rejected and everything else passes through — and
/// the tools are never dropped.
fn with_tools(choice: ToolChoice) -> CompletionRequest {
    let mut req = request();
    req.tool_choice = choice;
    req.tools.push(ToolSpec {
        tool_type: None,
        defer_loading: None,
        extra: serde_json::Value::Null,
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
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(req),
                &wire_api::context(
                    &profile(extra),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
            )
            .unwrap(),
    )
}

#[test]
fn a_forced_tool_choice_is_rejected_where_thinking_would_reject_it() {
    let p = profile(json!({"thinking_rejects_forced_tool_choice": true}));
    for forced in [
        ToolChoice::Any,
        ToolChoice::Tool {
            name: "read".into(),
        },
    ] {
        let req = with_tools(forced);
        let result = OpenAiChatCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(&p, &route().request_model, &RequestOptions::default()),
        );
        assert!(matches!(
            result,
            Err(lingxi_llm_client::protocol::LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[test]
fn none_is_preserved_because_it_is_supported() {
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
    let mut d = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    let mut out = Vec::new();
    for f in frames {
        out.extend(wire_api::decode_frame(&mut *d, f.as_bytes()).unwrap());
    }
    out.extend(wire_api::finish(&mut *d).unwrap());
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
        Some(StreamEvent::End {
            usage, stop_reason, ..
        }) => {
            assert_eq!(
                usage.usage.unwrap(),
                Usage {
                    // 11 prompt tokens of which 7 were cached: this wire folds
                    // them in, so only 4 are uncached input. Asserting 11 here
                    // is what billed the cached tokens twice.
                    input_tokens: 4,
                    output_tokens: 5,
                    cache_read_tokens: 7,
                    cache_write_tokens: 0,
                    cache_write_1h_tokens: 0,
                    reasoning_tokens: 0,
                    cost: None,
                    server_tool_usage: None,
                }
            );
            assert_eq!(*stop_reason, StopReason::EndTurn);
        }
        other => panic!("expected a terminal End, got {other:?}"),
    }
}

#[test]
fn streamed_chat_refusals_keep_the_text_and_refusal_stop_reason() {
    let events = decode(&[
        r#"{"model":"wire-m","choices":[{"delta":{"refusal":"I can't help with that request."}}]}"#,
        r#"{"choices":[{"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);

    let text = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "I can't help with that request.");
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::Refusal,
            ..
        })
    ));
}

#[test]
fn streamed_chat_refusals_preserve_a_more_specific_finish_reason() {
    let events = decode(&[
        r#"{"model":"wire-m","choices":[{"delta":{"refusal":"Partial refusal"}}]}"#,
        r#"{"choices":[{"finish_reason":"length"}]}"#,
        "[DONE]",
    ]);

    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::MaxTokens,
            ..
        })
    ));
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

    match OpenAiChatCodec.decode_response(&resp, &wire_api::decode_context()) {
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
            lingxi_llm_client::EncodeRequest::new(&request()),
            &wire_api::context(
                &profile(extra),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
            lingxi_llm_client::EncodeRequest::new(&request()),
            &wire_api::context(
                &profile(json!({
                    "body": {"model": "someone-elses-model", "messages": []},
                    "headers": {"Authorization": "Bearer sk-not-yours"},
                })),
                &route().request_model,
                &RequestOptions::default(),
            ),
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

#[test]
fn chat_preserves_explicit_tool_strictness() {
    for strict in [true, false] {
        let mut req = request();
        req.tools.push(ToolSpec {
            tool_type: None,
            defer_loading: None,
            extra: serde_json::Value::Null,
            name: "lookup".into(),
            description: "Lookup".into(),
            input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            strict,
        });
        let http = OpenAiChatCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile(Value::Null),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
            )
            .unwrap();
        assert_eq!(body_of(&http)["tools"][0]["function"]["strict"], strict);
    }
}

#[test]
fn billing_errors_fall_back_to_type_without_masking_known_codes() {
    for code in ["credit_balance_exhausted", "future_billing_code", ""] {
        let body = json!({"error":{"type":"insufficient_quota","code":code,"message":"balance exhausted"}});
        assert!(matches!(
            classify_error(429, &body, None),
            LlmError::QuotaExceeded { .. }
        ));
        assert!(matches!(
            lingxi_llm_client::codecs::openai::responses::classify_error(429, &body, None),
            LlmError::QuotaExceeded { .. }
        ));
    }
    assert!(matches!(
        classify_error(
            429,
            &json!({"error":{"code":"rate_limit_exceeded","type":"requests"}}),
            None
        ),
        LlmError::RateLimited { .. }
    ));
    assert!(matches!(
        classify_error(
            400,
            &json!({"error":{"code":"context_length_exceeded","type":"insufficient_quota"}}),
            None
        ),
        LlmError::ContextOverflow { .. }
    ));
}

fn chat_message_response(message: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: json!({"model":"wire-m","choices":[{"message":message,"finish_reason":"stop"}]})
            .to_string()
            .into(),
    }
}

#[test]
fn chat_native_reasoning_survives_history_serialization_and_tool_replay() {
    let details = json!([
        {"type":"reasoning.encrypted","data":"opaque-sig","index":0,"vendor_extension":{"version":2}},
        {"type":"reasoning.text","text":"signed text","signature":"sig","id":null,"index":1}
    ]);
    for message in [
        json!({"reasoning":"visible reasoning","reasoning_details":details,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read","arguments":"{}"}}]}),
        json!({"reasoning_details":details}),
        json!({"reasoning":"reasoning only","content":null}),
        json!({"reasoning_content":"legacy display","reasoning":"native display","reasoning_details":details,"content":"answer"}),
    ] {
        let decoded = OpenAiChatCodec
            .decode_response(
                &chat_message_response(message.clone()),
                &wire_api::decode_context(),
            )
            .unwrap();
        let history: ConversationMessage =
            serde_json::from_str(&serde_json::to_string(&decoded.message).unwrap()).unwrap();
        let wire = encoded(
            &request_with_assistant(history),
            json!({"preserve_reasoning_content":true}),
        );
        assert_eq!(
            wire["messages"].as_array().unwrap().len(),
            2,
            "reasoning-only messages must survive"
        );
        let assistant = &wire["messages"][1];
        for key in ["reasoning", "reasoning_details", "tool_calls"] {
            assert_eq!(assistant.get(key), message.get(key), "{key}");
        }
        assert!(
            assistant.get("reasoning_content").is_none(),
            "display text must not duplicate native replay"
        );
        if message.get("reasoning_content").is_some() {
            let thinking: Vec<_> = decoded
                .message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Thinking { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(thinking, ["legacy display"]);
        }
    }
}

#[test]
fn streaming_chat_reasoning_preserves_detail_chunks_and_emits_native_data_once() {
    let detail_chunks = [
        json!([{"type":"reasoning.text","index":0,"text":"first","signature":null}]),
        json!([{"type":"reasoning.text","index":0,"text":"second","signature":"sig"}, {"type":"reasoning.encrypted","data":"same-sig","index":1}]),
        json!([{"type":"reasoning.encrypted","data":"same-sig","index":1,"extra":true}]),
    ];
    let all_details: Vec<Value> = detail_chunks
        .iter()
        .flat_map(|chunk| chunk.as_array().unwrap().iter().cloned())
        .collect();
    for done_marker in [false, true] {
        let mut decoder = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
        let mut events = Vec::new();
        for (index, details) in detail_chunks.iter().enumerate() {
            events.extend(wire_api::decode_frame(&mut *decoder , json!({"choices":[{"delta":{"reasoning":format!("part-{index}"),"reasoning_details":details}}]}).to_string().as_bytes()).unwrap());
        }
        events.extend(wire_api::decode_frame(&mut *decoder , br#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#).unwrap());
        if done_marker {
            events.extend(wire_api::decode_frame(&mut *decoder, b"[DONE]").unwrap());
        }
        events.extend(wire_api::finish(&mut *decoder).unwrap());
        events.extend(wire_api::finish(&mut *decoder).unwrap());
        let native: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ProviderContent {
                    block,
                    protocol,
                    value,
                } => Some((*block, *protocol, value.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(native.len(), 1);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::End { .. }))
                .count(),
            1
        );
        assert!(matches!(events.last(), Some(StreamEvent::End { .. })));
        assert_eq!(native[0].2["reasoning"], "part-0part-1part-2");
        assert_eq!(native[0].2["reasoning_details"], json!(all_details));
        let (tool_block, tool) = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ToolCallDelta {
                    block,
                    id,
                    name,
                    arguments_fragment,
                    ..
                } => Some((
                    *block,
                    ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::from_str(arguments_fragment).unwrap(),
                        provider_id: None,
                        thought_signature: None,
                    },
                )),
                _ => None,
            })
            .unwrap();
        assert_ne!(native[0].0, tool_block);
        assert!(events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ReasoningDelta { block, .. } => Some(*block),
                _ => None,
            })
            .all(|block| block == native[0].0));
        let wire = encoded(
            &request_with_assistant(ConversationMessage::assistant(vec![
                tool,
                ContentBlock::ProviderContent {
                    protocol: native[0].1,
                    value: native[0].2.clone(),
                },
            ])),
            Value::Null,
        );
        assert_eq!(wire["messages"][1]["reasoning_details"], json!(all_details));
        assert_eq!(wire["messages"][1]["tool_calls"][0]["id"], "call-1");
    }
}

#[test]
fn encrypted_only_stream_reasoning_has_its_own_block_and_interruption_is_not_replayable() {
    let mut decoder = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    let mut events = wire_api::decode_frame(&mut *decoder , br#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"sig"}]}}]}"#).unwrap();
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ProviderContent { .. })));
    assert!(matches!(
        wire_api::finish(&mut *decoder),
        Err(LlmError::StreamInterrupted { .. })
    ));
    assert!(decoder.push_bytes(b"data: [DONE]\n\n").is_empty());
    let mut decoder = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(&mut *decoder, br#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"sig"}]}}]}"#).unwrap();
    events.extend(
        wire_api::decode_frame(
            &mut *decoder,
            br#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}"#,
        )
        .unwrap(),
    );
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    let native = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderContent { block, .. } => Some(*block),
            _ => None,
        })
        .unwrap();
    let text = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::TextDelta { block, .. } => Some(*block),
            _ => None,
        })
        .unwrap();
    assert_ne!(native, text);
}

#[test]
fn chat_replay_rejects_wrong_envelopes_roles_and_protocols() {
    let good = json!({"type":"chat_reasoning","reasoning_details":[{"type":"reasoning.encrypted","data":"sig"}]});
    for (role, protocol, value) in [
        (MessageRole::User, ProtocolFamily::OpenAiChat, good.clone()),
        (
            MessageRole::System,
            ProtocolFamily::OpenAiChat,
            good.clone(),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::AnthropicMessages,
            good.clone(),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiResponses,
            good.clone(),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiChat,
            json!({"type":"other","reasoning":"x"}),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiChat,
            json!({"type":"chat_reasoning","reasoning":false}),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiChat,
            json!({"type":"chat_reasoning","reasoning_details":{}}),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiChat,
            json!({"type":"chat_reasoning","reasoning_details":[],"reasoning":null}),
        ),
        (
            MessageRole::Assistant,
            ProtocolFamily::OpenAiChat,
            json!({"type":"chat_reasoning","reasoning":"x","role":"system"}),
        ),
    ] {
        let mut req = request();
        req.messages.push(ConversationMessage {
            role,
            content: vec![ContentBlock::ProviderContent { protocol, value }],
        });
        assert!(OpenAiChatCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile(Value::Null),
                    &route().request_model,
                    &RequestOptions::default()
                )
            )
            .is_err());
    }
    let block = ContentBlock::ProviderContent {
        protocol: ProtocolFamily::OpenAiChat,
        value: good,
    };
    let req = request_with_assistant(ConversationMessage::assistant(vec![block.clone(), block]));
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile(Value::Null),
                &route().request_model,
                &RequestOptions::default()
            )
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn chat_multimodal_parts_keep_order_across_tool_result_boundaries() {
    let mut req = request();
    req.messages = serde_json::from_value(json!([{"role":"user","content":[
        {"type":"text","text":"Reference:"},
        {"type":"image","source":{"type":"url","url":"https://example.test/reference.png"}},
        {"type":"text","text":"Candidate:"},
        {"type":"image","source":{"type":"url","url":"https://example.test/candidate.png"}},
        {"type":"tool_result","tool_use_id":"call-1","content":"tool result","is_error":false},
        {"type":"text","text":"Document:"},
        {"type":"document","source":{"type":"text","media_type":"text/plain","data":"sample"}}
    ]}]))
    .unwrap();
    let wire = encoded(&req, Value::Null);
    assert_eq!(
        wire["messages"][0]["content"],
        json!([
            {"type":"text","text":"Reference:"},
            {"type":"image_url","image_url":{"url":"https://example.test/reference.png"}},
            {"type":"text","text":"Candidate:"},
            {"type":"image_url","image_url":{"url":"https://example.test/candidate.png"}}
        ])
    );
    assert_eq!(
        wire["messages"][1],
        json!({"role":"tool","tool_call_id":"call-1","content":"tool result"})
    );
    assert_eq!(
        wire["messages"][2]["content"][0],
        json!({"type":"text","text":"Document:"})
    );
    assert_eq!(wire["messages"][2]["content"][1]["type"], "file");
}

#[test]
fn chat_reasoning_envelopes_stay_with_their_own_assistant_turns() {
    let mut req = request();
    for signature in ["first", "second"] {
        let decoded = OpenAiChatCodec.decode_response(&chat_message_response(json!({
            "content":"answer", "reasoning_details":[{"type":"reasoning.encrypted","data":signature}]
        })), &wire_api::decode_context()).unwrap();
        req.messages.push(decoded.message);
        req.messages
            .push(ConversationMessage::user_text("continue"));
    }
    let wire = encoded(&req, Value::Null);
    assert_eq!(wire["messages"][1]["reasoning_details"][0]["data"], "first");
    assert_eq!(
        wire["messages"][3]["reasoning_details"][0]["data"],
        "second"
    );
    assert!(wire["messages"][2].get("reasoning_details").is_none());
}
