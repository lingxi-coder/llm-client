//! Gemini `generateContent`, and the four hosted variants that reuse a wire
//! rather than defining one.
//!
//! Gate 31 on this family is the surprising one: a prompt over the context
//! window arrives as HTTP 400 `INVALID_ARGUMENT`, not 413.

use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, FailoverTriggers, LlmError, MessageRole,
    ModelCapabilities, ProviderId, ProviderProfile, StopReason, StreamEvent, SystemBlock,
    ToolChoice, ToolSpec, ToolUseId, Usage,
};
use lingxi_llm_client::codecs::gemini::classify_error;
use lingxi_llm_client::{
    AzureOpenAiCodec, FoundryClaudeCodec, GeminiCodec, PricingModelRef, RequestOptions,
    ResolvedRoute, VertexClaudeCodec, VertexGeminiCodec, WireCodec,
};
use serde_json::{json, Value};

fn profile(protocol: &str, base: &str, extra: Value) -> ProviderProfile {
    let mut v = json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": base,
        "protocol": protocol,
        "auth": "none",
        "models": [{"display_model": "m", "request_model": "wire-m", "billing_model": "wire-m"}],
    });
    if let Some(obj) = extra.as_object() {
        for (k, val) in obj {
            v[k] = val.clone();
        }
    }
    serde_json::from_value(v).unwrap()
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

fn request(messages: Vec<ConversationMessage>) -> CompletionRequest {
    CompletionRequest {
        model: "m".to_owned(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages,
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

fn user(text: &str) -> ConversationMessage {
    ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::Text {
            text: text.to_owned(),
            thought_signature: None,
        }],
    }
}

fn body(http: &lingxi_llm_client::HttpRequest) -> Value {
    serde_json::from_slice(&http.body).unwrap()
}

fn encode(
    codec: &dyn WireCodec,
    req: &CompletionRequest,
    p: &ProviderProfile,
    stream: bool,
) -> lingxi_llm_client::HttpRequest {
    codec
        .encode_request(
            req,
            p,
            &route(),
            &RequestOptions {
                stream,
                ..RequestOptions::default()
            },
        )
        .unwrap()
}

// --- the wire --------------------------------------------------------------

#[test]
fn the_assistants_role_on_this_wire_is_model() {
    let req = request(vec![
        user("hi"),
        ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text {
                text: "hello".to_owned(),
                thought_signature: None,
            }],
        },
    ]);
    let b = body(&encode(
        &GeminiCodec,
        &req,
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(b["contents"][0]["role"], "user");
    assert_eq!(b["contents"][1]["role"], "model");
}

#[test]
fn streaming_and_non_streaming_are_different_urls_not_a_body_flag() {
    let p = profile(
        "gemini_generate_content",
        "https://g.test/v1beta",
        Value::Null,
    );
    let req = request(vec![user("hi")]);

    let plain = encode(&GeminiCodec, &req, &p, false);
    let streamed = encode(&GeminiCodec, &req, &p, true);

    assert_eq!(
        plain.url,
        "https://g.test/v1beta/models/wire-m:generateContent"
    );
    assert_eq!(
        streamed.url,
        "https://g.test/v1beta/models/wire-m:streamGenerateContent?alt=sse"
    );
    assert!(body(&streamed).get("stream").is_none());
}

#[test]
fn a_tool_result_is_encoded_under_the_functions_name_not_the_call_id() {
    let req = request(vec![
        ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: ToolUseId::new("call-1"),
                name: "read_file".to_owned(),
                input: json!({"path": "a"}),
                provider_id: None,
                thought_signature: None,
            }],
        },
        ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: ToolUseId::new("call-1"),
                content: "contents".to_owned(),
                is_error: false,
                blocks: None,
            }],
        },
    ]);

    let b = body(&encode(
        &GeminiCodec,
        &req,
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(
        b["contents"][1]["parts"][0]["functionResponse"]["name"], "read_file",
        "this wire keys a result by function name, so the encoder has to find \
         the name from the call earlier in the transcript"
    );
}

#[test]
fn a_tool_result_with_no_matching_call_says_so_instead_of_guessing() {
    let req = request(vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("orphan"),
            content: "x".to_owned(),
            is_error: false,
            blocks: None,
        }],
    }]);

    let err = GeminiCodec
        .encode_request(
            &req,
            &profile(
                "gemini_generate_content",
                "https://g.test/v1beta",
                Value::Null,
            ),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(err, LlmError::InvalidRequest { ref message } if message.contains("orphan")),
        "{err:?}"
    );
}

#[test]
fn the_system_prompt_is_a_system_instruction() {
    let mut req = request(vec![user("hi")]);
    req.system = vec![SystemBlock {
        text: "be brief".to_owned(),
        cacheable: true,
    }];
    let b = body(&encode(
        &GeminiCodec,
        &req,
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(b["systemInstruction"]["parts"][0]["text"], "be brief");
    assert!(
        b["contents"].as_array().unwrap().len() == 1,
        "it is not smuggled in as a message"
    );
}

#[test]
fn tools_are_wrapped_in_function_declarations() {
    let mut req = request(vec![user("hi")]);
    req.tools.push(ToolSpec {
        name: "read".to_owned(),
        description: "read a file".to_owned(),
        input_schema: json!({"type": "object"}),
        strict: false,
    });
    let b = body(&encode(
        &GeminiCodec,
        &req,
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(b["tools"][0]["functionDeclarations"][0]["name"], "read");
    assert_eq!(b["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
}

// --- gate 31 ---------------------------------------------------------------

#[test]
fn a_prompt_over_the_context_window_arrives_as_a_400_and_must_still_compact() {
    let err = classify_error(
        400,
        &json!({"error": {"status": "INVALID_ARGUMENT",
                          "message": "The input token count (1050000) exceeds the maximum number of tokens allowed (1048576)."}}),
        None,
    );
    assert!(
        matches!(err, LlmError::ContextOverflow { .. }),
        "left as InvalidRequest the turn ends terminally instead of compacting \
         and retrying: {err:?}"
    );
}

#[test]
fn an_ordinary_bad_argument_is_not_routed_into_the_overflow_loop() {
    let err = classify_error(
        400,
        &json!({"error": {"status": "INVALID_ARGUMENT",
                          "message": "Invalid value at 'contents[0].role'"}}),
        None,
    );
    assert!(
        matches!(err, LlmError::InvalidRequest { .. }),
        "the match is deliberately tight — both 'token' and 'exceed' — so a \
         genuine argument error does not compact and retry forever: {err:?}"
    );
}

#[test]
fn the_google_status_codes_land_where_failover_expects_them() {
    for (status, want) in [
        ("UNAUTHENTICATED", "Authentication"),
        ("PERMISSION_DENIED", "PermissionDenied"),
        ("NOT_FOUND", "ModelUnavailable"),
        ("RESOURCE_EXHAUSTED", "RateLimited"),
        ("UNAVAILABLE", "Overloaded"),
        ("INTERNAL", "ProviderInternal"),
    ] {
        let err = classify_error(
            500,
            &json!({"error": {"status": status, "message": "x"}}),
            None,
        );
        assert!(format!("{err:?}").starts_with(want), "{status} -> {err:?}");
    }
}

// --- decoding --------------------------------------------------------------

#[test]
fn cached_tokens_are_subtracted_from_the_prompt_count() {
    let u = json!({
        "promptTokenCount": 1000,
        "cachedContentTokenCount": 400,
        "candidatesTokenCount": 50,
        "thoughtsTokenCount": 7,
    });
    let resp = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({"candidates": [], "usageMetadata": u}))
            .unwrap()
            .into(),
    };
    assert_eq!(
        GeminiCodec.response_usage(&resp),
        Some(Usage {
            input_tokens: 600,
            output_tokens: 57,
            cache_read_tokens: 400,
            cache_write_tokens: 0,
            reasoning_tokens: 7,
            cost: None,
        }),
        "this wire reports cached tokens as a subset of the prompt count; not \
         subtracting them bills the same tokens twice"
    );
}

#[test]
fn a_turn_that_called_a_tool_is_a_tool_turn_even_though_the_wire_says_stop() {
    let mut d = GeminiCodec.stream_decoder();
    let mut events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"path":"a"}}}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    events.extend(d.finish().unwrap());

    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::ToolCallDelta { name, .. } if name == "read"
    )));
    assert!(
        matches!(
            events.last(),
            Some(StreamEvent::End {
                stop_reason: StopReason::ToolUse,
                ..
            })
        ),
        "reporting EndTurn here would end the turn with a tool call unanswered: {events:?}"
    );
}

#[test]
fn a_thought_part_decodes_as_reasoning_not_as_text() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"m","candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"answer"}]}}]}"#,
        )
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ReasoningDelta { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { .. })));
}

// --- the hosted variants ---------------------------------------------------

#[test]
fn azure_puts_the_deployment_in_the_url_and_takes_model_out_of_the_body() {
    let p = profile(
        "azure_open_ai",
        "https://res.openai.azure.com",
        json!({"azure": {"api_version": "2024-02-01"}}),
    );
    let http = encode(&AzureOpenAiCodec, &request(vec![user("hi")]), &p, false);

    assert_eq!(
        http.url,
        "https://res.openai.azure.com/openai/deployments/wire-m/chat/completions?api-version=2024-02-01"
    );
    assert!(
        body(&http).get("model").is_none(),
        "the deployment is already in the path, and Azure rejects the key"
    );
}

#[test]
fn azure_without_an_api_version_names_the_profile_rather_than_guessing() {
    let p = profile("azure_open_ai", "https://res.openai.azure.com", Value::Null);
    let err = AzureOpenAiCodec
        .encode_request(
            &request(vec![user("hi")]),
            &p,
            &route(),
            &RequestOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(err, LlmError::InvalidRequest { ref message } if message.contains("api_version")),
        "{err:?}"
    );
}

#[test]
fn vertex_claude_moves_the_version_from_the_header_into_the_body() {
    let p = profile(
        "vertex_claude",
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
        Value::Null,
    );
    let http = encode(&VertexClaudeCodec, &request(vec![user("hi")]), &p, true);
    let b = body(&http);

    assert!(http
        .url
        .ends_with("/publishers/anthropic/models/wire-m:streamRawPredict"));
    assert!(
        !http.headers.iter().any(|(k, _)| k == "anthropic-version"),
        "Vertex reads the version from the body and rejects the request as \
         missing it when it stays in the header"
    );
    assert_eq!(b["anthropic_version"], "2023-06-01");
    assert!(b.get("model").is_none(), "the model is in the URL");
}

#[test]
fn vertex_gemini_reuses_the_generate_content_body_under_a_publisher_path() {
    let p = profile(
        "vertex_gemini",
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
        Value::Null,
    );
    let http = encode(&VertexGeminiCodec, &request(vec![user("hi")]), &p, false);

    assert!(http
        .url
        .ends_with("/publishers/google/models/wire-m:generateContent"));
    assert_eq!(body(&http)["contents"][0]["role"], "user");
}

#[test]
fn foundry_is_the_anthropic_wire_at_a_foundry_endpoint() {
    let p = profile(
        "foundry_claude",
        "https://res.services.ai.azure.com/anthropic",
        Value::Null,
    );
    let http = encode(&FoundryClaudeCodec, &request(vec![user("hi")]), &p, false);

    assert_eq!(
        http.url,
        "https://res.services.ai.azure.com/anthropic/v1/messages"
    );
    assert!(http
        .headers
        .iter()
        .any(|(k, v)| k == "anthropic-version" && v == "2023-06-01"));
}

#[test]
fn oversized_usage_counters_do_not_panic_or_wrap() {
    let mut decoder = GeminiCodec.stream_decoder();
    let frame = serde_json::to_vec(&json!({"usageMetadata": {
        "promptTokenCount": 0,
        "candidatesTokenCount": u64::MAX,
        "thoughtsTokenCount": 1,
    }}))
    .unwrap();
    decoder.decode_frame(&frame).unwrap();
    assert_eq!(decoder.observed_usage().unwrap().output_tokens, u64::MAX);
    assert!(!decoder.usage_is_complete());
}

fn gemini_response(parts: Value) -> lingxi_llm_client::HttpResponse {
    lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "modelVersion": "wire-m",
            "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]
        }))
        .unwrap()
        .into(),
    }
}

#[test]
fn parallel_same_name_calls_preserve_ids_and_signatures_on_replay() {
    let decoded = GeminiCodec
        .decode_response(&gemini_response(json!([
            {"functionCall": {"id": "call-a", "name": "read", "args": {"path": "a"}}, "thoughtSignature": "sig-a"},
            {"functionCall": {"id": "call-b", "name": "read", "args": {"path": "b"}}}
        ])))
        .unwrap();
    let calls: Vec<_> = decoded
        .message
        .tool_uses()
        .map(|(id, _, _)| id.clone())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].as_str(), "call-a");
    assert_eq!(calls[1].as_str(), "call-b");

    let serialized = serde_json::to_value(&decoded.message).unwrap();
    assert_eq!(serialized["content"][0]["provider_id"], "call-a");
    assert_eq!(serialized["content"][0]["thought_signature"], "sig-a");

    let mut req = request(vec![user("read both"), decoded.message]);
    req.messages.push(ConversationMessage {
        role: MessageRole::User,
        content: calls
            .iter()
            .map(|id| ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: "ok".into(),
                is_error: false,
                blocks: None,
            })
            .collect(),
    });
    let b = body(&encode(
        &GeminiCodec,
        &req,
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(b["contents"][1]["parts"][0]["thoughtSignature"], "sig-a");
    assert_eq!(b["contents"][1]["parts"][0]["functionCall"]["id"], "call-a");
    assert_eq!(b["contents"][1]["parts"][1]["functionCall"]["id"], "call-b");
    assert_eq!(
        b["contents"][2]["parts"][0]["functionResponse"]["id"],
        "call-a"
    );
    assert_eq!(
        b["contents"][2]["parts"][1]["functionResponse"]["id"],
        "call-b"
    );
}

#[test]
fn calls_without_provider_ids_get_unique_local_handles_without_wire_ids() {
    let decoded = GeminiCodec
        .decode_response(&gemini_response(json!([
            {"functionCall": {"name": "read", "args": {"path": "a"}}},
            {"functionCall": {"name": "read", "args": {"path": "b"}}}
        ])))
        .unwrap();
    let calls: Vec<_> = decoded.message.tool_uses().collect();
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].0, calls[1].0);
    let next_turn = GeminiCodec
        .decode_response(&gemini_response(json!([
            {"functionCall": {"name": "write", "args": {"path": "c"}}}
        ])))
        .unwrap();
    let next_id = next_turn.message.tool_uses().next().unwrap().0;
    assert_ne!(calls[0].0, next_id);
    assert_ne!(calls[1].0, next_id);
    let b = body(&encode(
        &GeminiCodec,
        &request(vec![decoded.message]),
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert!(b["contents"][0]["parts"][0]["functionCall"]
        .get("id")
        .is_none());
    assert!(b["contents"][0]["parts"][1]["functionCall"]
        .get("id")
        .is_none());
}

#[test]
fn text_part_thought_signature_survives_full_response_replay() {
    let decoded = GeminiCodec
        .decode_response(&gemini_response(json!([
            {"text": "answer", "thoughtSignature": "sig-text"}
        ])))
        .unwrap();
    let serialized = serde_json::to_value(&decoded.message).unwrap();
    assert_eq!(serialized["content"][0]["thought_signature"], "sig-text");
    let b = body(&encode(
        &GeminiCodec,
        &request(vec![decoded.message]),
        &profile(
            "gemini_generate_content",
            "https://g.test/v1beta",
            Value::Null,
        ),
        false,
    ));
    assert_eq!(b["contents"][0]["parts"][0]["thoughtSignature"], "sig-text");
}

#[test]
fn streaming_calls_have_distinct_ids_and_signatures_at_their_blocks() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"functionCall":{"id":"call-a","name":"read","args":{"path":"a"}},"thoughtSignature":"sig-a"},{"functionCall":{"id":"call-b","name":"read","args":{"path":"b"}}},{"text":"done","thoughtSignature":"sig-text"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ToolCallDelta { block: 0, id, provider_id: Some(provider_id), .. } if id.as_str() == "call-a" && provider_id == "call-a")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ToolCallDelta { block: 1, id, provider_id: Some(provider_id), .. } if id.as_str() == "call-b" && provider_id == "call-b")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 0, signature } if signature == "sig-a")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 2, signature } if signature == "sig-text")));
}

#[test]
fn streaming_legacy_calls_have_distinct_local_ids_without_provider_ids() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"path":"a"}}},{"functionCall":{"name":"read","args":{"path":"b"}}}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta {
                id, provider_id, ..
            } => Some((id, provider_id)),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].0, calls[1].0);
    assert!(calls.iter().all(|(_, provider_id)| provider_id.is_none()));
}

#[test]
fn streaming_signed_text_parts_keep_their_own_signature_blocks() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"text":"first","thoughtSignature":"sig-1"},{"text":"second","thoughtSignature":"sig-2"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 0, text } if text == "first")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 0, signature } if signature == "sig-1")));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 1, text } if text == "second")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 1, signature } if signature == "sig-2")));
}

#[test]
fn streaming_unsigned_then_signed_text_parts_do_not_share_a_block() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"text":"A"},{"text":"B","thoughtSignature":"sig-B"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 0, text } if text == "A")));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 1, text } if text == "B")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 1, signature } if signature == "sig-B")));
}

#[test]
fn streaming_unsigned_then_signed_reasoning_parts_do_not_share_a_block() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"text":"A","thought":true},{"text":"B","thought":true,"thoughtSignature":"sig-B"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ReasoningDelta { block: 0, text } if text == "A")));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ReasoningDelta { block: 1, text } if text == "B")));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 1, signature } if signature == "sig-B")));
}

#[test]
fn streaming_one_text_part_per_frame_keeps_its_open_block_until_signed() {
    let mut d = GeminiCodec.stream_decoder();
    let first = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"text":"A"}]}}]}"#,
        )
        .unwrap();
    let second = d
        .decode_frame(
            br#"{"candidates":[{"content":{"parts":[{"text":"B","thoughtSignature":"sig-AB"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(first
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 0, text } if text == "A")));
    assert!(second
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 0, text } if text == "B")));
    assert!(second.iter().any(|e| matches!(e, StreamEvent::ThoughtSignature { block: 0, signature } if signature == "sig-AB")));
}

#[test]
fn streaming_tool_call_separates_surrounding_text_blocks() {
    let mut d = GeminiCodec.stream_decoder();
    let events = d
        .decode_frame(
            br#"{"modelVersion":"wire-m","candidates":[{"content":{"parts":[{"text":"before"},{"functionCall":{"id":"call-a","name":"read","args":{}}},{"text":"after"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 0, text } if text == "before")));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ToolCallDelta { block: 1, .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { block: 2, text } if text == "after")));
}
