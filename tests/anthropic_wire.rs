//! Gate 18: a thinking signature round-trips through this codec.
//!
//! The provider rejects a replayed thinking block whose signature is missing,
//! and it does so on the *next* turn — far from whatever dropped it. So the
//! codec refuses to encode one instead, and these pin both directions: the
//! signature survives decoding, and its absence is an error rather than a
//! quietly-shorter block.
//!
//! Gate 31 is here too: this wire has two distinct ways of saying "the
//! transcript no longer fits", and both have to become `ContextOverflow`.

use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, FailoverTriggers, LlmError, MessageRole,
    ModelCapabilities, ProviderId, ProviderProfile, StopReason, StreamEvent, ThinkingConfig,
    ToolChoice, ToolUseId, Usage,
};
use lingxi_llm_client::codecs::anthropic::classify_error;
use lingxi_llm_client::{
    AnthropicMessagesCodec, HttpResponse, PricingModelRef, RequestOptions, ResolvedRoute, WireCodec,
};
use serde_json::{json, Value};

fn profile(extra: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": "https://api.acme.test",
        "protocol": "anthropic_messages",
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

fn request(content: Vec<ContentBlock>) -> CompletionRequest {
    CompletionRequest {
        model: "m".to_owned(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage {
            role: MessageRole::Assistant,
            content,
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

fn encode(req: &CompletionRequest, extra: Value) -> Result<Value, LlmError> {
    let http = AnthropicMessagesCodec.encode_request(
        req,
        &profile(extra),
        &route(),
        &RequestOptions::default(),
    )?;
    Ok(serde_json::from_slice(&http.body).unwrap())
}

fn decode_stream(frames: &[&str]) -> Vec<StreamEvent> {
    let mut d = AnthropicMessagesCodec.stream_decoder();
    let mut out = Vec::new();
    for f in frames {
        out.extend(d.decode_frame(f.as_bytes()).unwrap());
    }
    out.extend(d.finish().unwrap());
    out
}

// --- gate 18 ---------------------------------------------------------------

#[test]
fn a_signed_thinking_block_survives_the_round_trip() {
    let resp = HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "step one", "signature": "sig-abc"},
                {"type": "text", "text": "done"},
            ],
        }))
        .unwrap()
        .into(),
    };

    let decoded = AnthropicMessagesCodec.decode_response(&resp).unwrap();
    let signature = decoded.message.content.iter().find_map(|b| match b {
        ContentBlock::Thinking { signature, .. } => signature.clone(),
        _ => None,
    });
    assert_eq!(
        signature.as_deref(),
        Some("sig-abc"),
        "the signature has to survive decoding or the next turn cannot replay it"
    );

    // Replaying what was decoded produces the same signature on the wire.
    let body = encode(&request(decoded.message.content), Value::Null).unwrap();
    let blocks = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["signature"], "sig-abc");
    assert_eq!(blocks[0]["thinking"], "step one");
}

#[test]
fn a_thinking_block_without_its_signature_is_refused_not_sent() {
    let err = encode(
        &request(vec![ContentBlock::Thinking {
            text: "step one".to_owned(),
            signature: None,
        }]),
        Value::Null,
    )
    .unwrap_err();

    match err {
        LlmError::InvalidRequest { message } => assert!(
            message.contains("signature"),
            "the error has to name the cause: {message}"
        ),
        other => panic!(
            "sending it would fail on the next turn, far from here; expected a \
             refusal, got {other:?}"
        ),
    }
}

#[test]
fn a_signature_delta_reaches_the_transcript() {
    let events = decode_stream(&[
        r#"{"type":"message_start","message":{"model":"wire-m","usage":{"input_tokens":5}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-xyz"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    assert!(
        events.iter().any(|e| matches!(
            e,
            StreamEvent::ThoughtSignature { block: 0, signature } if signature == "sig-xyz"
        )),
        "the signature arrives after the thinking it signs, as its own delta: {events:?}"
    );
}

#[test]
fn redacted_thinking_round_trips_untouched() {
    let body = encode(
        &request(vec![ContentBlock::RedactedThinking {
            data: "opaque".to_owned(),
        }]),
        Value::Null,
    )
    .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "redacted_thinking", "data": "opaque"}),
        "the provider reads this back; it is not ours to reshape"
    );
}

// --- gate 31 ---------------------------------------------------------------

#[test]
fn both_ways_of_saying_the_transcript_does_not_fit_become_context_overflow() {
    let by_prompt = classify_error(
        400,
        &json!({"error": {"type": "invalid_request_error",
                          "message": "prompt is too long: 260085 tokens > 256000 maximum"}}),
        None,
    );
    match by_prompt {
        LlmError::ContextOverflow { actual, limit, .. } => {
            assert_eq!((actual, limit), (Some(260085), Some(256000)));
        }
        other => panic!("expected an overflow carrying its counts, got {other:?}"),
    }

    let by_size = classify_error(
        413,
        &json!({"error": {"type": "request_too_large",
                          "message": "exceeds the model's context window"}}),
        None,
    );
    assert!(matches!(by_size, LlmError::ContextOverflow { .. }));
}

#[test]
fn the_token_counts_are_read_from_the_raw_message_not_the_display_string() {
    // The display string is "400 prompt is too long: …". A parser that ran on
    // it would find 400 first and report a nonsense gap.
    let err = classify_error(
        400,
        &json!({"error": {"type": "invalid_request_error",
                          "message": "prompt is too long: 100 tokens > 50 maximum"}}),
        None,
    );
    match err {
        LlmError::ContextOverflow { actual, limit, .. } => {
            assert_eq!(actual, Some(100), "not the 400 from the status prefix");
            assert_eq!(limit, Some(50));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_request_too_large_that_is_not_about_context_stays_request_too_large() {
    let err = classify_error(
        413,
        &json!({"error": {"type": "request_too_large", "message": "too many bytes"}}),
        None,
    );
    assert!(
        matches!(err, LlmError::RequestTooLarge { .. }),
        "compaction never shrinks accumulated attachments: {err:?}"
    );
}

#[test]
fn the_error_types_land_where_failover_expects_them() {
    let cases = [
        ("authentication_error", "Authentication"),
        ("permission_error", "PermissionDenied"),
        ("not_found_error", "ModelUnavailable"),
        ("rate_limit_error", "RateLimited"),
        ("overloaded_error", "Overloaded"),
        ("api_error", "ProviderInternal"),
    ];
    for (kind, want) in cases {
        let err = classify_error(500, &json!({"error": {"type": kind, "message": "x"}}), None);
        assert!(
            format!("{err:?}").starts_with(want),
            "{kind} mapped to {err:?}, wanted {want}"
        );
    }
}

// --- the rest of the wire --------------------------------------------------

#[test]
fn the_system_prompt_is_a_top_level_array_and_keeps_its_cache_split() {
    let mut req = request(vec![ContentBlock::Text {
        text: "hi".to_owned(),
    }]);
    req.system = vec![
        lingxi_agent_api::protocol::SystemBlock {
            text: "stable".to_owned(),
            cacheable: true,
        },
        lingxi_agent_api::protocol::SystemBlock {
            text: "volatile".to_owned(),
            cacheable: false,
        },
    ];

    let http = AnthropicMessagesCodec
        .encode_request(
            &req,
            &profile(Value::Null),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&http.body).unwrap();

    assert_eq!(http.url, "https://api.acme.test/v1/messages");
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert!(
        body["system"][1].get("cache_control").is_none(),
        "partitioning the prompt is pointless if both halves are marked the same"
    );
}

#[test]
fn max_tokens_is_always_present_because_the_wire_requires_it() {
    let body = encode(&request(vec![]), Value::Null).unwrap();
    assert_eq!(body["max_tokens"], 4096);
}

#[test]
fn beta_headers_accumulate_instead_of_replacing_each_other() {
    let http = AnthropicMessagesCodec
        .encode_request(
            &request(vec![]),
            &profile(
                json!({"betas": ["computer-use-2025-01-24", "structured-outputs-2025-12-15"]}),
            ),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();

    let beta = http
        .headers
        .iter()
        .find(|(k, _)| k == "anthropic-beta")
        .map(|(_, v)| v.as_str());
    assert_eq!(
        beta,
        Some("computer-use-2025-01-24,structured-outputs-2025-12-15"),
        "a request needing two betas must not silently lose one"
    );
}

#[test]
fn a_tool_call_streamed_as_json_fragments_keeps_the_id_from_its_opening_frame() {
    let events = decode_stream(&[
        r#"{"type":"message_start","message":{"model":"m"}}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"p\":"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);

    let fragments: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta {
                id,
                arguments_fragment,
                ..
            } if id.as_str() == "toolu_1" => Some(arguments_fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        fragments,
        vec!["", "{\"p\":", "1}"],
        "only the opening frame names the call; the fragments refer to its index"
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
fn an_unknown_event_type_is_tolerated() {
    let events = decode_stream(&[
        r#"{"type":"message_start","message":{"model":"m"}}"#,
        r#"{"type":"ping"}"#,
        r#"{"type":"something_added_next_year","payload":{}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    assert_eq!(
        events.len(),
        2,
        "this wire's contract is that a client tolerates types it has never \
         seen: {events:?}"
    );
}

#[test]
fn cache_reads_and_writes_are_counted_separately() {
    let events = decode_stream(&[
        r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":10,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    match events.last() {
        Some(StreamEvent::End { usage, .. }) => assert_eq!(
            *usage,
            Usage {
                input_tokens: 10,
                output_tokens: 4,
                cache_read_tokens: 7,
                cache_write_tokens: 3,
                reasoning_tokens: 0,
                cost: None,
            },
            "a write costs more than a read; collapsing them misprices the turn"
        ),
        other => panic!("{other:?}"),
    }
}

#[test]
fn thinking_config_turns_the_budget_into_the_wire_shape() {
    let mut req = request(vec![]);
    req.thinking = Some(ThinkingConfig {
        budget_tokens: Some(2048),
    });
    let body = encode(&req, Value::Null).unwrap();
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );
}

#[test]
fn a_tool_result_carries_its_error_flag() {
    let body = encode(
        &request(vec![ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_1"),
            content: "no such file".to_owned(),
            is_error: true,
            blocks: None,
        }]),
        Value::Null,
    )
    .unwrap();
    let block = &body["messages"][0]["content"][0];
    assert_eq!(block["type"], "tool_result");
    assert_eq!(block["is_error"], true);
    assert_eq!(block["content"], "no such file");
}

// --- the usage merge -------------------------------------------------------

/// Decode a stream and answer with both the final usage and whether the decoder
/// considers the report complete.
fn decode_usage(frames: &[&str]) -> (Usage, bool) {
    let mut d = AnthropicMessagesCodec.stream_decoder();
    for f in frames {
        d.decode_frame(f.as_bytes()).unwrap();
    }
    d.finish().unwrap();
    (
        d.observed_usage().unwrap_or_default(),
        d.usage_is_complete(),
    )
}

/// This wire reports usage twice and the closing report restates only some
/// counters. Merging on value — taking the later number only when it is
/// non-zero — cannot tell a counter that is genuinely zero now from one this
/// frame simply did not mention.
///
/// A turn that reads from the cache without writing to it is exactly that case:
/// it reports `cache_creation_input_tokens: 0` at the end, and keeping the
/// seed's figure bills a cache write that did not happen.
#[test]
fn a_closing_zero_replaces_the_seed_rather_than_being_ignored() {
    let (usage, _) = decode_usage(&[
        r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":10,"cache_creation_input_tokens":500,"cache_read_input_tokens":0}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":500}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    assert_eq!(
        usage.cache_write_tokens, 0,
        "the provider said zero writes; a write costs many times a read"
    );
    assert_eq!(usage.cache_read_tokens, 500);
    assert_eq!(usage.output_tokens, 4);
}

/// The other half of the same rule: silence is not zero.
#[test]
fn a_counter_the_closing_frame_omits_keeps_what_the_seed_said() {
    let (usage, _) = decode_usage(&[
        r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":10,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens
        ),
        (10, 7, 3),
        "the closing frame restated only the output count"
    );
    assert_eq!(usage.output_tokens, 4);
}

#[test]
fn a_stream_that_never_reported_its_output_is_not_a_complete_report() {
    // Cut off after the seed: the counts are still worth showing, but nothing
    // may be billed or budgeted from them.
    let mut decoder = AnthropicMessagesCodec.stream_decoder();
    decoder
        .decode_frame(
            br#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":10}}}"#,
        )
        .unwrap();
    assert!(matches!(
        decoder.finish(),
        Err(LlmError::StreamInterrupted { .. })
    ));
    assert_eq!(
        decoder.observed_usage().unwrap().input_tokens,
        10,
        "what was seen is still reported"
    );
    assert!(!decoder.usage_is_complete());

    let (_, complete) = decode_usage(&[
        r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":10}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    assert!(complete, "both counters stated: a measurement");
}

#[test]
fn a_cache_split_that_does_not_sum_to_its_own_total_is_not_complete() {
    let split = |five: u64, hour: u64, total: u64| {
        format!(
            r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn"}},"usage":{{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":{total},"cache_creation":{{"ephemeral_5m_input_tokens":{five},"ephemeral_1h_input_tokens":{hour}}}}}}}"#
        )
    };
    let start = r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":1}}}"#;

    let (_, complete) = decode_usage(&[start, &split(30, 70, 100)]);
    assert!(complete, "30 + 70 = 100");

    let (_, complete) = decode_usage(&[start, &split(30, 70, 200)]);
    assert!(
        !complete,
        "the two tariffs are priced differently, so a split that does not \
         account for the total cannot be apportioned"
    );
}

/// The exception to that rule, and it only runs one way: the 5m figure is the
/// total minus the 1h figure, so a report may leave it out. The reverse is not
/// derivable — what is left over after the 5m part could be 1h or could be
/// nothing, and those cost differently.
#[test]
fn only_the_derivable_half_of_a_cache_split_may_be_omitted() {
    let start = r#"{"type":"message_start","message":{"model":"m","usage":{"input_tokens":1}}}"#;
    let one_part = |part: &str, value: u64, total: u64| {
        format!(
            r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn"}},"usage":{{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":{total},"cache_creation":{{"{part}":{value}}}}}}}"#
        )
    };

    let (_, complete) = decode_usage(&[start, &one_part("ephemeral_1h_input_tokens", 70, 100)]);
    assert!(complete, "the 5m part is 100 - 70");

    let (_, complete) = decode_usage(&[start, &one_part("ephemeral_5m_input_tokens", 30, 100)]);
    assert!(
        !complete,
        "the remaining 70 cannot be attributed to a tariff"
    );
}
