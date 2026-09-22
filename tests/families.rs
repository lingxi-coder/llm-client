//! Every protocol family has a codec, and the two wires that are not variants
//! of the other seven behave the way their providers expect.

use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, FailoverTriggers, LlmError, MessageRole,
    ModelCapabilities, ProtocolFamily, ProviderId, ProviderProfile, ResponseId, StopReason,
    StreamEvent, SystemBlock, ToolChoice, ToolUseId, Usage,
};
use lingxi_llm_client::codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
use lingxi_llm_client::framing::eventstream::crc32;
use lingxi_llm_client::{
    AnthropicMessagesCodec, BedrockClaudeCodec, Clock, GeminiCodec, LlmClientBuilder, LlmServices,
    PricingModelRef, RequestOptions, ResolvedRoute, WireCodec,
};
use serde_json::{json, Value};

mod support;
use support::NoHttp;

fn profile(protocol: &str, base: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": base,
        "protocol": protocol,
        "auth": "none",
        "models": [{"display_model": "m", "request_model": "wire-m", "billing_model": "wire-m"}],
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

fn request(messages: Vec<ConversationMessage>) -> CompletionRequest {
    CompletionRequest {
        model: "m".to_owned(),
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
        }],
    }
}

fn body(http: &lingxi_llm_client::HttpRequest) -> Value {
    serde_json::from_slice(&http.body).unwrap()
}

// --- the closed set --------------------------------------------------------

#[test]
fn every_protocol_family_has_a_codec() {
    let services = LlmServices {
        http: std::sync::Arc::new(NoHttp),
        clock: std::sync::Arc::new(Now),
    };
    let families: Vec<ProtocolFamily> = LlmClientBuilder::new(&services, &[])
        .build()
        .unwrap()
        .codec_families();

    for family in ProtocolFamily::ALL {
        assert!(
            families.contains(&family),
            "{family:?} has no codec, so a profile naming it cannot be built \
             (gate 33) — and nothing else would have said so"
        );
    }
    assert_eq!(families.len(), 9);
}

struct Now;
impl Clock for Now {
    fn now(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }
}

// --- the Responses wire ----------------------------------------------------

#[test]
fn the_conversation_is_a_flat_list_of_items() {
    let req = request(vec![
        user("hi"),
        ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: ToolUseId::new("call-1"),
                name: "read".to_owned(),
                input: json!({"path": "a"}),
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

    let http = OpenAiResponsesCodec
        .encode_request(
            &req,
            &profile("open_ai_responses", "https://api.acme.test/v1"),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap();
    let b = body(&http);
    let kinds: Vec<&str> = b["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["type"].as_str().unwrap())
        .collect();

    assert_eq!(http.url, "https://api.acme.test/v1/responses");
    assert_eq!(
        kinds,
        vec!["message", "function_call", "function_call_output"],
        "a call is a sibling item, not a field on a message"
    );
    assert_eq!(
        b["input"][1]["arguments"], "{\"path\":\"a\"}",
        "arguments are a JSON string here too"
    );
    assert_eq!(b["input"][1]["call_id"], "call-1");
}

#[test]
fn the_system_prompt_is_instructions_not_a_message() {
    let mut req = request(vec![user("hi")]);
    req.system = vec![SystemBlock {
        text: "be brief".to_owned(),
        cacheable: true,
    }];
    let b = body(
        &OpenAiResponsesCodec
            .encode_request(
                &req,
                &profile("open_ai_responses", "https://api.acme.test/v1"),
                &route(),
                &RequestOptions::default(),
            )
            .unwrap(),
    );
    assert_eq!(b["instructions"], "be brief");
    assert_eq!(b["input"].as_array().unwrap().len(), 1);
}

#[test]
fn a_declared_stateful_endpoint_receives_the_typed_continuation_id() {
    let mut req = request(vec![user("only the new turn")]);
    req.previous_response_id = Some(ResponseId::new("resp_previous"));
    req.system = vec![SystemBlock {
        text: "repeat this every turn".to_owned(),
        cacheable: false,
    }];
    let mut p = profile("open_ai_responses", "https://api.acme.test/v1");
    p.extra = json!({"supports_previous_response_id": true});

    let b = body(
        &OpenAiResponsesCodec
            .encode_request(&req, &p, &route(), &RequestOptions::default())
            .unwrap(),
    );
    assert_eq!(b["previous_response_id"], "resp_previous");
    assert_eq!(b["instructions"], "repeat this every turn");
    assert_eq!(b["input"].as_array().unwrap().len(), 1);
}

#[test]
fn a_responses_profile_must_opt_in_before_it_can_receive_a_continuation_id() {
    let mut req = request(vec![user("next")]);
    req.previous_response_id = Some(ResponseId::new("resp_previous"));
    let err = OpenAiResponsesCodec
        .encode_request(
            &req,
            &profile("open_ai_responses", "https://api.acme.test/v1"),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(err, LlmError::UnsupportedCapability { .. }));
}

#[test]
fn non_responses_wires_refuse_a_continuation_id_instead_of_dropping_it() {
    let mut req = request(vec![user("next")]);
    req.previous_response_id = Some(ResponseId::new("resp_previous"));
    for (codec, family) in [
        (&OpenAiChatCodec as &dyn WireCodec, "open_ai_chat"),
        (
            &AnthropicMessagesCodec as &dyn WireCodec,
            "anthropic_messages",
        ),
        (&GeminiCodec as &dyn WireCodec, "gemini_generate_content"),
    ] {
        let err = codec
            .encode_request(
                &req,
                &profile(family, "https://api.acme.test"),
                &route(),
                &RequestOptions::default(),
            )
            .unwrap_err();
        assert!(matches!(err, LlmError::UnsupportedCapability { .. }));
    }
}

#[test]
fn a_stop_sequence_is_refused_rather_than_dropped() {
    let mut req = request(vec![user("hi")]);
    req.stop_sequences.push("STOP".to_owned());

    let err = OpenAiResponsesCodec
        .encode_request(
            &req,
            &profile("open_ai_responses", "https://api.acme.test/v1"),
            &route(),
            &RequestOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(err, LlmError::InvalidRequest { ref message } if message.contains("stop")),
        "this wire has no stop parameter; sending the request anyway would run \
         the model without a limit the caller asked for: {err:?}"
    );
}

#[test]
fn the_stream_ends_at_response_completed_without_a_sentinel() {
    let mut d = OpenAiResponsesCodec.stream_decoder();
    let mut events = Vec::new();
    for f in [
        r#"{"type":"response.created","response":{"id":"resp_stream","model":"wire-m"}}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"delta":"hel"}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"delta":"lo"}"#,
        r#"{"type":"response.in_progress"}"#,
        r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":2}}}"#,
    ] {
        events.extend(d.decode_frame(f.as_bytes()).unwrap());
    }
    // A gateway may append [DONE] after that; it must not produce a second end.
    events.extend(d.decode_frame(b"[DONE]").unwrap());
    events.extend(d.finish().unwrap());

    assert!(matches!(
        events.first(),
        Some(StreamEvent::Start {
            response_id: Some(id),
            ..
        }) if id.as_str() == "resp_stream"
    ));

    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::End { .. }))
            .count(),
        1
    );
    match events.last() {
        Some(StreamEvent::End { usage, stop_reason }) => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            assert_eq!(usage.input_tokens, 4);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_function_call_is_named_once_and_its_arguments_stream_after() {
    let mut d = OpenAiResponsesCodec.stream_decoder();
    let mut events = Vec::new();
    for f in [
        r#"{"type":"response.created","response":{"model":"m"}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"fc_1","name":"read"}}"#,
        r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"p\":"}"#,
        r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"1}"}"#,
        r#"{"type":"response.completed","response":{"status":"completed"}}"#,
    ] {
        events.extend(d.decode_frame(f.as_bytes()).unwrap());
    }

    let fragments: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta {
                id,
                arguments_fragment,
                ..
            } if id.as_str() == "fc_1" => Some(arguments_fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(fragments, vec!["", "{\"p\":", "1}"]);
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::ToolUse,
            ..
        })
    ));
}

#[test]
fn a_refusal_part_is_not_reported_as_the_answer() {
    let resp = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "id": "resp_complete",
            "model": "wire-m",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "I can't help with that"},
                {"type": "output_text", "text": "here is what I can do"},
            ]}],
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiResponsesCodec.decode_response(&resp).unwrap();
    assert_eq!(
        decoded.message.content.len(),
        1,
        "only output_text parts carry model text"
    );
    assert_eq!(
        decoded.response_id.as_ref().map(ResponseId::as_str),
        Some("resp_complete")
    );
}

// --- Bedrock ---------------------------------------------------------------

/// Build one AWS event-stream frame carrying an Anthropic event.
fn bedrock_frame(event_json: &str) -> Vec<u8> {
    use base64::Engine;
    let payload = serde_json::to_vec(&json!({
        "bytes": base64::engine::general_purpose::STANDARD.encode(event_json),
    }))
    .unwrap();
    let name = ":message-type";
    let value = "event";
    let mut h = Vec::new();
    h.push(name.len() as u8);
    h.extend_from_slice(name.as_bytes());
    h.push(7);
    h.extend_from_slice(&(value.len() as u16).to_be_bytes());
    h.extend_from_slice(value.as_bytes());

    let total = (12 + h.len() + payload.len() + 4) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(&(h.len() as u32).to_be_bytes());
    out.extend_from_slice(&crc32(&out[0..8]).to_be_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&payload);
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

#[test]
fn bedrock_puts_the_model_in_the_url_and_its_own_version_in_the_body() {
    let http = BedrockClaudeCodec
        .encode_request(
            &request(vec![user("hi")]),
            &profile(
                "bedrock_claude",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
            &route(),
            &RequestOptions {
                stream: true,
                ..RequestOptions::default()
            },
        )
        .unwrap();
    let b = body(&http);

    assert_eq!(
        http.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/wire-m/invoke-with-response-stream"
    );
    assert_eq!(
        b["anthropic_version"], "bedrock-2023-05-31",
        "this host has its own version constant, not the first-party date"
    );
    assert!(b.get("model").is_none());
    assert!(
        !http.headers.iter().any(|(k, _)| k == "anthropic-version"),
        "the header is rejected here"
    );
}

#[test]
fn bedrock_unwraps_event_stream_frames_into_the_anthropic_decoder() {
    let mut d = BedrockClaudeCodec.stream_decoder();
    let mut events = Vec::new();
    for json in [
        r#"{"type":"message_start","message":{"model":"wire-m","usage":{"input_tokens":3}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
        r#"{"type":"message_stop"}"#,
    ] {
        events.extend(d.decode_frame(&bedrock_frame(json)).unwrap());
    }
    events.extend(d.finish().unwrap());

    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::TextDelta { text, .. } if text == "hello"
    )));
    match events.last() {
        Some(StreamEvent::End { usage, stop_reason }) => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            assert_eq!(
                *usage,
                Usage {
                    input_tokens: 3,
                    output_tokens: 2,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: 0,
                    cost: None,
                }
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_truncated_bedrock_stream_is_an_interruption_not_a_clean_end() {
    let bytes = bedrock_frame(r#"{"type":"message_stop"}"#);
    let mut d = BedrockClaudeCodec.stream_decoder();
    d.decode_frame(&bytes[..bytes.len() - 4]).unwrap();

    let err = d.finish().unwrap_err();
    assert!(
        matches!(err, LlmError::StreamInterrupted { .. }),
        "ending on a half-frame must not look like a finished turn: {err:?}"
    );
}

/// One logical turn, described by each wire in its own dialect, must decode to
/// one `Usage`. The wires disagree about what their counters contain — OpenAI
/// folds cached tokens into the prompt count and reasoning into the completion
/// count, Gemini folds cached in but keeps thoughts *out*, Anthropic folds
/// nothing — so a codec that passes the numbers through unchanged silently
/// reports a different bill per provider.
///
/// The turn: 600 uncached input, 400 read from cache, 50 answer tokens and 7
/// thinking tokens.
#[test]
fn every_wire_decodes_the_same_turn_to_the_same_usage() {
    let expected = Usage {
        input_tokens: 600,
        output_tokens: 57,
        cache_read_tokens: 400,
        cache_write_tokens: 0,
        reasoning_tokens: 7,
        cost: None,
    };

    let openai = OpenAiChatCodec.response_usage(&json_response(&json!({
        "usage": {
            "prompt_tokens": 1000,           // 600 uncached + 400 cached
            "completion_tokens": 57,         // 50 answer + 7 thinking
            "prompt_tokens_details": {"cached_tokens": 400},
            "completion_tokens_details": {"reasoning_tokens": 7},
        }
    })));
    let gemini = lingxi_llm_client::GeminiCodec.response_usage(&json_response(&json!({
        "candidates": [],
        "usageMetadata": {
            "promptTokenCount": 1000,        // cached folded in
            "cachedContentTokenCount": 400,
            "candidatesTokenCount": 50,      // thoughts NOT folded in
            "thoughtsTokenCount": 7,
        }
    })));
    let anthropic =
        lingxi_llm_client::AnthropicMessagesCodec.response_usage(&json_response(&json!({
            "usage": {
                "input_tokens": 600,         // already excludes the cached read
                "output_tokens": 57,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 0,
            }
        })));

    let responses = OpenAiResponsesCodec.response_usage(&json_response(&json!({
        "usage": {
            "input_tokens": 1000,            // cached folded in, as on chat
            "output_tokens": 57,
            "input_tokens_details": {"cached_tokens": 400},
            "output_tokens_details": {"reasoning_tokens": 7},
        }
    })));

    assert_eq!(openai, Some(expected), "openai chat");
    assert_eq!(responses, Some(expected), "openai responses");
    assert_eq!(gemini, Some(expected), "gemini");
    assert_eq!(
        anthropic,
        Some(Usage {
            reasoning_tokens: 0,
            ..expected
        }),
        "this wire publishes no thinking count, so it alone cannot fill it"
    );
    for (who, got) in [
        ("openai", openai),
        ("responses", responses),
        ("gemini", gemini),
    ] {
        assert_eq!(
            got.unwrap().total(),
            1057,
            "{who} must bill 1057 tokens, not 1457 with the cache counted twice"
        );
    }
}

fn json_response(body: &Value) -> lingxi_llm_client::HttpResponse {
    lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(body).unwrap().into(),
    }
}

/// Only an aggregator knows what a call cost, because only it knows which
/// upstream served it. The first-party wires send token counts and no money, so
/// a cost is absent far more often than present — and absent must read as
/// unknown, not as free.
#[test]
fn a_cost_is_read_when_the_provider_sends_one_and_absent_when_it_does_not() {
    let priced = OpenAiChatCodec.response_usage(&json_response(&json!({
        "usage": {
            "prompt_tokens": 194,
            "completion_tokens": 2,
            "cost": 0.95,
        }
    })));
    assert_eq!(
        priced.unwrap().cost,
        Some(lingxi_agent_api::protocol::ReportedCost {
            nano_usd: 950_000_000
        })
    );

    let unpriced = OpenAiChatCodec.response_usage(&json_response(&json!({
        "usage": {"prompt_tokens": 194, "completion_tokens": 2}
    })));
    assert_eq!(
        unpriced.unwrap().cost,
        None,
        "the same wire, a provider that does not price it: unknown, not zero"
    );

    let nonsense = OpenAiChatCodec.response_usage(&json_response(&json!({
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": -3.0}
    })));
    assert_eq!(
        nonsense.unwrap().cost,
        None,
        "a negative amount is not a cost, and must not wrap into a huge one"
    );
}
