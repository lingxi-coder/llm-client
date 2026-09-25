//! Every protocol family has a codec, and the two wires that are not variants
//! of the other seven behave the way their providers expect.

#[path = "support/wire_api.rs"]
mod wire_api;

use lingxi_llm_client::codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
use lingxi_llm_client::framing::eventstream::crc32;
use lingxi_llm_client::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, FailoverTriggers, LlmError, MessageRole,
    ModelCapabilitySupport, ProtocolFamily, ProviderId, ProviderProfile, ResponseId, StopReason,
    StreamEvent, SystemBlock, ToolChoice, ToolUseId, Usage,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, BedrockClaudeCodec, GeminiCodec, LlmClientBuilder, PricingModelRef,
    RequestOptions, ResolvedRoute, VertexClaudeCodec, WireCodec,
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
        capability_support: ModelCapabilitySupport::default(),
        connection_chain: vec![],
        failover: FailoverTriggers::default(),
    }
}

fn request(messages: Vec<ConversationMessage>) -> CompletionRequest {
    CompletionRequest {
        controls: Default::default(),
        service_tier: None,
        model: "m".to_owned(),
        web_search: None,
        file_search: None,
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

// --- the closed set --------------------------------------------------------

#[test]
fn every_protocol_family_has_a_codec() {
    let http = std::sync::Arc::new(NoHttp);
    let families: Vec<ProtocolFamily> = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
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

    let http = OpenAiResponsesCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("open_ai_responses", "https://api.acme.test/v1"),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
        cache_control: None,
        text: "be brief".to_owned(),
        cacheable: true,
    }];
    let b = body(
        &OpenAiResponsesCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile("open_ai_responses", "https://api.acme.test/v1"),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
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
        cache_control: None,
        text: "repeat this every turn".to_owned(),
        cacheable: false,
    }];
    let mut p = profile("open_ai_responses", "https://api.acme.test/v1");
    p.extra = json!({"supports_previous_response_id": true});

    let b = body(
        &OpenAiResponsesCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(&p, &route().request_model, &RequestOptions::default()),
            )
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
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("open_ai_responses", "https://api.acme.test/v1"),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile(family, "https://api.acme.test"),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
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
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("open_ai_responses", "https://api.acme.test/v1"),
                &route().request_model,
                &RequestOptions::default(),
            ),
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
    let mut d = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for f in [
        r#"{"type":"response.created","response":{"id":"resp_stream","model":"wire-m"}}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"delta":"hel"}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"delta":"lo"}"#,
        r#"{"type":"response.in_progress"}"#,
        r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":2}}}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *d, f.as_bytes()).unwrap());
    }
    // A gateway may append [DONE] after that; it must not produce a second end.
    events.extend(wire_api::decode_frame(&mut *d, b"[DONE]").unwrap());
    events.extend(wire_api::finish(&mut *d).unwrap());

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
        Some(StreamEvent::End {
            usage, stop_reason, ..
        }) => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            assert_eq!(usage.usage.as_ref().unwrap().input_tokens, 4);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_function_call_is_named_once_and_its_arguments_stream_after() {
    let mut d = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for f in [
        r#"{"type":"response.created","response":{"model":"m"}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"fc_1","name":"read"}}"#,
        r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"p\":"}"#,
        r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"1}"}"#,
        r#"{"type":"response.completed","response":{"status":"completed"}}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *d, f.as_bytes()).unwrap());
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
fn buffered_and_streamed_responses_preserve_all_reasoning_summary_parts() {
    let summary = json!([
        {"type": "summary_text", "text": "first"},
        {"type": "summary_text", "text": "second"},
    ]);
    let response = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "status": "completed",
            "output": [{"type": "reasoning", "summary": summary}],
        }))
        .unwrap()
        .into(),
    };
    let buffered = OpenAiResponsesCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
    let buffered_summary = buffered
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for frame in [
        r#"{"type":"response.created","response":{"model":"wire-m"}}"#,
        r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"first"}"#,
        r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"second"}"#,
        r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"first"},{"type":"summary_text","text":"second"}]}]}}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    let streamed_summary = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(buffered_summary, vec!["first", "second"]);
    assert_eq!(buffered_summary, streamed_summary);
}

#[test]
fn an_incomplete_function_call_retains_its_truncation_reason() {
    let response = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "id": "resp_incomplete",
            "model": "wire-m",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "function_call", "call_id": "call-1", "name": "read", "arguments": "{\"path\":"}]
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiResponsesCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
    assert_eq!(decoded.stop_reason, StopReason::MaxTokens);
    assert!(matches!(
        decoded.message.content.as_slice(),
        [ContentBlock::ProviderContent { .. }]
    ));

    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for frame in [
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call-1","name":"read"}}"#,
        r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":"}"#,
        r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::MaxTokens,
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
    let decoded = OpenAiResponsesCodec
        .decode_response(&resp, &wire_api::decode_context())
        .unwrap();
    assert!(
        matches!(
            decoded.message.content.as_slice(),
            [ContentBlock::Text { text, .. }] if text == "here is what I can do"
        ),
        "only output_text parts carry model text: {:?}",
        decoded.message.content
    );
    assert_eq!(
        decoded.response_id.as_ref().map(ResponseId::as_str),
        Some("resp_complete")
    );
    assert_eq!(
        decoded.stop_reason,
        StopReason::Refusal,
        "a refusal part determines the turn result while output_text policy stays unchanged"
    );
}

#[test]
fn a_completed_refusal_only_response_has_no_answer_text_but_keeps_refusal_status() {
    let response = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "model": "wire-m",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "I can't help with that"}
            ]}],
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiResponsesCodec
        .decode_response(&response, &wire_api::decode_context())
        .unwrap();
    assert!(decoded.message.content.is_empty());
    assert_eq!(decoded.stop_reason, StopReason::Refusal);
}

#[test]
fn responses_streamed_refusal_keeps_its_stop_reason() {
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for frame in [
        r#"{"type":"response.created","response":{"model":"wire-m"}}"#,
        r#"{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"I can't help with that"}"#,
        r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"I can't help with that"}]}]}}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());

    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::Refusal,
            ..
        })
    ));
}

#[test]
fn responses_refusal_does_not_replace_truncation_or_provider_errors() {
    let truncated = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "partial refusal"}
            ]}],
        }))
        .unwrap()
        .into(),
    };
    let decoded = OpenAiResponsesCodec
        .decode_response(&truncated, &wire_api::decode_context())
        .unwrap();
    assert_eq!(decoded.stop_reason, StopReason::MaxTokens);

    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(&mut *decoder , br#"{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"partial refusal"}"#)
        .unwrap();
    let events = wire_api::decode_frame(&mut *decoder , br#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#)
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::MaxTokens,
            ..
        })
    ));

    let mut failed = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(&mut *failed , br#"{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"refusal"}"#)
        .unwrap();
    assert!(matches!(
        wire_api::decode_frame(&mut *failed , br#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"generation failed"}}}"#),
        Err(LlmError::ProviderInternal { .. })
    ));
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
            lingxi_llm_client::EncodeRequest::new(&request(vec![user("hi")])),
            &wire_api::context(
                &profile(
                    "bedrock_claude",
                    "https://bedrock-runtime.us-east-1.amazonaws.com",
                ),
                &route().request_model,
                &wire_api::EncodingOptions {
                    stream: true,
                    ..wire_api::EncodingOptions::default()
                },
            ),
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
        b.get("stream").is_none(),
        "Bedrock selects streaming through the invoke URL, not the Anthropic body field"
    );
    assert!(
        !http.headers.iter().any(|(k, _)| k == "anthropic-version"),
        "the header is rejected here"
    );
}

#[test]
fn bedrock_uses_one_path_segment_for_model_ids_and_omits_the_stream_body_field() {
    let base = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let cases = [
        (
            "wire-m",
            false,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/wire-m/invoke",
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
            true,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Fus.anthropic.claude-sonnet-4-20250514-v1:0/invoke-with-response-stream",
        ),
    ];

    for (model_id, stream, expected_url) in cases {
        let mut selected = route();
        selected.request_model = model_id.to_owned();
        let http = BedrockClaudeCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&request(vec![user("hi")])),
                &wire_api::context(
                    &profile("bedrock_claude", base),
                    &selected.request_model,
                    &wire_api::EncodingOptions {
                        stream,
                        ..wire_api::EncodingOptions::default()
                    },
                ),
            )
            .unwrap();

        assert_eq!(http.url, expected_url);
        assert!(
            body(&http).get("stream").is_none(),
            "stream mode is selected by the URL for model ID {model_id}"
        );
    }
}

#[test]
fn vertex_claude_uses_its_platform_version_in_both_request_modes() {
    let p = profile(
        "vertex_claude",
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
    );

    for stream in [false, true] {
        let http = VertexClaudeCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&request(vec![user("hi")])),
                &wire_api::context(
                    &p,
                    &route().request_model,
                    &wire_api::EncodingOptions {
                        stream,
                        ..wire_api::EncodingOptions::default()
                    },
                ),
            )
            .unwrap();

        assert_eq!(body(&http)["anthropic_version"], "vertex-2023-10-16");
        assert!(
            !http
                .headers
                .iter()
                .any(|(name, _)| name == "anthropic-version"),
            "Vertex receives anthropic_version in the body"
        );
    }
}

#[test]
fn vertex_claude_preserves_an_explicit_profile_api_version_override() {
    let mut p = profile(
        "vertex_claude",
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
    );
    p.extra = json!({"api_version": "vertex-custom-version"});

    let http = VertexClaudeCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&request(vec![user("hi")])),
            &wire_api::context(&p, &route().request_model, &RequestOptions::default()),
        )
        .unwrap();

    assert_eq!(body(&http)["anthropic_version"], "vertex-custom-version");
}

#[test]
fn bedrock_unwraps_event_stream_frames_into_the_anthropic_decoder() {
    let mut d = BedrockClaudeCodec.stream_decoder(&wire_api::decode_context());
    let mut events = Vec::new();
    for json in [
        r#"{"type":"message_start","message":{"model":"wire-m","usage":{"input_tokens":3}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
        r#"{"type":"message_stop"}"#,
    ] {
        events.extend(wire_api::decode_frame(&mut *d, &bedrock_frame(json)).unwrap());
    }
    events.extend(wire_api::finish(&mut *d).unwrap());

    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::TextDelta { text, .. } if text == "hello"
    )));
    match events.last() {
        Some(StreamEvent::End {
            usage, stop_reason, ..
        }) => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            assert_eq!(
                usage.usage.unwrap(),
                Usage {
                    input_tokens: 3,
                    output_tokens: 2,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    cache_write_1h_tokens: 0,
                    reasoning_tokens: 0,
                    cost: None,
                    server_tool_usage: None,
                }
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_truncated_bedrock_stream_is_an_interruption_not_a_clean_end() {
    let bytes = bedrock_frame(r#"{"type":"message_stop"}"#);
    let mut d = BedrockClaudeCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(&mut *d, &bytes[..bytes.len() - 4]).unwrap();

    let err = wire_api::finish(&mut *d).unwrap_err();
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
        cache_write_1h_tokens: 0,
        reasoning_tokens: 7,
        cost: None,
        server_tool_usage: None,
    };

    let openai = wire_api::response_usage(
        &OpenAiChatCodec,
        &json_response(&json!({
            "usage": {
                "prompt_tokens": 1000,           // 600 uncached + 400 cached
                "completion_tokens": 57,         // 50 answer + 7 thinking
                "prompt_tokens_details": {"cached_tokens": 400},
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        })),
        &wire_api::decode_context(),
    );
    let gemini = wire_api::response_usage(
        &lingxi_llm_client::GeminiCodec,
        &json_response(&json!({
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 1000,        // cached folded in
                "cachedContentTokenCount": 400,
                "candidatesTokenCount": 50,      // thoughts NOT folded in
                "thoughtsTokenCount": 7,
            }
        })),
        &wire_api::decode_context(),
    );
    let anthropic = wire_api::response_usage(
        &lingxi_llm_client::AnthropicMessagesCodec,
        &json_response(&json!({
            "usage": {
                "input_tokens": 600,         // already excludes the cached read
                "output_tokens": 57,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 0,
            }
        })),
        &wire_api::decode_context(),
    );

    let responses = wire_api::response_usage(
        &OpenAiResponsesCodec,
        &json_response(&json!({
            "usage": {
                "input_tokens": 1000,            // cached folded in, as on chat
                "output_tokens": 57,
                "input_tokens_details": {"cached_tokens": 400},
                "output_tokens_details": {"reasoning_tokens": 7},
            }
        })),
        &wire_api::decode_context(),
    );

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
    let priced = wire_api::response_usage(
        &OpenAiChatCodec,
        &json_response(&json!({
            "usage": {
                "prompt_tokens": 194,
                "completion_tokens": 2,
                "cost": 0.95,
            }
        })),
        &wire_api::decode_context(),
    );
    assert_eq!(
        priced.unwrap().cost,
        Some(lingxi_llm_client::protocol::ReportedCost {
            nano_usd: 950_000_000
        })
    );

    let unpriced = wire_api::response_usage(
        &OpenAiChatCodec,
        &json_response(&json!({
            "usage": {"prompt_tokens": 194, "completion_tokens": 2}
        })),
        &wire_api::decode_context(),
    );
    assert_eq!(
        unpriced.unwrap().cost,
        None,
        "the same wire, a provider that does not price it: unknown, not zero"
    );

    let nonsense = wire_api::response_usage(
        &OpenAiChatCodec,
        &json_response(&json!({
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": -3.0}
        })),
        &wire_api::decode_context(),
    );
    assert_eq!(
        nonsense.unwrap().cost,
        None,
        "a negative amount is not a cost, and must not wrap into a huge one"
    );
}

#[test]
fn codecs_reject_malformed_success_bodies_and_redirects() {
    let codecs: Vec<Box<dyn WireCodec>> = vec![
        Box::new(AnthropicMessagesCodec),
        Box::new(GeminiCodec),
        Box::new(OpenAiResponsesCodec),
    ];
    for codec in codecs {
        for (status, body) in [(200, "not JSON"), (200, "null"), (200, "{}"), (302, "{}")] {
            let response = lingxi_llm_client::HttpResponse {
                status,
                headers: vec![],
                body: body.as_bytes().to_vec().into(),
            };
            assert!(
                codec
                    .decode_response(&response, &wire_api::decode_context())
                    .is_err(),
                "{:?}: {status} {body}",
                codec.family()
            );
        }
    }
}

#[test]
fn responses_failed_event_preserves_provider_error_classification() {
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    let error = wire_api::decode_frame(&mut *decoder , br#"{"type":"response.failed","response":{"error":{"code":"context_length_exceeded","message":"context exceeded"}}}"#).unwrap_err();
    assert!(matches!(error, LlmError::ContextOverflow { .. }));
}

#[test]
fn chat_and_gemini_stream_error_frames_are_errors() {
    let mut chat = OpenAiChatCodec.stream_decoder(&wire_api::decode_context());
    assert!(matches!(
        wire_api::decode_frame(
            &mut *chat,
            br#"{"error":{"code":"invalid_api_key","message":"bad key"}}"#
        ),
        Err(LlmError::Authentication { .. })
    ));
    let mut gemini = GeminiCodec.stream_decoder(&wire_api::decode_context());
    assert!(matches!(
        wire_api::decode_frame(
            &mut *gemini,
            br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"too many requests"}}"#
        ),
        Err(LlmError::RateLimited { .. })
    ));
}

#[test]
fn text_documents_are_base64_encoded_on_gemini_and_responses() {
    use base64::Engine;
    use lingxi_llm_client::protocol::DocumentSource;
    let text = "hello 世界";
    let req = request(vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::Text {
                media_type: "text/plain".to_owned(),
                data: text.to_owned(),
            },
            title: Some("sample.txt".to_owned()),
        }],
    }]);
    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    let gemini = GeminiCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("gemini_generate_content", "https://example.test"),
                &route().request_model,
                &RequestOptions::default(),
            ),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&gemini.body).unwrap();
    assert_eq!(
        body["contents"][0]["parts"][0]["inlineData"]["data"],
        encoded
    );
    let responses = OpenAiResponsesCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("open_ai_responses", "https://example.test"),
                &route().request_model,
                &RequestOptions::default(),
            ),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&responses.body).unwrap();
    assert_eq!(
        body["input"][0]["content"][0]["file_data"],
        format!("data:text/plain;base64,{encoded}")
    );
}

#[test]
fn responses_document_urls_use_file_url() {
    use lingxi_llm_client::protocol::DocumentSource;
    let req = request(vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::Url {
                url: "https://example.test/doc.pdf".to_owned(),
            },
            title: None,
        }],
    }]);
    let encoded = OpenAiResponsesCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &wire_api::context(
                &profile("open_ai_responses", "https://example.test"),
                &route().request_model,
                &RequestOptions::default(),
            ),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    let part = &body["input"][0]["content"][0];
    assert_eq!(part["file_url"], "https://example.test/doc.pdf");
    assert!(part.get("file_data").is_none());
}

#[test]
fn responses_failed_body_is_an_error_even_with_http_success() {
    let response = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "status": "failed", "output": [],
            "error": {"code": "insufficient_quota", "message": "quota exhausted"}
        }))
        .unwrap()
        .into(),
    };
    assert!(matches!(
        OpenAiResponsesCodec.decode_response(&response, &wire_api::decode_context()),
        Err(LlmError::QuotaExceeded { .. })
    ));
}

#[test]
fn stream_eof_without_terminal_evidence_is_interrupted() {
    let cases: Vec<(Box<dyn WireCodec>, &[u8])> = vec![
        (Box::new(OpenAiChatCodec), br#"{"choices":[{"delta":{"content":"partial"}}]}"#),
        (Box::new(AnthropicMessagesCodec), br#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#),
        (Box::new(OpenAiResponsesCodec), br#"{"type":"response.output_text.delta","output_index":0,"delta":"partial"}"#),
        (Box::new(GeminiCodec), br#"{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}"#),
    ];
    for (codec, frame) in cases {
        let mut empty = codec.stream_decoder(&wire_api::decode_context());
        assert!(
            matches!(
                wire_api::finish(&mut *empty),
                Err(LlmError::StreamInterrupted { .. })
            ),
            "empty {:?}",
            codec.family()
        );
        let mut partial = codec.stream_decoder(&wire_api::decode_context());
        wire_api::decode_frame(&mut *partial, frame).unwrap();
        assert!(
            matches!(
                wire_api::finish(&mut *partial),
                Err(LlmError::StreamInterrupted { .. })
            ),
            "partial {:?}",
            codec.family()
        );
    }
}

#[test]
fn terminal_evidence_allows_clean_eof() {
    let cases: Vec<(Box<dyn WireCodec>, &[u8])> = vec![
        (
            Box::new(OpenAiChatCodec),
            br#"{"choices":[{"finish_reason":"stop"}]}"#,
        ),
        (
            Box::new(AnthropicMessagesCodec),
            br#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        ),
        (
            Box::new(OpenAiResponsesCodec),
            br#"{"type":"response.incomplete","response":{"status":"incomplete"}}"#,
        ),
        (
            Box::new(GeminiCodec),
            br#"{"candidates":[{"finishReason":"STOP"}]}"#,
        ),
    ];
    for (codec, frame) in cases {
        let mut decoder = codec.stream_decoder(&wire_api::decode_context());
        let mut events = wire_api::decode_frame(&mut *decoder, frame).unwrap();
        events.extend(wire_api::finish(&mut *decoder).unwrap());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::End { .. }))
                .count(),
            1
        );
        assert!(wire_api::finish(&mut *decoder).unwrap().is_empty());
    }
}

#[test]
fn gemini_explicit_prompt_block_is_a_refusal_not_truncation() {
    let blocked = json!({
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [],
    });
    let response = lingxi_llm_client::HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&blocked).unwrap().into(),
    };
    assert_eq!(
        GeminiCodec
            .decode_response(&response, &wire_api::decode_context())
            .unwrap()
            .stop_reason,
        StopReason::Refusal,
        "prompt-level safety blocks are refusals in buffered responses too"
    );

    let mut decoder = GeminiCodec.stream_decoder(&wire_api::decode_context());
    wire_api::decode_frame(&mut *decoder, &serde_json::to_vec(&blocked).unwrap()).unwrap();
    assert!(matches!(
        wire_api::finish(&mut *decoder).unwrap().as_slice(),
        [StreamEvent::End {
            stop_reason: StopReason::Refusal,
            ..
        }]
    ));

    for feedback in [
        json!({}),
        json!({"blockReason": "BLOCK_REASON_UNSPECIFIED"}),
    ] {
        let body = json!({
            "promptFeedback": feedback,
            "candidates": [{"finishReason": "STOP"}],
        });
        let response = lingxi_llm_client::HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&body).unwrap().into(),
        };
        assert_eq!(
            GeminiCodec
                .decode_response(&response, &wire_api::decode_context())
                .unwrap()
                .stop_reason,
            StopReason::EndTurn,
            "empty and unspecified prompt feedback must not be interpreted as a block"
        );

        let mut decoder = GeminiCodec.stream_decoder(&wire_api::decode_context());
        wire_api::decode_frame(&mut *decoder, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert!(matches!(
            wire_api::finish(&mut *decoder).unwrap().as_slice(),
            [StreamEvent::End {
                stop_reason: StopReason::EndTurn,
                ..
            }]
        ));
    }
}

#[test]
fn responses_flat_error_event_keeps_error_code() {
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
    assert!(matches!(
        wire_api::decode_frame(
            &mut *decoder,
            br#"{"type":"error","code":"context_length_exceeded","message":"context exceeded"}"#
        ),
        Err(LlmError::ContextOverflow { .. })
    ));
}

#[test]
fn responses_reasoning_round_trips_before_tool_outputs() {
    for summary in [json!([]), json!([{"type":"summary_text","text":"plan"}])] {
        let reasoning =
            json!({"type":"reasoning","id":"rs_1","summary":summary,"encrypted_content":"opaque"});
        let response = json_response(
            &json!({"model":"wire-m","status":"completed","output":[reasoning, {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]}),
        );
        let decoded = OpenAiResponsesCodec
            .decode_response(&response, &wire_api::decode_context())
            .unwrap();
        let req = request(vec![
            decoded.message,
            ConversationMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: ToolUseId::new("call_1"),
                    content: "result".into(),
                    is_error: false,
                    blocks: None,
                }],
            },
        ]);
        let http = OpenAiResponsesCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile("open_ai_responses", "https://test.invalid/v1"),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["input"][0], reasoning);
        assert_eq!(body["input"][1]["type"], "function_call");
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert!(OpenAiChatCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile("open_ai_chat", "https://test.invalid/v1"),
                    &route().request_model,
                    &RequestOptions::default()
                )
            )
            .is_err());
    }
}

#[test]
fn responses_stream_preserves_native_reasoning_once_with_terminal_fallback() {
    for item_done in [true, false] {
        let reasoning =
            json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"});
        let mut decoder = OpenAiResponsesCodec.stream_decoder(&wire_api::decode_context());
        let mut events = vec![];
        if item_done {
            events.extend(
                wire_api::decode_frame(
                    &mut *decoder,
                    json!({"type":"response.output_item.done","output_index":0,"item":reasoning})
                        .to_string()
                        .as_bytes(),
                )
                .unwrap(),
            );
        }
        events.extend(wire_api::decode_frame(&mut *decoder , json!({"type":"response.completed","response":{"status":"completed","output":[reasoning]}}).to_string().as_bytes()).unwrap());
        let native: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ProviderContent {
                    block,
                    protocol,
                    value,
                } => Some((*block, *protocol, value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            native,
            vec![(0, ProtocolFamily::OpenAiResponses, &reasoning)]
        );
        assert!(matches!(events.last(), Some(StreamEvent::End { .. })));
    }
}

#[test]
fn responses_tool_results_preserve_structured_and_text_outputs() {
    for blocks in [
        None,
        Some(vec![
            json!({"type":"input_image","image_url":"https://example.test/screenshot.png"}),
            json!({"type":"input_file","file_id":"file-result"}),
        ]),
    ] {
        let req = request(vec![ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: ToolUseId::new("call_1"),
                content: "fallback".into(),
                is_error: false,
                blocks: blocks.clone(),
            }],
        }]);
        let http = OpenAiResponsesCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(
                    &profile("open_ai_responses", "https://test.invalid/v1"),
                    &route().request_model,
                    &RequestOptions::default(),
                ),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(
            body["input"][0]["output"],
            blocks.map_or(json!("fallback"), |v| json!(v))
        );
    }
}
