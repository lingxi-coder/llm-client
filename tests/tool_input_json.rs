//! Exact tool arguments survive provider decode, stream assembly and replay.
use lingxi_llm_client::codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
use lingxi_llm_client::codecs::{CodecContext, EncodeRequest, RequestMode};
use lingxi_llm_client::exact_json::{
    parse_request_body_json, parse_tool_input_json, serialize_for_request_with_raw_subtrees,
    JsonEncoding,
};
use lingxi_llm_client::protocol::*;
use lingxi_llm_client::{
    AnthropicMessagesCodec, AzureOpenAiCodec, BedrockClaudeCodec, FoundryClaudeCodec, GeminiCodec,
    GeminiInteractionsCodec, HttpResponse, StreamAccumulator, VertexClaudeCodec, VertexGeminiCodec,
    WireCodec,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
const RAW: &str = r#"{"\ud800":{"\udc01":"value \ud802","n":1.2300e2},"literal":"�"}"#;
fn profile(protocol: &str) -> ProviderProfile {
    let mut value = json!({"provider_id":"test","profile_name":"test","base_url":"https://example.invalid/v1","protocol":protocol,"auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m"}]});
    if protocol == "azure_open_ai" {
        value["azure"] = json!({"api_version":"2024-02-01","deployment":"m"});
    }
    serde_json::from_value(value).unwrap()
}
fn request(content: Vec<ContentBlock>) -> ChatRequest {
    ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        controls: Default::default(),
        service_tier: None,
        model: "m".into(),
        native_options: vec![],
        hosted_tools: vec![],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage::assistant(content)],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: Some(64),
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}
fn tool() -> ContentBlock {
    ContentBlock::ToolUse {
        id: "t1".into(),
        name: "read".into(),
        input: parse_tool_input_json(RAW).unwrap(),
        input_json: Some(RAW.into()),
        provider_id: Some("t1".into()),
        caller: None,
        toolset_name: None,
        thought_signature: None,
    }
}
#[test]
fn all_current_function_wires_replay_exact_argument_json() {
    for (family, codec) in [
        (
            "anthropic_messages",
            Box::new(AnthropicMessagesCodec) as Box<dyn WireCodec>,
        ),
        ("gemini_generate_content", Box::new(GeminiCodec)),
        ("open_ai_chat", Box::new(OpenAiChatCodec)),
        ("open_ai_responses", Box::new(OpenAiResponsesCodec)),
        ("gemini_interactions", Box::new(GeminiInteractionsCodec)),
        ("vertex_claude", Box::new(VertexClaudeCodec)),
        ("vertex_gemini", Box::new(VertexGeminiCodec)),
        ("bedrock_claude", Box::new(BedrockClaudeCodec)),
        ("foundry_claude", Box::new(FoundryClaudeCodec)),
        ("azure_open_ai", Box::new(AzureOpenAiCodec)),
    ] {
        let req = request(vec![tool()]);
        let profile = profile(family);
        let context = CodecContext::new(&profile, "m", RequestMode::Complete)
            .with_account_scope(Some("fixture"));
        let encoded = codec
            .encode_request(EncodeRequest::new(&req), &context)
            .unwrap();
        let bytes = String::from_utf8(encoded.body.to_vec()).unwrap();
        if family.starts_with("open_ai") || family == "azure_open_ai" {
            let value: Value = serde_json::from_str(&bytes).unwrap();
            let arguments = if family == "open_ai_chat" || family == "azure_open_ai" {
                &value["messages"][0]["tool_calls"][0]["function"]["arguments"]
            } else {
                &value["input"][0]["arguments"]
            };
            assert_eq!(arguments.as_str(), Some(RAW), "{family}");
        } else {
            assert!(bytes.contains(RAW), "{family}: {bytes}");
        }
        assert!(!bytes.contains("input_json"));
        assert!(!bytes.contains("__llmClientUtf16KeyV1_"));
    }
}
#[test]
fn full_responses_retain_exact_subtrees() {
    let cases=[("anthropic_messages",Box::new(AnthropicMessagesCodec) as Box<dyn WireCodec>,format!(r#"{{"model":"m","content":[{{"type":"tool_use","id":"t1","name":"read","input":{RAW}}}],"stop_reason":"tool_use"}}"#)),("gemini_generate_content",Box::new(GeminiCodec),format!(r#"{{"candidates":[{{"content":{{"parts":[{{"functionCall":{{"id":"t1","name":"read","args":{RAW}}}}}]}}}}]}}"#)),("open_ai_chat",Box::new(OpenAiChatCodec),json!({"model":"m","choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"t1","function":{"name":"read","arguments":RAW}}]}}]}).to_string()),("open_ai_responses",Box::new(OpenAiResponsesCodec),json!({"id":"resp1","model":"m","status":"completed","output":[{"type":"function_call","id":"fc1","call_id":"t1","name":"read","arguments":RAW}]}).to_string()),("gemini_interactions",Box::new(GeminiInteractionsCodec),format!(r#"{{"id":"interaction1","model":"m","status":"requires_action","steps":[{{"type":"function_call","id":"t1","name":"read","arguments":{RAW}}}]}}"#))];
    for (family, codec, body) in cases {
        let profile = profile(family);
        let context = CodecContext::new(&profile, "m", RequestMode::Complete);
        let response = codec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: body.into(),
                },
                &context,
            )
            .unwrap();
        let block = response
            .message
            .content
            .iter()
            .find(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .unwrap();
        let ContentBlock::ToolUse {
            input, input_json, ..
        } = block
        else {
            unreachable!()
        };
        assert_eq!(input_json.as_deref(), Some(RAW), "{family}");
        assert_eq!(*input, parse_tool_input_json(RAW).unwrap());
    }
}
#[test]
fn stale_or_injected_raw_input_is_rejected_before_encoding() {
    let profile = profile("anthropic_messages");
    let context = CodecContext::new(&profile, "m", RequestMode::Complete);
    let mut block = tool();
    if let ContentBlock::ToolUse { input, .. } = &mut block {
        *input = json!({"changed":true});
    }
    assert!(AnthropicMessagesCodec
        .validate_request(&request(vec![block.clone()]), &context)
        .is_err());
    assert!(AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request(vec![block])), &context)
        .is_err());
    for raw in [r#"{},"injected":true"#, r#"{"broken":}"#, r#"{} {}"#] {
        assert!(parse_tool_input_json(raw).is_err());
    }
}
#[test]
fn request_projection_supports_body_rewrites_without_losing_input_keys() {
    let body = format!(
        r#"{{"model":"m","messages":[{{"role":"assistant","content":[{{"type":"tool_use","id":"t1","name":"read","input":{RAW}}}]}}]}}"#
    );
    let mut projection = parse_request_body_json(body.as_bytes()).unwrap();
    projection.value["max_tokens"] = json!(64);
    let encoded = serialize_for_request_with_raw_subtrees(
        &projection.value,
        &projection.string_overrides,
        &projection.raw_subtrees,
        JsonEncoding::JavaScript,
        Some(ProtocolFamily::AnthropicMessages),
        lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestKind::Main,
    )
    .unwrap();
    assert!(String::from_utf8(encoded).unwrap().contains(RAW));
    let mut bad = projection.raw_subtrees.clone();
    bad.insert("/messages/00/content/0/input".into(), RAW.into());
    assert!(serialize_for_request_with_raw_subtrees(
        &projection.value,
        &BTreeMap::new(),
        &bad,
        JsonEncoding::Serde,
        None,
        Default::default()
    )
    .is_err());
}
#[test]
fn fragmented_arguments_assemble_into_an_exact_replayable_call() {
    let mut stream = StreamAccumulator::new();
    for fragment in [
        r#"{"\ud800":{"#,
        r#""\udc01":"value \ud802","n":1.2300e2},"literal":"�"}"#,
    ] {
        stream.observe(&StreamEvent::ToolCallDelta {
            block: 0,
            id: "t1".into(),
            name: "read".into(),
            arguments_fragment: fragment.into(),
            provider_id: None,
            caller: None,
            toolset_name: None,
        });
    }
    stream.observe(&StreamEvent::BlockEnd { block: 0 });
    let block = stream.content_at(0).unwrap();
    let ContentBlock::ToolUse {
        input, input_json, ..
    } = block
    else {
        unreachable!()
    };
    assert_eq!(input_json.as_deref(), Some(RAW));
    assert_eq!(input, parse_tool_input_json(RAW).unwrap());
}
#[test]
fn surrogate_key_placeholders_never_collide_with_real_provider_keys() {
    let raw = r#"{"\ud800":1,"__llmClientUtf16KeyV1_1__":2,"\ud800":3,"�":4}"#;
    let display = parse_tool_input_json(raw).unwrap();
    let map = display.as_object().unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(map["__llmClientUtf16KeyV1_1__"], 2);
    assert_eq!(map["�"], 4);
    assert!(map
        .iter()
        .any(|(key, value)| key.starts_with("__llmClientUtf16KeyV1_") && value == &json!(3)));
}

#[test]
fn equivalent_javascript_numbers_can_change_rust_representation_without_losing_raw_json() {
    let raw = r#"{"n":1.2300e2}"#;
    let req = request(vec![ContentBlock::ToolUse {
        id: "t1".into(),
        name: "read".into(),
        input: json!({"n":123}),
        input_json: Some(raw.into()),
        provider_id: None,
        caller: None,
        toolset_name: None,
        thought_signature: None,
    }]);
    let profile = profile("anthropic_messages");
    let context = CodecContext::new(&profile, "m", RequestMode::Complete);
    let encoded = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&req), &context)
        .unwrap();
    assert!(String::from_utf8(encoded.body.to_vec())
        .unwrap()
        .contains(raw));
}

#[test]
fn mutated_large_integer_carrier_is_rejected_without_float_rounding() {
    for (raw, display) in [
        (
            r#"{"n":9007199254740993}"#,
            json!({"n":9007199254740992u64}),
        ),
        (
            r#"{"n":18446744073709551615}"#,
            json!({"n":18446744073709551614u64}),
        ),
        (
            r#"{"n":-9007199254740993}"#,
            json!({"n":-9007199254740992i64}),
        ),
    ] {
        let mut block = tool();
        if let ContentBlock::ToolUse {
            input, input_json, ..
        } = &mut block
        {
            *input = display;
            *input_json = Some(raw.into());
        }
        let req = request(vec![block]);
        let p = profile("anthropic_messages");
        let context = CodecContext::new(&p, "m", RequestMode::Complete);
        assert!(AnthropicMessagesCodec
            .validate_request(&req, &context)
            .is_err());
        assert!(AnthropicMessagesCodec
            .encode_request(EncodeRequest::new(&req), &context)
            .is_err());
    }
}

fn sse(raw: &str) -> Vec<u8> {
    format!("data: {raw}\n\n").into_bytes()
}

#[test]
fn interactions_exact_terminal_arguments_and_snapshots_are_reconciled_by_id() {
    let p = profile("gemini_interactions");
    let context = CodecContext::new(&p, "m", RequestMode::Stream);
    for (start, snapshot, terminal, expected) in [
        (r#"{"x":"\ud800"}"#, None, r#"{"x":"\ud801"}"#, false),
        (r#"{"\ud800":1}"#, None, r#"{"\ud801":1}"#, false),
        (
            r#"{"x":"\ud800"}"#,
            Some(r#"{"x":"\ud801"}"#),
            r#"{"x":"\ud801"}"#,
            true,
        ),
        (
            r#"{"x":"\ud800"}"#,
            Some(r#"{"x":"\ud801"}"#),
            r#"{"x":"\ud800"}"#,
            false,
        ),
    ] {
        let mut decoder = GeminiInteractionsCodec.stream_decoder(&context);
        let mut events = Vec::new();
        for raw in [
            r#"{"event_type":"interaction.created","interaction":{"id":"int1","model":"m"}}"#
                .into(),
            format!(
                r#"{{"event_type":"step.start","index":0,"step":{{"type":"function_call","id":"t1","name":"read","arguments":{start}}}}}"#
            ),
        ] {
            events.extend(decoder.push_bytes(&sse(&raw)));
        }
        if let Some(snapshot) = snapshot {
            events.extend(decoder.push_bytes(&sse(&format!(r#"{{"event_type":"step.delta","index":0,"delta":{{"type":"function_call","arguments":{snapshot}}}}}"#))));
        }
        events.extend(decoder.push_bytes(&sse(r#"{"event_type":"step.stop","index":0}"#)));
        events.extend(decoder.push_bytes(&sse(&format!(r#"{{"event_type":"interaction.completed","interaction":{{"id":"int1","status":"requires_action","steps":[{{"type":"function_call","id":"t1","name":"read","arguments":{terminal}}}]}}}}"#))));
        assert_eq!(
            events.iter().all(Result::is_ok),
            expected,
            "{start} / {snapshot:?} / {terminal}"
        );
        if expected {
            let mut accumulator = StreamAccumulator::new();
            for event in events.into_iter().map(Result::unwrap) {
                accumulator.observe(&event);
            }
            let ContentBlock::ToolUse { input_json, .. } = accumulator.content_at(0).unwrap()
            else {
                panic!("missing call")
            };
            assert_eq!(input_json.as_deref(), Some(terminal));
        } else {
            assert!(!events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::ToolCallDelta { .. }))));
        }
    }
}

#[test]
fn argument_escape_state_survives_provider_sse_fragment_boundaries() {
    let prefix = serde_json::to_string("{\"x\":\"\\").unwrap();
    let suffix = r#""\ud800\"}""#;
    for (family, codec) in [
        (
            "anthropic_messages",
            Box::new(AnthropicMessagesCodec) as Box<dyn WireCodec>,
        ),
        ("open_ai_chat", Box::new(OpenAiChatCodec)),
        ("open_ai_responses", Box::new(OpenAiResponsesCodec)),
        ("gemini_interactions", Box::new(GeminiInteractionsCodec)),
    ] {
        let p = profile(family);
        let context = CodecContext::new(&p, "m", RequestMode::Stream);
        let mut decoder = codec.stream_decoder(&context);
        let start: Vec<String> = match family {
            "anthropic_messages" => vec![r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"read","input":{}}}"#.into()],
            "open_ai_chat" => vec![],
            "open_ai_responses" => vec![r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"t1","name":"read","arguments":""}}"#.into()],
            _ => vec![r#"{"event_type":"interaction.created","interaction":{"id":"int1"}}"#.into(), r#"{"event_type":"step.start","index":0,"step":{"type":"function_call","id":"t1","name":"read","arguments":{}}}"#.into()],
        };
        for raw in start {
            assert!(
                decoder.push_bytes(&sse(&raw)).iter().all(Result::is_ok),
                "{family}"
            );
        }
        let frame = |fragment: &str| match family {
            "anthropic_messages" => format!(
                r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{fragment}}}}}"#
            ),
            "open_ai_chat" => format!(
                r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"t1","function":{{"name":"read","arguments":{fragment}}}}}]}}}}]}}"#
            ),
            "open_ai_responses" => format!(
                r#"{{"type":"response.function_call_arguments.delta","output_index":0,"delta":{fragment}}}"#
            ),
            _ => format!(
                r#"{{"event_type":"step.delta","index":0,"delta":{{"type":"arguments_delta","arguments":{fragment}}}}}"#
            ),
        };
        assert!(
            decoder
                .push_bytes(&sse(&frame(&prefix)))
                .iter()
                .all(Result::is_ok),
            "{family}"
        );
        assert!(
            decoder
                .push_bytes(&sse(&frame(suffix)))
                .iter()
                .any(Result::is_err),
            "{family}"
        );
    }
}

#[test]
fn full_argument_ingress_rejects_backslash_before_literal_lone_surrogate() {
    let malformed = r#""{\"x\":\"\\\ud800\"}""#;
    for (family, codec, body) in [
        (
            "open_ai_chat",
            Box::new(OpenAiChatCodec) as Box<dyn WireCodec>,
            format!(
                r#"{{"choices":[{{"message":{{"tool_calls":[{{"id":"t1","function":{{"name":"read","arguments":{malformed}}}}}]}}}}]}}"#
            ),
        ),
        (
            "open_ai_responses",
            Box::new(OpenAiResponsesCodec),
            format!(
                r#"{{"id":"r1","model":"m","status":"completed","output":[{{"type":"function_call","id":"fc1","call_id":"t1","name":"read","arguments":{malformed}}}]}}"#
            ),
        ),
    ] {
        let p = profile(family);
        let context = CodecContext::new(&p, "m", RequestMode::Complete);
        assert!(
            codec
                .decode_response(
                    &HttpResponse {
                        status: 200,
                        headers: vec![],
                        body: body.into()
                    },
                    &context
                )
                .is_err(),
            "{family}"
        );
    }
}

#[test]
fn interactions_reordered_terminal_calls_keep_their_exact_arguments() {
    let p = profile("gemini_interactions");
    let context = CodecContext::new(&p, "m", RequestMode::Stream);
    let mut decoder = GeminiInteractionsCodec.stream_decoder(&context);
    let first = r#"{"type":"function_call","id":"t1","name":"read","arguments":{"x":"\ud800"}}"#;
    let second = r#"{"type":"function_call","id":"t2","name":"read","arguments":{"x":"\ud801"}}"#;
    let frames = [
        r#"{"event_type":"interaction.created","interaction":{"id":"int1"}}"#.into(),
        format!(r#"{{"event_type":"step.start","index":0,"step":{first}}}"#),
        r#"{"event_type":"step.stop","index":0}"#.into(),
        format!(r#"{{"event_type":"step.start","index":1,"step":{second}}}"#),
        r#"{"event_type":"step.stop","index":1}"#.into(),
        format!(
            r#"{{"event_type":"interaction.completed","interaction":{{"id":"int1","status":"requires_action","steps":[{second},{first}]}}}}"#
        ),
    ];
    let mut accumulator = StreamAccumulator::new();
    for raw in frames {
        for event in decoder.push_bytes(&sse(&raw)) {
            accumulator.observe(&event.unwrap());
        }
    }
    for (index, expected_id, expected_raw) in [
        (0, "t2", r#"{"x":"\ud801"}"#),
        (1, "t1", r#"{"x":"\ud800"}"#),
    ] {
        let ContentBlock::ToolUse { id, input_json, .. } = accumulator.content_at(index).unwrap()
        else {
            panic!("missing call")
        };
        assert_eq!(id.as_str(), expected_id);
        assert_eq!(input_json.as_deref(), Some(expected_raw));
    }
}

fn exact_result(raw: &str) -> ContentBlock {
    let display = lingxi_llm_client::exact_json::parse_tool_output_json(raw).unwrap();
    ContentBlock::ToolResult {
        cache_reference: None,
        tool_use_id: "t1".into(),
        content: display.as_str().unwrap_or("display text").into(),
        blocks: display.as_array().cloned(),
        output_json: Some(raw.into()),
        is_error: None,
        toolset_name: None,
    }
}
fn request_with_result(raw: &str) -> ChatRequest {
    let mut req = request(vec![tool()]);
    req.messages.push(ConversationMessage {
        role: MessageRole::User,
        content: vec![exact_result(raw)],
        native_options: vec![],
    });
    req
}
#[test]
fn exact_tool_results_preserve_strings_and_content_keys_on_every_codec() {
    for raw in [
        r#""value \ud800""#,
        r#"[{"type":"text","text":"value \ud800","\ud801":{"n":9007199254740993}}]"#,
    ] {
        for (family, codec) in [
            (
                "anthropic_messages",
                Box::new(AnthropicMessagesCodec) as Box<dyn WireCodec>,
            ),
            ("gemini_generate_content", Box::new(GeminiCodec)),
            ("open_ai_chat", Box::new(OpenAiChatCodec)),
            ("open_ai_responses", Box::new(OpenAiResponsesCodec)),
            ("gemini_interactions", Box::new(GeminiInteractionsCodec)),
            ("vertex_claude", Box::new(VertexClaudeCodec)),
            ("vertex_gemini", Box::new(VertexGeminiCodec)),
            ("bedrock_claude", Box::new(BedrockClaudeCodec)),
            ("foundry_claude", Box::new(FoundryClaudeCodec)),
            ("azure_open_ai", Box::new(AzureOpenAiCodec)),
        ] {
            let req = request_with_result(raw);
            let p = profile(family);
            let context = CodecContext::new(&p, "m", RequestMode::Complete)
                .with_account_scope(Some("fixture"));
            let encoded = codec
                .encode_request(EncodeRequest::new(&req), &context)
                .unwrap_or_else(|error| panic!("{family}: {error}"));
            let bytes = String::from_utf8(encoded.body.to_vec()).unwrap();
            if matches!(family, "open_ai_chat" | "azure_open_ai") && raw.starts_with('[') {
                let body: Value = serde_json::from_str(&bytes).unwrap();
                let tool = body["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|message| message["role"] == "tool")
                    .unwrap();
                assert_eq!(tool["content"].as_str(), Some(raw), "{family}");
            } else if family == "gemini_interactions" && raw.starts_with('"') {
                assert!(
                    bytes.contains(r#""text":"value \ud800""#),
                    "{family}: {bytes}"
                );
            } else {
                assert!(bytes.contains(raw), "{family}: {bytes}");
            }
            assert!(!bytes.contains("output_json"), "{family}");
            assert!(
                !bytes.contains("__llmClientUtf16KeyV1_"),
                "{family}: {bytes}"
            );
            assert_eq!(
                codec
                    .encoded_body_len(EncodeRequest::new(&req), &context)
                    .unwrap(),
                encoded.body.len(),
                "{family}"
            );
        }
    }
}

#[test]
fn exact_tool_result_carriers_reject_stale_and_malformed_outputs() {
    let p = profile("anthropic_messages");
    let context = CodecContext::new(&p, "m", RequestMode::Complete);
    for raw in [
        r#""value \ud800""#,
        r#"[{"type":"text","text":"value \ud800"}]"#,
    ] {
        let mut req = request_with_result(raw);
        if let ContentBlock::ToolResult {
            content, blocks, ..
        } = &mut req.messages[1].content[0]
        {
            if let Some(blocks) = blocks {
                blocks[0]["text"] = json!("mutated");
            } else {
                *content = "mutated".into();
            }
        }
        assert!(AnthropicMessagesCodec
            .validate_request(&req, &context)
            .is_err());
        assert!(AnthropicMessagesCodec
            .encode_request(EncodeRequest::new(&req), &context)
            .is_err());
    }
    for raw in ["{}", "true", "1", r#""ok" {}"#, "[}"] {
        assert!(
            lingxi_llm_client::exact_json::parse_tool_output_json(raw).is_err(),
            "{raw}"
        );
    }
}

#[test]
fn exact_tool_schemas_are_encoded_by_name_without_private_wire_fields() {
    let raw = r#"{"type":"object","properties":{"\ud800":{"type":"string","description":"value \ud801"}},"required":["\ud800"]}"#;
    for (family, codec) in [
        (
            "anthropic_messages",
            Box::new(AnthropicMessagesCodec) as Box<dyn WireCodec>,
        ),
        ("gemini_generate_content", Box::new(GeminiCodec)),
        ("open_ai_chat", Box::new(OpenAiChatCodec)),
        ("open_ai_responses", Box::new(OpenAiResponsesCodec)),
        ("gemini_interactions", Box::new(GeminiInteractionsCodec)),
        ("vertex_claude", Box::new(VertexClaudeCodec)),
        ("vertex_gemini", Box::new(VertexGeminiCodec)),
        ("bedrock_claude", Box::new(BedrockClaudeCodec)),
        ("foundry_claude", Box::new(FoundryClaudeCodec)),
        ("azure_open_ai", Box::new(AzureOpenAiCodec)),
    ] {
        let mut req = request(vec![ContentBlock::Text {
            text: "ready".into(),
            thought_signature: None,
            citations: None,
        }]);
        let mut tool: ToolSpec = serde_json::from_value(json!({"name":"read","description":"read","input_schema":parse_tool_input_json(raw).unwrap()})).unwrap();
        tool.input_schema_json = Some(raw.into());
        req.tools.push(tool);
        let p = profile(family);
        let context =
            CodecContext::new(&p, "m", RequestMode::Complete).with_account_scope(Some("fixture"));
        let encoded = codec
            .encode_request(EncodeRequest::new(&req), &context)
            .unwrap_or_else(|error| panic!("{family}: {error}"));
        let bytes = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(bytes.contains(raw), "{family}: {bytes}");
        assert!(!bytes.contains("input_schema_json"), "{family}");
        assert!(!bytes.contains("__llmClientUtf16KeyV1_"), "{family}");
        assert_eq!(
            codec
                .encoded_body_len(EncodeRequest::new(&req), &context)
                .unwrap(),
            encoded.body.len(),
            "{family}"
        );
        req.tools[0].input_schema["required"] = json!([]);
        assert!(codec.validate_request(&req, &context).is_err(), "{family}");
        assert!(
            codec
                .encode_request(EncodeRequest::new(&req), &context)
                .is_err(),
            "{family}"
        );
    }
}

#[test]
fn anthropic_cache_reference_uses_the_same_exact_result_receipt() {
    let raw = r#"[{"type":"text","text":"value \ud800","\ud801":1}]"#;
    let mut req = request_with_result(raw);
    if let ContentBlock::ToolResult {
        cache_reference, ..
    } = &mut req.messages[1].content[0]
    {
        *cache_reference = Some("receipt1".into());
    }
    let p = profile("anthropic_messages");
    let context = CodecContext::new(&p, "m", RequestMode::Complete);
    let encoded = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&req), &context)
        .unwrap();
    let bytes = String::from_utf8(encoded.body.to_vec()).unwrap();
    assert!(bytes.contains(raw));
    assert!(bytes.contains(r#""cache_reference":"receipt1""#));
    assert!(!bytes.contains("output_json"));
}
