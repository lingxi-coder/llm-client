use lingxi_llm_client::protocol::{
    ChatRequest, ContentBlock, ConversationMessage, HostedTool, LlmError, ProtocolFamily,
    ProviderProfile, StreamEvent, ToolChoice, ToolSpec,
};
use lingxi_llm_client::providers::anthropic::types::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, BedrockClaudeCodec, CodecContext, EncodeRequest, FoundryClaudeCodec,
    GeminiCodec, HttpResponse, OpenAiChatCodec, OpenAiResponsesCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

#[path = "support/wire_api.rs"]
mod wire_api;

fn profile(protocol: ProtocolFamily) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "anthropic",
        "profile_name": "anthropic",
        "base_url": "https://api.anthropic.com",
        "protocol": protocol,
        "auth": "none",
        "models": [{"display_model": "claude-opus-5-5", "request_model": "claude-opus-5-5", "billing_model": "claude-opus-5-5"}],
        "extra": {}
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": "claude-opus-5-5",
        "messages": [{"role":"user","content":[{"type":"text","text":"Find the right tool"}]}]
    }))
    .unwrap()
}

fn encode_anthropic(req: &ChatRequest) -> Result<Value, LlmError> {
    let p = profile(ProtocolFamily::AnthropicMessages);
    let context = CodecContext::new(&p, "claude-opus-5-5", RequestMode::Complete);
    let http = AnthropicMessagesCodec.encode_request(EncodeRequest::new(req), &context)?;
    Ok(serde_json::from_slice(&http.body).unwrap())
}

fn tool_search(strategy: AnthropicToolSearchStrategy) -> HostedTool {
    lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
        AnthropicToolSearchConfig { strategy },
    )
    .into()
}

fn tool(name: &str, defer_loading: bool) -> ToolSpec {
    ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: name.into(),
        description: format!("Use {name} to find or change matching records"),
        input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        strict: false,
        defer_loading,
        native_options: Vec::new(),
    }
}

#[test]
fn regex_and_bm25_encode_versioned_server_tools_and_deferred_definitions() {
    for (strategy, kind, name) in [
        (
            AnthropicToolSearchStrategy::Regex,
            "tool_search_tool_regex_20251119",
            "tool_search_tool_regex",
        ),
        (
            AnthropicToolSearchStrategy::Bm25,
            "tool_search_tool_bm25_20251119",
            "tool_search_tool_bm25",
        ),
    ] {
        let mut req = request();
        req.hosted_tools.push(tool_search(strategy));
        req.tools.push(tool("find_events", true));
        req.tools.push(tool("list_profiles", false));

        let body = encode_anthropic(&req).unwrap();
        assert_eq!(body["tools"][2], json!({"type":kind,"name":name}));
        assert_eq!(body["tools"][0]["name"], "find_events");
        assert_eq!(body["tools"][0]["defer_loading"], true);
        assert_eq!(body["tools"][0]["input_schema"], req.tools[0].input_schema);
        assert!(body["tools"][1].get("defer_loading").is_none());
        assert_eq!(body["tool_choice"], json!({"type":"auto"}));
    }
}

#[test]
fn deferred_tools_require_search_and_tool_choice_is_preserved() {
    let mut req = request();
    req.tools.push(tool("find_events", true));
    assert!(matches!(
        encode_anthropic(&req),
        Err(LlmError::InvalidRequest { message }) if message.contains("defer_loading")
    ));

    req.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Bm25));
    req.tool_choice = ToolChoice::None;
    let body = encode_anthropic(&req).unwrap();
    assert_eq!(body["tool_choice"], json!({"type":"none"}));
    assert_eq!(body["tools"][1]["type"], "tool_search_tool_bm25_20251119");
}

#[test]
fn deferred_catalog_and_midconversation_reference_compose_with_hosted_search() {
    let mut req = request();
    req.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Bm25));
    req.tools.push(tool("find_calendar_events", true));
    req.tools.push(tool("get_current_profile", false));
    req.messages.push(ConversationMessage {
        native_options: Vec::new(),
        role: lingxi_llm_client::protocol::MessageRole::System,
        content: vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"tool_addition",
                "tool":{"type":"tool_reference","name":"find_calendar_events"}
            }),
        }],
    });

    let p = profile(ProtocolFamily::AnthropicMessages);
    let context = CodecContext::new(&p, "claude-opus-5-5", RequestMode::Complete);
    let http = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&req), &context)
        .unwrap();
    assert!(http.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value
                .split(',')
                .any(|beta| beta == "inline-tools-2026-09-15")
    }));
    let body: Value = serde_json::from_slice(&http.body).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 3);
    assert_eq!(body["tools"][0]["name"], "find_calendar_events");
    assert_eq!(body["tools"][0]["defer_loading"], true);
    assert_eq!(body["tools"][0]["input_schema"], req.tools[0].input_schema);
    assert_eq!(body["tools"][1]["name"], "get_current_profile");
    assert!(body["tools"][1].get("defer_loading").is_none());
    assert_eq!(body["tools"][2]["type"], "tool_search_tool_bm25_20251119");
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({
            "type":"tool_addition",
            "tool":{"type":"tool_reference","name":"find_calendar_events"}
        })
    );
}

#[test]
fn search_configuration_is_rejected_by_unsupported_codecs_before_http() {
    let mut req = request();
    req.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Regex));
    let openai_profile = profile(ProtocolFamily::OpenAiResponses);
    let openai_context =
        CodecContext::new(&openai_profile, "claude-opus-5-5", RequestMode::Complete);
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(EncodeRequest::new(&req), &openai_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let chat_profile = profile(ProtocolFamily::OpenAiChat);
    let chat_context = CodecContext::new(&chat_profile, "claude-opus-5-5", RequestMode::Complete);
    assert!(matches!(
        OpenAiChatCodec.encode_request(EncodeRequest::new(&req), &chat_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let gemini_profile = profile(ProtocolFamily::GeminiGenerateContent);
    let gemini_context =
        CodecContext::new(&gemini_profile, "claude-opus-5-5", RequestMode::Complete);
    assert!(matches!(
        GeminiCodec.encode_request(EncodeRequest::new(&req), &gemini_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let hosted_profile = profile(ProtocolFamily::FoundryClaude);
    let hosted_context =
        CodecContext::new(&hosted_profile, "claude-opus-5-5", RequestMode::Complete);
    assert!(matches!(
        FoundryClaudeCodec.encode_request(EncodeRequest::new(&req), &hosted_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut compatible_vendor = profile(ProtocolFamily::AnthropicMessages);
    compatible_vendor.provider_id = "deepseek".into();
    let vendor_context =
        CodecContext::new(&compatible_vendor, "claude-opus-5-5", RequestMode::Complete);
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(EncodeRequest::new(&req), &vendor_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut deferred_only = request();
    deferred_only.tools.push(tool("find_events", true));
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(EncodeRequest::new(&deferred_only), &openai_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn bedrock_invoke_model_uses_the_same_tool_search_body() {
    let mut req = request();
    req.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Regex));
    req.tools.push(tool("find_events", true));
    let p = profile(ProtocolFamily::BedrockClaude);
    let ctx = CodecContext::new(&p, "claude-opus-5-5", RequestMode::Complete);
    let http = BedrockClaudeCodec
        .encode_request(EncodeRequest::new(&req), &ctx)
        .unwrap();
    assert!(http.url.ends_with("/model/claude-opus-5-5/invoke"));
    let body: Value = serde_json::from_slice(&http.body).unwrap();
    assert!(body.get("model").is_none());
    assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
    assert_eq!(body["tools"][1]["type"], "tool_search_tool_regex_20251119");
}

#[test]
fn tool_search_response_blocks_round_trip_unchanged() {
    let server_use = json!({
        "type":"server_tool_use",
        "id":"srvtoolu_01ABC123",
        "name":"tool_search_tool_bm25",
        "input":{"query":"calendar events about a launch"}
    });
    let search_result = json!({
        "type":"tool_search_tool_result",
        "tool_use_id":"srvtoolu_01ABC123",
        "content": {
            "type":"tool_search_tool_search_result",
            "tool_references":[{"type":"tool_reference","tool_name":"find_events"}]
        }
    });
    let body = json!({
        "model":"claude-opus-5-5",
        "stop_reason":"tool_use",
        "content":[server_use.clone(), search_result.clone()]
    });
    let p = profile(ProtocolFamily::AnthropicMessages);
    let ctx = CodecContext::new(&p, "claude-opus-5-5", RequestMode::Complete);
    let response = AnthropicMessagesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &ctx,
        )
        .unwrap();
    assert_eq!(
        response.message.content,
        vec![
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: server_use.clone(),
            },
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: search_result.clone(),
            },
        ]
    );

    let mut next = request();
    next.messages = vec![ConversationMessage::assistant(response.message.content)];
    next.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Bm25));
    let encoded = encode_anthropic(&next).unwrap();
    assert_eq!(
        encoded["messages"][0]["content"],
        json!([server_use, search_result])
    );
}

#[test]
fn stream_emits_complete_tool_search_native_blocks() {
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&wire_api::decode_context());
    let events = [
        json!({"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":5}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"tool_search_tool_regex"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"pattern\":"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"calendar\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"find_events"}]}}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}),
        json!({"type":"message_stop"}),
    ];
    let mut decoded = Vec::new();
    for event in events {
        decoded
            .extend(wire_api::decode_frame(&mut *decoder, event.to_string().as_bytes()).unwrap());
    }
    decoded.extend(wire_api::finish(&mut *decoder).unwrap());
    let native = decoded
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderContent { block, value, .. } => Some((*block, value)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(native.len(), 2);
    assert_eq!(native[0].0, 0);
    assert_eq!(
        native[0].1,
        &json!({
            "type":"server_tool_use",
            "id":"srvtoolu_1",
            "name":"tool_search_tool_regex",
            "input":{"pattern":"calendar"}
        })
    );
    assert_eq!(native[1].0, 1);
    assert_eq!(native[1].1["type"], "tool_search_tool_result");

    let mut replay = request();
    replay.messages = vec![ConversationMessage::assistant(
        native
            .iter()
            .map(|(_, value)| ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: (*value).clone(),
            })
            .collect(),
    )];
    replay
        .hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Regex));
    let replay_body = encode_anthropic(&replay).unwrap();
    assert_eq!(
        replay_body["messages"][0]["content"],
        json!([
            {
                "type":"server_tool_use",
                "id":"srvtoolu_1",
                "name":"tool_search_tool_regex",
                "input":{"pattern":"calendar"}
            },
            {
                "type":"tool_search_tool_result",
                "tool_use_id":"srvtoolu_1",
                "content": {
                    "type":"tool_search_tool_search_result",
                    "tool_references":[{"type":"tool_reference","tool_name":"find_events"}]
                }
            }
        ])
    );
    assert!(decoded.iter().any(|event| matches!(
        event,
        StreamEvent::End {
            stop_reason: lingxi_llm_client::protocol::StopReason::ToolUse,
            ..
        }
    )));
}

#[test]
fn hosted_tool_search_config_round_trips_as_request_json() {
    let mut req = request();
    req.hosted_tools
        .push(tool_search(AnthropicToolSearchStrategy::Regex));
    let value = serde_json::to_value(&req).unwrap();
    assert_eq!(
        value["hosted_tools"][0],
        json!({"type":"native","config":{"format":"anthropic.hosted_tool.v1","data":{"type":"tool_search","config":{"strategy":"regex"}}}})
    );
    let decoded: ChatRequest = serde_json::from_value(value).unwrap();
    assert_eq!(
        decoded.hosted_anthropic_tool_search().unwrap().strategy,
        AnthropicToolSearchStrategy::Regex
    );
}
