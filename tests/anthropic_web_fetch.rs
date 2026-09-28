use async_trait::async_trait;
use lingxi_llm_client::protocol::*;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::*;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
#[path = "support/wire_api.rs"]
mod wire_api;
const MODEL: &str = "claude-opus-5-5";
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"anthropic","profile_name":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","regions":["international"],"models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]})).unwrap()
}
fn request(config: AnthropicWebFetchConfig) -> ChatRequest {
    let mut r:ChatRequest=serde_json::from_value(json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"text","text":"Read https://example.com/page"}]}]})).unwrap();
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(config)
            .into(),
    );
    r
}
fn encode(
    r: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(r),
        &CodecContext::new(p, model, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}
fn schema() -> OutputFormat {
    OutputFormat::JsonSchema {
        name: "answer".into(),
        schema: json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}),
        strict: true,
    }
}
fn native(v: Value) -> ContentBlock {
    ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: v,
    }
}
#[test]
fn latest_fetch_parameters_encode_without_an_extra_beta_or_host_executor() {
    let r = request(AnthropicWebFetchConfig {
        max_uses: Some(3),
        allowed_domains: vec!["example.com".into()],
        citations: Some(true),
        max_content_tokens: Some(4096),
        use_cache: Some(false),
        response_inclusion: Some(AnthropicFetchResponseInclusion::Excluded),
        ..Default::default()
    });
    let (body, headers) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type":"web_fetch_20260318","name":"web_fetch","max_uses":3,"allowed_domains":["example.com"],"citations":{"enabled":true},"max_content_tokens":4096,"use_cache":false,"response_inclusion":"excluded"}])
    );
    assert!(!headers
        .iter()
        .any(|(n, _)| n.eq_ignore_ascii_case("anthropic-beta")));
}
#[test]
fn four_live_versions_have_explicit_feature_gates() {
    for (version, wire) in [
        (AnthropicWebFetchVersion::V20250910, "web_fetch_20250910"),
        (AnthropicWebFetchVersion::V20260209, "web_fetch_20260209"),
        (AnthropicWebFetchVersion::V20260309, "web_fetch_20260309"),
        (AnthropicWebFetchVersion::V20260318, "web_fetch_20260318"),
    ] {
        let r = request(AnthropicWebFetchConfig {
            version,
            ..Default::default()
        });
        let (body, _) = encode(&r, &profile(), MODEL).unwrap();
        assert_eq!(body["tools"][0]["type"], wire);
        let r = request(AnthropicWebFetchConfig {
            version,
            use_cache: Some(true),
            ..Default::default()
        });
        assert_eq!(
            encode(&r, &profile(), MODEL).is_ok(),
            matches!(
                version,
                AnthropicWebFetchVersion::V20260309 | AnthropicWebFetchVersion::V20260318
            )
        );
        let r = request(AnthropicWebFetchConfig {
            version,
            response_inclusion: Some(AnthropicFetchResponseInclusion::Full),
            ..Default::default()
        });
        assert_eq!(
            encode(&r, &profile(), MODEL).is_ok(),
            version == AnthropicWebFetchVersion::V20260318
        );
    }
}
#[test]
fn wrong_routes_duplicates_name_collisions_and_domain_conflicts_fail() {
    let mut r = request(Default::default());
    let mut p = profile();
    p.base_url = "https://gateway.example.test".into();
    assert!(encode(&r, &p, MODEL).is_err());
    p = profile();
    p.extra = json!({"body":{"tools":[{"type":"web_fetch_20260318","name":"web_fetch"}]}});
    assert!(encode(&r, &p, MODEL).is_err());
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(
            Default::default(),
        )
        .into(),
    );
    assert!(encode(&r, &profile(), MODEL).is_err());
    r = request(Default::default());
    r.tools.push(
        serde_json::from_value(
            json!({"name":"web_fetch","description":"collision","input_schema":{"type":"object"}}),
        )
        .unwrap(),
    );
    assert!(encode(&r, &profile(), MODEL).is_err());
    r = request(AnthropicWebFetchConfig {
        allowed_domains: vec!["example.com".into()],
        blocked_domains: vec!["blocked.example.com".into()],
        ..Default::default()
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
    for domain in [
        "https://example.com",
        "*.example.com",
        "bad domain",
        "example.com\n",
    ] {
        r = request(AnthropicWebFetchConfig {
            allowed_domains: vec![domain.into()],
            ..Default::default()
        });
        assert!(encode(&r, &profile(), MODEL).is_err(), "{domain:?}");
    }
}
#[test]
fn dynamic_filtering_does_not_claim_support_on_older_models() {
    assert!(encode(
        &request(AnthropicWebFetchConfig {
            allowed_callers: vec![AnthropicFetchCaller::Direct],
            ..Default::default()
        }),
        &profile(),
        "claude-haiku-4-5"
    )
    .is_ok());
    assert!(encode(&request(Default::default()), &profile(), "claude-haiku-4-5").is_err());
    assert!(encode(&request(Default::default()), &profile(), "claude-opus-4-5").is_err());
    assert!(encode(
        &request(AnthropicWebFetchConfig {
            version: AnthropicWebFetchVersion::V20250910,
            ..Default::default()
        }),
        &profile(),
        "claude-haiku-4-5"
    )
    .is_ok());
}
#[test]
fn citations_conflict_with_json_schema_in_config_and_native_replay() {
    let mut r = request(AnthropicWebFetchConfig {
        citations: Some(true),
        ..Default::default()
    });
    r.output_format = schema();
    assert!(encode(&r, &profile(), MODEL).is_err());
    r = request(AnthropicWebFetchConfig {
        citations: Some(false),
        ..Default::default()
    });
    r.output_format = schema();
    assert!(encode(&r, &profile(), MODEL).is_ok());
    r.messages.insert(0,ConversationMessage::assistant(vec![native(json!({"type":"web_fetch_tool_result","tool_use_id":"srv_1","content":{"type":"web_fetch_result","url":"https://example.com","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"Page"},"citations":{"enabled":true}}}}))]));
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn fetch_can_coexist_with_search_code_execution_and_inline_references() {
    let mut r = request(Default::default());
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            Default::default(),
        )
        .into(),
    );
    r.hosted_tools
        .push(HostedTool::WebSearch(Default::default()));
    let mut p = profile();
    p.extra = json!({"web_search":"anthropic"});
    r.messages.push(
        AnthropicToolChange::remove(AnthropicToolReference::tool("web_fetch"))
            .into_system_message(),
    );
    let (body, _) = encode(&r, &p, MODEL).unwrap();
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v["name"] == "web_fetch"));
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v["name"] == "web_search"));
}
#[test]
fn fetch_results_errors_and_usage_are_native_not_client_calls() {
    let blocks = json!([{"type":"server_tool_use","id":"srv_1","name":"web_fetch","input":{"url":"https://example.com"}},{"type":"web_fetch_tool_result","tool_use_id":"srv_1","content":{"type":"web_fetch_tool_result_error","error_code":"url_not_accessible"}}]);
    let p = profile();
    let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
    let resp=AnthropicMessagesCodec.decode_response(&HttpResponse{status:200,headers:vec![],body:bytes::Bytes::from(serde_json::to_vec(&json!({"id":"msg_1","model":MODEL,"content":blocks,"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":4,"server_tool_use":{"web_fetch_requests":1}}})).unwrap())},&ctx).unwrap();
    assert_eq!(resp.message.tool_uses().count(), 0);
    assert_eq!(
        resp.anthropic_usage().unwrap()["server_tool_use"]["web_fetch_requests"],
        1
    );
    assert_eq!(
        resp.usage
            .usage
            .unwrap()
            .server_tool_usage
            .unwrap()
            .web_fetch_requests,
        Some(1)
    );
    let mut r = request(Default::default());
    r.messages.insert(0, resp.message);
    assert_eq!(
        encode(&r, &p, MODEL).unwrap().0["messages"][0]["content"],
        blocks
    );
}
#[test]
fn streamed_fetch_json_and_result_survive_with_raw_usage() {
    let p = profile();
    let ctx = CodecContext::new(&p, MODEL, RequestMode::Stream);
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&ctx);
    let mut events = vec![];
    let result = json!({"type":"web_fetch_tool_result","tool_use_id":"srv_1","content":{"type":"web_fetch_result","url":"https://example.com","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"Page"}}}});
    for frame in [
        json!({"type":"message_start","message":{"model":MODEL,"usage":{"input_tokens":3,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_fetch","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"url\":\"https://example.com\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":result}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4,"server_tool_use":{"web_fetch_requests":1}}}),
        json!({"type":"message_stop"}),
    ] {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    assert_eq!(
        wire_api::observed_usage(&decoder)
            .unwrap()
            .server_tool_usage
            .unwrap()
            .web_fetch_requests,
        Some(1)
    );
    assert_eq!(
        events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ProviderContent {
                    block: 0, value, ..
                } => Some(value),
                _ => None,
            })
            .unwrap()["input"],
        json!({"url":"https://example.com"})
    );
    assert_eq!(
        events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ProviderContent {
                    block: 1, value, ..
                } => Some(value),
                _ => None,
            })
            .unwrap(),
        &result
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(events.iter().any(|event|matches!(event,StreamEvent::ProviderEvent{payload,..} if payload.pointer("/usage/server_tool_use/web_fetch_requests")==Some(&json!(1)))));
}
#[derive(Default)]
struct FailTransport(Mutex<usize>);
#[async_trait]
impl Transport for FailTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        *self.0.lock().unwrap() += 1;
        Err(LlmError::Transport {
            message: "unknown outcome".into(),
        })
    }
}
#[tokio::test]
async fn uncertain_fetch_is_not_failed_over_or_retried() {
    let mut a = profile();
    a.profile_name = "primary".into();
    a.connection.group = Some("fetch".into());
    a.connection.order = 0;
    a.connection.failover.network = true;
    let mut b = a.clone();
    b.profile_name = "secondary".into();
    b.connection.order = 1;
    let transport = Arc::new(FailTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[a, b])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(client
        .chat()
        .complete(&request(Default::default()), &Default::default())
        .await
        .is_err());
    assert_eq!(*transport.0.lock().unwrap(), 1);
}

#[tokio::test]
async fn invalid_fetch_and_citations_fail_before_attachment_resolution() {
    struct Never;
    #[async_trait]
    impl Transport for Never {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            panic!("must not send invalid fetch")
        }
    }
    #[async_trait]
    impl AttachmentResolver for Never {
        async fn resolve(&self, _: &AttachmentRef) -> Result<bytes::Bytes, LlmError> {
            panic!("must not read attachment for invalid fetch")
        }
    }
    let mut builder = LlmClientBuilder::with_transport(Arc::new(Never), &[profile()])
        .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(Never));
    let client = builder.build().unwrap();
    for config in [
        AnthropicWebFetchConfig {
            version: AnthropicWebFetchVersion::V20250910,
            use_cache: Some(false),
            ..Default::default()
        },
        AnthropicWebFetchConfig {
            citations: Some(true),
            ..Default::default()
        },
    ] {
        let mut r = request(config);
        r.output_format = schema();
        r.messages[0].content.push(ContentBlock::Document {
            source: DocumentSource::Attachment {
                attachment: AttachmentRef {
                    attachment_id: "doc".into(),
                    revision: "1".into(),
                    filename: "doc.pdf".into(),
                    media_type: "application/pdf".into(),
                    size_bytes: 8,
                },
            },
            title: None,
        });
        assert!(client
            .chat()
            .complete(&r, &Default::default())
            .await
            .is_err());
    }
}

#[test]
fn fetch_cache_markers_follow_mcp_wire_order_and_share_the_four_slot_limit() {
    let mut r = request(AnthropicWebFetchConfig {
        cache_control: Some(CacheTtl::FiveMinutes),
        ..Default::default()
    });
    let mcp = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_cache_control(AnthropicMcpCacheControl {
            ttl: Some(AnthropicMcpCacheTtl::OneHour),
        });
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp).into(),
    );
    let (body, _) = encode(&r, &profile(), MODEL).unwrap();
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools[0]["type"], "mcp_toolset");
    assert_eq!(tools[1]["name"], "web_fetch");
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    assert!(encode(&r, &profile(), MODEL).is_ok());
    r.messages
        .push(ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "More".into(),
            thought_signature: None,
        }]));
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
    let mut bad = request(AnthropicWebFetchConfig {
        cache_control: Some(CacheTtl::OneHour),
        ..Default::default()
    });
    bad.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
                .unwrap()
                .with_cache_control(AnthropicMcpCacheControl::default()),
        )
        .into(),
    );
    assert!(encode(&bad, &profile(), MODEL).is_err());
}
#[test]
fn deferred_fetch_cannot_cache_but_can_use_tool_search() {
    let mut r = request(AnthropicWebFetchConfig {
        defer_loading: true,
        cache_control: Some(CacheTtl::FiveMinutes),
        ..Default::default()
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
    r = request(AnthropicWebFetchConfig {
        defer_loading: true,
        ..Default::default()
    });
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
            AnthropicToolSearchConfig {
                strategy: AnthropicToolSearchStrategy::Regex,
            },
        )
        .into(),
    );
    let (body, _) = encode(&r, &profile(), MODEL).unwrap();
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "web_fetch" && tool["defer_loading"] == true));
}
#[test]
fn other_codecs_reject_fetch_instead_of_omitting_it() {
    let codecs: Vec<Box<dyn WireCodec>> = vec![
        Box::new(OpenAiChatCodec),
        Box::new(OpenAiResponsesCodec),
        Box::new(GeminiCodec),
        Box::new(AzureOpenAiCodec),
        Box::new(FoundryClaudeCodec),
        Box::new(VertexClaudeCodec),
        Box::new(VertexGeminiCodec),
        Box::new(BedrockClaudeCodec),
    ];
    let r = request(Default::default());
    for codec in codecs {
        let mut p = profile();
        p.provider_id = "other".into();
        p.protocol = codec.family();
        p.base_url = "https://other.example.test".into();
        let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
        assert!(
            codec.validate_request(&r, &ctx).is_err(),
            "{:?}",
            codec.family()
        );
        assert!(
            codec.encode_request(EncodeRequest::new(&r), &ctx).is_err(),
            "{:?}",
            codec.family()
        );
    }
}

#[test]
fn source_filters_callers_and_zero_limits_remain_explicit() {
    let sources = json!({"user_input":{"type":"none"},"client_tool_results":{"type":"only","tools":[{"type":"tool_reference","name":"lookup"}]},"server_tool_results":{"type":"only","tools":[{"type":"tool_reference","name":"web_fetch"}]}});
    let mut r = request(AnthropicWebFetchConfig {
        allowed_callers: vec![
            AnthropicFetchCaller::Direct,
            AnthropicFetchCaller::CodeExecution20250825,
        ],
        max_uses: Some(0),
        max_content_tokens: Some(0),
        strict: true,
        url_sources: Some(serde_json::from_value(sources.clone()).unwrap()),
        ..Default::default()
    });
    r.tools.push(
        serde_json::from_value(
            json!({"name":"lookup","description":"Lookup","input_schema":{"type":"object"}}),
        )
        .unwrap(),
    );
    let (body, _) = encode(&r, &profile(), MODEL).unwrap();
    let fetch = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == "web_fetch")
        .unwrap();
    assert_eq!(fetch["url_sources"], sources);
    assert_eq!(fetch["max_uses"], 0);
    assert_eq!(fetch["max_content_tokens"], 0);
    assert_eq!(fetch["strict"], true);
    assert_eq!(
        fetch["allowed_callers"],
        json!(["direct", "code_execution_20250825"])
    );
    r.tools.clear();
    assert!(encode(&r, &profile(), MODEL).is_err());
    let empty = request(AnthropicWebFetchConfig {
        url_sources: Some(
            serde_json::from_value(json!({"client_tool_results":{"type":"only","tools":[]}}))
                .unwrap(),
        ),
        ..Default::default()
    });
    assert_eq!(
        encode(&empty, &profile(), MODEL).unwrap().0["tools"][0]["url_sources"]
            ["client_tool_results"]["tools"],
        json!([])
    );
}
#[test]
fn strict_fetch_participates_in_the_shared_strict_tool_limit() {
    let mut r = request(AnthropicWebFetchConfig {
        strict: true,
        ..Default::default()
    });
    r.tools=(0..20).map(|i|serde_json::from_value(json!({"name":format!("tool_{i}"),"description":"Tool","strict":true,"input_schema":{"type":"object","properties":{},"additionalProperties":false}})).unwrap()).collect();
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.tools.pop();
    assert!(encode(&r, &profile(), MODEL).is_ok());
}
#[test]
fn raw_hosted_fetch_is_rejected_even_without_a_typed_declaration() {
    let mut r = request(Default::default());
    r.hosted_tools.clear();
    let mut p = profile();
    p.extra = json!({"body":{"tools":[{"type":"web_fetch_20260318","name":"web_fetch"}]}});
    assert!(encode(&r, &p, MODEL).is_err());
    p.extra = json!({"body":{"tools":[{"name":"web_fetch","description":"Caller function","input_schema":{"type":"object"}}]}});
    assert!(encode(&r, &p, MODEL).is_ok());
}
#[test]
fn fetch_usage_keeps_zero_distinct_from_unreported() {
    let p = profile();
    let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
    for value in [Some(json!(0)), None] {
        let mut usage = json!({"input_tokens":1,"output_tokens":1});
        if let Some(value) = value.clone() {
            usage["server_tool_use"] = json!({"web_fetch_requests":value});
        }
        let response=AnthropicMessagesCodec.decode_response(&HttpResponse{status:200,headers:vec![],body:bytes::Bytes::from(serde_json::to_vec(&json!({"content":[],"model":MODEL,"stop_reason":"end_turn","usage":usage})).unwrap())},&ctx).unwrap();
        assert_eq!(
            response
                .usage
                .usage
                .unwrap()
                .server_tool_usage
                .and_then(|u| u.web_fetch_requests),
            value.map(|_| 0)
        );
    }
}

#[test]
fn source_filter_shapes_are_typed_and_do_not_accept_unknown_options() {
    for invalid in [
        json!({"user_input":{"type":"only","tools":[]}}),
        json!({"client_tool_results":{"type":"only"}}),
        json!({"server_tool_results":{"type":"only","tools":[{"type":"mcp_tool_reference","server_name":"x","name":"lookup"}]}}),
        json!({"unknown":true}),
    ] {
        assert!(serde_json::from_value::<AnthropicFetchUrlSources>(invalid).is_err());
    }
}
#[test]
fn domain_path_patterns_are_preserved_not_promoted_to_host_wildcards() {
    let r = request(AnthropicWebFetchConfig {
        allowed_domains: vec!["example.com/*/articles".into()],
        ..Default::default()
    });
    assert_eq!(
        encode(&r, &profile(), MODEL).unwrap().0["tools"][0]["allowed_domains"],
        json!(["example.com/*/articles"])
    );
}
