use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
fn req() -> ChatRequest {
    serde_json::from_value(json!({"model":"test","system":[{"text":"system"}],"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"tools":[{"name":"f","description":"f","input_schema":{"type":"object"}}]})).unwrap()
}
fn encode(request: &ChatRequest, provider: &str, codec: &dyn WireCodec) -> Result<Value, LlmError> {
    let p: ProviderProfile = serde_json::from_value(json!({"provider_id":provider,"profile_name":"test","base_url":"https://test.invalid","protocol":codec.family(),"auth":"none"})).unwrap();
    encode_profile(request, &p, codec)
}

fn encode_profile(
    request: &ChatRequest,
    profile: &ProviderProfile,
    codec: &dyn WireCodec,
) -> Result<Value, LlmError> {
    codec
        .encode_request(
            EncodeRequest::new(request),
            &CodecContext::new(profile, "test", RequestMode::Complete),
        )
        .map(|r| serde_json::from_slice(&r.body).unwrap())
}

fn first_party_profile(extra: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic",
        "profile_name":"test",
        "base_url":"https://api.anthropic.com",
        "protocol":"anthropic_messages",
        "auth":"none",
        "extra":extra
    }))
    .unwrap()
}

fn anthropic_mcp(ttl: AnthropicMcpCacheTtl) -> HostedTool {
    lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
        AnthropicMcpConfig::new("server", "https://mcp.example.test/sse")
            .unwrap()
            .with_cache_control(AnthropicMcpCacheControl { ttl: Some(ttl) }),
    )
    .into()
}
#[test]
fn tool_system_message_breakpoints_and_automatic_cache_encode_without_losing_content() {
    let mut r = req();
    r.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![
            CacheBreakpoint {
                scope: None,
                position: CachePosition::Tool { index: 0 },
                ttl: CacheTtl::OneHour,
            },
            CacheBreakpoint {
                scope: None,
                position: CachePosition::System { index: 0 },
                ttl: CacheTtl::FiveMinutes,
            },
            CacheBreakpoint {
                scope: None,
                position: CachePosition::Message { index: 0, block: 0 },
                ttl: CacheTtl::FiveMinutes,
            },
        ],
        ..PromptCachePolicy::default()
    };
    let body = encode(&r, "anthropic", &AnthropicMessagesCodec).unwrap();
    assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(body["cache_control"]["type"], "ephemeral");
}
#[test]
fn invalid_locations_ttl_order_and_unsupported_protocol_are_rejected() {
    let mut r = req();
    r.prompt_cache.breakpoints = vec![CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 4, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    }];
    assert!(encode(&r, "anthropic", &AnthropicMessagesCodec).is_err());
    r.prompt_cache.breakpoints = vec![
        CacheBreakpoint {
            scope: None,
            position: CachePosition::System { index: 0 },
            ttl: CacheTtl::FiveMinutes,
        },
        CacheBreakpoint {
            scope: None,
            position: CachePosition::Message { index: 0, block: 0 },
            ttl: CacheTtl::OneHour,
        },
    ];
    assert!(encode(&r, "anthropic", &AnthropicMessagesCodec).is_err());
    r.prompt_cache.breakpoints.truncate(1);
    assert!(encode(&r, "openai", &OpenAiChatCodec).is_err());
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    assert!(encode(&r, "minimax", &AnthropicMessagesCodec).is_err());
}

#[test]
fn anthropic_mcp_breakpoint_composes_with_policy_in_wire_prefix_order() {
    let mut request = req();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::OneHour));
    request.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![CacheBreakpoint {
            scope: None,
            position: CachePosition::System { index: 0 },
            ttl: CacheTtl::FiveMinutes,
        }],
        ..PromptCachePolicy::default()
    };

    let body = encode_profile(
        &request,
        &first_party_profile(Value::Null),
        &AnthropicMessagesCodec,
    )
    .unwrap();
    assert_eq!(body["tools"][1]["type"], "mcp_toolset");
    assert_eq!(body["tools"][1]["cache_control"]["ttl"], "1h");
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["cache_control"]["type"], "ephemeral");
}

#[test]
fn anthropic_mcp_marker_counts_toward_four_and_orders_before_messages() {
    let mut request = req();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::OneHour));
    request.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![
            CacheBreakpoint {
                scope: None,
                position: CachePosition::Tool { index: 0 },
                ttl: CacheTtl::OneHour,
            },
            CacheBreakpoint {
                scope: None,
                position: CachePosition::System { index: 0 },
                ttl: CacheTtl::FiveMinutes,
            },
            CacheBreakpoint {
                scope: None,
                position: CachePosition::Message { index: 0, block: 0 },
                ttl: CacheTtl::FiveMinutes,
            },
        ],
        ..PromptCachePolicy::default()
    };
    let profile = first_party_profile(Value::Null);
    assert!(matches!(
        encode_profile(&request, &profile, &AnthropicMessagesCodec),
        Err(LlmError::InvalidRequest { .. })
    ));

    // A toolset 5-minute breakpoint is before messages, so a later 1-hour
    // policy breakpoint must fail preflight instead of reaching Anthropic.
    request.prompt_cache = PromptCachePolicy {
        breakpoints: vec![CacheBreakpoint {
            scope: None,
            position: CachePosition::Message { index: 0, block: 0 },
            ttl: CacheTtl::OneHour,
        }],
        ..PromptCachePolicy::default()
    };
    request.hosted_tools.clear();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::FiveMinutes));
    assert!(matches!(
        encode_profile(&request, &profile, &AnthropicMessagesCodec),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn anthropic_native_and_typed_message_cache_controls_are_deduplicated_and_preserved() {
    let mut request = req();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::OneHour));
    request.messages[0].content[0] = ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({
            "type":"mcp_tool_use",
            "id":"mcptoolu_1",
            "name":"read_doc",
            "server_name":"server",
            "input":{},
            "cache_control":{"type":"ephemeral","ttl":"5m"}
        }),
    };
    request.prompt_cache.breakpoints = vec![CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    }];

    let body = encode_profile(
        &request,
        &first_party_profile(Value::Null),
        &AnthropicMessagesCodec,
    )
    .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type":"ephemeral","ttl":"5m"})
    );

    request.prompt_cache.breakpoints[0].ttl = CacheTtl::OneHour;
    assert!(matches!(
        encode_profile(
            &request,
            &first_party_profile(Value::Null),
            &AnthropicMessagesCodec,
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn anthropic_raw_automatic_duplicate_counts_once_and_keeps_native_form() {
    let mut request = req();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::OneHour));
    request.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![
            CacheBreakpoint {
                scope: None,
                position: CachePosition::Tool { index: 0 },
                ttl: CacheTtl::OneHour,
            },
            CacheBreakpoint {
                scope: None,
                position: CachePosition::System { index: 0 },
                ttl: CacheTtl::FiveMinutes,
            },
        ],
        ..PromptCachePolicy::default()
    };
    let profile = first_party_profile(json!({
        "body":{"cache_control":{"type":"ephemeral","ttl":"5m"}}
    }));

    // The typed and profile-level controls are the same automatic breakpoint.
    // Counting them twice would incorrectly reject this four-breakpoint request.
    let body = encode_profile(&request, &profile, &AnthropicMessagesCodec).unwrap();
    assert_eq!(
        body["cache_control"],
        json!({"type":"ephemeral","ttl":"5m"})
    );
}

#[test]
fn anthropic_mcp_preflight_counts_merged_system_but_ignores_refused_raw_tools() {
    let mut request = req();
    request.system.clear();
    request
        .hosted_tools
        .push(anthropic_mcp(AnthropicMcpCacheTtl::OneHour));
    request.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![CacheBreakpoint {
            scope: None,
            position: CachePosition::Tool { index: 0 },
            ttl: CacheTtl::OneHour,
        }],
        ..PromptCachePolicy::default()
    };
    let profile = first_party_profile(json!({
        "body":{
            "tools":[
                {"name":"ignored-a","cache_control":{"type":"ephemeral","ttl":"1h"}},
                {"name":"ignored-b","cache_control":{"type":"ephemeral"}},
                {"name":"ignored-c","cache_control":{"type":"ephemeral"}}
            ],
            "system":[{
                "type":"text","text":"profile system",
                "cache_control":{"type":"ephemeral","ttl":"5m"}
            }]
        }
    }));

    // The raw tools array is refused because MCP created the typed tools array;
    // the raw system array is merged because the request has no system blocks.
    // Actual count: user tool + MCP toolset + merged system + automatic = four.
    let body = encode_profile(&request, &profile, &AnthropicMessagesCodec).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    assert_eq!(body["system"][0]["text"], "profile system");
    assert_eq!(body["system"][0]["cache_control"]["ttl"], "5m");
}
