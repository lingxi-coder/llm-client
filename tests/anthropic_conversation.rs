use lingxi_llm_client::protocol::{
    CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, ContentBlock, ConversationMessage,
    MessageRole, ProtocolFamily, ProviderProfile,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, RequestMode, WireCodec,
};
use serde_json::{json, Value};

const MODEL: &str = "claude-opus-5-5";
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic", "profile_name":"anthropic",
        "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages",
        "auth":"none", "regions":["international"],
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap()
}
fn native(value: Value) -> ContentBlock {
    ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value,
    }
}
fn system(content: Vec<ContentBlock>) -> ConversationMessage {
    ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::System,
        content,
    }
}
fn text(value: &str) -> ContentBlock {
    ContentBlock::Text {
        text: value.into(),
        thought_signature: None,
    }
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]})).unwrap()
}
fn encode(
    req: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
) -> Result<(Value, Vec<(String, String)>), lingxi_llm_client::protocol::LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(req),
        &CodecContext::new(p, model, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}
fn definition(name: &str) -> Value {
    json!({"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":name,"description":"A tool","input_schema":{"type":"object","properties":{}}}}})
}
fn reference(kind: &str, name: &str) -> Value {
    json!({"type":kind,"tool":{"type":"tool_reference","name":name}})
}
#[test]
fn plain_and_consecutive_system_messages_preserve_role_and_prefix() {
    let mut req = request();
    let (before, _) = encode(&req, &profile(), MODEL).unwrap();
    req.messages
        .push(system(vec![text("Use concise language")]));
    req.messages.push(system(vec![text("Include units")]));
    let (after, headers) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(before["messages"][0], after["messages"][0]);
    assert_eq!(after["messages"][1]["role"], "system");
    assert_eq!(after["messages"][2]["content"][0]["text"], "Include units");
    assert!(!headers.iter().any(|(_, v)| v.contains("inline-tools-")));
    req.messages
        .push(ConversationMessage::assistant(vec![text("Understood")]));
    assert!(encode(&req, &profile(), MODEL).is_ok());
}
#[test]
fn unsupported_model_gateway_and_invalid_placement_are_rejected() {
    let mut req = request();
    req.messages.push(system(vec![text("Reminder")]));
    for model in ["claude-sonnet-5", "claude-opus-4-6", "claude-opus-5-50"] {
        assert!(encode(&req, &profile(), model).is_err(), "{model}");
    }
    let mut gateway = profile();
    gateway.provider_id = "gateway".into();
    gateway.base_url = "https://gateway.example.test".into();
    assert!(encode(&req, &gateway, MODEL).is_err());
    req.messages
        .push(ConversationMessage::user_text("Another user"));
    assert!(encode(&req, &profile(), MODEL).is_err());
    req.messages = vec![system(vec![text("Initial system")])];
    assert!(encode(&req, &profile(), MODEL).is_err());
    req.messages = vec![
        ConversationMessage::assistant(vec![text("Answer")]),
        system(vec![text("Reminder")]),
    ];
    assert!(encode(&req, &profile(), MODEL).is_err());
}
#[test]
fn inline_definition_update_removal_and_reference_keep_wire_content() {
    let mut req = request();
    let blocks = vec![
        definition("lookup"),
        reference("tool_removal", "lookup"),
        reference("tool_addition", "lookup"),
    ];
    req.messages
        .push(system(blocks.iter().cloned().map(native).collect()));
    let (body, headers) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"], json!(blocks));
    assert!(headers
        .iter()
        .any(|(n, v)| n.eq_ignore_ascii_case("anthropic-beta")
            && v.split(',').any(|b| b == "inline-tools-2026-09-15")));
}
#[test]
fn malformed_or_unresolved_changes_and_changes_outside_system_fail() {
    for block in [
        reference("tool_addition", "missing"),
        reference("tool_removal", "missing"),
        json!({"type":"tool_removal","tool":{"type":"tool_definition","definition":{"name":"x","input_schema":{"type":"object"}}}}),
        json!({"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"x","input_schema":false}}}),
    ] {
        let mut req = request();
        req.messages.push(system(vec![native(block)]));
        assert!(encode(&req, &profile(), MODEL).is_err());
    }
    for role in [MessageRole::User, MessageRole::Assistant] {
        let mut req = request();
        req.messages.push(ConversationMessage {
            native_options: Vec::new(),
            role,
            content: vec![native(definition("x"))],
        });
        assert!(encode(&req, &profile(), MODEL).is_err());
    }
}
#[test]
fn paused_server_result_allows_text_but_not_tool_changes() {
    let mut req = request();
    req.messages
        .push(ConversationMessage::assistant(vec![native(
            json!({"type":"web_search_tool_result","tool_use_id":"srv_1","content":[]}),
        )]));
    req.messages.push(system(vec![text("Continue concisely")]));
    assert!(encode(&req, &profile(), MODEL).is_ok());
    req.messages
        .last_mut()
        .unwrap()
        .content
        .push(native(definition("lookup")));
    assert!(encode(&req, &profile(), MODEL).is_err());
}
#[test]
fn nested_inline_cache_controls_count_in_message_order() {
    let mut req = request();
    let mut a = definition("a");
    a["tool"]["definition"]["cache_control"] = json!({"type":"ephemeral","ttl":"1h"});
    let mut b = definition("b");
    b["cache_control"] = json!({"type":"ephemeral"});
    req.messages
        .push(system(vec![native(a.clone()), native(b.clone())]));
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"][0], a);
    req.messages[1].content.reverse();
    assert!(encode(&req, &profile(), MODEL).is_err());
    req.messages[1].content = vec![native(a.clone())];
    req.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::OneHour,
    });
    assert!(encode(&req, &profile(), MODEL).is_err());
    req.prompt_cache.breakpoints.clear();
    a["cache_control"] = json!({"type":"ephemeral","ttl":"1h"});
    req.messages[1].content = vec![native(a)];
    assert!(encode(&req, &profile(), MODEL).is_err());
    let mut deferred = definition("deferred");
    deferred["tool"]["definition"]["defer_loading"] = json!(true);
    req.messages[1].content = vec![native(deferred)];
    req.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(encode(&req, &profile(), MODEL).is_err());
}
#[test]
fn five_nested_definition_breakpoints_are_rejected() {
    let mut req = request();
    req.messages.push(system(
        (0..5)
            .map(|i| {
                let mut d = definition(&format!("tool_{i}"));
                d["tool"]["definition"]["cache_control"] = json!({"type":"ephemeral"});
                native(d)
            })
            .collect(),
    ));
    assert!(encode(&req, &profile(), MODEL).is_err());
}

#[test]
fn documented_wire_models_accept_system_messages_without_a_beta() {
    let mut req = request();
    req.messages.push(system(vec![text("Reminder")]));
    for model in [
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5-5",
        "claude-opus-4-8",
        "claude-opus-5",
    ] {
        let (_, headers) = encode(&req, &profile(), model).unwrap();
        assert!(!headers.iter().any(|(_, v)| v.contains("inline-tools-")));
    }
}

#[tokio::test]
async fn invalid_system_placement_fails_before_transport() {
    struct NoTransport;
    #[async_trait::async_trait]
    impl lingxi_llm_client::Transport for NoTransport {
        async fn send(
            &self,
            _: lingxi_llm_client::HttpRequest,
        ) -> Result<lingxi_llm_client::StreamResponse, lingxi_llm_client::protocol::LlmError>
        {
            panic!("invalid history must fail before transport")
        }
    }
    let client = lingxi_llm_client::LlmClientBuilder::with_transport(
        std::sync::Arc::new(NoTransport),
        &[profile()],
    )
    .with_region(lingxi_llm_client::protocol::Region::International)
    .build()
    .unwrap();
    let mut req = request();
    req.messages
        .insert(0, system(vec![text("Invalid first message")]));
    let result = client.chat().complete(&req, &Default::default()).await;
    assert!(matches!(
        result,
        Err(lingxi_llm_client::protocol::LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn deferred_declared_tools_can_be_added_without_tool_search() {
    let mut req = request();
    req.tools.push(serde_json::from_value(json!({"name":"lookup","description":"Lookup","input_schema":{"type":"object"},"defer_loading":true})).unwrap());
    req.messages
        .push(system(vec![native(reference("tool_addition", "lookup"))]));
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["tools"][0]["defer_loading"], true);
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
}

#[test]
fn custom_definition_replacement_is_valid_but_limits_apply_after_each_message() {
    let mut req = request();
    let mut updated = definition("lookup");
    updated["tool"]["definition"]["description"] = json!("New schema");
    req.messages.push(system(vec![
        native(definition("lookup")),
        native(updated.clone()),
    ]));
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"][1], updated);
    let mut large = definition("large");
    large["tool"]["definition"]["description"] = json!("x".repeat(4_194_304));
    req.messages[1].content = vec![native(large)];
    assert!(encode(&req, &profile(), MODEL).is_err());
}

#[test]
fn inline_mcp_keeps_connection_top_level_and_toolset_only_in_history() {
    use lingxi_llm_client::providers::anthropic::types::{
        AnthropicMcpCacheControl, AnthropicMcpCacheTtl, AnthropicMcpConfig,
    };
    let config = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_inline_toolset(true)
        .with_cache_control(AnthropicMcpCacheControl {
            ttl: Some(AnthropicMcpCacheTtl::OneHour),
        });
    let mut req = request();
    req.messages
        .push(config.inline_tool_addition_message().unwrap());
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    let (body, headers) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["mcp_servers"][0]["name"], "calendar");
    assert!(body
        .get("tools")
        .is_none_or(|v| v.as_array().unwrap().is_empty()));
    assert_eq!(
        body["messages"][1]["content"][0]["tool"]["definition"]["mcp_server_name"],
        "calendar"
    );
    let beta = headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("anthropic-beta"))
        .unwrap()
        .1
        .split(',')
        .collect::<Vec<_>>();
    assert!(beta.contains(&"inline-tools-2026-09-15"));
    assert!(beta.contains(&"mcp-client-2026-09-15"));
    // One inline MCP marker plus three ordinary markers is exactly the limit.
    req.messages
        .push(ConversationMessage::assistant(vec![text("Done")]));
    req.messages.push(ConversationMessage::user_text("Next"));
    for index in [0, 2, 3] {
        req.prompt_cache.breakpoints.push(CacheBreakpoint {
            scope: None,
            position: CachePosition::Message { index, block: 0 },
            ttl: if index == 0 {
                CacheTtl::OneHour
            } else {
                CacheTtl::FiveMinutes
            },
        });
    }
    assert!(encode(&req, &profile(), MODEL).is_ok());
    // A missing inline declaration must not silently fall back to top-level tools.
    req.messages.remove(1);
    req.prompt_cache.breakpoints.clear();
    assert!(encode(&req, &profile(), MODEL).is_err());
}

#[test]
fn typed_change_helpers_emit_documented_reference_shapes() {
    use lingxi_llm_client::providers::anthropic::types::{
        AnthropicToolChange, AnthropicToolReference,
    };
    let tool = serde_json::from_value(
        json!({"name":"lookup","description":"Lookup","input_schema":{"type":"object"}}),
    )
    .unwrap();
    let mut req = request();
    req.messages
        .push(AnthropicToolChange::define_tool(tool).into_system_message());
    req.messages.push(
        AnthropicToolChange::remove(AnthropicToolReference::tool("lookup")).into_system_message(),
    );
    req.messages.push(
        AnthropicToolChange::add_reference(AnthropicToolReference::tool("lookup"))
            .into_system_message(),
    );
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(
        body["messages"][2]["content"][0],
        reference("tool_removal", "lookup")
    );
    assert_eq!(
        body["messages"][3]["content"][0],
        reference("tool_addition", "lookup")
    );
}

#[test]
fn inline_definition_policies_are_not_silently_bypassed() {
    for (field, value) in [
        ("strict", json!(true)),
        ("allowed_callers", json!(["code_execution_20260521"])),
        ("allowed_callers", json!("direct")),
        ("defer_loading", json!("yes")),
    ] {
        let mut block = definition("lookup");
        block["tool"]["definition"][field] = value;
        let mut req = request();
        req.messages.push(system(vec![native(block)]));
        assert!(encode(&req, &profile(), MODEL).is_err(), "{field}");
    }
}

#[test]
fn deferred_mcp_toolset_can_be_surfaced_by_reference_without_search() {
    use lingxi_llm_client::providers::anthropic::types::{
        AnthropicMcpConfig, AnthropicMcpToolConfig, AnthropicToolChange, AnthropicToolReference,
    };
    let config = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_default_config(AnthropicMcpToolConfig {
            enabled: Some(true),
            defer_loading: Some(true),
        });
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    req.messages.push(
        AnthropicToolChange::add_reference(AnthropicToolReference::mcp_toolset("calendar"))
            .into_system_message(),
    );
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["tools"][0]["default_config"]["defer_loading"], true);
}

#[test]
fn inline_only_tools_preserve_explicit_tool_choice() {
    let mut req = request();
    req.messages
        .push(system(vec![native(definition("lookup"))]));
    req.tool_choice = lingxi_llm_client::protocol::ToolChoice::None;
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["tool_choice"]["type"], "none");
}

#[test]
fn native_text_and_repeated_identical_inline_mcp_are_preserved() {
    use lingxi_llm_client::providers::anthropic::types::AnthropicMcpConfig;
    let config = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_inline_toolset(true);
    let mut req = request();
    req.messages.push(system(vec![
        native(json!({"type":"text","text":"Reminder"})),
        config.inline_tool_addition().unwrap(),
        config.inline_tool_addition().unwrap(),
    ]));
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    let (body, _) = encode(&req, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"].as_array().unwrap().len(), 3);
}

#[test]
fn inline_mcp_references_require_prior_definition() {
    use lingxi_llm_client::providers::anthropic::types::{
        AnthropicMcpConfig, AnthropicToolChange, AnthropicToolReference,
    };
    let config = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_inline_toolset(true);
    let mut req = request();
    req.messages.push(
        AnthropicToolChange::add_reference(AnthropicToolReference::mcp_toolset("calendar"))
            .into_system_message(),
    );
    req.messages
        .push(config.inline_tool_addition_message().unwrap());
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    assert!(encode(&req, &profile(), MODEL).is_err());
}

#[test]
fn pinned_mcp_limits_use_effective_members_after_each_message() {
    use lingxi_llm_client::providers::anthropic::types::{
        AnthropicMcpConfig, AnthropicMcpTool, AnthropicToolChange, AnthropicToolReference,
    };
    let config = AnthropicMcpConfig::new("many", "https://mcp.example.test/many")
        .unwrap()
        .with_inline_toolset(true)
        .with_tools((0..10_001).map(|i| AnthropicMcpTool {
            name: format!("tool_{i}"),
            description: None,
            input_schema: json!({"type":"object"}),
        }))
        .unwrap();
    let mut req = request();
    req.messages.push(system(vec![
        config.inline_tool_addition().unwrap(),
        AnthropicToolChange::remove(AnthropicToolReference::mcp_tool("many", "tool_0"))
            .into_content_block(),
    ]));
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    assert!(encode(&req, &profile(), MODEL).is_ok());
    let definition = req.messages[1].content[0].clone();
    req.messages.push(system(vec![definition]));
    assert!(encode(&req, &profile(), MODEL).is_err());
    req.messages.pop();
    req.messages[1].content.pop();
    assert!(encode(&req, &profile(), MODEL).is_err());
}

#[test]
fn inline_and_mcp_betas_preserve_other_tokens_without_reviving_old_mcp() {
    use lingxi_llm_client::providers::anthropic::types::AnthropicMcpConfig;
    let config = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_inline_toolset(true);
    let mut req = request();
    req.messages
        .push(config.inline_tool_addition_message().unwrap());
    req.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    let mut p = profile();
    p.extra =
        json!({"headers":{"anthropic-beta":"token-counting-2024-11-01,mcp-client-2025-11-20"}});
    let (_, headers) = encode(&req, &p, MODEL).unwrap();
    let beta = &headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("anthropic-beta"))
        .unwrap()
        .1;
    assert!(beta.contains("token-counting-2024-11-01"));
    assert!(!beta.contains("mcp-client-2025-11-20"));
    assert_eq!(beta.matches("inline-tools-2026-09-15").count(), 1);
    assert_eq!(beta.matches("mcp-client-2026-09-15").count(), 1);
}

#[test]
fn definition_byte_limit_counts_retained_definitions_not_history_copies() {
    let mut large = definition("lookup");
    large["tool"]["definition"]["description"] = json!("x".repeat(2_100_000));
    let mut req = request();
    req.messages
        .push(system(vec![native(large.clone()), native(large)]));
    assert!(encode(&req, &profile(), MODEL).is_ok());
    let mut oversized = definition("lookup");
    oversized["tool"]["definition"]["description"] = json!("x".repeat(4_194_304));
    req.messages[1].content = vec![
        native(oversized),
        native(reference("tool_removal", "lookup")),
    ];
    assert!(encode(&req, &profile(), MODEL).is_ok());
}
