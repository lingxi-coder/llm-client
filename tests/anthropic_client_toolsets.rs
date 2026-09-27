use async_trait::async_trait;
use lingxi_llm_client::protocol::*;
use lingxi_llm_client::*;
use serde_json::{json, Value};
use std::sync::Arc;

const MODEL: &str = "claude-opus-5-5";
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"anthropic","profile_name":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","regions":["international"],"models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]})).unwrap()
}
fn request() -> ChatRequest {
    let mut r: ChatRequest = serde_json::from_value(json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"text","text":"Inspect the page"}]}]})).unwrap();
    r.anthropic_client_toolsets = vec![
        AnthropicClientToolset::Browser(Default::default()),
        AnthropicClientToolset::Computer(Default::default()),
    ];
    r
}
fn encode(r: &ChatRequest, p: &ProviderProfile, model: &str) -> Result<Value, LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(r),
        &CodecContext::new(p, model, RequestMode::Complete),
    )?;
    Ok(serde_json::from_slice(&wire.body).unwrap())
}
fn browser(r: &mut ChatRequest) -> &mut AnthropicBrowserToolsetConfig {
    match &mut r.anthropic_client_toolsets[0] {
        AnthropicClientToolset::Browser(config) => config,
        _ => unreachable!(),
    }
}
fn tool(name: &str) -> ToolSpec {
    serde_json::from_value(json!({"name":name,"description":"A custom tool","input_schema":{"type":"object","properties":{}}})).unwrap()
}
fn calls_and_results(
    member: &str,
    result_set: Option<&str>,
    blocks: Value,
    error: bool,
) -> ChatRequest {
    let mut r = request();
    r.messages.push(serde_json::from_value(json!({"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":member,"input":{},"toolset_name":"browser"}]})).unwrap());
    r.messages.push(serde_json::from_value(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","toolset_name":result_set,"content":"result","is_error":error,"blocks":blocks}]})).unwrap());
    r
}
#[test]
fn stable_client_toolsets_encode_separately_from_hosted_tools() {
    let r = request();
    assert!(r.hosted_tools.is_empty());
    let wire = AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&r),
            &CodecContext::new(&profile(), MODEL, RequestMode::Complete),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type":"browser_toolset_20260801"},{"type":"computer_toolset_20260801"}])
    );
    assert!(!wire
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));
}
#[test]
fn sparse_configs_and_explicit_direct_caller_preserve_wire_intent() {
    let mut r = request();
    browser(&mut r).configs.insert(
        AnthropicBrowserMember::JavascriptExec,
        AnthropicClientToolConfig {
            enabled: Some(true),
            defer_loading: None,
        },
    );
    browser(&mut r).allowed_callers = Some(vec![AnthropicToolCaller::Direct]);
    let body = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(
        body["tools"][0]["configs"],
        json!({"javascript_exec":{"enabled":true}})
    );
    assert_eq!(body["tools"][0]["allowed_callers"], json!(["direct"]));
    browser(&mut r).allowed_callers = Some(vec![]);
    assert!(encode(&r, &profile(), MODEL).is_err());
    browser(&mut r).allowed_callers = Some(vec![AnthropicToolCaller::CodeExecution20260120]);
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn serde_rejects_unsupported_toolset_and_member_fields() {
    for v in [
        json!({"strict":true}),
        json!({"defer_loading":true}),
        json!({"input_examples":[{}]}),
        json!({"configs":{"missing":{}}}),
        json!({"configs":{"screenshot":{"strict":true}}}),
    ] {
        assert!(serde_json::from_value::<AnthropicBrowserToolsetConfig>(v).is_err());
    }
    assert!(serde_json::from_value::<AnthropicClientToolConfig>(
        json!({"enabled":null,"defer_loading":null})
    )
    .is_ok());
}
#[test]
fn deferred_members_require_uniform_policy_search_and_no_cache() {
    let names = [
        "close_tab",
        "double_click",
        "file_upload",
        "find",
        "form_input",
        "get_page_text",
        "hold_key",
        "hover",
        "javascript_exec",
        "key",
        "left_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "list_tabs",
        "middle_click",
        "mouse_move",
        "navigate",
        "new_tab",
        "read_console",
        "read_network",
        "read_page",
        "right_click",
        "screenshot",
        "scroll",
        "scroll_to",
        "switch_tab",
        "triple_click",
        "type",
        "wait",
        "zoom",
    ];
    let configs = names
        .into_iter()
        .map(|name| (name.to_owned(), json!({"enabled":false})))
        .collect::<serde_json::Map<_, _>>();
    let mut r = request();
    r.anthropic_client_toolsets = vec![AnthropicClientToolset::Browser(
        serde_json::from_value(json!({"configs":configs})).unwrap(),
    )];
    assert!(encode(&r, &profile(), MODEL).is_err());
    browser(&mut r).configs.insert(
        AnthropicBrowserMember::Screenshot,
        AnthropicClientToolConfig {
            enabled: Some(true),
            defer_loading: Some(true),
        },
    );
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.hosted_tools
        .push(HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Regex,
        }));
    assert!(encode(&r, &profile(), MODEL).is_ok());
    browser(&mut r).cache_control = Some(AnthropicMcpCacheControl::default());
    assert!(encode(&r, &profile(), MODEL).is_err());
    browser(&mut r).cache_control = None;
    browser(&mut r).configs.insert(
        AnthropicBrowserMember::Navigate,
        AnthropicClientToolConfig {
            enabled: Some(true),
            defer_loading: Some(false),
        },
    );
    assert!(encode(&r, &profile(), MODEL).is_err());
    browser(&mut r).configs.insert(
        AnthropicBrowserMember::Navigate,
        AnthropicClientToolConfig {
            enabled: Some(false),
            defer_loading: Some(false),
        },
    );
    assert!(encode(&r, &profile(), MODEL).is_ok());
}
#[test]
fn deferred_limit_counts_each_toolset_as_one_definition() {
    let mut r = request();
    for member in AnthropicBrowserMember::ALL {
        browser(&mut r).configs.insert(
            *member,
            AnthropicClientToolConfig {
                enabled: None,
                defer_loading: Some(true),
            },
        );
    }
    if let AnthropicClientToolset::Computer(config) = &mut r.anthropic_client_toolsets[1] {
        for member in AnthropicComputerMember::ALL {
            config.configs.insert(
                *member,
                AnthropicClientToolConfig {
                    enabled: None,
                    defer_loading: Some(true),
                },
            );
        }
    }
    r.hosted_tools
        .push(HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Regex,
        }));
    r.tools = (0..9_998)
        .map(|index| {
            let mut t = tool(&format!("lookup_{index}"));
            t.defer_loading = true;
            t
        })
        .collect();
    assert!(encode(&r, &profile(), MODEL).is_ok());
    let mut extra = tool("last_lookup");
    extra.defer_loading = true;
    r.tools.push(extra);
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn duplicate_sets_names_and_forced_member_choice_are_rejected() {
    let mut r = request();
    r.anthropic_client_toolsets
        .push(AnthropicClientToolset::Browser(Default::default()));
    assert!(encode(&r, &profile(), MODEL).is_err());
    for name in ["browser", "computer"] {
        let mut r = request();
        r.tools.push(tool(name));
        assert!(encode(&r, &profile(), MODEL).is_err());
    }
    let mut r = request();
    r.tools.push(tool("screenshot"));
    assert!(encode(&r, &profile(), MODEL).is_ok());
    for name in ["browser", "computer", "screenshot", "navigate"] {
        r.tool_choice = ToolChoice::Tool { name: name.into() };
        assert!(encode(&r, &profile(), MODEL).is_err());
    }
    r.tools.push(tool("lookup"));
    r.tool_choice = ToolChoice::Tool {
        name: "lookup".into(),
    };
    assert!(encode(&r, &profile(), MODEL).is_ok());
}
#[test]
fn inline_definitions_cannot_take_declared_toolset_names() {
    for (name, allowed) in [
        ("browser", false),
        ("computer", false),
        ("screenshot", true),
    ] {
        let mut r = request();
        r.messages
            .push(ConversationMessage::system_text("New tool"));
        r.messages.last_mut().unwrap().content.push(ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":name,"description":"Inline client tool","input_schema":{"type":"object","properties":{}}}}}),
        });
        let result = encode(&r, &profile(), MODEL);
        assert_eq!(result.is_ok(), allowed, "{name}: {result:?}");
    }
}
#[test]
fn unsupported_models_fine_grained_header_and_raw_entries_fail() {
    assert!(encode(&request(), &profile(), "claude-opus-4-6").is_err());
    let mut history = calls_and_results("screenshot", Some("browser"), Value::Null, true);
    history.anthropic_client_toolsets.clear();
    assert!(encode(&history, &profile(), MODEL).is_ok());
    assert!(encode(&history, &profile(), "claude-opus-4-6").is_err());
    let mut p = profile();
    p.extra = json!({"betas":["fine-grained-tool-streaming-2025-05-14"]});
    assert!(encode(&request(), &p, MODEL).is_err());
    let mut r = request();
    r.anthropic_client_toolsets.clear();
    p.extra = json!({"body":{"tools":[{"type":"browser_toolset_20260801"}]}});
    assert!(encode(&r, &p, MODEL).is_err());
}
#[test]
fn declaration_and_replay_metadata_are_rejected_by_other_protocols() {
    let codecs: Vec<Box<dyn WireCodec>> = vec![
        Box::new(OpenAiChatCodec),
        Box::new(OpenAiResponsesCodec),
        Box::new(GeminiCodec),
        Box::new(AzureOpenAiCodec),
        Box::new(FoundryClaudeCodec),
        Box::new(VertexGeminiCodec),
        Box::new(BedrockClaudeCodec),
    ];
    let mut history = calls_and_results(
        "screenshot",
        Some("browser"),
        json!([{"type":"text","text":"done"}]),
        false,
    );
    history.anthropic_client_toolsets.clear();
    for codec in codecs {
        let mut p = profile();
        p.provider_id = "other".into();
        p.protocol = codec.family();
        p.base_url = "https://other.example.test".into();
        let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
        for r in [&request(), &history] {
            assert!(
                codec.validate_request(r, &ctx).is_err(),
                "{:?}",
                codec.family()
            );
            assert!(
                codec.encode_request(EncodeRequest::new(r), &ctx).is_err(),
                "{:?}",
                codec.family()
            );
        }
    }
}
#[test]
fn toolset_cache_markers_follow_fetch_and_share_global_limit() {
    let mut r = request();
    r.hosted_tools
        .push(HostedTool::AnthropicWebFetch(AnthropicWebFetchConfig {
            cache_control: Some(CacheTtl::OneHour),
            ..Default::default()
        }));
    browser(&mut r).cache_control = Some(AnthropicMcpCacheControl::default());
    let body = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["tools"][0]["name"], "web_fetch");
    assert_eq!(body["tools"][1]["type"], "browser_toolset_20260801");
    if let AnthropicClientToolset::Computer(config) = &mut r.anthropic_client_toolsets[1] {
        config.cache_control = Some(AnthropicMcpCacheControl::default());
    }
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    assert!(encode(&r, &profile(), MODEL).is_ok());
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.prompt_cache.breakpoints.clear();
    if let HostedTool::AnthropicWebFetch(config) = &mut r.hosted_tools[0] {
        config.cache_control = Some(CacheTtl::FiveMinutes);
    }
    browser(&mut r).cache_control = Some(AnthropicMcpCacheControl {
        ttl: Some(AnthropicMcpCacheTtl::OneHour),
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn member_results_echo_the_namespace_and_keep_browser_state() {
    let state = json!({"type":"browser_state","tabs":[]});
    let r = calls_and_results("list_tabs", Some("browser"), json!([state]), false);
    let body = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["toolset_name"], "browser");
    assert_eq!(body["messages"][2]["content"][0]["toolset_name"], "browser");
    assert_eq!(body["messages"][2]["content"][0]["content"], json!([state]));
    for result_set in [None, Some("computer")] {
        assert!(encode(
            &calls_and_results("list_tabs", result_set, json!([state]), false),
            &profile(),
            MODEL
        )
        .is_err());
    }
}
#[test]
fn browser_tab_success_and_error_have_distinct_result_shapes() {
    assert!(encode(
        &calls_and_results(
            "list_tabs",
            Some("browser"),
            json!([{"type":"text","text":"tabs"}]),
            false
        ),
        &profile(),
        MODEL
    )
    .is_err());
    assert!(encode(
        &calls_and_results("list_tabs", Some("browser"), Value::Null, true),
        &profile(),
        MODEL
    )
    .is_ok());
    assert!(encode(
        &calls_and_results(
            "list_tabs",
            Some("browser"),
            json!([{"type":"browser_state","tabs":[]}]),
            true
        ),
        &profile(),
        MODEL
    )
    .is_err());
    assert!(encode(
        &calls_and_results(
            "screenshot",
            Some("browser"),
            json!([{"type":"text","text":"done"}]),
            false
        ),
        &profile(),
        MODEL
    )
    .is_ok());
}
#[test]
fn browser_state_inventory_and_new_tab_contracts_are_validated() {
    let null_changes = json!({"type":"browser_state","tabs":[],"state_changes":null});
    assert!(encode(
        &calls_and_results("list_tabs", Some("browser"), json!([null_changes]), false),
        &profile(),
        MODEL
    )
    .is_ok());
    let valid = json!({"type":"browser_state","tabs":[{"tab_id":"tab1","title":"Page","url":"https://example.com","active":true}],"state_changes":[{"type":"tab_opened","tab_id":"tab1"}]});
    assert!(encode(
        &calls_and_results("new_tab", Some("browser"), json!([valid]), false),
        &profile(),
        MODEL
    )
    .is_ok());
    for bad in [
        json!({"type":"browser_state","tabs":[{"tab_id":"a","title":"Page","url":"https://example.com"}]}),
        json!({"type":"browser_state","tabs":[],"state_changes":[]}),
        json!({"type":"browser_state","tabs":[{"tab_id":"a\n","title":"Page","url":"https://example.com","active":true}]}),
        json!({"type":"browser_state","tabs":[{"tab_id":"a","active":true},{"tab_id":"a"}]}),
    ] {
        assert!(encode(
            &calls_and_results("list_tabs", Some("browser"), json!([bad]), false),
            &profile(),
            MODEL
        )
        .is_err());
    }
    let no_open = json!({"type":"browser_state","tabs":[{"tab_id":"a","title":"Page","url":"https://example.com","active":true}]});
    assert!(encode(
        &calls_and_results("new_tab", Some("browser"), json!([no_open]), false),
        &profile(),
        MODEL
    )
    .is_err());
}
#[test]
fn namespace_metadata_cannot_turn_plain_or_computer_results_into_browser_state() {
    let mut r = calls_and_results(
        "screenshot",
        Some("browser"),
        json!([{"type":"browser_state","tabs":[]}]),
        false,
    );
    r.anthropic_client_toolsets.clear();
    for m in &mut r.messages {
        for b in &mut m.content {
            match b {
                ContentBlock::ToolUse { toolset_name, .. }
                | ContentBlock::ToolResult { toolset_name, .. } => *toolset_name = None,
                _ => {}
            }
        }
    }
    assert!(encode(&r, &profile(), MODEL).is_err());
    for m in &mut r.messages {
        for b in &mut m.content {
            match b {
                ContentBlock::ToolUse { toolset_name, .. }
                | ContentBlock::ToolResult { toolset_name, .. } => {
                    *toolset_name = Some("computer".into())
                }
                _ => {}
            }
        }
    }
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn member_calls_are_direct_and_download_metadata_remains_native() {
    let blocks = json!([{"type":"browser_state","tabs":[],"state_changes":[{"type":"download_completed","download_id":"d1","url":"https://example.com/file","path":null,"size_bytes":null}]}]);
    let mut r = calls_and_results("screenshot", Some("browser"), blocks.clone(), false);
    if let ContentBlock::ToolUse { caller, .. } = &mut r.messages[1].content[0] {
        *caller = Some(json!({"type":"direct"}));
    }
    let body = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][2]["content"][0]["content"], blocks);
    let fractional = json!([{"type":"browser_state","tabs":[],"state_changes":[{"type":"download_completed","download_id":"d1","url":"https://example.com/file","size_bytes":0.5}]}]);
    assert!(encode(
        &calls_and_results("screenshot", Some("browser"), fractional, false),
        &profile(),
        MODEL
    )
    .is_err());
    if let ContentBlock::ToolUse { caller, .. } = &mut r.messages[1].content[0] {
        *caller = Some(json!({"type":"code_execution_20260120","tool_id":"srv1"}));
    }
    assert!(encode(&r, &profile(), MODEL).is_err());
    let duplicate = json!([{"type":"browser_state","tabs":[],"state_changes":[{"type":"download_started","download_id":"d1","url":"https://example.com/file"},{"type":"download_completed","download_id":"d1","url":"https://example.com/file"}]}]);
    assert!(encode(
        &calls_and_results("screenshot", Some("browser"), duplicate, false),
        &profile(),
        MODEL
    )
    .is_err());
}
#[tokio::test]
async fn invalid_declaration_fails_before_attachment_read_or_transport() {
    struct Never;
    #[async_trait]
    impl Transport for Never {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            panic!("must not send invalid client toolsets")
        }
    }
    #[async_trait]
    impl AttachmentResolver for Never {
        async fn resolve(&self, _: &AttachmentRef) -> Result<bytes::Bytes, LlmError> {
            panic!("must not resolve invalid client toolsets")
        }
    }
    let mut builder = LlmClientBuilder::with_transport(Arc::new(Never), &[profile()])
        .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(Never));
    let client = builder.build().unwrap();
    let mut r = request();
    r.anthropic_client_toolsets
        .push(AnthropicClientToolset::Browser(Default::default()));
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
    r.anthropic_client_toolsets = vec![AnthropicClientToolset::Browser(Default::default())];
    for member in AnthropicBrowserMember::ALL {
        browser(&mut r).configs.insert(
            *member,
            AnthropicClientToolConfig {
                enabled: None,
                defer_loading: Some(true),
            },
        );
    }
    r.tools = (0..9_999)
        .map(|index| {
            let mut t = tool(&format!("lookup_{index}"));
            t.defer_loading = true;
            t
        })
        .collect();
    r.hosted_tools
        .push(HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Regex,
        }));
    r.hosted_tools.push(HostedTool::AnthropicMcp(
        AnthropicMcpConfig::new("remote", "https://mcp.example.test/sse")
            .unwrap()
            .with_default_config(AnthropicMcpToolConfig {
                enabled: None,
                defer_loading: Some(true),
            })
            .with_tools([AnthropicMcpTool {
                name: "remote_lookup".into(),
                description: None,
                input_schema: json!({"type":"object","properties":{}}),
            }])
            .unwrap(),
    ));
    let error = client
        .chat()
        .complete(&r, &Default::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, LlmError::UnsupportedCapability { message } if message.contains("10,000"))
    );
}
