//! Hosted search stays opt-in, explicit per endpoint, and separate from client tools.
use lingxi_agent_api::protocol::{
    CompletionRequest, CompletionResponse, LlmError, ProviderProfile, ToolChoice, ToolSpec,
    WebSearchConfig,
};
use lingxi_llm_client::codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
use lingxi_llm_client::{
    AnthropicMessagesCodec, BedrockClaudeCodec, GeminiCodec, LlmClientBuilder, RequestOptions,
    WireCodec,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
mod support;

const ADAPTERS: &[(&str, &str)] = &[
    ("openai_responses", "open_ai_responses"),
    ("openai_chat", "open_ai_chat"),
    ("anthropic", "anthropic_messages"),
    ("gemini", "gemini_generate_content"),
    ("openrouter", "open_ai_chat"),
    ("xai", "open_ai_responses"),
];
fn profile(adapter: &str, protocol: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"custom", "profile_name":"custom", "base_url":"https://custom.test/v1",
        "protocol":protocol, "auth":"none", "extra":{"web_search":adapter},
        "models":[{"display_model":"m", "request_model":"m", "billing_model":"m"}]
    }))
    .unwrap()
}
fn request() -> CompletionRequest {
    serde_json::from_value(json!({"model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"latest news"}]}], "web_search":{}})).unwrap()
}
fn encode(req: &CompletionRequest, p: &ProviderProfile) -> Result<Value, LlmError> {
    let client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), std::slice::from_ref(p))
            .build()
            .unwrap();
    let route = client.resolve("m").unwrap();
    use lingxi_agent_api::protocol::ProtocolFamily::*;
    let codec: &dyn WireCodec = match p.protocol {
        OpenAiResponses => &OpenAiResponsesCodec,
        OpenAiChat => &OpenAiChatCodec,
        AnthropicMessages => &AnthropicMessagesCodec,
        GeminiGenerateContent => &GeminiCodec,
        BedrockClaude => &BedrockClaudeCodec,
        _ => panic!("unhandled fixture"),
    };
    codec
        .encode_request(req, p, &route, &RequestOptions::default())
        .map(|http| serde_json::from_slice(&http.body).unwrap())
}
fn client_tool() -> ToolSpec {
    ToolSpec {
        name: "lookup_local".into(),
        description: "Read local data".into(),
        input_schema: json!({"type":"object","properties":{}}),
        strict: false,
    }
}

#[test]
fn each_adapter_emits_its_documented_hosted_search_shape() {
    for &(adapter, protocol) in ADAPTERS {
        let body = encode(&request(), &profile(adapter, protocol)).unwrap();
        match adapter {
            "openai_chat" => assert_eq!(body["web_search_options"], json!({})),
            "anthropic" => assert_eq!(
                body["tools"][0],
                json!({"type":"web_search_20250305","name":"web_search"})
            ),
            "gemini" => assert_eq!(body["tools"][0], json!({"googleSearch":{}})),
            "openrouter" => assert_eq!(body["tools"][0], json!({"type":"openrouter:web_search"})),
            _ => assert_eq!(body["tools"][0], json!({"type":"web_search"})),
        }
        assert!(
            body.get("web_search").is_none(),
            "adapter config must not leak to wire"
        );
    }
}

#[test]
fn declaring_adapter_does_not_enable_search_until_requested() {
    for &(adapter, protocol) in ADAPTERS {
        let mut req = request();
        req.web_search = None;
        let configured = profile(adapter, protocol);
        let mut plain = configured.clone();
        plain.extra = Value::Null;
        assert_eq!(
            encode(&req, &configured).unwrap(),
            encode(&req, &plain).unwrap()
        );
    }
}

#[test]
fn hosted_search_coexists_with_client_function_tools() {
    for &(adapter, protocol) in ADAPTERS {
        let mut req = request();
        req.tools.push(client_tool());
        let body = encode(&req, &profile(adapter, protocol)).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(
            tools.len(),
            if adapter == "openai_chat" { 1 } else { 2 },
            "{adapter}: {body}"
        );
        assert!(tools[0].to_string().contains("lookup_local"));
    }
}

#[test]
fn explicit_none_is_honored_or_rejected_never_silently_ignored() {
    for &(adapter, protocol) in ADAPTERS {
        for with_client_tool in [false, true] {
            let mut req = request();
            req.tool_choice = ToolChoice::None;
            if with_client_tool {
                req.tools.push(client_tool());
            }
            let result = encode(&req, &profile(adapter, protocol));
            if matches!(adapter, "gemini" | "openai_chat") {
                assert!(matches!(
                    result,
                    Err(LlmError::UnsupportedCapability { .. })
                ));
            } else {
                let body = result.unwrap();
                assert_eq!(
                    body["tool_choice"],
                    if adapter == "anthropic" {
                        json!({"type":"none"})
                    } else {
                        json!("none")
                    }
                );
            }
        }
    }
}

#[test]
fn undeclared_unknown_incompatible_and_bedrock_search_are_rejected() {
    let mut undeclared = profile("openai_responses", "open_ai_responses");
    undeclared.extra = json!({});
    for p in [
        undeclared,
        profile("unknown", "open_ai_chat"),
        profile("anthropic", "open_ai_chat"),
        profile("anthropic", "bedrock_claude"),
    ] {
        assert!(matches!(
            encode(&request(), &p),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[test]
fn supported_domain_filters_and_anthropic_budget_are_preserved() {
    for &(adapter, protocol) in ADAPTERS
        .iter()
        .filter(|(a, _)| matches!(*a, "openai_responses" | "anthropic" | "xai"))
    {
        let mut req = request();
        req.web_search.as_mut().unwrap().allowed_domains = vec!["example.com".into()];
        let body = encode(&req, &profile(adapter, protocol)).unwrap();
        let tool = &body["tools"][0];
        assert_eq!(
            if adapter == "anthropic" {
                &tool["allowed_domains"]
            } else {
                &tool["filters"]["allowed_domains"]
            },
            &json!(["example.com"])
        );
    }
    for (adapter, protocol, key) in [
        ("anthropic", "anthropic_messages", "blocked_domains"),
        ("xai", "open_ai_responses", "excluded_domains"),
    ] {
        let mut req = request();
        req.web_search.as_mut().unwrap().blocked_domains = vec!["example.com".into()];
        let body = encode(&req, &profile(adapter, protocol)).unwrap();
        assert_eq!(
            if adapter == "anthropic" {
                &body["tools"][0][key]
            } else {
                &body["tools"][0]["filters"][key]
            },
            &json!(["example.com"])
        );
    }
    let mut req = request();
    req.web_search.as_mut().unwrap().max_uses = Some(3);
    assert_eq!(
        encode(&req, &profile("anthropic", "anthropic_messages")).unwrap()["tools"][0]["max_uses"],
        3
    );
}

#[test]
fn unsupported_controls_fail_instead_of_disappearing() {
    for &(adapter, protocol) in ADAPTERS {
        let mut req = request();
        req.web_search.as_mut().unwrap().max_uses = Some(2);
        if adapter != "anthropic" {
            assert!(matches!(
                encode(&req, &profile(adapter, protocol)),
                Err(LlmError::UnsupportedCapability { .. })
            ));
        }
        for blocked in [false, true] {
            if matches!(adapter, "gemini" | "openai_chat")
                || (adapter == "openai_responses" && blocked)
            {
                let mut req = request();
                let search = req.web_search.as_mut().unwrap();
                if blocked {
                    search.blocked_domains.push("example.com".into());
                } else {
                    search.allowed_domains.push("example.com".into());
                }
                assert!(matches!(
                    encode(&req, &profile(adapter, protocol)),
                    Err(LlmError::UnsupportedCapability { .. })
                ));
            }
        }
    }
}

#[test]
fn malformed_filters_conflicting_filters_and_zero_budget_are_invalid() {
    for config in [
        json!({"allowed_domains":["example.com"],"blocked_domains":["other.com"]}),
        json!({"max_uses":0}),
        json!({"allowed_domains":[""]}),
        json!({"allowed_domains":["https://example.com"]}),
        json!({"blocked_domains":["example.com/path"]}),
        json!({"allowed_domains":["example .com"]}),
    ] {
        let mut req = request();
        req.web_search = Some(serde_json::from_value(config).unwrap());
        assert!(matches!(
            encode(&req, &profile("anthropic", "anthropic_messages")),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    for (adapter, limit) in [("xai", 5), ("openai_responses", 100)] {
        let mut req = request();
        req.web_search.as_mut().unwrap().allowed_domains =
            (0..=limit).map(|n| format!("{n}.example.com")).collect();
        assert!(matches!(
            encode(&req, &profile(adapter, "open_ai_responses")),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[test]
fn search_tool_names_cannot_collide_with_hosted_tools() {
    for (adapter, protocol) in [
        ("anthropic", "anthropic_messages"),
        ("openrouter", "open_ai_chat"),
    ] {
        let mut req = request();
        let mut tool = client_tool();
        tool.name = "web_search".into();
        req.tools.push(tool);
        assert!(matches!(
            encode(&req, &profile(adapter, protocol)),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[test]
fn older_request_and_response_json_remain_valid_without_search_fields() {
    let req: CompletionRequest =
        serde_json::from_value(json!({"model":"m","messages":[]})).unwrap();
    assert!(req.web_search.is_none());
    assert!(serde_json::to_value(req)
        .unwrap()
        .get("web_search")
        .is_none());
    let response: CompletionResponse = serde_json::from_value(json!({"message":{"role":"assistant","content":[]},"stop_reason":"end_turn","usage":{"input_tokens":0,"output_tokens":0,"cache_read_tokens":0,"cache_write_tokens":0},"model":"m"})).unwrap();
    assert!(response.web_search.is_none());
    assert!(serde_json::to_value(response)
        .unwrap()
        .get("web_search")
        .is_none());
    let req = request();
    assert_eq!(req.web_search, Some(WebSearchConfig::default()));
    assert_eq!(
        serde_json::from_value::<CompletionRequest>(serde_json::to_value(&req).unwrap()).unwrap(),
        req
    );
}

#[test]
fn builtin_search_adapters_are_declared_only_on_verified_endpoints() {
    let declared: std::collections::BTreeMap<_, _> = lingxi_llm_client::presets::builtin()
        .unwrap()
        .into_iter()
        .filter_map(|p| {
            p.extra
                .get("web_search")
                .and_then(Value::as_str)
                .map(|adapter| (p.profile_name.clone(), adapter.to_owned()))
        })
        .collect();
    let expected = [
        ("anthropic", "anthropic"),
        ("gemini", "gemini"),
        ("deepseek-search", "deepseek"),
        ("glm", "glm"),
        ("kimi-search", "kimi"),
        ("kimi-search-intl", "kimi"),
        ("minimax", "minimax"),
        ("minimax-intl", "minimax"),
        ("qwen-search", "qwen"),
        ("qwen-search-intl", "qwen"),
        ("qwen-search-us", "qwen"),
        ("qwen-search-hk", "qwen"),
        ("zai", "glm"),
        ("openai", "openai_responses"),
        ("openrouter", "openrouter"),
    ]
    .into_iter()
    .map(|(p, a)| (p.to_owned(), a.to_owned()))
    .collect();
    assert_eq!(declared, expected);
}

#[test]
fn openrouter_domain_filters_use_server_tool_parameters() {
    for (field, native) in [
        ("allowed_domains", "allowed_domains"),
        ("blocked_domains", "excluded_domains"),
    ] {
        let mut req = request();
        req.web_search = Some(serde_json::from_value(json!({field:["example.com"]})).unwrap());
        let body = encode(&req, &profile("openrouter", "open_ai_chat")).unwrap();
        assert_eq!(
            body["tools"][0]["parameters"][native],
            json!(["example.com"])
        );
    }
}

#[test]
fn native_hosted_search_history_replays_only_on_its_own_wire() {
    let native = json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"news"}});
    let mut req = request();
    req.web_search = None;
    req.messages = serde_json::from_value(json!([{"role":"assistant","content":[{"type":"provider_content","protocol":"anthropic_messages","value":native}]}])).unwrap();
    let body = encode(&req, &profile("anthropic", "anthropic_messages")).unwrap();
    assert_eq!(body["messages"][0]["content"][0], native);
    for (adapter, protocol) in [
        ("openai_responses", "open_ai_responses"),
        ("openai_chat", "open_ai_chat"),
        ("gemini", "gemini_generate_content"),
    ] {
        assert!(matches!(
            encode(&req, &profile(adapter, protocol)),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

struct CaptureSearchRequests(Mutex<Vec<Value>>);

#[async_trait::async_trait]
impl lingxi_llm_client::Transport for CaptureSearchRequests {
    async fn execute(
        &self,
        req: lingxi_llm_client::HttpRequest,
    ) -> Result<lingxi_llm_client::HttpResponse, LlmError> {
        self.0
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&req.body).unwrap());
        Err(LlmError::Transport {
            message: "captured".into(),
        })
    }

    async fn open_stream(
        &self,
        req: lingxi_llm_client::HttpRequest,
    ) -> Result<lingxi_llm_client::StreamResponse, LlmError> {
        self.0
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&req.body).unwrap());
        Err(LlmError::Transport {
            message: "captured".into(),
        })
    }

    async fn open_responses_websocket_session(
        &self,
        _req: lingxi_llm_client::HttpRequest,
    ) -> Result<Box<dyn lingxi_llm_client::WebSocketSession>, LlmError> {
        unreachable!()
    }
}

#[test]
fn public_web_search_methods_enable_search_without_mutating_request() {
    use futures::executor::block_on;
    let transport = Arc::new(CaptureSearchRequests(Mutex::new(Vec::new())));
    let client = LlmClientBuilder::with_transport(
        transport.clone(),
        &[profile("openai_chat", "open_ai_chat")],
    )
    .build()
    .unwrap();
    let mut req = request();
    req.web_search = Some(WebSearchConfig {
        allowed_domains: vec!["example.com".into()],
        ..WebSearchConfig::default()
    });
    let config = WebSearchConfig::default();
    assert!(matches!(
        block_on(client.web_search(&req, config.clone(), &RequestOptions::default())),
        Err(LlmError::Transport { .. })
    ));
    assert!(matches!(
        block_on(client.web_search_stream(&req, config, &RequestOptions::default())),
        Err(LlmError::Transport { .. })
    ));
    assert_eq!(
        req.web_search.as_ref().unwrap().allowed_domains,
        vec!["example.com"]
    );
    let bodies = transport.0.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    for body in bodies.iter() {
        assert_eq!(body["web_search_options"], json!({}));
    }
    assert!(bodies[0].get("stream").is_none());
    assert_eq!(bodies[1]["stream"], true);
}

#[test]
fn failover_cannot_silently_drop_requested_search_on_an_unsupported_connection() {
    use futures::executor::block_on;
    let mut first = profile("openai_responses", "open_ai_responses");
    first.connection.group = Some("search-group".into());
    first.connection.failover.network = true;
    let mut fallback = first.clone();
    fallback.profile_name = "fallback".into();
    fallback.connection.order = 1;
    fallback.extra = json!({});
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[first, fallback])
        .build()
        .unwrap();
    let route = client.resolve("m").unwrap();
    assert_eq!(route.connection_chain.len(), 1);
    let error = block_on(client.complete(&request(), &RequestOptions::default())).unwrap_err();
    assert!(
        matches!(error, LlmError::UnsupportedCapability { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("fallback"));
    let error = match block_on(client.stream(&request(), &RequestOptions::default())) {
        Ok(_) => panic!("unsupported fallback should fail"),
        Err(error) => error,
    };
    assert!(
        matches!(error, LlmError::UnsupportedCapability { .. }),
        "{error}"
    );
}
