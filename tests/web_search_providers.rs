//! Verified first-party search endpoints and their distinct wire contracts.
use lingxi_agent_api::protocol::{
    AuthStrategy, BillingMode, CompletionRequest, CredentialConfig, FileSearchConfig, LlmError,
    ProviderId, ProviderProfile, StreamEvent, ToolChoice, ToolSpec,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, HttpResponse, LlmClientBuilder, OpenAiChatCodec, OpenAiResponsesCodec,
    RequestOptions, WireCodec,
};
use serde_json::{json, Value};
use std::sync::Arc;
mod support;

const ADAPTERS: &[(&str, &str)] = &[
    ("kimi", "open_ai_responses"),
    ("deepseek", "anthropic_messages"),
    ("glm", "open_ai_chat"),
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
    serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"latest news"}]}],"web_search":{}})).unwrap()
}

fn encode(req: &CompletionRequest, p: &ProviderProfile) -> Result<Value, LlmError> {
    let client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), std::slice::from_ref(p))
            .with_region(lingxi_agent_api::protocol::Region::International)
            .build()
            .unwrap();
    let route = client.resolve("m").unwrap();
    use lingxi_agent_api::protocol::ProtocolFamily::*;
    let codec: &dyn WireCodec = match p.protocol {
        OpenAiResponses => &OpenAiResponsesCodec,
        OpenAiChat => &OpenAiChatCodec,
        AnthropicMessages => &AnthropicMessagesCodec,
        _ => panic!("unhandled fixture"),
    };
    codec
        .encode_request(req, p, &route, &RequestOptions::default())
        .map(|http| serde_json::from_slice(&http.body).unwrap())
}

fn response(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![],
        body: body.to_string().into(),
    }
}

#[test]
fn deepseek_unsigned_thinking_and_search_results_replay_without_loosening_claude() {
    let content = json!([
        {"type":"thinking", "thinking":"Need current sources"},
        {"type":"server_tool_use","id":"s1","name":"web_search","input":{"query":"news"}},
        {"type":"web_search_tool_result","tool_use_id":"s1","content":[{"type":"web_search_result","url":"https://example.com","title":"News","encrypted_content":"opaque"}]}
    ]);
    let decoded = AnthropicMessagesCodec
        .decode_response(&response(json!({
            "content":content, "stop_reason":"pause_turn"
        })))
        .unwrap();
    let mut req = request();
    req.messages.push(decoded.message);
    let mut p = profile("deepseek", "anthropic_messages");
    p.extra["supports_unsigned_thinking"] = json!(true);
    assert_eq!(encode(&req, &p).unwrap()["messages"][1]["content"], content);
    assert!(matches!(
        encode(&req, &profile("anthropic", "anthropic_messages")),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn provider_tools_are_opt_in_and_coexist_with_functions() {
    for &(adapter, protocol) in ADAPTERS {
        let p = profile(adapter, protocol);
        let mut req = request();
        req.tools.push(ToolSpec {
            name: "lookup_local".into(),
            description: "Local lookup".into(),
            input_schema: json!({"type":"object","properties":{}}),
            strict: false,
        });
        let body = encode(&req, &p).unwrap();
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert!(body["tools"][0].to_string().contains("lookup_local"));
        let expected = match adapter {
            "kimi" => json!({"type":"web_search"}),
            "deepseek" => json!({"type":"web_search_20250305","name":"web_search"}),
            _ => {
                json!({"type":"web_search","web_search":{"enable":true,"search_result":true,"search_engine":"search_pro"}})
            }
        };
        assert_eq!(body["tools"][1], expected, "{adapter}");
        if adapter == "kimi" {
            assert!(body["include"]
                .as_array()
                .unwrap()
                .contains(&json!("web_search_call.action.sources")));
        }
        assert!(body.get("web_search_engine").is_none());
        req.web_search = None;
        let mut plain = p.clone();
        plain.extra = Value::Null;
        assert_eq!(encode(&req, &p).unwrap(), encode(&req, &plain).unwrap());
    }
}

#[test]
fn provider_domain_limits_and_engine_are_explicit() {
    for (adapter, protocol, count) in [
        ("kimi", "open_ai_responses", 100),
        ("glm", "open_ai_chat", 1),
    ] {
        let mut req = request();
        req.web_search.as_mut().unwrap().allowed_domains =
            (0..count).map(|n| format!("{n}.example.com")).collect();
        let mut p = profile(adapter, protocol);
        if adapter == "glm" {
            p.extra["web_search_engine"] = json!("search-prime");
        }
        let body = encode(&req, &p).unwrap();
        if adapter == "kimi" {
            assert_eq!(
                body["tools"][0]["filters"]["allowed_domains"]
                    .as_array()
                    .unwrap()
                    .len(),
                count
            );
        } else {
            assert_eq!(
                body["tools"][0]["web_search"]["search_domain_filter"],
                "0.example.com"
            );
            assert_eq!(
                body["tools"][0]["web_search"]["search_engine"],
                "search-prime"
            );
        }
        req.web_search
            .as_mut()
            .unwrap()
            .allowed_domains
            .push("overflow.example.com".into());
        let error = encode(&req, &p).unwrap_err();
        if adapter == "kimi" {
            assert!(matches!(error, LlmError::InvalidRequest { .. }));
        } else {
            assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
        }
    }
}

#[test]
fn unsupported_controls_and_protocols_fail_before_http() {
    for &(adapter, protocol) in ADAPTERS {
        for config in [
            json!({"blocked_domains":["example.com"]}),
            json!({"max_uses":2}),
        ] {
            let mut req = request();
            req.web_search = Some(serde_json::from_value(config).unwrap());
            assert!(matches!(
                encode(&req, &profile(adapter, protocol)),
                Err(LlmError::UnsupportedCapability { .. })
            ));
        }
        for choice in [ToolChoice::None, ToolChoice::Any] {
            let mut req = request();
            req.tool_choice = choice;
            assert!(matches!(
                encode(&req, &profile(adapter, protocol)),
                Err(LlmError::UnsupportedCapability { .. })
            ));
        }
        let wrong = if protocol == "open_ai_chat" {
            "open_ai_responses"
        } else {
            "open_ai_chat"
        };
        assert!(matches!(
            encode(&request(), &profile(adapter, wrong)),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    let mut req = request();
    req.web_search
        .as_mut()
        .unwrap()
        .allowed_domains
        .push("example.com".into());
    assert!(matches!(
        encode(&req, &profile("deepseek", "anthropic_messages")),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn search_presets_use_separate_verified_endpoints_and_credentials() {
    let profiles = lingxi_llm_client::presets::builtin().unwrap();
    for (name, url, adapter, env) in [
        (
            "kimi-search",
            "https://api.moonshot.cn/v1",
            "kimi",
            "MOONSHOT_API_KEY",
        ),
        (
            "deepseek-search",
            "https://api.deepseek.com/anthropic",
            "deepseek",
            "DEEPSEEK_API_KEY",
        ),
        (
            "glm",
            "https://open.bigmodel.cn/api/paas/v4",
            "glm",
            "ZHIPU_API_KEY",
        ),
    ] {
        let p = profiles.iter().find(|p| p.profile_name == name).unwrap();
        assert_eq!(p.base_url, url);
        assert_eq!(p.extra["web_search"], adapter);
        assert_eq!(p.auth, AuthStrategy::Bearer);
        assert!(matches!(&p.credential, CredentialConfig::Env { var } if var == env));
        assert_eq!(p.pricing.billing_mode, BillingMode::PerToken);
        if name == "kimi-search" {
            assert_eq!(p.models.len(), 1);
            assert_eq!(p.models[0].request_model, "kimi-k3");
        }
    }
    for name in ["kimi", "deepseek", "kimi-code"] {
        let p = profiles.iter().find(|p| p.profile_name == name).unwrap();
        assert!(p.extra.get("web_search").is_none(), "{name}");
    }
    for (name, url) in [
        (
            "qwen-search",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "qwen-search-intl",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "qwen-search-us",
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "qwen-search-hk",
            "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    ] {
        let p = profiles.iter().find(|p| p.profile_name == name).unwrap();
        assert_eq!(p.base_url, url);
        assert_eq!(p.extra["web_search"], "qwen");
        assert_eq!(p.extra["file_search"], "qwen");
    }
    let zai = profiles.iter().find(|p| p.profile_name == "zai").unwrap();
    assert_eq!(zai.extra["web_search_engine"], "search-prime");
}

#[test]
fn glm_results_survive_full_and_metadata_only_stream_frames() {
    let hits = json!([{"title":"Source","link":"https://example.com/news","content":"Relevant excerpt","media":"Example","icon":"https://example.com/icon.png","refer":"ref_1"}]);
    let full = OpenAiChatCodec.decode_response(&response(json!({"choices":[{"message":{"content":"News"},"finish_reason":"stop"}],"web_search":hits}))).unwrap();
    let search = full.web_search.unwrap();
    assert_eq!(search.citations[0].url, "https://example.com/news");
    assert_eq!(search.citations[0].title.as_deref(), Some("Source"));
    assert_eq!(search.metadata["web_search"], hits);
    for choices in [
        None,
        Some(json!([])),
        Some(json!([{"delta":{"content":"News"}}])),
    ] {
        let mut decoder = OpenAiChatCodec.stream_decoder();
        let mut frame = json!({"web_search":hits});
        if let Some(choices) = choices {
            frame["choices"] = choices;
        }
        let bytes = frame.to_string();
        let events = decoder.decode_frame(bytes.as_bytes()).unwrap();
        let search = events
            .iter()
            .find_map(|e| {
                if let StreamEvent::WebSearch { result } = e {
                    Some(result)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(search.citations.len(), 1);
        assert_eq!(search.metadata["web_search"], hits);
        assert!(!events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallDelta { .. })));
        assert!(!decoder
            .decode_frame(bytes.as_bytes())
            .unwrap()
            .iter()
            .any(|e| matches!(e, StreamEvent::WebSearch { .. })));
    }
}

#[test]
fn kimi_source_only_search_calls_yield_attribution_in_full_and_stream() {
    let call = json!({"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"news","sources":[{"type":"url","url":"https://example.com/news","title":"Source"}]}});
    let full = OpenAiResponsesCodec
        .decode_response(&response(json!({"output":[call]})))
        .unwrap();
    let search = full.web_search.unwrap();
    assert_eq!(search.citations[0].url, "https://example.com/news");
    assert_eq!(search.citations[0].title.as_deref(), Some("Source"));
    assert_eq!(search.metadata["web_search_calls"][0], call);
    for frame in [
        json!({"type":"response.output_item.done","output_index":0,"item":call}),
        json!({"type":"response.completed","response":{"output":[call]}}),
    ] {
        let mut decoder = OpenAiResponsesCodec.stream_decoder();
        let events = decoder.decode_frame(frame.to_string().as_bytes()).unwrap();
        let citations: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::WebSearch { result } = e {
                    Some(&result.citations)
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].url, "https://example.com/news");
        assert!(!events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallDelta { .. })));
    }
}

#[test]
fn qwen_search_and_workspace_file_search_share_responses_and_use_the_regional_host() {
    let mut p = profile("qwen", "open_ai_responses");
    p.provider_id = ProviderId::new("qwen");
    p.profile_name = "qwen-search-intl".into();
    p.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".into();
    p.models[0].display_model = "Qwen Max".into();
    p.models[0].request_model = "qwen3.8-max".into();
    p.extra["file_search"] = json!("qwen");
    let client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), std::slice::from_ref(&p))
            .with_region(lingxi_agent_api::protocol::Region::International)
            .build()
            .unwrap();
    let route = client.resolve("qwen3.8-max").unwrap();
    let mut req = request();
    req.model = "qwen3.8-max".into();
    req.file_search = Some(FileSearchConfig {
        knowledge_base_id: "kb-123".into(),
        workspace_id: "ws-demo-1".into(),
    });
    let http = OpenAiResponsesCodec
        .encode_request(&req, &p, &route, &RequestOptions::default())
        .unwrap();
    assert_eq!(
        http.url,
        "https://ws-demo-1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/responses"
    );
    let body: Value = serde_json::from_slice(&http.body).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    assert_eq!(body["tools"][0]["type"], "web_search");
    assert_eq!(
        body["tools"][1],
        json!({"type":"file_search","vector_store_ids":["kb-123"]})
    );
    for (base_url, region_domain) in [
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "cn-beijing.maas.aliyuncs.com",
        ),
        (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "ap-southeast-1.maas.aliyuncs.com",
        ),
        (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            "us-east-1.maas.aliyuncs.com",
        ),
        (
            "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
            "cn-hongkong.maas.aliyuncs.com",
        ),
    ] {
        let mut regional_profile = p.clone();
        regional_profile.base_url = base_url.into();
        let regional_request = OpenAiResponsesCodec
            .encode_request(&req, &regional_profile, &route, &RequestOptions::default())
            .unwrap();
        assert!(regional_request.url.starts_with(&format!(
            "https://ws-demo-1.{region_domain}/compatible-mode/v1/responses"
        )));
    }

    let search_call = json!({
        "type":"file_search_call",
        "queries":["service terms"],
        "results":[{"file_id":"file-1","filename":"terms.pdf","score":0.91,"text":"Found clause"}]
    });
    let response = OpenAiResponsesCodec
        .decode_response(&response(json!({
            "id":"resp-qwen","status":"completed","model":"qwen3.8-max",
            "output":[search_call],
            "usage":{"input_tokens":11,"output_tokens":4,"total_tokens":15,"x_tools":{"web_search":{"count":1},"file_search":{"count":1}}}
        })))
        .unwrap();
    assert_eq!(
        response.file_search.as_ref().unwrap().queries,
        ["service terms"]
    );
    assert_eq!(
        response.file_search.as_ref().unwrap().hits[0]
            .filename
            .as_deref(),
        Some("terms.pdf")
    );
    assert_eq!(
        response
            .usage
            .server_tool_usage
            .unwrap()
            .file_search_requests,
        Some(1)
    );

    let mut decoder = OpenAiResponsesCodec.stream_decoder();
    let mut streamed = decoder
        .decode_frame(
            json!({"type":"response.output_item.done","output_index":1,"item":search_call})
                .to_string()
                .as_bytes(),
        )
        .unwrap();
    streamed.extend(
        decoder
            .decode_frame(
                json!({"type":"response.completed","response":{"output":[search_call]}})
                    .to_string()
                    .as_bytes(),
            )
            .unwrap(),
    );
    assert_eq!(
        streamed
            .iter()
            .filter(|event| matches!(event, StreamEvent::FileSearch { .. }))
            .count(),
        1
    );
}

#[test]
fn minimax_server_search_uses_anthropic_tool_without_unsupported_controls() {
    let p = profile("minimax", "anthropic_messages");
    let mut req = request();
    let body = encode(&req, &p).unwrap();
    assert_eq!(
        body["tools"][0],
        json!({"type":"web_search_20250305","name":"web_search"})
    );
    assert_eq!(body["tool_choice"], json!({"type":"auto"}));

    req.web_search.as_mut().unwrap().allowed_domains = vec!["example.com".into()];
    assert!(matches!(
        encode(&req, &p),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    req.web_search = Some(Default::default());
    req.tool_choice = ToolChoice::Any;
    assert!(matches!(
        encode(&req, &p),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let decoded = AnthropicMessagesCodec
        .decode_response(&response(json!({
            "content":[
                {"type":"server_tool_use","id":"s1","name":"web_search","input":{"query":"news"}},
                {"type":"web_search_tool_result","tool_use_id":"s1","content":[{"type":"web_search_result","url":"https://example.com/news","title":"News"}]}
            ],
            "usage":{"input_tokens":10,"output_tokens":4,"server_tool_use":{"web_search_requests":1}}
        })))
        .unwrap();
    assert!(decoded.web_search.is_some());
    assert_eq!(
        decoded.usage.server_tool_usage.unwrap().web_search_requests,
        Some(1)
    );
}
