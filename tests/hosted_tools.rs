use lingxi_llm_client::protocol::{
    ChatRequest, HostedTool, LlmError, ProviderProfile, Region, WebSearchConfig,
};
use lingxi_llm_client::providers::openai::types::CodeInterpreterConfig;
use lingxi_llm_client::providers::qwen::types::FileSearchConfig;
use lingxi_llm_client::{LlmClientBuilder, RequestOptions};
use serde_json::json;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"m",
        "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    }))
    .unwrap()
}

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"test","profile_name":"test",
        "base_url":"https://test.invalid/v1","protocol":"open_ai_responses",
        "auth":"none","extra":{"web_search":"openai_responses","file_search":"qwen"},
        "models":[{"display_model":"m","request_model":"m","billing_model":"m"}]
    }))
    .unwrap()
}

#[tokio::test]
async fn duplicate_hosted_tools_fail_before_transport() {
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    for tools in [
        vec![
            HostedTool::WebSearch(WebSearchConfig::default()),
            HostedTool::WebSearch(WebSearchConfig::default()),
        ],
        vec![
            lingxi_llm_client::providers::qwen::native::QwenHostedTool::FileSearch(
                FileSearchConfig {
                    knowledge_base_id: "kb-a".into(),
                    workspace_id: "ws-a".into(),
                },
            )
            .into(),
            lingxi_llm_client::providers::qwen::native::QwenHostedTool::FileSearch(
                FileSearchConfig {
                    knowledge_base_id: "kb-b".into(),
                    workspace_id: "ws-b".into(),
                },
            )
            .into(),
        ],
        vec![
            lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
                CodeInterpreterConfig::default(),
            )
            .into(),
            lingxi_llm_client::providers::openai::native::OpenAiHostedTool::CodeInterpreter(
                CodeInterpreterConfig::default(),
            )
            .into(),
        ],
    ] {
        let mut req = request();
        req.hosted_tools = tools;
        assert!(matches!(
            client
                .chat()
                .complete(&req, &RequestOptions::default())
                .await,
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[test]
fn hosted_tools_are_typed_and_replacement_preserves_other_kinds() {
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::qwen::native::QwenHostedTool::FileSearch(FileSearchConfig {
            knowledge_base_id: "kb".into(),
            workspace_id: "ws".into(),
        })
        .into(),
    );
    req.set_hosted_web_search(Some(WebSearchConfig::default()));
    req.set_hosted_web_search(Some(WebSearchConfig {
        allowed_domains: vec!["example.com".into()],
        ..Default::default()
    }));
    assert_eq!(req.hosted_tools.len(), 2);
    assert_eq!(req.hosted_file_search().unwrap().workspace_id, "ws");
    assert_eq!(
        req.hosted_web_search().unwrap().allowed_domains,
        vec!["example.com"]
    );
    let serialized = serde_json::to_value(&req).unwrap();
    assert_eq!(serialized["hosted_tools"].as_array().unwrap().len(), 2);
    assert_eq!(
        serde_json::from_value::<ChatRequest>(serialized).unwrap(),
        req
    );

    assert!(serde_json::from_value::<ChatRequest>(json!({
        "model":"m", "messages":[], "web_search":{}
    }))
    .is_err());
}
