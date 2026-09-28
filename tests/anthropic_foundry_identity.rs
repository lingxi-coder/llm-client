use async_trait::async_trait;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::{
    codecs::{CodecContext, EncodeRequest, RequestMode},
    hosting::FoundryClaudeCodec,
    protocol::*,
    *,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"custom-foundry", "profile_name":"foundry", "protocol":"foundry_claude", "base_url":"https://test.services.ai.azure.com/anthropic", "auth":"none", "regions":["international"], "models":[{"display_model":"supported", "request_model":"custom-deployment", "billing_model":"unrelated-price-id", "aliases":["allowed"], "foundry":{"hosting":"azure", "model_id":"claude-opus-5-5"}}]})).unwrap()
}
fn request(model: &str) -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Find information"}]}]})).unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
            AnthropicToolSearchConfig {
                strategy: AnthropicToolSearchStrategy::Regex,
            },
        )
        .into(),
    );
    request
}
#[derive(Default)]
struct Recording(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for Recording {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let stream = body["stream"] == true;
        self.0.lock().unwrap().push(request);
        let body = if stream {
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"custom-deployment\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".as_bytes().to_vec()
        } else {
            serde_json::to_vec(&json!({"id":"msg1","type":"message","role":"assistant","model":"custom-deployment","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}})).unwrap()
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.into(),
        }
        .into())
    }
}
struct Never;
#[async_trait]
impl AttachmentResolver for Never {
    async fn resolve(&self, _: &AttachmentRef) -> Result<bytes::Bytes, LlmError> {
        panic!("invalid identity read an attachment")
    }
}

#[tokio::test]
async fn selected_alias_identity_survives_complete_stream_and_catalog_order() {
    for reverse in [false, true] {
        let mut profile = profile();
        let mut other = profile.models[0].clone();
        other.display_model = "unsupported".into();
        other.aliases = vec!["denied".into()];
        other.foundry.as_mut().unwrap().model_id = "claude-sonnet-5".into();
        profile.models.push(other);
        if reverse {
            profile.models.reverse();
        }
        let transport = Arc::new(Recording::default());
        let client = LlmClientBuilder::with_transport(transport.clone(), &[profile])
            .with_region(Region::International)
            .build()
            .unwrap();
        let r = request("denied");
        assert!(client
            .chat()
            .complete(&r, &Default::default())
            .await
            .is_err());
        assert!(client.chat().stream(&r, &Default::default()).await.is_err());
        assert!(transport.0.lock().unwrap().is_empty());
        let r = request("allowed");
        client
            .chat()
            .complete(&r, &Default::default())
            .await
            .unwrap();
        let mut stream = client.chat().stream(&r, &Default::default()).await.unwrap();
        let mut ended = false;
        while let Some(event) = stream.next().await {
            ended |= matches!(event.unwrap(), StreamEvent::End { .. });
        }
        assert!(ended);
        let sent = transport.0.lock().unwrap();
        assert_eq!(sent.len(), 2);
        for request in sent.iter() {
            assert_eq!(
                request.url,
                "https://test.services.ai.azure.com/anthropic/v1/messages"
            );
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            assert_eq!(body["model"], "custom-deployment");
            assert!(body.get("foundry").is_none());
            assert!(body.get("model_id").is_none());
        }
    }
}

#[test]
fn wire_only_context_rejects_conflicting_deployment_identities() {
    let mut profile = profile();
    let mut other = profile.models[0].clone();
    other.foundry.as_mut().unwrap().hosting = FoundryHosting::Anthropic;
    profile.models.push(other);
    let r = request("custom-deployment");
    for mode in [RequestMode::Complete, RequestMode::Stream] {
        let context = CodecContext::new(&profile, "custom-deployment", mode);
        assert!(FoundryClaudeCodec.validate_request(&r, &context).is_err());
        assert!(FoundryClaudeCodec
            .encode_request(EncodeRequest::new(&r), &context)
            .is_err());
        for model in &profile.models {
            let selected = CodecContext::for_model(&profile, model, mode);
            FoundryClaudeCodec.validate_request(&r, &selected).unwrap();
            FoundryClaudeCodec
                .encode_request(EncodeRequest::new(&r), &selected)
                .unwrap();
        }
    }
}

#[tokio::test]
async fn missing_identity_fails_before_attachment_reads_and_transport() {
    let mut profile = profile();
    profile.models[0].foundry = None;
    let transport = Arc::new(Recording::default());
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile])
        .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(Never));
    let client = builder.build().unwrap();
    let mut r = request("allowed");
    r.messages[0].content.push(ContentBlock::Document {
        source: DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "document".into(),
                revision: "1".into(),
                filename: "document.pdf".into(),
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
    assert!(client.chat().stream(&r, &Default::default()).await.is_err());
    assert!(transport.0.lock().unwrap().is_empty());
}

#[derive(Default)]
struct Failing(Mutex<usize>);
#[async_trait]
impl Transport for Failing {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        *self.0.lock().unwrap() += 1;
        Err(LlmError::Transport {
            message: "unknown execution outcome".into(),
        })
    }
}
#[tokio::test]
async fn foundry_mcp_and_fetch_are_not_retried_or_failed_over() {
    for fetch in [false, true] {
        let mut primary = profile();
        primary.connection.group = Some("foundry-actions".into());
        primary.connection.order = 0;
        primary.connection.failover.network = true;
        let mut backup = primary.clone();
        backup.profile_name = "backup".into();
        backup.connection.order = 1;
        let transport = Arc::new(Failing::default());
        let client = LlmClientBuilder::with_transport(transport.clone(), &[primary, backup])
            .with_region(Region::International)
            .build()
            .unwrap();
        let mut r = request("allowed");
        r.hosted_tools.clear();
        if fetch {
            let config = AnthropicWebFetchConfig {
                version: AnthropicWebFetchVersion::V20250910,
                ..Default::default()
            };
            r.hosted_tools.push(
                lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(
                    config,
                )
                .into(),
            );
        } else {
            r.hosted_tools.push(
                lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
                    AnthropicMcpConfig::new("docs", "https://mcp.example.test/sse").unwrap(),
                )
                .into(),
            );
        }
        assert!(client
            .chat()
            .complete(&r, &Default::default())
            .await
            .is_err());
        assert_eq!(*transport.0.lock().unwrap(), 1);
    }
}

fn fetch_and_mcp_cache() -> ChatRequest {
    let mut r = request("allowed");
    r.hosted_tools.clear();
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(
            AnthropicWebFetchConfig {
                version: AnthropicWebFetchVersion::V20250910,
                cache_control: Some(CacheTtl::FiveMinutes),
                ..Default::default()
            },
        )
        .into(),
    );
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            AnthropicMcpConfig::new("docs", "https://mcp.example.test/sse")
                .unwrap()
                .with_cache_control(AnthropicMcpCacheControl {
                    ttl: Some(AnthropicMcpCacheTtl::OneHour),
                }),
        )
        .into(),
    );
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    r
}
#[test]
fn foundry_mcp_fetch_and_messages_share_cache_count_and_ttl_order() {
    let p = profile();
    let ctx = CodecContext::for_model(&p, &p.models[0], RequestMode::Complete);
    let mut r = fetch_and_mcp_cache();
    FoundryClaudeCodec.validate_request(&r, &ctx).unwrap();
    let wire = FoundryClaudeCodec
        .encode_request(EncodeRequest::new(&r), &ctx)
        .unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(body["tools"][0]["type"], "mcp_toolset");
    assert_eq!(body["tools"][1]["type"], "web_fetch_20250910");
    r.system.push(SystemBlock {
        text: "Instruction".into(),
    });
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(FoundryClaudeCodec.validate_request(&r, &ctx).is_err());
    assert!(FoundryClaudeCodec
        .encode_request(EncodeRequest::new(&r), &ctx)
        .is_err());
    let mut bad = fetch_and_mcp_cache();
    bad.hosted_tools = vec![
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(
            AnthropicWebFetchConfig {
                version: AnthropicWebFetchVersion::V20250910,
                cache_control: Some(CacheTtl::OneHour),
                ..Default::default()
            },
        )
        .into(),
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            AnthropicMcpConfig::new("docs", "https://mcp.example.test/sse")
                .unwrap()
                .with_cache_control(Default::default()),
        )
        .into(),
    ];
    assert!(FoundryClaudeCodec.validate_request(&bad, &ctx).is_err());
    assert!(FoundryClaudeCodec
        .encode_request(EncodeRequest::new(&bad), &ctx)
        .is_err());
}
#[tokio::test]
async fn invalid_foundry_combined_cache_fails_before_attachment_reads() {
    let p = profile();
    let transport = Arc::new(Recording::default());
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[p])
        .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(Never));
    let client = builder.build().unwrap();
    let mut r = fetch_and_mcp_cache();
    r.system.push(SystemBlock {
        text: "Instruction".into(),
    });
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
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
    assert!(transport.0.lock().unwrap().is_empty());
}

#[test]
fn vertex_client_toolsets_share_the_same_cache_marker_budget() {
    let p:ProviderProfile = serde_json::from_value(json!({"provider_id":"vertex","profile_name":"vertex","base_url":"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1","protocol":"vertex_claude","auth":"none"})).unwrap();
    let ctx = CodecContext::new(&p, "claude-opus-5-5", RequestMode::Complete);
    let mut r = request("claude-opus-5-5");
    r.hosted_tools.clear();
    r.set_anthropic_client_toolsets(vec![
        AnthropicClientToolset::Browser(AnthropicBrowserToolsetConfig {
            cache_control: Some(AnthropicMcpCacheControl::default()),
            ..Default::default()
        }),
        AnthropicClientToolset::Computer(AnthropicComputerToolsetConfig {
            cache_control: Some(AnthropicMcpCacheControl::default()),
            ..Default::default()
        }),
    ]);
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    VertexClaudeCodec.validate_request(&r, &ctx).unwrap();
    r.system.push(SystemBlock {
        text: "Instruction".into(),
    });
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(VertexClaudeCodec.validate_request(&r, &ctx).is_err());
    assert!(VertexClaudeCodec
        .encode_request(EncodeRequest::new(&r), &ctx)
        .is_err());
}
