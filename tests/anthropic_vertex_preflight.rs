use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::protocol::*;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::*;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

const MODEL: &str = "claude-opus-5-5";
fn profile(model: &str) -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"custom-google-label","profile_name":"vertex","base_url":"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1","protocol":"vertex_claude","auth":"none","regions":["international"],"models":[{"display_model":model,"request_model":model,"billing_model":model}]})).unwrap()
}
fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]})).unwrap()
}
fn effort() -> ConversationMessage {
    let mut message =
        ConversationMessage::system_text("").with_anthropic_options(AnthropicMessageOptions {
            clear_at: None,
            effort: Some(AnthropicMessageEffort::High),
        });
    message.content.clear();
    message
}
fn attach(r: &mut ChatRequest) {
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
}
#[derive(Default)]
struct Recording(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for Recording {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.lock().unwrap().push(request);
        let body = serde_json::to_vec(&json!({"id":"msg1","type":"message","role":"assistant","model":MODEL,"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}})).unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(bytes::Bytes::from(body)) }).boxed(),
        })
    }
}
struct Never;
#[async_trait]
impl Transport for Never {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("invalid request reached transport")
    }
}
#[async_trait]
impl AttachmentResolver for Never {
    async fn resolve(&self, _: &AttachmentRef) -> Result<bytes::Bytes, LlmError> {
        panic!("invalid request reached attachment resolver")
    }
}
#[tokio::test]
async fn vertex_effort_and_tool_search_reach_the_wire_and_report_active_effort() {
    let transport = Arc::new(Recording::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile(MODEL)])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut r = request(MODEL);
    r.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
            AnthropicToolSearchConfig {
                strategy: AnthropicToolSearchStrategy::Regex,
            },
        )
        .into(),
    );
    r.messages.push(effort());
    r.messages.push(ConversationMessage::user_text("Continue"));
    let response = client
        .chat()
        .complete(&r, &Default::default())
        .await
        .unwrap();
    assert_eq!(
        response.inference.requested_effort,
        Some(ReasoningEffort::High)
    );
    let sent = transport.0.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert!(sent[0]
        .url
        .ends_with("/publishers/anthropic/models/claude-opus-5-5:rawPredict"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert!(body.get("model").is_none());
    assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
    assert_eq!(body["messages"][1]["output_config"]["effort"], "high");
    assert_eq!(body["tools"][0]["type"], "tool_search_tool_regex_20251119");
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("anthropic-beta")
            && value.contains("mid-conversation-output-config-2026-07-01")));
}
#[tokio::test]
async fn unsupported_vertex_models_and_invalid_effort_fail_before_attachments() {
    for model in [MODEL, "claude-sonnet-5"] {
        let mut builder = LlmClientBuilder::with_transport(Arc::new(Never), &[profile(model)])
            .with_region(Region::International);
        builder.with_attachment_resolver(Arc::new(Never));
        let client = builder.build().unwrap();
        let mut r = request(model);
        attach(&mut r);
        if model == MODEL {
            r.messages.push(effort());
            r.messages.push(ConversationMessage::user_text("Continue"));
            r.thinking = Some(ThinkingConfig {
                mode: Some(ThinkingMode::Disabled),
                ..Default::default()
            });
        } else {
            r.hosted_tools.push(
                lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
                    AnthropicToolSearchConfig {
                        strategy: AnthropicToolSearchStrategy::Regex,
                    },
                )
                .into(),
            );
        }
        assert!(client
            .chat()
            .complete(&r, &Default::default())
            .await
            .is_err());
        let mut r = request(model);
        attach(&mut r);
        r.thinking = Some(ThinkingConfig {
            mode: Some(ThinkingMode::Disabled),
            budget: Some(ThinkingBudget::Tokens(1024)),
            ..Default::default()
        });
        assert!(client
            .chat()
            .complete(&r, &Default::default())
            .await
            .is_err());
    }
}
#[tokio::test]
async fn vertex_system_and_tool_change_placement_are_checked_before_attachments() {
    let mut builder = LlmClientBuilder::with_transport(Arc::new(Never), &[profile(MODEL)])
        .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(Never));
    let client = builder.build().unwrap();
    let mut r = request(MODEL);
    attach(&mut r);
    r.messages.insert(
        0,
        ConversationMessage::system_text("Invalid initial system"),
    );
    assert!(client
        .chat()
        .complete(&r, &Default::default())
        .await
        .is_err());
    let mut r = request(MODEL);
    attach(&mut r);
    r.tools.push(serde_json::from_value(json!({"name":"lookup","description":"Lookup","input_schema":{"type":"object","properties":{}}})).unwrap());
    r.messages[0].content.push(
        AnthropicToolChange::remove(AnthropicToolReference::tool("lookup")).into_content_block(),
    );
    let error = client
        .chat()
        .complete(&r, &Default::default())
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::InvalidRequest { message } if message.contains("system")));
}
#[test]
fn vertex_client_toolsets_encode_and_replay_without_opening_other_cloud_routes() {
    let mut r = request(MODEL);
    r.set_anthropic_client_toolsets(vec![
        AnthropicClientToolset::Browser(Default::default()),
        AnthropicClientToolset::Computer(Default::default()),
    ]);
    r.messages.push(serde_json::from_value(json!({"role":"assistant","content":[{"type":"tool_use","id":"call1","name":"list_tabs","input":{},"toolset_name":"browser"}]})).unwrap());
    r.messages.push(serde_json::from_value(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call1","toolset_name":"browser","content":"tabs","is_error":false,"blocks":[{"type":"browser_state","tabs":[]}]}]})).unwrap());
    for mode in [RequestMode::Complete, RequestMode::Stream] {
        let ctx = CodecContext::new(&profile(MODEL), MODEL, mode);
        VertexClaudeCodec.validate_request(&r, &ctx).unwrap();
        let wire = VertexClaudeCodec
            .encode_request(EncodeRequest::new(&r), &ctx)
            .unwrap();
        let body: Value = serde_json::from_slice(&wire.body).unwrap();
        assert_eq!(body["tools"][0]["type"], "browser_toolset_20260801");
        assert_eq!(body["tools"][1]["type"], "computer_toolset_20260801");
        assert_eq!(body["messages"][2]["content"][0]["toolset_name"], "browser");
        assert!(!wire
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));
        assert!(wire.url.ends_with(if mode == RequestMode::Stream {
            ":streamRawPredict"
        } else {
            ":rawPredict"
        }));
    }
    let mut foundry = profile(MODEL);
    foundry.protocol = ProtocolFamily::FoundryClaude;
    assert!(FoundryClaudeCodec
        .encode_request(
            EncodeRequest::new(&r),
            &CodecContext::new(&foundry, MODEL, RequestMode::Complete)
        )
        .is_err());
    let mut unsupported = r.clone();
    unsupported.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(
            Default::default(),
        )
        .into(),
    );
    assert!(VertexClaudeCodec
        .validate_request(
            &unsupported,
            &CodecContext::new(&profile(MODEL), MODEL, RequestMode::Complete)
        )
        .is_err());
}
