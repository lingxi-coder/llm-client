use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    codecs::{openai::responses::OpenAiResponsesCodec, EncodeRequest, WireCodec},
    protocol::{
        ChatRequest, ContentBlock, HostedTool, LlmError, ProtocolFamily, ProviderProfile, Region,
        StopReason, StreamEvent,
    },
    transport::{HttpRequest, StreamResponse, Transport},
    HttpResponse, LlmClientBuilder, RequestOptions,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[path = "support/wire_api.rs"]
mod wire_api;

const XAI_BASE: &str = "https://api.x.ai/v1";
const TOKEN: &str = "Bearer request-scoped-mcp-token";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"xai",
        "profile_name":"grok-responses",
        "base_url":XAI_BASE,
        "protocol":"open_ai_responses",
        "auth":"none",
        "extra":{"xai_remote_mcp":"xai_responses"},
        "models":[{"display_model":"grok-4.7","request_model":"grok-4.7","billing_model":"grok-4.7"}]
    }))
    .unwrap()
}

fn config() -> lingxi_llm_client::protocol::llm::XaiRemoteMcpConfig {
    lingxi_llm_client::protocol::llm::XaiRemoteMcpConfig::new(
        "docs",
        "https://mcp.example.test/mcp",
    )
    .unwrap()
    .with_description("Read-only documentation tools")
    .with_allowed_tools(["search", "fetch"])
    .unwrap()
}

fn request() -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":"grok-4.7",
        "messages":[{"role":"user","content":[{"type":"text","text":"Find the API guide."}]}]
    }))
    .unwrap();
    request
        .hosted_tools
        .push(HostedTool::XaiRemoteMcp(config()));
    request
}

fn encoded(request: &ChatRequest, profile: &ProviderProfile) -> Result<HttpRequest, LlmError> {
    OpenAiResponsesCodec.encode_request(
        EncodeRequest::new(request),
        &wire_api::context(profile, "grok-4.7", &RequestOptions::default()),
    )
}

#[test]
fn xai_mcp_wire_uses_only_the_documented_configuration_fields() {
    let request = request();
    let profile = profile();
    let outgoing = encoded(&request, &profile).unwrap();
    assert_eq!(outgoing.url, "https://api.x.ai/v1/responses");
    let body: Value = serde_json::from_slice(&outgoing.body).unwrap();
    assert_eq!(
        body["tools"][0],
        json!({
            "type":"mcp",
            "server_url":"https://mcp.example.test/mcp",
            "server_label":"docs",
            "server_description":"Read-only documentation tools",
            "allowed_tools":["search","fetch"]
        })
    );
    assert!(body["tools"][0].get("authorization").is_none());
    assert!(body["tools"][0].get("require_approval").is_none());
    assert!(body["tools"][0].get("connector_id").is_none());
    assert!(!serde_json::to_string(&request)
        .unwrap()
        .contains("authorization"));
    assert!(
        serde_json::from_value::<lingxi_llm_client::protocol::llm::XaiRemoteMcpConfig>(json!({
            "server_label":"docs",
            "server_url":"https://mcp.example.test/mcp",
            "require_approval":"always"
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<lingxi_llm_client::protocol::llm::XaiRemoteMcpConfig>(json!({
            "server_label":"docs",
            "server_url":"https://mcp.example.test/mcp",
            "connector_id":"connector-1"
        }))
        .is_err()
    );
}

#[test]
fn xai_route_and_provider_specific_configs_are_preflighted() {
    let request = request();
    let mut wrong_profile = profile();
    wrong_profile.extra = json!({});
    assert!(encoded(&request, &wrong_profile).is_err());

    let mut wrong_host = profile();
    wrong_host.base_url = "https://api.openai.com/v1".into();
    assert!(encoded(&request, &wrong_host).is_err());

    let mut mixed = request;
    mixed.hosted_tools.push(HostedTool::RemoteMcp(
        lingxi_llm_client::protocol::RemoteMcpConfig::new(
            "openai-server",
            "https://mcp.openai.example.test/mcp",
        )
        .unwrap(),
    ));
    assert!(encoded(&mixed, &profile()).is_err());
}

#[tokio::test]
async fn disabled_xai_profile_is_rejected_before_transport() {
    let mut disabled = profile();
    disabled.extra = json!({});
    let transport = Arc::new(RecordingTransport {
        sent: Mutex::new(Vec::new()),
        response: json!({"id":"unused","status":"completed","output":[]}),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[disabled])
        .with_region(Region::International)
        .build()
        .unwrap();
    let error = client
        .chat()
        .complete(&request(), &RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
    assert!(transport.sent.lock().unwrap().is_empty());
}

struct RecordingTransport {
    sent: Mutex<Vec<HttpRequest>>,
    response: Value,
}

#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let response = self.response.clone();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move {
                Ok(Bytes::from(serde_json::to_vec(&response).unwrap()))
            })
            .boxed(),
        })
    }
}

#[tokio::test]
async fn authorization_is_injected_per_request_and_native_mcp_output_is_preserved() {
    let transport = Arc::new(RecordingTransport {
        sent: Mutex::new(Vec::new()),
        response: json!({
            "id":"resp_xai_mcp",
            "model":"grok-4.7",
            "status":"completed",
            "output":[
                {"id":"mcp_call_1","type":"mcp_call","server_label":"docs","name":"search","arguments":"{\"query\":\"API guide\"}","status":"completed"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"The guide is available."}]}
            ]
        }),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let request = request();
    let request_history = serde_json::to_string(&request).unwrap();
    assert!(!request_history.contains(TOKEN));

    let mut options = RequestOptions::default();
    options.mcp_authorizations.insert(
        "docs".into(),
        lingxi_llm_client::protocol::Secret::new(TOKEN.into()),
    );
    let response = client.chat().complete(&request, &options).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.message.text(), "The guide is available.");
    assert!(response.message.content.iter().any(|block| matches!(
        block,
        ContentBlock::ProviderContent { protocol: ProtocolFamily::OpenAiResponses, value }
            if value["type"] == "mcp_call" && value["name"] == "search"
    )));
    assert!(!response
        .message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolUse { .. })));
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "https://api.x.ai/v1/responses");
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["tools"][0]["authorization"], TOKEN);
    assert!(!format!("{:?}", sent[0]).contains(TOKEN));
}

#[test]
fn xai_approval_shaped_output_stays_native_and_does_not_pause_for_openai_approval() {
    let context = wire_api::context(&profile(), "grok-4.7", &RequestOptions::default());
    let approval = json!({
        "id":"mcp_approval_1",
        "type":"mcp_approval_request",
        "server_label":"docs",
        "name":"search",
        "arguments":"{}"
    });
    let mcp_call = json!({
        "id":"mcp_call_1",
        "type":"mcp_call",
        "server_label":"docs",
        "name":"search",
        "arguments":"{}",
        "status":"completed"
    });
    let body = json!({
        "id":"resp_approval_shape",
        "model":"grok-4.7",
        "status":"completed",
        "output":[mcp_call, approval]
    });
    let response = OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &context,
        )
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(response.message.content.iter().any(|block| matches!(
        block,
        ContentBlock::ProviderContent { value, .. }
            if value["type"] == "mcp_approval_request"
    )));

    let mut decoder = OpenAiResponsesCodec.stream_decoder(&context);
    let frames = [
        json!({"type":"response.output_item.done","output_index":0,"item":mcp_call}),
        json!({"type":"response.output_item.done","output_index":1,"item":approval}),
        json!({"type":"response.completed","response":body}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        let wire = format!("data: {frame}\n\n");
        events.extend(
            decoder
                .push_bytes(wire.as_bytes())
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        );
    }
    events.extend(
        decoder
            .finish()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ProviderContent { value, .. } if matches!(value["type"].as_str(), Some("mcp_approval_request" | "mcp_call"))))
            .count(),
        2
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::EndTurn,
            ..
        })
    ));
}
