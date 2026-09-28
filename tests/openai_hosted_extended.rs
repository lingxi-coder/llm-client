use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::providers::openai::types::{
    McpApprovalPolicy, McpApprovalRequest, McpApprovalResponse, RemoteMcpConfig,
};
use lingxi_llm_client::{
    codecs::{openai::responses::OpenAiResponsesCodec, EncodeRequest, WireCodec},
    protocol::{
        ChatRequest, ChatResponse, ContentBlock, ContinuationRef, HostedTool, LlmError,
        ProtocolFamily, ProviderId, ProviderProfile, Region, ResponseId, StopReason, StreamEvent,
        WebSearchConfig,
    },
    transport::{HttpRequest, StreamResponse, Transport},
    HttpResponse, LlmClientBuilder, RequestOptions,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[path = "support/wire_api.rs"]
mod wire_api;

const OPENAI_BASE: &str = "https://api.openai.com/v1";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "openai",
        "profile_name": "openai",
        "base_url": OPENAI_BASE,
        "protocol": "open_ai_responses",
        "auth": "none",
        "extra": {
            "supports_previous_response_id": true,
            "web_search": "openai_responses",
            "remote_mcp": "openai_responses"
        },
        "models": [{
            "display_model": "gpt-6-astra",
            "request_model": "gpt-6-astra",
            "billing_model": "gpt-6-astra"
        }]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": "gpt-6-astra",
        "messages": [{"role":"user","content":[{"type":"text","text":"hello"}]}]
    }))
    .unwrap()
}

fn mcp() -> RemoteMcpConfig {
    RemoteMcpConfig::new("docs", "https://mcp.example.test/mcp")
        .unwrap()
        .with_description("Read-only documentation tools")
        .with_allowed_tools(["search", "fetch"])
        .unwrap()
        .with_require_approval(McpApprovalPolicy::Always)
}

fn encode(request: &ChatRequest, profile: &ProviderProfile) -> Result<Value, LlmError> {
    let context = wire_api::context(profile, "gpt-6-astra", &RequestOptions::default());
    let http = OpenAiResponsesCodec.encode_request(EncodeRequest::new(request), &context)?;
    Ok(serde_json::from_slice(&http.body).unwrap())
}

fn decode_response(body: Value, profile: &ProviderProfile) -> ChatResponse {
    let context = wire_api::context(profile, "gpt-6-astra", &RequestOptions::default());
    OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &context,
        )
        .unwrap()
}

#[test]
fn hosted_web_search_uses_confirmed_filter_names_and_retains_open_page_output() {
    let mut request = request();
    request
        .hosted_tools
        .push(HostedTool::WebSearch(WebSearchConfig {
            allowed_domains: vec!["openai.com".into()],
            blocked_domains: vec!["example.invalid".into()],
            ..Default::default()
        }));
    let wire = encode(&request, &profile()).unwrap();
    assert_eq!(wire["tools"][0]["type"], "web_search");
    assert_eq!(
        wire["tools"][0]["filters"]["allowed_domains"],
        json!(["openai.com"])
    );
    assert_eq!(
        wire["tools"][0]["filters"]["blocked_domains"],
        json!(["example.invalid"])
    );
    assert_eq!(wire["include"], json!(["web_search_call.action.sources"]));

    let search = json!({
        "type":"web_search_call",
        "id":"ws_call_1",
        "status":"completed",
        "action": {
            "type":"open_page",
            "url":"https://openai.com/docs",
            "sources":[{"url":"https://openai.com/docs","title":"API docs"}]
        }
    });
    let response = decode_response(
        json!({
            "id":"resp-web",
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[
                search,
                {"type":"message","role":"assistant","content":[{
                    "type":"output_text","text":"See the docs.",
                    "annotations":[{"type":"url_citation","url":"https://openai.com/docs","title":"API docs","start_index":8,"end_index":12}]
                }]}
            ]
        }),
        &profile(),
    );
    assert_eq!(response.message.text(), "See the docs.");
    let native_search = response
        .message
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ProviderContent { value, .. }
                if value["type"].as_str() == Some("web_search_call") =>
            {
                Some(value)
            }
            _ => None,
        });
    assert_eq!(native_search.unwrap()["action"]["type"], "open_page");
    let attribution = response.web_search.unwrap();
    assert_eq!(attribution.citations[0].url, "https://openai.com/docs");
    assert_eq!(
        attribution.metadata["web_search_calls"][0]["action"]["type"],
        "open_page"
    );
}

#[test]
fn remote_mcp_tool_is_native_provider_execution_with_no_serialized_secret() {
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    );
    let serialized = serde_json::to_string(&request).unwrap();
    assert!(!serialized.contains("authorization"));
    assert!(!serialized.contains("oauth-token"));

    let wire = encode(&request, &profile()).unwrap();
    assert_eq!(
        wire["tools"][0],
        json!({
            "type":"mcp",
            "server_label":"docs",
            "server_url":"https://mcp.example.test/mcp",
            "server_description":"Read-only documentation tools",
            "allowed_tools":["search","fetch"],
            "require_approval":"always"
        })
    );
    assert!(wire["tools"][0].get("authorization").is_none());
}

#[test]
fn approval_required_and_mcp_output_are_retained_without_host_tool_calls() {
    let approval_item = json!({
        "id":"mcpr_123",
        "type":"mcp_approval_request",
        "arguments":"{\"path\":\"guide.md\"}",
        "name":"fetch",
        "server_label":"docs"
    });
    let response = decode_response(
        json!({
            "id":"resp-approval",
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[approval_item]
        }),
        &profile(),
    );
    assert_eq!(
        response.stop_reason,
        StopReason::Other("requires_action".into())
    );
    assert_eq!(response.message.tool_uses().count(), 0);
    let native = response
        .message
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ProviderContent { value, .. } => Some(value),
            _ => None,
        })
        .unwrap();
    assert_eq!(native, &approval_item);

    let approval = McpApprovalRequest::from_native_item(native).unwrap();
    assert_eq!(approval.approval_request_id, "mcpr_123");
    assert_eq!(approval.server_label, "docs");
    assert_eq!(approval.name, "fetch");
    assert_eq!(approval.arguments, "{\"path\":\"guide.md\"}");
    let answer = approval.respond(true);
    assert_eq!(
        answer,
        McpApprovalResponse {
            approval_request_id: "mcpr_123".into(),
            approve: true,
        }
    );

    // Replaying the host's explicit decision yields the documented Responses
    // input item. It remains a provider-native item, never a local ToolUse.
    let mut follow_up = request();
    follow_up.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    );
    follow_up.continuation = Some(ContinuationRef {
        response_id: ResponseId::new("resp-approval"),
        provider_id: ProviderId::new("openai"),
        profile_name: "openai".into(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            OPENAI_BASE,
        ),
        account_scope: "account-a".into(),
        request_model: "gpt-6-astra".into(),
        workspace_id: None,
    });
    follow_up.messages = vec![answer.into_assistant_message()];
    let wire = encode(&follow_up, &profile()).unwrap();
    assert_eq!(wire["previous_response_id"], "resp-approval");
    assert_eq!(
        wire["input"],
        json!([{
            "type":"mcp_approval_response",
            "approval_request_id":"mcpr_123",
            "approve":true
        }])
    );
}

#[test]
fn mcp_list_and_call_outputs_remain_native_and_never_become_client_tool_use() {
    let list = json!({
        "id":"mcpl_1",
        "type":"mcp_list_tools",
        "server_label":"docs",
        "tools":[{"name":"fetch","description":"Fetch a page","input_schema":{"type":"object"}}]
    });
    let call = json!({
        "id":"mcp_1",
        "type":"mcp_call",
        "approval_request_id":null,
        "arguments":"{\"path\":\"guide.md\"}",
        "error":null,
        "name":"fetch",
        "output":"page text",
        "server_label":"docs"
    });
    let response = decode_response(
        json!({
            "id":"resp-mcp",
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[list, call]
        }),
        &profile(),
    );
    assert_eq!(response.message.tool_uses().count(), 0);
    let native = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ProviderContent { value, .. } => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(native, vec![&list, &call]);
    assert_eq!(native[1]["output"], "page text");
}

#[test]
fn mcp_settings_and_native_replay_are_validated_before_encoding() {
    assert!(RemoteMcpConfig::new("docs", "http://mcp.example.test/mcp").is_err());
    assert!(RemoteMcpConfig::new("", "https://mcp.example.test/mcp").is_err());

    let mut duplicate = request();
    duplicate.hosted_tools = vec![
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    ];
    assert!(duplicate.validate_hosted_tools().is_err());

    let mut malformed_approval = request();
    malformed_approval.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    );
    malformed_approval.messages = vec![McpApprovalResponse {
        approval_request_id: "".into(),
        approve: true,
    }
    .into_assistant_message()];
    assert!(encode(&malformed_approval, &profile()).is_err());

    let mut foreign_native = request();
    foreign_native.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    );
    foreign_native.messages = vec![lingxi_llm_client::protocol::ConversationMessage::assistant(
        vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiChat,
            value: json!({
                "type":"mcp_approval_response",
                "approval_request_id":"mcpr_123",
                "approve":true
            }),
        }],
    )];
    assert!(encode(&foreign_native, &profile()).is_err());
}

#[test]
fn approval_items_stream_as_native_content_and_end_as_requires_action() {
    let profile = profile();
    let context = wire_api::context(&profile, "gpt-6-astra", &RequestOptions::default());
    let mut decoder = OpenAiResponsesCodec.stream_decoder(&context);
    let approval_item = json!({
        "id":"mcpr_stream",
        "type":"mcp_approval_request",
        "arguments":"{}",
        "name":"fetch",
        "server_label":"docs"
    });
    let web_search_item = json!({
        "type":"web_search_call",
        "id":"ws_stream",
        "status":"completed",
        "action":{"type":"open_page","url":"https://openai.com/docs"}
    });
    let mut events = Vec::new();
    for frame in [
        json!({"type":"response.created","response":{"id":"resp-stream","model":"gpt-6-astra"}}),
        json!({"type":"response.output_item.done","output_index":0,"item":approval_item}),
        json!({"type":"response.output_item.done","output_index":1,"item":web_search_item}),
        json!({"type":"response.completed","response":{
            "id":"resp-stream","model":"gpt-6-astra","status":"completed",
            "output":[approval_item,web_search_item],"usage":{"input_tokens":0,"output_tokens":0}
        }}),
    ] {
        events.extend(
            wire_api::decode_frame(decoder.as_mut(), frame.to_string().as_bytes()).unwrap(),
        );
    }
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ProviderContent { value, .. } if value["type"] == "mcp_approval_request"))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderContent { value, .. }
            if value["type"] == "web_search_call" && value["action"]["type"] == "open_page"
    )));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::End {
            stop_reason: StopReason::Other(reason), ..
        }) if reason == "requires_action"
    ));
}

struct RecordingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let body = serde_json::to_vec(&json!({
            "id":"resp_auth",
            "model":"gpt-6-astra",
            "status":"completed",
            "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]
        })).unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
        })
    }
}

#[tokio::test]
async fn request_options_inject_mcp_oauth_without_serializing_it_into_chat_history() {
    let profile = profile();
    let transport = Arc::new(RecordingTransport {
        sent: Mutex::new(Vec::new()),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::openai::native::OpenAiHostedTool::RemoteMcp(mcp()).into(),
    );
    let serialized = serde_json::to_string(&request).unwrap();
    assert!(!serialized.contains("Bearer"));
    assert!(!serialized.contains("oauth-token"));

    let mut options = RequestOptions::default();
    options.mcp_authorizations.insert(
        "docs".into(),
        lingxi_llm_client::protocol::Secret::new("Bearer oauth-token".into()),
    );
    client.chat().complete(&request, &options).await.unwrap();
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["tools"][0]["authorization"], "Bearer oauth-token");
    assert!(!format!("{:?}", sent[0]).contains("oauth-token"));
}
