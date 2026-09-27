use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::protocol::{
    AnthropicMcpConfig, AnthropicMcpToolConfig, ChatRequest, ConnectionSpec, ContentBlock,
    ConversationMessage, FailoverTriggers, HostedTool, LlmError, ProtocolFamily, ProviderProfile,
    Region, Secret,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, FoundryClaudeCodec, HttpRequest,
    LlmClientBuilder, RequestMode, RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

const DEPLOYMENT: &str = "custom-opus-deployment";
const MCP_TOKEN: &str = "Bearer foundry-mcp-secret";

fn foundry_profile(extra: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "azure_foundry",
        "profile_name": "foundry",
        "base_url": "https://example.services.ai.azure.com/anthropic",
        "protocol": "foundry_claude",
        "auth": "none",
        "regions": ["international"],
        "models": [{
            "display_model": "Claude Opus 5.5 deployment",
            "request_model": DEPLOYMENT,
            "billing_model": "claude-opus-5-5"
        }],
        "extra": extra
    }))
    .unwrap()
}

fn make_request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": DEPLOYMENT,
        "messages": [{"role":"user","content":[{"type":"text","text":"Use the MCP server."}]}]
    }))
    .unwrap()
}

fn mcp_config(name: &str) -> AnthropicMcpConfig {
    AnthropicMcpConfig::new(name, "https://mcp.example.test/sse").unwrap()
}

type EncodedTools = (String, Value, Vec<(String, String)>);

fn encode(request: &ChatRequest, profile: &ProviderProfile) -> Result<EncodedTools, LlmError> {
    let context = CodecContext::new(profile, DEPLOYMENT, RequestMode::Complete);
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(request), &context)?;
    Ok((
        wire.url,
        serde_json::from_slice(&wire.body).unwrap(),
        wire.headers,
    ))
}

fn beta_values(headers: &[(String, String)]) -> Vec<String> {
    headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .flat_map(|(_, value)| value.split(','))
        .map(str::trim)
        .map(str::to_owned)
        .collect()
}

#[test]
fn foundry_uses_the_2025_connector_and_preserves_custom_deployment_and_beta_merge() {
    let mut request = make_request();
    request.hosted_tools.push(HostedTool::AnthropicMcp(
        mcp_config("workspace")
            .with_default_config(AnthropicMcpToolConfig {
                enabled: Some(false),
                defer_loading: None,
            })
            .with_tool_config(
                "search",
                AnthropicMcpToolConfig {
                    enabled: Some(true),
                    defer_loading: None,
                },
            )
            .unwrap(),
    ));
    let profile = foundry_profile(json!({
        "betas": ["prompt-caching-2024-07-31", "mcp-client-2025-11-20"],
        "headers": {"anthropic-beta":"token-counting-2024-11-01,mcp-client-2025-11-20"}
    }));

    let (url, body, headers) = encode(&request, &profile).unwrap();
    assert_eq!(
        url,
        "https://example.services.ai.azure.com/anthropic/v1/messages"
    );
    assert_eq!(body["model"], DEPLOYMENT);
    assert_eq!(
        body["mcp_servers"],
        json!([{"type":"url","name":"workspace","url":"https://mcp.example.test/sse"}])
    );
    assert_eq!(
        body["tools"][0],
        json!({
            "type":"mcp_toolset",
            "mcp_server_name":"workspace",
            "default_config":{"enabled":false},
            "configs":{"search":{"enabled":true}}
        })
    );
    let betas = beta_values(&headers);
    assert!(betas.contains(&"mcp-client-2025-11-20".to_owned()));
    assert_eq!(
        betas
            .iter()
            .filter(|beta| beta.as_str() == "mcp-client-2025-11-20")
            .count(),
        1
    );
    assert!(betas.contains(&"prompt-caching-2024-07-31".to_owned()));
    assert!(betas.contains(&"token-counting-2024-11-01".to_owned()));
    assert!(!betas
        .iter()
        .any(|beta| beta.starts_with("mcp-client-2026-")));
}

#[test]
fn foundry_rejects_pinned_lists_inline_toolsets_and_listing_replay() {
    let profile = foundry_profile(Value::Null);
    let mut pinned_empty = make_request();
    pinned_empty.hosted_tools.push(HostedTool::AnthropicMcp(
        mcp_config("workspace").with_tools(Vec::new()).unwrap(),
    ));
    assert!(matches!(
        encode(&pinned_empty, &profile),
        Err(LlmError::UnsupportedCapability { message }) if message.contains("pinned MCP tool lists")
    ));

    let mut inline = make_request();
    inline.hosted_tools.push(HostedTool::AnthropicMcp(
        mcp_config("workspace").with_inline_toolset(true),
    ));
    assert!(matches!(
        encode(&inline, &profile),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut listing_history = make_request();
    listing_history.messages.insert(
        0,
        ConversationMessage::assistant(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"mcp_tool_listing",
                "mcp_server_name":"workspace",
                "tools":[]
            }),
        }]),
    );
    assert!(matches!(
        encode(&listing_history, &profile),
        Err(LlmError::UnsupportedCapability { message }) if message.contains("Claude API-only")
    ));

    let mut inline_addition = make_request();
    let mut system_message = ConversationMessage::system_text("Add the MCP server here.");
    system_message.content.push(ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({
            "type":"tool_addition",
            "tool":{
                "type":"tool_definition",
                "definition":{"type":"mcp_toolset","mcp_server_name":"workspace"}
            }
        }),
    });
    inline_addition.messages.push(system_message);
    assert!(matches!(
        encode(&inline_addition, &profile),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn foundry_rejects_raw_mcp_body_and_newer_profile_beta_injection() {
    let raw_body = foundry_profile(json!({
        "body":{"mcp_servers":[{"type":"url","name":"workspace","url":"https://mcp.example.test/sse"}]}
    }));
    assert!(matches!(
        encode(&make_request(), &raw_body),
        Err(LlmError::InvalidRequest { message }) if message.contains("typed request configuration")
    ));

    for extra in [
        json!({"headers":{"Anthropic-Beta":"mcp-client-2026-09-15"}}),
        json!({"betas":["inline-tools-2026-09-15"]}),
    ] {
        assert!(matches!(
            encode(&make_request(), &foundry_profile(extra)),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

fn native_mcp_history() -> ChatRequest {
    let mut request = make_request();
    request.messages.push(ConversationMessage::assistant(vec![
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"mcp_tool_use",
                "id":"mcptoolu_call1",
                "name":"workspace__search",
                "server_name":"workspace",
                "input":{"query":"launch plan"}
            }),
        },
    ]));
    let mut tool_result_message = ConversationMessage::user_text("");
    tool_result_message.content = vec![ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json!({
            "type":"mcp_tool_result",
            "tool_use_id":"mcptoolu_call1",
            "is_error":false,
            "content":[{"type":"text","text":"Found the plan."}]
        }),
    }];
    request.messages.push(tool_result_message);
    request
}

#[test]
fn foundry_replays_mcp_uses_and_results_with_the_legacy_connector_beta() {
    let profile = foundry_profile(Value::Null);
    let request = native_mcp_history();
    let (_, body, headers) = encode(&request, &profile).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["type"], "mcp_tool_use");
    assert_eq!(body["messages"][2]["content"][0]["type"], "mcp_tool_result");
    let betas = beta_values(&headers);
    assert!(betas.contains(&"mcp-client-2025-11-20".to_owned()));
    assert!(!betas
        .iter()
        .any(|beta| beta.starts_with("mcp-client-2026-")));
}

#[derive(Default)]
struct RecordingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let body = serde_json::to_vec(&json!({
            "id":"msg_foundry_mcp",
            "type":"message",
            "role":"assistant",
            "model":DEPLOYMENT,
            "content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn",
            "usage":{"input_tokens":1,"output_tokens":1}
        }))
        .unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
        })
    }
}

#[tokio::test]
async fn foundry_mcp_authorization_is_injected_per_request_and_mismatch_fails_preflight() {
    let transport = Arc::new(RecordingTransport::default());
    let profile = foundry_profile(Value::Null);
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = make_request();
    request
        .hosted_tools
        .push(HostedTool::AnthropicMcp(mcp_config("workspace")));
    assert!(!serde_json::to_string(&request).unwrap().contains(MCP_TOKEN));

    let mut good_options = RequestOptions::default();
    good_options
        .mcp_authorizations
        .insert("workspace".into(), Secret::new(MCP_TOKEN.to_owned()));
    client
        .chat()
        .complete(&request, &good_options)
        .await
        .unwrap();
    {
        let sent = transport.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(
            sent[0].url,
            "https://example.services.ai.azure.com/anthropic/v1/messages"
        );
        let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
        assert_eq!(body["mcp_servers"][0]["authorization_token"], MCP_TOKEN);
        assert!(sent[0].headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("anthropic-beta") && value.contains("mcp-client-2025-11-20")
        }));
    }

    let mut bad_options = RequestOptions::default();
    bad_options.mcp_authorizations.insert(
        "different-server".into(),
        Secret::new("Bearer wrong-server".into()),
    );
    assert!(matches!(
        client.chat().complete(&request, &bad_options).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[derive(Default)]
struct FailingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        Err(LlmError::Transport {
            message: "Foundry outcome unavailable".into(),
        })
    }
}

#[tokio::test]
async fn uncertain_foundry_mcp_replay_is_not_retried_or_failed_over() {
    let mut primary = foundry_profile(Value::Null);
    primary.profile_name = "foundry-primary".into();
    primary.connection = ConnectionSpec {
        group: Some("foundry-mcp".into()),
        connection_id: Some("primary".into()),
        order: 0,
        hidden: false,
        failover: FailoverTriggers {
            network: true,
            ..Default::default()
        },
    };
    let mut secondary = foundry_profile(Value::Null);
    secondary.profile_name = "foundry-secondary".into();
    secondary.connection = ConnectionSpec {
        group: Some("foundry-mcp".into()),
        connection_id: Some("secondary".into()),
        order: 1,
        hidden: false,
        failover: FailoverTriggers {
            network: true,
            ..Default::default()
        },
    };
    let transport = Arc::new(FailingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[primary, secondary])
        .with_region(Region::International)
        .build()
        .unwrap();

    assert!(client
        .chat()
        .complete(&native_mcp_history(), &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[test]
fn typed_mcp_stays_rejected_on_a_non_foundry_compatible_messages_profile() {
    let mut request = make_request();
    request
        .hosted_tools
        .push(HostedTool::AnthropicMcp(mcp_config("workspace")));
    let mut profile = foundry_profile(Value::Null);
    profile.provider_id = "compatible_gateway".into();
    profile.protocol = ProtocolFamily::AnthropicMessages;
    profile.base_url = "https://gateway.example.test".into();
    let context = CodecContext::new(&profile, DEPLOYMENT, RequestMode::Complete);
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request), &context),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}
