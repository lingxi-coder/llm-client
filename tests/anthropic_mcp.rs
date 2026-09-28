use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::protocol::{
    AuthStrategy, ChatRequest, ContentBlock, ConversationMessage, LlmError, ProtocolFamily,
    ProviderProfile, Region, Secret, StopReason,
};
use lingxi_llm_client::providers::anthropic::types::{
    AnthropicMcpCacheControl, AnthropicMcpCacheTtl, AnthropicMcpConfig, AnthropicMcpTool,
    AnthropicMcpToolConfig, AnthropicToolSearchConfig, AnthropicToolSearchStrategy,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, Authenticator, CodecContext, EncodeRequest, HttpRequest,
    LlmClientBuilder, RequestMode, RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const MODEL: &str = "claude-opus-5-5";
const TOKEN: &str = "Bearer mcp-oauth-secret";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic", "profile_name":"anthropic",
        "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages",
        "auth":"none", "regions":["international"],
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}],
        "extra":{"betas":["prompt-caching-2024-07-31","mcp-client-2025-11-20"],
            "headers":{"anthropic-beta":"token-counting-2024-11-01,mcp-client-2025-11-20"}}
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":MODEL,
        "messages":[{"role":"user","content":[{"type":"text","text":"Use the MCP server."}]}]
    }))
    .unwrap()
}

fn context(profile: &ProviderProfile) -> CodecContext {
    CodecContext::new(profile, MODEL, RequestMode::Complete)
}

fn mcp_config(name: &str) -> AnthropicMcpConfig {
    AnthropicMcpConfig::new(name, "https://mcp.example.test/sse").unwrap()
}

fn encode(
    request: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let wire =
        AnthropicMessagesCodec.encode_request(EncodeRequest::new(request), &context(profile))?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

#[test]
fn typed_connector_encodes_one_server_and_matching_toolset_with_current_beta() {
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("workspace")
                .with_default_config(AnthropicMcpToolConfig {
                    enabled: Some(false),
                    defer_loading: None,
                })
                .with_tool_config(
                    "dynamic_remote_tool",
                    AnthropicMcpToolConfig {
                        enabled: Some(true),
                        defer_loading: Some(true),
                    },
                )
                .unwrap()
                .with_tools([AnthropicMcpTool {
                    name: "search".into(),
                    description: Some("Search the workspace".into()),
                    input_schema: json!({"type":"object","properties":{"query":{"type":"string"}}}),
                }])
                .unwrap()
                .with_cache_control(AnthropicMcpCacheControl {
                    ttl: Some(AnthropicMcpCacheTtl::OneHour),
                }),
        )
        .into(),
    );
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
            AnthropicToolSearchConfig {
                strategy: AnthropicToolSearchStrategy::Bm25,
            },
        )
        .into(),
    );

    let (body, headers) = encode(&request, &profile()).unwrap();
    assert_eq!(
        body["mcp_servers"],
        json!([{
            "type":"url", "name":"workspace", "url":"https://mcp.example.test/sse"
        }])
    );
    assert_eq!(body["tools"][0]["type"], "tool_search_tool_bm25_20251119");
    assert_eq!(
        body["tools"][1],
        json!({
            "type":"mcp_toolset", "mcp_server_name":"workspace",
            "default_config":{"enabled":false},
            "configs":{"dynamic_remote_tool":{"enabled":true,"defer_loading":true}},
            "tools":[{"name":"search","description":"Search the workspace",
                "input_schema":{"type":"object","properties":{"query":{"type":"string"}}}}],
            "cache_control":{"type":"ephemeral","ttl":"1h"}
        })
    );
    let beta = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .unwrap()
        .1
        .split(',')
        .collect::<Vec<_>>();
    assert!(beta.contains(&"mcp-client-2026-09-15"));
    assert!(beta.contains(&"prompt-caching-2024-07-31"));
    assert!(beta.contains(&"token-counting-2024-11-01"));
    assert!(!beta
        .iter()
        .any(|value| value.starts_with("mcp-client-2025-")));
    assert_eq!(
        beta.iter()
            .filter(|value| **value == "mcp-client-2026-09-15")
            .count(),
        1
    );
}

#[test]
fn pinned_tool_list_distinguishes_unpinned_from_explicit_empty() {
    let profile = profile();
    let mut unpinned = request();
    unpinned.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );
    let (unpinned_body, _) = encode(&unpinned, &profile).unwrap();
    assert!(unpinned_body["tools"][0].get("tools").is_none());

    let mut pinned_empty = request();
    pinned_empty.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("workspace").with_tools(Vec::new()).unwrap(),
        )
        .into(),
    );
    let (pinned_body, _) = encode(&pinned_empty, &profile).unwrap();
    assert_eq!(pinned_body["tools"][0]["tools"], json!([]));
}

#[test]
fn replayed_listing_is_pinned_on_first_party_and_stays_generic_on_gateways() {
    let mut request = request();
    request.messages.insert(
        0,
        ConversationMessage::assistant(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({"type":"mcp_tool_listing","mcp_server_name":"workspace","tools":[]}),
        }]),
    );
    let (body, headers) = encode(&request, &profile()).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({
            "type":"mcp_tool_listing","mcp_server_name":"workspace","tools":[]
        })
    );
    assert!(headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta") && value.contains("mcp-client-2026-09-15")
    }));

    let mut gateway = profile();
    gateway.provider_id = "compatible-gateway".into();
    gateway.base_url = "https://gateway.example.test".into();
    let (gateway_body, gateway_headers) = encode(&request, &gateway).unwrap();
    assert_eq!(
        gateway_body["messages"][0]["content"][0],
        body["messages"][0]["content"][0]
    );
    assert!(!gateway_headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta") && value.contains("mcp-client-2026-09-15")
    }));
}

#[test]
fn only_effectively_deferred_mcp_tools_require_anthropic_tool_search() {
    let profile = profile();
    let mut disabled_default = request();
    disabled_default.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("disabled").with_default_config(AnthropicMcpToolConfig {
                enabled: Some(false),
                defer_loading: Some(true),
            }),
        )
        .into(),
    );
    assert!(encode(&disabled_default, &profile).is_ok());

    let mut pinned_empty = request();
    pinned_empty.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("empty")
                .with_default_config(AnthropicMcpToolConfig {
                    enabled: Some(true),
                    defer_loading: Some(true),
                })
                .with_tools(Vec::new())
                .unwrap(),
        )
        .into(),
    );
    assert!(encode(&pinned_empty, &profile).is_ok());

    let mut override_disabled = request();
    override_disabled.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("override")
                .with_default_config(AnthropicMcpToolConfig {
                    enabled: Some(true),
                    defer_loading: Some(true),
                })
                .with_tool_config(
                    "search",
                    AnthropicMcpToolConfig {
                        enabled: Some(false),
                        defer_loading: None,
                    },
                )
                .unwrap()
                .with_tools([AnthropicMcpTool {
                    name: "search".into(),
                    description: None,
                    input_schema: json!({"type":"object"}),
                }])
                .unwrap(),
        )
        .into(),
    );
    assert!(encode(&override_disabled, &profile).is_ok());

    let mut deferred = request();
    deferred.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("deferred").with_default_config(AnthropicMcpToolConfig {
                enabled: Some(true),
                defer_loading: Some(true),
            }),
        )
        .into(),
    );
    assert!(matches!(
        encode(&deferred, &profile),
        Err(LlmError::InvalidRequest { message }) if message.contains("tool search")
    ));
}

#[test]
fn invalid_routes_raw_connector_injection_and_cross_provider_use_are_rejected() {
    assert!(AnthropicMcpConfig::new("workspace", "http://mcp.example.test").is_err());
    let mut typed = request();
    typed.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );
    let mut gateway = profile();
    gateway.base_url = "https://gateway.example.test".into();
    assert!(matches!(
        encode(&typed, &gateway),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let openai: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"openai", "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses", "auth":"none"
    }))
    .unwrap();
    let openai_context = CodecContext::new(&openai, MODEL, RequestMode::Complete);
    assert!(matches!(
        lingxi_llm_client::OpenAiResponsesCodec
            .encode_request(EncodeRequest::new(&typed), &openai_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut raw = profile();
    raw.extra = json!({"body":{"mcp_servers":[{"type":"url","name":"raw","url":"https://mcp.example.test","authorization_token":"secret"}]}});
    assert!(matches!(
        encode(&typed, &raw),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn unknown_remote_names_are_not_rejected_locally_and_server_limit_is_enforced() {
    let mut configured = request();
    let config = mcp_config("workspace")
        .with_tool_config(
            "not_returned_yet",
            AnthropicMcpToolConfig {
                enabled: Some(false),
                defer_loading: None,
            },
        )
        .unwrap();
    configured.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(config).into(),
    );
    assert_eq!(
        encode(&configured, &profile()).unwrap().0["tools"][0]["configs"]["not_returned_yet"]
            ["enabled"],
        false
    );

    let mut too_many = request();
    for index in 0..21 {
        too_many.hosted_tools.push(
            lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
                &format!("server-{index}"),
            ))
            .into(),
        );
    }
    assert!(matches!(
        encode(&too_many, &profile()),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[derive(Default)]
struct RecordingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let response = json!({
            "id":"msg_mcp_test","type":"message","role":"assistant","model":MODEL,
            "content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}
        });
        let body = serde_json::to_vec(&response).unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
        })
    }
}

#[tokio::test]
async fn authorization_is_injected_per_request_and_never_serialized_in_chat_history() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );
    assert!(!serde_json::to_string(&request).unwrap().contains(TOKEN));
    let mut options = RequestOptions::default();
    options
        .mcp_authorizations
        .insert("workspace".into(), Secret::new(TOKEN.to_owned()));
    let response = client.chat().complete(&request, &options).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "https://api.anthropic.com/v1/messages");
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["mcp_servers"][0]["authorization_token"], TOKEN);
    assert!(!format!("{:?}", sent[0]).contains(TOKEN));
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta") && value.contains("mcp-client-2026-09-15")
    }));
}

#[tokio::test]
async fn malformed_or_unmatched_per_request_tokens_fail_before_transport() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );

    for (name, token) in [
        ("workspace", "Bearer bad\ntoken"),
        ("different-server", "Bearer valid-shaped-token"),
    ] {
        let mut options = RequestOptions::default();
        options
            .mcp_authorizations
            .insert(name.into(), Secret::new(token.into()));
        assert!(matches!(
            client.chat().complete(&request, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn deferred_toolset_without_search_fails_in_request_preflight() {
    let transport = Arc::new(RecordingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(
            mcp_config("workspace").with_default_config(AnthropicMcpToolConfig {
                enabled: Some(true),
                defer_loading: Some(true),
            }),
        )
        .into(),
    );
    assert!(matches!(
        client.chat().complete(&request, &RequestOptions::default()).await,
        Err(LlmError::InvalidRequest { message }) if message.contains("tool search")
    ));
    assert!(transport.sent.lock().unwrap().is_empty());
}

struct CountingAuthenticator(Arc<AtomicUsize>);

#[async_trait]
impl Authenticator for CountingAuthenticator {
    async fn apply(
        &self,
        _request: &mut HttpRequest,
        _profile: &ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn injected_token_crossing_anthropic_body_limit_fails_before_auth_or_transport() {
    const BODY_LIMIT: usize = 32_000_000;
    let mut profile = profile();
    profile.auth = AuthStrategy::ApiKey;
    let mut chat = request();
    chat.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );

    let key_bytes = serde_json::to_vec("authorization_token").unwrap().len();
    let token_bytes = serde_json::to_vec("t").unwrap().len();
    let authorization_field_bytes = 1 + key_bytes + 1 + token_bytes;
    let initial_len = AnthropicMessagesCodec
        .encoded_body_len(EncodeRequest::new(&chat), &context(&profile))
        .unwrap();
    let target_body_len = BODY_LIMIT - authorization_field_bytes + 1;
    let padding = target_body_len - initial_len;
    let ContentBlock::Text { text, .. } = &mut chat.messages[0].content[0] else {
        unreachable!()
    };
    text.extend(std::iter::repeat_n('x', padding));
    assert_eq!(
        AnthropicMessagesCodec
            .encoded_body_len(EncodeRequest::new(&chat), &context(&profile))
            .unwrap(),
        target_body_len
    );

    let transport = Arc::new(RecordingTransport::default());
    let auth_calls = Arc::new(AtomicUsize::new(0));
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile]);
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(CountingAuthenticator(auth_calls.clone())),
    );
    let client = builder.with_region(Region::International).build().unwrap();
    let mut options = RequestOptions {
        credential: Some(Secret::new("api-key".into())),
        ..Default::default()
    };
    options
        .mcp_authorizations
        .insert("workspace".into(), Secret::new("t".into()));

    assert!(matches!(
        client.chat().complete(&chat, &options).await,
        Err(LlmError::RequestTooLarge { .. })
    ));
    assert_eq!(auth_calls.load(Ordering::SeqCst), 0);
    assert!(transport.sent.lock().unwrap().is_empty());
}

struct FailingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        Err(LlmError::Transport {
            message: "outcome unavailable".into(),
        })
    }
}

#[tokio::test]
async fn uncertain_mcp_execution_is_not_retried_or_failed_over() {
    let mut first = profile();
    first.profile_name = "anthropic-primary".into();
    first.connection.group = Some("anthropic-route".into());
    first.connection.order = 0;
    first.connection.failover = lingxi_llm_client::protocol::FailoverTriggers {
        network: true,
        ..Default::default()
    };
    first.connection.connection_id = Some("primary".into());
    let mut second = profile();
    second.profile_name = "anthropic-secondary".into();
    second.connection.group = Some("anthropic-route".into());
    second.connection.order = 1;
    second.connection.failover = lingxi_llm_client::protocol::FailoverTriggers {
        network: true,
        ..Default::default()
    };
    second.connection.connection_id = Some("secondary".into());
    let transport = Arc::new(FailingTransport {
        sent: Mutex::new(Vec::new()),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[first, second])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut configured = request();
    configured.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::Mcp(mcp_config(
            "workspace",
        ))
        .into(),
    );
    assert!(client
        .chat()
        .complete(&configured, &RequestOptions::default())
        .await
        .is_err());

    let mut replay = request();
    replay.messages.insert(
        0,
        ConversationMessage::assistant(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({"type":"mcp_tool_listing","mcp_server_name":"workspace","tools":[]}),
        }]),
    );
    assert!(client
        .chat()
        .complete(&replay, &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.sent.lock().unwrap().len(), 2);
}
