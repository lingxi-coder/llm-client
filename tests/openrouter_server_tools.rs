use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, ConnectionSpec, ContentBlock, ConversationMessage,
    HostedTool, LlmError, MessageRole, ProtocolFamily, ProviderProfile, Region, Secret,
    StreamEvent, ToolChoice, ToolSpec, ToolUseId,
};
use lingxi_llm_client::providers::anthropic::types::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy,
};
use lingxi_llm_client::providers::openrouter::types::{
    OpenRouterContainerRef, OpenRouterContainerScope, OpenRouterShellConfig, OpenRouterShellEngine,
    OpenRouterShellEnvironment, OpenRouterShellNetworkPolicy, OpenRouterToolSearchConfig,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, Authenticator, CodecContext, EncodeRequest, FoundryClaudeCodec,
    GeminiCodec, HttpRequest, HttpResponse, LlmClientBuilder, OpenAiChatCodec,
    OpenAiResponsesCodec, RequestMode, RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[path = "support/wire_api.rs"]
mod wire_api;

fn profile(name: &str, protocol: ProtocolFamily, base_url: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openrouter",
        "profile_name":name,
        "base_url":base_url,
        "protocol":protocol,
        "auth":"none",
        "regions":["international"],
        "models":[{"display_model":"any/model","request_model":"any/model","billing_model":"any/model"}],
        "extra":{}
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"any/model",
        "messages":[{"role":"user","content":[{"type":"text","text":"Use the right tool."}]}]
    }))
    .unwrap()
}

fn encode(codec: &dyn WireCodec, req: &ChatRequest, profile: &ProviderProfile) -> Value {
    let context = CodecContext::new(profile, "any/model", RequestMode::Complete);
    let request = codec
        .encode_request(EncodeRequest::new(req), &context)
        .unwrap();
    serde_json::from_slice(&request.body).unwrap()
}

fn deferred_tool(name: &str) -> ToolSpec {
    ToolSpec {
        tool_type: None,
        extra: serde_json::Value::Null,
        name: name.into(),
        description: format!("Find or update {name}"),
        input_schema: json!({
            "type":"object",
            "properties":{"query":{"type":"string"}},
            "required":["query"]
        }),
        strict: false,
        defer_loading: true,
        native_options: Vec::new(),
    }
}

fn shell(config: OpenRouterShellConfig) -> HostedTool {
    lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::Shell(config).into()
}

#[test]
fn tool_search_is_encoded_for_responses_and_messages_with_deferred_client_tools() {
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig {
                max_results: Some(12),
            },
        )
        .into(),
    );
    req.tools.push(deferred_tool("find_events"));

    let responses_profile = profile(
        "or-responses",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    let responses = encode(&OpenAiResponsesCodec, &req, &responses_profile);
    assert_eq!(
        responses["tools"],
        json!([
            {
                "type":"function",
                "name":"find_events",
                "description":"Find or update find_events",
                "parameters":req.tools[0].input_schema,
                "strict":false,
                "defer_loading":true
            },
            {"type":"openrouter:tool_search","parameters":{"max_results":12}}
        ])
    );
    assert_eq!(responses["tool_choice"], "auto");

    let messages_profile = profile(
        "or-messages",
        ProtocolFamily::AnthropicMessages,
        "https://openrouter.ai/api/v1",
    );
    let messages = encode(&AnthropicMessagesCodec, &req, &messages_profile);
    assert_eq!(messages["tools"][0]["name"], "find_events");
    assert_eq!(messages["tools"][0]["defer_loading"], true);
    assert_eq!(
        messages["tools"][0]["input_schema"],
        req.tools[0].input_schema
    );
    assert_eq!(
        messages["tools"][1],
        json!({
            "type":"openrouter:tool_search",
            "parameters":{"max_results":12}
        })
    );
    assert_eq!(messages["tool_choice"], json!({"type":"auto"}));
}

#[test]
fn shell_parameters_and_scoped_container_ids_use_the_documented_wire_shape() {
    let mut req = request();
    req.hosted_tools.push(shell(OpenRouterShellConfig {
        engine: Some(OpenRouterShellEngine::OpenRouter),
        environment: Some(OpenRouterShellEnvironment::ContainerAuto {
            network_policy: Some(OpenRouterShellNetworkPolicy::Allowlist {
                allowed_domains: vec!["pypi.org".into(), "*.pythonhosted.org".into()],
            }),
        }),
        timeout_ms: Some(30_000),
        max_output_length: Some(8_000),
    }));

    let responses_profile = profile(
        "or-shell",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    let body = encode(&OpenAiResponsesCodec, &req, &responses_profile);
    assert_eq!(
        body["tools"][0],
        json!({
            "type":"openrouter:shell",
            "parameters":{
                "engine":"openrouter",
                "environment":{
                    "type":"container_auto",
                    "network_policy":{
                        "type":"allowlist",
                        "allowed_domains":["pypi.org","*.pythonhosted.org"]
                    }
                },
                "timeout_ms":30_000,
                "max_output_length":8_000
            }
        })
    );

    let scope =
        OpenRouterContainerScope::new("or-shell", "https://openrouter.ai/api/v1", "tenant-a")
            .unwrap();
    let reference = OpenRouterContainerRef::new("container_123", scope).unwrap();
    req.hosted_tools = vec![shell(OpenRouterShellConfig {
        environment: Some(OpenRouterShellEnvironment::ContainerReference {
            container: reference,
            network_policy: Some(OpenRouterShellNetworkPolicy::Disabled),
        }),
        ..Default::default()
    })];
    let body = encode(&OpenAiResponsesCodec, &req, &responses_profile);
    assert_eq!(
        body["tools"][0]["parameters"]["environment"],
        json!({
            "type":"container_reference",
            "container_id":"container_123",
            "network_policy":{"type":"disabled"}
        })
    );
    assert!(body.to_string().contains("container_123"));
    assert!(!body.to_string().contains("tenant-a"));
}

#[test]
fn a_container_scope_matches_the_profile_root_with_a_trailing_slash_after_serde_roundtrip() {
    let responses_profile = profile(
        "or-shell",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1/",
    );
    let scope =
        OpenRouterContainerScope::new("or-shell", "https://openrouter.ai/api/v1/", "tenant-a")
            .unwrap();
    let reference = OpenRouterContainerRef::new("container_123", scope).unwrap();
    let persisted_reference: OpenRouterContainerRef =
        serde_json::from_value(serde_json::to_value(&reference).unwrap()).unwrap();

    for container in [reference, persisted_reference] {
        let mut req = request();
        req.hosted_tools.push(shell(OpenRouterShellConfig {
            environment: Some(OpenRouterShellEnvironment::ContainerReference {
                container,
                network_policy: None,
            }),
            ..Default::default()
        }));
        let body = encode(&OpenAiResponsesCodec, &req, &responses_profile);
        assert_eq!(
            body["tools"][0]["parameters"]["environment"]["container_id"],
            "container_123"
        );
    }
}

#[test]
fn search_deferred_choice_and_protocol_limits_are_rejected_without_loosening_native_messages() {
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig::default(),
        )
        .into(),
    );
    req.tools.push(deferred_tool("find_events"));
    req.tool_choice = ToolChoice::None;
    let responses_profile = profile(
        "or-responses",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(
            EncodeRequest::new(&req),
            &CodecContext::new(&responses_profile, "any/model", RequestMode::Complete),
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut too_many_results = request();
    too_many_results.hosted_tools.push(
        lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig {
                max_results: Some(51),
            },
        )
        .into(),
    );
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(
            EncodeRequest::new(&too_many_results),
            &CodecContext::new(&responses_profile, "any/model", RequestMode::Complete),
        ),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut chat_request = request();
    chat_request
        .hosted_tools
        .push(shell(OpenRouterShellConfig::default()));
    let chat_profile = profile(
        "or-chat",
        ProtocolFamily::OpenAiChat,
        "https://openrouter.ai/api/v1",
    );
    let chat_context = CodecContext::new(&chat_profile, "any/model", RequestMode::Complete);
    assert!(OpenAiChatCodec
        .encode_request(EncodeRequest::new(&chat_request), &chat_context)
        .is_err());
    assert!(OpenAiChatCodec
        .validate_request(&chat_request, &chat_context)
        .is_err());

    let gemini_profile = profile(
        "gemini",
        ProtocolFamily::GeminiGenerateContent,
        "https://generativelanguage.googleapis.com/v1beta",
    );
    let gemini_context = CodecContext::new(&gemini_profile, "any/model", RequestMode::Complete);
    assert!(GeminiCodec
        .encode_request(EncodeRequest::new(&chat_request), &gemini_context)
        .is_err());
    assert!(GeminiCodec
        .validate_request(&chat_request, &gemini_context)
        .is_err());

    let mut shell_only_deferred = request();
    shell_only_deferred
        .hosted_tools
        .push(shell(OpenRouterShellConfig::default()));
    shell_only_deferred.tools.push(deferred_tool("find_events"));
    let messages_profile = profile(
        "or-messages",
        ProtocolFamily::AnthropicMessages,
        "https://openrouter.ai/api/v1",
    );
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(
            EncodeRequest::new(&shell_only_deferred),
            &CodecContext::new(&responses_profile, "any/model", RequestMode::Complete),
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(
            EncodeRequest::new(&shell_only_deferred),
            &CodecContext::new(&messages_profile, "any/model", RequestMode::Complete),
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut shell_with_search = shell_only_deferred.clone();
    shell_with_search.hosted_tools.push(
        lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig::default(),
        )
        .into(),
    );
    assert!(OpenAiResponsesCodec
        .encode_request(
            EncodeRequest::new(&shell_with_search),
            &CodecContext::new(&responses_profile, "any/model", RequestMode::Complete),
        )
        .is_ok());
    assert!(AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&shell_with_search),
            &CodecContext::new(&messages_profile, "any/model", RequestMode::Complete),
        )
        .is_ok());

    let regional_profile = profile(
        "or-regional",
        ProtocolFamily::OpenAiResponses,
        "https://us.openrouter.ai/api/v1",
    );
    assert!(OpenAiResponsesCodec
        .encode_request(
            EncodeRequest::new(&chat_request),
            &CodecContext::new(&regional_profile, "any/model", RequestMode::Complete),
        )
        .is_err());

    // OpenRouter's search tool must not turn off the already-supported
    // Anthropic-native deferral path when no OpenRouter tool was requested.
    let mut native = request();
    native.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::ToolSearch(
            AnthropicToolSearchConfig {
                strategy: AnthropicToolSearchStrategy::Regex,
            },
        )
        .into(),
    );
    native.tools.push(deferred_tool("find_events"));
    native.tool_choice = ToolChoice::None;
    let mut anthropic = profile(
        "anthropic",
        ProtocolFamily::AnthropicMessages,
        "https://api.anthropic.com",
    );
    anthropic.provider_id = "anthropic".into();
    let encoded = AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&native),
            &CodecContext::new(&anthropic, "claude-opus-4-6", RequestMode::Complete),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["tool_choice"], json!({"type":"none"}));
}

#[test]
fn openrouter_messages_container_metadata_is_raw_and_explicitly_importable() {
    let server_use = json!({
        "type":"server_tool_use",
        "id":"srvtoolu_1",
        "name":"openrouter:shell",
        "input":{"commands":["pwd"]}
    });
    let shell_result = json!({
        "type":"openrouter_shell_tool_result",
        "tool_use_id":"srvtoolu_1",
        "output":[{"stdout":"/workspace","stderr":"","outcome":{"type":"exit","exit_code":0}}]
    });
    let container = json!({"id":"container_123","expires_at":1_900_000_000,"skills":[]});
    let payload = json!({
        "id":"msg_1",
        "model":"any/model",
        "content":[server_use.clone(),shell_result.clone()],
        "container":container.clone(),
        "stop_reason":"end_turn",
        "usage":{"input_tokens":3,"output_tokens":4}
    });
    let messages_profile = profile(
        "or-messages",
        ProtocolFamily::AnthropicMessages,
        "https://openrouter.ai/api/v1",
    );
    let context = CodecContext::new(&messages_profile, "any/model", RequestMode::Complete);
    let response = AnthropicMessagesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: Bytes::from(payload.to_string()),
            },
            &context,
        )
        .unwrap();

    assert_eq!(
        response.message.content,
        vec![
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: server_use.clone(),
            },
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: shell_result.clone(),
            }
        ]
    );
    assert_eq!(response.openrouter_container().unwrap().envelope, container);

    let scope =
        OpenRouterContainerScope::new("or-messages", "https://openrouter.ai/api/v1", "tenant-a")
            .unwrap();
    let imported = response
        .openrouter_container()
        .unwrap()
        .reference_for(scope)
        .unwrap();
    assert_eq!(imported.id(), "container_123");

    let mut replay = request();
    replay.messages = vec![ConversationMessage::assistant(response.message.content)];
    replay.hosted_tools.push(shell(OpenRouterShellConfig {
        environment: Some(OpenRouterShellEnvironment::ContainerReference {
            container: imported,
            network_policy: None,
        }),
        ..Default::default()
    }));
    let body = encode(&AnthropicMessagesCodec, &replay, &messages_profile);
    assert_eq!(
        body["messages"][0]["content"],
        json!([server_use, shell_result])
    );
    assert_eq!(
        body["tools"][0]["parameters"]["environment"]["container_id"],
        "container_123"
    );
    assert!(!body.to_string().contains("tenant-a"));

    let native_anthropic = profile(
        "anthropic-native",
        ProtocolFamily::AnthropicMessages,
        "https://api.anthropic.com",
    );
    let generic = AnthropicMessagesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: Bytes::from(payload.to_string()),
            },
            &CodecContext::new(&native_anthropic, "any/model", RequestMode::Complete),
        )
        .unwrap();
    assert!(generic.openrouter_container().is_none());
}

#[test]
fn openrouter_messages_replays_direct_and_unknown_native_tool_callers() {
    let messages_profile = profile(
        "or-messages",
        ProtocolFamily::AnthropicMessages,
        "https://openrouter.ai/api/v1",
    );
    let context = CodecContext::new(&messages_profile, "any/model", RequestMode::Complete);

    for (id, caller) in [
        (
            "tool_direct",
            json!({"type":"direct","future":{"keep":true}}),
        ),
        (
            "tool_future",
            json!({"type":"future_native_caller_v1","opaque":[1,"keep"]}),
        ),
    ] {
        let payload = json!({
            "id":"msg_1",
            "model":"any/model",
            "content":[{
                "type":"tool_use",
                "id":id,
                "name":"lookup",
                "input":{"query":"recent"},
                "caller":caller.clone()
            }],
            "stop_reason":"tool_use",
            "usage":{"input_tokens":3,"output_tokens":4}
        });
        let response = AnthropicMessagesCodec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: Bytes::from(payload.to_string()),
                },
                &context,
            )
            .unwrap();

        let mut replay = request();
        replay.messages.push(response.message);
        replay.messages.push(ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: ToolUseId::new(id),
                content: "Found one record.".into(),
                is_error: false,
                blocks: None,
                toolset_name: None,
            }],
        });
        let body = encode(&AnthropicMessagesCodec, &replay, &messages_profile);
        assert_eq!(body["messages"][1]["content"][0]["caller"], caller);
    }

    let caller = json!({"type":"direct","future":{"wrapper":"keep"}});
    let foundry_payload = json!({
        "id":"msg_foundry",
        "model":"any/model",
        "content":[{
            "type":"tool_use",
            "id":"tool_foundry",
            "name":"lookup",
            "input":{"query":"recent"},
            "caller":caller.clone()
        }],
        "stop_reason":"tool_use",
        "usage":{"input_tokens":3,"output_tokens":4}
    });
    let mut foundry_profile = profile(
        "foundry",
        ProtocolFamily::FoundryClaude,
        "https://resource.example.test/anthropic",
    );
    foundry_profile.provider_id = "azure_foundry".into();
    let foundry_context = CodecContext::new(&foundry_profile, "any/model", RequestMode::Complete);
    let response = FoundryClaudeCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: Bytes::from(foundry_payload.to_string()),
            },
            &foundry_context,
        )
        .unwrap();
    let mut replay = request();
    replay.messages.push(response.message);
    replay.messages.push(ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("tool_foundry"),
            content: "Found one record.".into(),
            is_error: false,
            blocks: None,
            toolset_name: None,
        }],
    });
    let wire_request = FoundryClaudeCodec
        .encode_request(EncodeRequest::new(&replay), &foundry_context)
        .unwrap();
    let body: Value = serde_json::from_slice(&wire_request.body).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["caller"], caller);

    let non_anthropic_caller = ContentBlock::ToolUse {
        id: ToolUseId::new("tool_direct"),
        name: "lookup".into(),
        input: json!({"query":"recent"}),
        provider_id: None,
        caller: Some(json!({"type":"direct"})),
        toolset_name: None,
        thought_signature: None,
    };
    let mut non_anthropic_request = request();
    non_anthropic_request
        .messages
        .push(ConversationMessage::assistant(vec![non_anthropic_caller]));
    let mut openai_profile = profile(
        "openai",
        ProtocolFamily::OpenAiChat,
        "https://api.openai.com/v1",
    );
    openai_profile.provider_id = "openai".into();
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            EncodeRequest::new(&non_anthropic_request),
            &CodecContext::new(&openai_profile, "any/model", RequestMode::Complete),
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn openrouter_responses_server_tool_output_stays_native_and_replays() {
    let native_item = json!({
        "type":"openrouter:shell",
        "id":"shell_1",
        "commands":["pwd"],
        "output":[{"stdout":"/workspace","stderr":"","outcome":{"type":"exit","exit_code":0}}]
    });
    let body = json!({
        "id":"resp_1",
        "status":"completed",
        "model":"any/model",
        "output":[native_item.clone()],
        "usage":{"input_tokens":3,"output_tokens":4}
    });
    let responses_profile = profile(
        "or-responses",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    let response = OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: Bytes::from(body.to_string()),
            },
            &CodecContext::new(&responses_profile, "any/model", RequestMode::Complete),
        )
        .unwrap();
    assert_eq!(
        response.message.content,
        vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: native_item.clone(),
        }]
    );
    assert!(!response
        .message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolUse { .. })));

    let mut replay = request();
    replay.messages = vec![ConversationMessage::assistant(response.message.content)];
    replay
        .hosted_tools
        .push(shell(OpenRouterShellConfig::default()));
    let body = encode(&OpenAiResponsesCodec, &replay, &responses_profile);
    assert_eq!(body["input"][0], native_item);
}

#[test]
fn messages_stream_preserves_server_tool_content_events_and_container_metadata() {
    let messages_profile = profile(
        "or-messages",
        ProtocolFamily::AnthropicMessages,
        "https://openrouter.ai/api/v1",
    );
    let context = CodecContext::new(&messages_profile, "any/model", RequestMode::Stream);
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&context);
    let server_use = json!({
        "type":"server_tool_use",
        "id":"srvtoolu_1",
        "name":"openrouter:shell",
        "input":{"commands":["pwd"]}
    });
    let shell_result = json!({
        "type":"openrouter_shell_tool_result",
        "tool_use_id":"srvtoolu_1",
        "output":[{"stdout":"/workspace","stderr":"","outcome":{"type":"exit","exit_code":0}}]
    });
    let frames = [
        json!({"type":"message_start","message":{"model":"any/model","container":{"id":"container_123","expires_at":1_900_000_000,"skills":[]},"usage":{"input_tokens":1}}}),
        json!({"type":"content_block_start","index":0,"content_block":server_use.clone()}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":shell_result.clone()}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderEvent { protocol: ProtocolFamily::AnthropicMessages, payload }
            if payload["message"]["container"]["id"] == "container_123"
    )));
    let blocks = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ProviderContent { block, value, .. } => Some((*block, value.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(blocks, vec![(0, server_use), (1, shell_result)]);
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
}

struct Counts {
    resolver: AtomicUsize,
    auth: AtomicUsize,
    transport: AtomicUsize,
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            resolver: AtomicUsize::new(0),
            auth: AtomicUsize::new(0),
            transport: AtomicUsize::new(0),
        }
    }
}

struct CountingResolver(Arc<Counts>);

#[async_trait]
impl lingxi_llm_client::AttachmentResolver for CountingResolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.0.resolver.fetch_add(1, Ordering::SeqCst);
        Ok(Bytes::from_static(b"document"))
    }
}

struct CountingAuth(Arc<Counts>);

#[async_trait]
impl Authenticator for CountingAuth {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.auth.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct CountingTransport(Arc<Counts>);

#[async_trait]
impl Transport for CountingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.transport.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "unexpected network request".into(),
        })
    }
}

#[tokio::test]
async fn missing_container_account_scope_fails_before_attachment_auth_or_transport() {
    let mut profile = profile(
        "or-responses",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    profile.auth = AuthStrategy::ApiKey;
    let counts = Arc::new(Counts::default());
    let transport = Arc::new(CountingTransport(counts.clone()));
    let mut builder = LlmClientBuilder::with_transport(transport, std::slice::from_ref(&profile));
    builder.with_attachment_resolver(Arc::new(CountingResolver(counts.clone())));
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(CountingAuth(counts.clone())));
    let client = builder.with_region(Region::International).build().unwrap();

    let mut req = request();
    req.messages[0].content.push(ContentBlock::Document {
        source: lingxi_llm_client::protocol::DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "doc".into(),
                revision: "1".into(),
                filename: "document.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 8,
            },
        },
        title: None,
    });
    req.hosted_tools.push(shell(OpenRouterShellConfig {
        environment: Some(OpenRouterShellEnvironment::ContainerReference {
            container: OpenRouterContainerRef::new(
                "container_123",
                OpenRouterContainerScope::new(
                    "or-responses",
                    "https://openrouter.ai/api/v1",
                    "tenant-a",
                )
                .unwrap(),
            )
            .unwrap(),
            network_policy: None,
        }),
        ..Default::default()
    }));

    assert!(matches!(
        client
            .chat()
            .complete(&req, &RequestOptions::default())
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(counts.resolver.load(Ordering::SeqCst), 0);
    assert_eq!(counts.auth.load(Ordering::SeqCst), 0);
    assert_eq!(counts.transport.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn shell_without_tool_search_rejects_deferred_tools_before_any_request_side_effect() {
    let mut profile = profile(
        "or-responses",
        ProtocolFamily::OpenAiResponses,
        "https://openrouter.ai/api/v1",
    );
    profile.auth = AuthStrategy::ApiKey;
    let counts = Arc::new(Counts::default());
    let transport = Arc::new(CountingTransport(counts.clone()));
    let mut builder = LlmClientBuilder::with_transport(transport, std::slice::from_ref(&profile));
    builder.with_attachment_resolver(Arc::new(CountingResolver(counts.clone())));
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(CountingAuth(counts.clone())));
    let client = builder.with_region(Region::International).build().unwrap();

    let mut req = request();
    req.messages[0].content.push(ContentBlock::Document {
        source: lingxi_llm_client::protocol::DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "doc".into(),
                revision: "1".into(),
                filename: "document.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 8,
            },
        },
        title: None,
    });
    req.tools.push(deferred_tool("find_events"));
    req.hosted_tools
        .push(shell(OpenRouterShellConfig::default()));

    assert!(matches!(
        client
            .chat()
            .complete(&req, &RequestOptions::default())
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(counts.resolver.load(Ordering::SeqCst), 0);
    assert_eq!(counts.auth.load(Ordering::SeqCst), 0);
    assert_eq!(counts.transport.load(Ordering::SeqCst), 0);
}

struct FailingTransport(AtomicUsize);

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "simulated network interruption after dispatch".into(),
        })
    }
}

#[tokio::test]
async fn a_server_tool_request_is_not_retried_on_a_sibling_connection() {
    let make_connection = |name: &str, order: u32| {
        let mut profile = profile(
            name,
            ProtocolFamily::OpenAiResponses,
            "https://openrouter.ai/api/v1",
        );
        profile.connection = ConnectionSpec {
            group: Some("openrouter-server-tools".into()),
            connection_id: Some(name.into()),
            order,
            hidden: false,
            failover: lingxi_llm_client::protocol::FailoverTriggers {
                network: true,
                server_error: true,
                ..Default::default()
            },
        };
        profile
    };
    let profiles = [
        make_connection("or-primary", 0),
        make_connection("or-secondary", 1),
    ];
    let transport = Arc::new(FailingTransport(AtomicUsize::new(0)));
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut req = request();
    req.hosted_tools.push(
        lingxi_llm_client::providers::openrouter::native::OpenRouterHostedTool::ToolSearch(
            OpenRouterToolSearchConfig::default(),
        )
        .into(),
    );

    let result = client
        .chat()
        .complete(&req, &RequestOptions::default())
        .await;
    assert!(matches!(result, Err(LlmError::Transport { .. })));
    assert_eq!(transport.0.load(Ordering::SeqCst), 1);
}
