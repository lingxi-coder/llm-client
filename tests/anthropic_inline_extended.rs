use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::protocol::{
    AnthropicCodeExecutionConfig, AnthropicContainerRef, AnthropicContainerScope,
    AnthropicWebFetchConfig, CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, ContentBlock,
    ConversationMessage, HostedTool, LlmError, MessageRole, PromptCachePolicy, ProtocolFamily,
    ProviderFileSource, ProviderProfile, Region, StreamEvent, ToolChoice, ToolSpec, ToolUseId,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, HttpRequest, LlmClientBuilder,
    RequestMode, RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const MODEL: &str = "claude-opus-5-5";
type Encoded = (Value, Vec<(String, String)>);

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic", "profile_name":"anthropic",
        "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages",
        "auth":"none", "regions":["international"],
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap()
}

fn inline_failover_profiles() -> [ProviderProfile; 2] {
    std::array::from_fn(|index| {
        let mut value = serde_json::to_value(profile()).unwrap();
        value["profile_name"] = json!(format!("anthropic-inline-{index}"));
        value["connection"] = json!({
            "group":"anthropic-inline",
            "order":index,
            "hidden":false,
            "failover":{
                "rateLimit":true,
                "overloaded":true,
                "serverError":true,
                "network":true,
                "auth":true
            }
        });
        serde_json::from_value(value).unwrap()
    })
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":MODEL,
        "max_tokens":1024,
        "messages":[{"role":"user","content":[{"type":"text","text":"Continue."}]}]
    }))
    .unwrap()
}

fn encode(request: &ChatRequest) -> Result<Encoded, LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(request),
        &CodecContext::new(&profile(), MODEL, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

fn encode_with_account(request: &ChatRequest, account: &str) -> Result<Encoded, LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(request),
        &CodecContext::new(&profile(), MODEL, RequestMode::Complete)
            .with_account_scope(Some(account)),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

fn encode_with_profile(
    request: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<Encoded, LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(request),
        &CodecContext::new(profile, model, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

fn native(value: Value) -> ContentBlock {
    ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value,
    }
}

fn system(blocks: Vec<ContentBlock>) -> ConversationMessage {
    ConversationMessage {
        anthropic: None,
        role: MessageRole::System,
        content: blocks,
    }
}

fn addition(definition: Value) -> ContentBlock {
    native(json!({
        "type":"tool_addition",
        "tool":{"type":"tool_definition","definition":definition}
    }))
}

fn schema() -> Value {
    json!({
        "type":"object",
        "properties":{"id":{"type":"string"}},
        "required":["id"],
        "additionalProperties":false
    })
}

fn custom_definition(name: &str) -> Value {
    json!({
        "name":name,
        "description":"Look up a record",
        "input_schema":schema()
    })
}

fn code_execution_definition() -> Value {
    json!({"type":"code_execution_20260521","name":"code_execution"})
}

fn inline_tool(name: &str, input_schema: Value, strict: bool) -> ContentBlock {
    let mut definition = custom_definition(name);
    definition["input_schema"] = input_schema;
    if strict {
        definition["strict"] = json!(true);
    }
    addition(definition)
}

fn spec(name: String, input_schema: Value, strict: bool) -> ToolSpec {
    ToolSpec {
        name,
        description: "Tool".into(),
        input_schema,
        strict,
        defer_loading: false,
        allowed_callers: Vec::new(),
    }
}

fn strict_schema_with_optional_fields(count: usize) -> Value {
    let properties = (0..count)
        .map(|index| (format!("field_{index}"), json!({"type":"string"})))
        .collect::<serde_json::Map<_, _>>();
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

fn code_execution_request() -> ChatRequest {
    let mut request = request();
    request
        .hosted_tools
        .push(HostedTool::AnthropicCodeExecution(
            AnthropicCodeExecutionConfig::default(),
        ));
    request
}

fn programmatic_definition(callers: Value) -> Value {
    let mut definition = custom_definition("lookup");
    definition["allowed_callers"] = callers;
    definition
}

fn programmatic_server_call() -> ContentBlock {
    native(json!({
        "type":"server_tool_use",
        "id":"srvtoolu_inline",
        "name":"code_execution",
        "input":{"code":"await lookup({'id': '1'})"}
    }))
}

fn programmatic_tool_call(caller_type: &str) -> ContentBlock {
    ContentBlock::ToolUse {
        id: ToolUseId::new("toolu_inline"),
        name: "lookup".into(),
        input: json!({"id":"1"}),
        provider_id: None,
        caller: Some(json!({
            "type":caller_type,
            "tool_id":"srvtoolu_inline"
        })),
        toolset_name: None,
        thought_signature: None,
    }
}

fn user_tool_result() -> ConversationMessage {
    ConversationMessage {
        anthropic: None,
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_inline"),
            content: "found".into(),
            is_error: false,
            blocks: None,
            toolset_name: None,
        }],
    }
}

#[derive(Default)]
struct CountingTransport(AtomicUsize);

#[async_trait]
impl Transport for CountingTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "unexpected dispatch".into(),
        })
    }
}

#[derive(Default)]
struct ClientResponseTransport(Mutex<Vec<HttpRequest>>);

#[async_trait]
impl Transport for ClientResponseTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let request_body: Value = serde_json::from_slice(&request.body).unwrap();
        self.0.lock().unwrap().push(request);
        let body = if request_body["stream"] == true {
            let frames = [
                json!({
                    "type":"message_start",
                    "message":{
                        "id":"msg_inline",
                        "type":"message",
                        "role":"assistant",
                        "model":MODEL,
                        "content":[],
                        "stop_reason":null,
                        "usage":{"input_tokens":3,"output_tokens":0},
                        "container":{"id":"container_inline"}
                    }
                }),
                json!({
                    "type":"content_block_start",
                    "index":0,
                    "content_block":{"type":"text","text":""}
                }),
                json!({
                    "type":"content_block_delta",
                    "index":0,
                    "delta":{"type":"text_delta","text":"continued"}
                }),
                json!({"type":"content_block_stop","index":0}),
                json!({
                    "type":"message_delta",
                    "delta":{"stop_reason":"end_turn"},
                    "usage":{"output_tokens":2}
                }),
                json!({"type":"message_stop"}),
            ];
            frames
                .into_iter()
                .map(|frame| format!("data: {}\n\n", serde_json::to_string(&frame).unwrap()))
                .collect::<String>()
                .into_bytes()
        } else {
            serde_json::to_vec(&json!({
                "id":"msg_inline",
                "type":"message",
                "role":"assistant",
                "model":MODEL,
                "content":[{"type":"text","text":"continued"}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":3,"output_tokens":2},
                "container":{"id":"container_inline"}
            }))
            .unwrap()
        };
        Ok(StreamResponse {
            status: 200,
            headers: Vec::new(),
            body: futures::stream::iter([Ok(Bytes::from(body))]).boxed(),
        })
    }
}

fn completed_programmatic_history(initial_definition: Value) -> ChatRequest {
    let mut request = code_execution_request();
    request
        .messages
        .push(system(vec![addition(initial_definition)]));
    request.messages.push(ConversationMessage::assistant(vec![
        programmatic_server_call(),
        programmatic_tool_call("code_execution_20260120"),
    ]));
    request.messages.push(user_tool_result());
    request
}

#[test]
fn strict_inline_custom_tool_uses_shared_schema_validation_and_beta() {
    let mut strict_request = request();
    let mut definition = custom_definition("lookup");
    definition["strict"] = json!(true);
    strict_request
        .messages
        .push(system(vec![addition(definition.clone())]));

    let (body, headers) = encode(&strict_request).unwrap();
    assert_eq!(
        body["messages"][1]["content"][0]["tool"]["definition"],
        definition
    );
    assert!(headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value
                .split(',')
                .any(|beta| beta.trim() == "inline-tools-2026-09-15")
    }));

    let mut unsupported_schema = schema();
    unsupported_schema["patternProperties"] = json!({"^x": {"type":"string"}});
    let mut invalid = request();
    invalid.messages.push(system(vec![inline_tool(
        "lookup",
        unsupported_schema,
        true,
    )]));
    assert!(matches!(
        encode(&invalid),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn strict_inline_schemas_share_request_wide_count_and_complexity_limits() {
    let mut exactly_twenty_tools = request();
    exactly_twenty_tools.tools = (0..19)
        .map(|index| spec(format!("top_{index}"), schema(), true))
        .collect();
    exactly_twenty_tools
        .messages
        .push(system(vec![inline_tool("inline", schema(), true)]));
    assert!(encode(&exactly_twenty_tools).is_ok());

    let mut too_many_tools = request();
    too_many_tools.tools = (0..20)
        .map(|index| spec(format!("top_{index}"), schema(), true))
        .collect();
    too_many_tools
        .messages
        .push(system(vec![inline_tool("inline", schema(), true)]));
    assert!(matches!(
        encode(&too_many_tools),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut too_many_optional = request();
    too_many_optional.tools.push(spec(
        "top".into(),
        strict_schema_with_optional_fields(13),
        true,
    ));
    too_many_optional.messages.push(system(vec![inline_tool(
        "inline",
        strict_schema_with_optional_fields(12),
        true,
    )]));
    assert!(matches!(
        encode(&too_many_optional),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut exactly_twenty_four_optional = request();
    exactly_twenty_four_optional.tools.push(spec(
        "top".into(),
        strict_schema_with_optional_fields(12),
        true,
    ));
    exactly_twenty_four_optional
        .messages
        .push(system(vec![inline_tool(
            "inline",
            strict_schema_with_optional_fields(12),
            true,
        )]));
    assert!(encode(&exactly_twenty_four_optional).is_ok());
}

#[test]
fn inline_programmatic_callers_require_typed_code_execution_and_valid_schemas() {
    let mut valid_request = code_execution_request();
    valid_request
        .messages
        .push(system(vec![addition(programmatic_definition(json!([
            "code_execution_20260521"
        ])))]));
    assert!(encode(&valid_request).is_ok());

    let mut missing_code_execution = request();
    missing_code_execution
        .messages
        .push(system(vec![addition(programmatic_definition(json!([
            "code_execution_20260521"
        ])))]));
    assert!(matches!(
        encode(&missing_code_execution),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut strict_programmatic = code_execution_request();
    let mut definition = programmatic_definition(json!(["code_execution_20260120"]));
    definition["strict"] = json!(true);
    strict_programmatic
        .messages
        .push(system(vec![addition(definition)]));
    assert!(matches!(
        encode(&strict_programmatic),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut recursive_programmatic = code_execution_request();
    let mut definition = programmatic_definition(json!(["code_execution_20260120"]));
    definition["input_schema"] = json!({
        "type":"object",
        "$defs":{"node":{"type":"object","properties":{"child":{"$ref":"#/$defs/node"}}}},
        "properties":{"root":{"$ref":"#/$defs/node"}}
    });
    recursive_programmatic
        .messages
        .push(system(vec![addition(definition)]));
    assert!(matches!(
        encode(&recursive_programmatic),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn pending_inline_programmatic_call_resumes_with_scoped_container() {
    let account = "workspace-a";
    let scope =
        AnthropicContainerScope::new("anthropic", "https://api.anthropic.com", account, MODEL)
            .unwrap();
    let container = AnthropicContainerRef::new("container_inline", scope).unwrap();
    let mut request = code_execution_request();
    let execution = request
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap();
    execution.container = Some(container);
    request
        .messages
        .push(system(vec![addition(programmatic_definition(json!([
            "code_execution_20260120"
        ])))]));
    request.messages.push(ConversationMessage::assistant(vec![
        programmatic_server_call(),
        programmatic_tool_call("code_execution_20260120"),
    ]));
    request.messages.push(user_tool_result());

    assert!(encode_with_account(&request, account).is_ok());
}

#[tokio::test]
async fn full_client_preflights_pending_inline_ptc_for_complete_and_stream() {
    let account = "workspace-a";
    let scope =
        AnthropicContainerScope::new("anthropic", "https://api.anthropic.com", account, MODEL)
            .unwrap();
    let container = AnthropicContainerRef::new("container_inline", scope).unwrap();
    let mut request = code_execution_request();
    let execution = request
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap();
    execution.container = Some(container);
    request
        .messages
        .push(system(vec![addition(programmatic_definition(json!([
            "code_execution_20260120"
        ])))]));
    request.messages.push(ConversationMessage::assistant(vec![
        programmatic_server_call(),
        programmatic_tool_call("code_execution_20260120"),
    ]));
    request.messages.push(user_tool_result());

    let transport = Arc::new(ClientResponseTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        account_scope: Some(account.into()),
        ..Default::default()
    };
    let completed = client.chat().complete(&request, &options).await.unwrap();
    assert!(completed
        .message
        .content
        .iter()
        .any(|block| { matches!(block, ContentBlock::Text { text, .. } if text == "continued") }));

    let mut stream = client.chat().stream(&request, &options).await.unwrap();
    let mut saw_end = false;
    while let Some(event) = stream.next().await {
        if matches!(event.unwrap(), StreamEvent::End { .. }) {
            saw_end = true;
        }
    }
    assert!(saw_end);

    let sent = transport.0.lock().unwrap();
    assert_eq!(sent.len(), 2);
    for (index, request) in sent.iter().enumerate() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["container"], "container_inline");
        assert_eq!(
            body["messages"][1]["content"][0]["tool"]["definition"]["name"],
            "lookup"
        );
        assert_eq!(body["stream"].as_bool().unwrap_or(false), index == 1);
    }
}

#[test]
fn programmatic_history_uses_the_definition_active_at_each_tool_call() {
    let mut completed_history =
        completed_programmatic_history(programmatic_definition(json!(["code_execution_20260120"])));
    completed_history
        .messages
        .push(ConversationMessage::assistant(vec![native(json!({
            "type":"code_execution_tool_result",
            "tool_use_id":"srvtoolu_inline",
            "content":{"type":"code_execution_result","stdout":"found","stderr":"","return_code":0,"content":[]}
        }))]));
    completed_history
        .messages
        .push(ConversationMessage::user_text("Continue."));
    assert!(encode(&completed_history).is_ok());

    let mut future_definition =
        completed_programmatic_history(programmatic_definition(json!(["direct"])));
    let next = programmatic_definition(json!(["code_execution_20260120"]));
    future_definition
        .messages
        .push(system(vec![addition(next)]));
    future_definition.messages.push(ConversationMessage::assistant(vec![native(
        json!({
            "type":"code_execution_tool_result",
            "tool_use_id":"srvtoolu_inline",
            "content":{"type":"code_execution_result","stdout":"found","stderr":"","return_code":0,"content":[]}
        }),
    )]));
    future_definition
        .messages
        .push(ConversationMessage::user_text("Continue."));
    assert!(matches!(
        encode(&future_definition),
        Err(LlmError::InvalidRequest { .. })
    ));

    for last_change in [
        native(json!({
            "type":"tool_removal",
            "tool":{"type":"tool_reference","name":"lookup"}
        })),
        addition(programmatic_definition(json!(["direct"]))),
    ] {
        let mut pending = code_execution_request();
        pending
            .messages
            .push(system(vec![addition(programmatic_definition(json!([
                "code_execution_20260120"
            ])))]));
        pending.messages.push(ConversationMessage::assistant(vec![
            programmatic_server_call(),
            programmatic_tool_call("code_execution_20260120"),
        ]));
        pending.messages.push(user_tool_result());
        pending.messages.push(system(vec![last_change]));
        assert!(matches!(
            encode(&pending),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}

#[test]
fn provider_defined_server_tools_require_their_typed_feature_path() {
    for (kind, name) in [
        ("code_execution_20260521", "code_execution"),
        ("web_fetch_20260318", "web_fetch"),
        ("web_search_20260318", "web_search"),
        ("tool_search_tool_regex_20251119", "tool_search_tool_regex"),
        ("advisor_20260301", "advisor"),
        ("future_server_tool_20990101", "future_tool"),
    ] {
        let mut request = request();
        request.messages.push(system(vec![addition(json!({
            "type":kind,
            "name":name
        }))]));
        assert!(
            matches!(
                encode(&request),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "{kind}"
        );
    }
}

#[tokio::test]
async fn rejected_inline_server_definition_never_reaches_transport() {
    let transport = Arc::new(CountingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.messages.push(system(vec![addition(json!({
        "type":"code_execution_20260521",
        "name":"code_execution"
    }))]));

    assert!(client
        .chat()
        .complete(&request, &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.0.load(Ordering::SeqCst), 0);
}

#[test]
fn typed_server_tool_inline_placement_requires_exact_matching_config() {
    let mut execution = code_execution_request();
    let code_execution = execution
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap();
    code_execution.inline_definition = true;
    execution
        .messages
        .push(system(vec![addition(code_execution_definition())]));
    let (body, headers) = encode(&execution).unwrap();
    assert!(!body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["type"] == "code_execution_20260521"));
    assert_eq!(
        body["messages"][1]["content"][0]["tool"]["definition"],
        code_execution_definition()
    );
    assert!(headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value
                .split(',')
                .any(|beta| beta.trim() == "inline-tools-2026-09-15")
    }));

    let mut missing = code_execution_request();
    missing
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap()
        .inline_definition = true;
    assert!(matches!(
        encode(&missing),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut mismatch = code_execution_request();
    mismatch
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap()
        .inline_definition = true;
    let mut wrong_version = code_execution_definition();
    wrong_version["type"] = json!("code_execution_20260120");
    mismatch
        .messages
        .push(system(vec![addition(wrong_version)]));
    assert!(matches!(
        encode(&mismatch),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut fetch = request();
    let mut inline_fetch_config = AnthropicWebFetchConfig {
        max_uses: Some(3),
        ..Default::default()
    };
    inline_fetch_config.inline_definition = true;
    fetch
        .hosted_tools
        .push(HostedTool::AnthropicWebFetch(inline_fetch_config));
    fetch.messages.push(system(vec![addition(json!({
        "type":"web_fetch_20260318",
        "name":"web_fetch",
        "max_uses":3
    }))]));
    let (body, _) = encode(&fetch).unwrap();
    assert!(!body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "web_fetch"));
    assert_eq!(
        body["messages"][1]["content"][0]["tool"]["definition"],
        json!({"type":"web_fetch_20260318","name":"web_fetch","max_uses":3})
    );

    let mut wrong_fetch = fetch;
    wrong_fetch.messages[1].content = vec![addition(json!({
        "type":"web_fetch_20260318",
        "name":"web_fetch",
        "max_uses":4
    }))];
    assert!(matches!(
        encode(&wrong_fetch),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[tokio::test]
async fn inline_typed_server_tools_are_first_party_only_and_model_gated_before_file_preparation() {
    let mut execution = code_execution_request();
    let config = execution
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap();
    config.inline_definition = true;
    config.files.push(ProviderFileSource {
        protocol: ProtocolFamily::AnthropicMessages,
        provider_id: "anthropic".into(),
        profile_name: "anthropic".into(),
        endpoint_fingerprint: "intentionally-stale".into(),
        account_scope: Some("workspace-a".into()),
        file_id: "file-stale".into(),
        uri: None,
        expires_at: None,
        processing_status: None,
        media_type: Some("text/csv".into()),
        purpose: None,
    });
    execution
        .messages
        .push(system(vec![addition(code_execution_definition())]));
    let mut unsupported_model = profile();
    unsupported_model.models[0].display_model = "claude-haiku-4-5".into();
    unsupported_model.models[0].request_model = "claude-haiku-4-5".into();
    unsupported_model.models[0].billing_model = "claude-haiku-4-5".into();
    execution.model = "claude-haiku-4-5".into();
    let transport = Arc::new(CountingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[unsupported_model])
        .with_region(Region::International)
        .build()
        .unwrap();
    let result = client
        .chat()
        .complete(&execution, &RequestOptions::default())
        .await;
    assert!(matches!(
        result,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(transport.0.load(Ordering::SeqCst), 0);

    let mut foundry = profile();
    foundry.provider_id = "microsoft-foundry".into();
    foundry.profile_name = "foundry-anthropic".into();
    foundry.base_url = "https://example.services.ai.azure.com".into();
    foundry.protocol = ProtocolFamily::FoundryClaude;
    let mut foundry_request = execution.clone();
    foundry_request.model = MODEL.into();
    assert!(matches!(
        encode_with_profile(&foundry_request, &foundry, MODEL),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut foundry_fetch = request();
    foundry_fetch
        .hosted_tools
        .push(HostedTool::AnthropicWebFetch(AnthropicWebFetchConfig {
            inline_definition: true,
            ..Default::default()
        }));
    foundry_fetch.messages.push(system(vec![addition(json!({
        "type":"web_fetch_20260318",
        "name":"web_fetch"
    }))]));
    assert!(matches!(
        encode_with_profile(&foundry_fetch, &foundry, MODEL),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[tokio::test]
async fn inline_hosted_execution_and_fetch_keep_no_retry_routing_pinned() {
    let profiles = inline_failover_profiles();
    let transport = Arc::new(CountingTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();

    let mut execution = code_execution_request();
    execution
        .hosted_tools
        .iter_mut()
        .find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
        .unwrap()
        .inline_definition = true;
    execution
        .messages
        .push(system(vec![addition(code_execution_definition())]));
    assert!(client
        .chat()
        .complete(&execution, &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.0.load(Ordering::SeqCst), 1);

    let mut fetch = request();
    fetch
        .hosted_tools
        .push(HostedTool::AnthropicWebFetch(AnthropicWebFetchConfig {
            inline_definition: true,
            ..Default::default()
        }));
    fetch.messages.push(system(vec![addition(json!({
        "type":"web_fetch_20260318",
        "name":"web_fetch"
    }))]));
    assert!(client
        .chat()
        .complete(&fetch, &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.0.load(Ordering::SeqCst), 2);
}

#[test]
fn inline_web_fetch_cache_marker_is_counted_at_its_message_position_only() {
    let mut request = request();
    request.tools = (0..3)
        .map(|index| spec(format!("tool_{index}"), schema(), false))
        .collect();
    request
        .hosted_tools
        .push(HostedTool::AnthropicWebFetch(AnthropicWebFetchConfig {
            cache_control: Some(CacheTtl::FiveMinutes),
            inline_definition: true,
            ..Default::default()
        }));
    let mut definition = json!({
        "type":"web_fetch_20260318",
        "name":"web_fetch",
    });
    definition["cache_control"] = json!({"type":"ephemeral"});
    request.messages.push(system(vec![addition(definition)]));
    request.prompt_cache = PromptCachePolicy {
        breakpoints: (0..3)
            .map(|index| CacheBreakpoint {
                position: CachePosition::Tool { index },
                ttl: CacheTtl::FiveMinutes,
            })
            .collect(),
        ..Default::default()
    };

    let (body, _) = encode(&request).unwrap();
    assert_eq!(
        body["messages"][1]["content"][0]["tool"]["definition"]["cache_control"]["type"],
        "ephemeral"
    );

    request.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(matches!(
        encode(&request),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn typed_tool_choice_is_not_changed_by_inline_custom_tools() {
    let mut request = request();
    request.tool_choice = ToolChoice::None;
    request
        .messages
        .push(system(vec![inline_tool("lookup", schema(), false)]));
    let (body, _) = encode(&request).unwrap();
    assert_eq!(body["tool_choice"]["type"], "none");
}
