use lingxi_llm_client::protocol::*;
use lingxi_llm_client::{CodecContext, EncodeRequest, RequestMode, VertexClaudeCodec, WireCodec};
use serde_json::{json, Value};

const MODEL: &str = "claude-opus-5-5";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "vertex",
        "base_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/test/locations/us-central1",
        "protocol": "vertex_claude",
        "auth": "none",
        "models": [{"display_model": MODEL, "request_model": MODEL, "billing_model": MODEL}],
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role":"user","content":[{"type":"text","text":"Hello"}]}]
    }))
    .unwrap()
}

fn encode(
    request: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let wire = VertexClaudeCodec.encode_request(
        EncodeRequest::new(request),
        &CodecContext::new(profile, model, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

fn has_beta(headers: &[(String, String)], beta: &str) -> bool {
    headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value.split(',').any(|candidate| candidate.trim() == beta)
    })
}

fn system_text(text: &str) -> ConversationMessage {
    ConversationMessage::system_text(text)
}

fn system(content: Vec<ContentBlock>) -> ConversationMessage {
    ConversationMessage {
        role: MessageRole::System,
        content,
        anthropic: None,
    }
}

fn native(value: Value) -> ContentBlock {
    ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value,
    }
}

#[test]
fn documented_vertex_models_accept_mid_conversation_system_messages() {
    let supported = [
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5-5",
        "claude-opus-4-8",
        "claude-opus-5",
    ];
    let mut request = request();
    request.messages.push(system_text("Use metric units."));

    for model in supported {
        let (body, _) = encode(&request, &profile(), model).unwrap();
        assert_eq!(body["messages"][1]["role"], "system", "{model}");
        assert_eq!(
            body["messages"][1]["content"][0]["text"],
            "Use metric units."
        );
    }
    for unsupported in [
        "claude-sonnet-5",
        "claude-opus-4-6",
        "claude-opus-5-50",
        "claude-fable-5.1",
        "claude-opus-4-5@20251101",
    ] {
        assert!(
            encode(&request, &profile(), unsupported).is_err(),
            "{unsupported}"
        );
    }
}

#[test]
fn vertex_clear_at_and_per_message_effort_use_their_beta_headers() {
    let mut scoped = request();
    scoped.messages.push(
        ConversationMessage::system_text("Use concise answers.").with_anthropic_options(
            AnthropicMessageOptions {
                clear_at: Some(AnthropicClearAt::NextUserMessage),
                effort: None,
            },
        ),
    );
    let (body, headers) = encode(&scoped, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["clear_at"], "next_user_message");
    assert!(has_beta(
        &headers,
        "mid-conversation-system-clear-at-2026-08-21"
    ));

    let mut effort = request();
    let mut empty_system =
        ConversationMessage::system_text("").with_anthropic_options(AnthropicMessageOptions {
            clear_at: None,
            effort: Some(AnthropicMessageEffort::High),
        });
    empty_system.content.clear();
    effort.messages.insert(0, empty_system);
    let (body, headers) = encode(&effort, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][0]["output_config"]["effort"], "high");
    assert!(has_beta(
        &headers,
        "mid-conversation-output-config-2026-07-01"
    ));

    for model in [
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
    ] {
        assert!(encode(&effort, &profile(), model).is_ok(), "{model}");
    }
    for model in ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5"] {
        assert!(encode(&effort, &profile(), model).is_err(), "{model}");
    }
}

#[test]
fn vertex_tool_changes_are_reference_only_and_use_the_cloud_beta() {
    let mut request = request();
    request.tools = serde_json::from_value(json!([{
        "name":"lookup",
        "description":"Look up a record",
        "input_schema":{"type":"object","properties":{}}
    }]))
    .unwrap();
    request.messages.push(system(vec![
        native(json!({
            "type":"tool_addition",
            "tool":{"type":"tool_reference","name":"lookup"}
        })),
        native(json!({
            "type":"tool_removal",
            "tool":{"type":"tool_reference","name":"lookup"}
        })),
    ]));

    let (body, headers) = encode(&request, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["type"], "tool_addition");
    assert!(has_beta(
        &headers,
        "mid-conversation-tool-changes-2026-07-01"
    ));
    assert!(!has_beta(&headers, "inline-tools-2026-09-15"));
}

#[test]
fn vertex_rejects_inline_custom_definitions_and_mcp_toolsets() {
    let mut custom = request();
    custom.messages.push(system(vec![native(json!({
        "type":"tool_addition",
        "tool":{"type":"tool_definition","definition":{
            "name":"lookup",
            "description":"Look up a record",
            "input_schema":{"type":"object","properties":{}}
        }}
    }))]));
    assert!(encode(&custom, &profile(), MODEL).is_err());

    let mut mcp = request();
    let server = AnthropicMcpConfig::new("calendar", "https://mcp.example.test/calendar")
        .unwrap()
        .with_inline_toolset(true);
    mcp.messages
        .push(server.inline_tool_addition_message().unwrap());
    mcp.hosted_tools.push(HostedTool::AnthropicMcp(server));
    assert!(encode(&mcp, &profile(), MODEL).is_err());
}
