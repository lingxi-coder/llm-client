use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, ConnectionSpec, ContentBlock, ConversationMessage,
    DocumentSource, FailoverTriggers, LlmError, MessageRole, ProtocolFamily, ProviderFileSource,
    ProviderProfile, Region, Secret, StopReason, StreamEvent, ToolChoice, ToolSpec, ToolUseId,
};
use lingxi_llm_client::providers::anthropic::types::{
    AnthropicCodeExecutionConfig, AnthropicContainerMetadata, AnthropicContainerRef,
    AnthropicContainerScope, AnthropicSkillRef, AnthropicSkillScope, AnthropicToolCaller,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, Authenticator, CodecContext, EncodeRequest, GeminiCodec, HttpRequest,
    HttpResponse, LlmClientBuilder, OpenAiChatCodec, OpenAiResponsesCodec, RequestMode,
    RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[path = "support/wire_api.rs"]
mod wire_api;

const MODEL: &str = "claude-opus-5-5";
const ENDPOINT: &str = "https://api.anthropic.com";
const ACCOUNT: &str = "workspace-a";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"anthropic", "profile_name":"anthropic", "base_url":ENDPOINT,
        "protocol":"anthropic_messages", "auth":"none", "regions":["international"],
        "models":[{"display_model":MODEL, "request_model":MODEL, "billing_model":MODEL,
            "metadata":{"input_modalities":["text","image","file"]}}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":MODEL, "max_tokens":4096,
        "messages":[{"role":"user","content":[{"type":"text","text":"Compute the mean using Python."}]}]
    })).unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            AnthropicCodeExecutionConfig::default(),
        )
        .into(),
    );
    request
}

struct ExecutionEdit<'a> {
    request: &'a mut ChatRequest,
    config: AnthropicCodeExecutionConfig,
}
impl std::ops::Deref for ExecutionEdit<'_> {
    type Target = AnthropicCodeExecutionConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
impl std::ops::DerefMut for ExecutionEdit<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.config
    }
}
impl Drop for ExecutionEdit<'_> {
    fn drop(&mut self) {
        use lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool;
        self.request
            .hosted_tools
            .iter_mut()
            .find(|tool| {
                matches!(
                    tool.native::<AnthropicHostedTool>(),
                    Some(AnthropicHostedTool::CodeExecution(_))
                )
            })
            .unwrap()
            .edit_native::<AnthropicHostedTool, _>(|tool| {
                if let AnthropicHostedTool::CodeExecution(config) = tool {
                    *config = self.config.clone();
                }
            })
            .unwrap();
    }
}
fn execution(request: &mut ChatRequest) -> ExecutionEdit<'_> {
    let config = request.hosted_anthropic_code_execution().unwrap().clone();
    ExecutionEdit { request, config }
}

fn scope() -> AnthropicContainerScope {
    AnthropicContainerScope::new("anthropic", ENDPOINT, ACCOUNT, MODEL).unwrap()
}

fn container() -> AnthropicContainerRef {
    AnthropicContainerRef::new("container_123", scope()).unwrap()
}

fn skill_scope() -> AnthropicSkillScope {
    AnthropicSkillScope::new("anthropic", ENDPOINT, ACCOUNT).unwrap()
}

fn context(profile: &ProviderProfile) -> CodecContext {
    CodecContext::new(profile, MODEL, RequestMode::Complete)
        .with_account_scope(Some(ACCOUNT))
        .with_file_scope(Some(ACCOUNT))
}

fn encode(request: &ChatRequest, context: &CodecContext) -> Result<Value, LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(EncodeRequest::new(request), context)?;
    Ok(serde_json::from_slice(&wire.body).unwrap())
}

fn file(id: &str, media_type: &str) -> ProviderFileSource {
    ProviderFileSource {
        protocol: ProtocolFamily::AnthropicMessages,
        provider_id: "anthropic".into(),
        profile_name: "anthropic".into(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            ENDPOINT,
        ),
        account_scope: Some(ACCOUNT.into()),
        expires_at: None,
        processing_status: None,
        file_id: id.into(),
        uri: None,
        media_type: Some(media_type.into()),
        purpose: None,
    }
}

fn native_call() -> Value {
    json!({"type":"server_tool_use","id":"srvtoolu_1","name":"bash_code_execution",
        "input":{"command":"printf 37 > /tmp/number.txt"},"future":{"keep":true}})
}

fn native_result() -> Value {
    json!({"type":"bash_code_execution_tool_result","tool_use_id":"srvtoolu_1",
        "content":{"type":"bash_code_execution_result","stdout":"37","stderr":"",
            "return_code":0,"content":[{"type":"bash_code_execution_output","file_id":"file_artifact"}]}})
}

fn programmatic_server_call() -> Value {
    json!({"type":"server_tool_use","id":"srvtoolu_programmatic","name":"code_execution",
        "input":{"code":"rows = await lookup({})"}})
}

fn programmatic_tool_call(caller: Value) -> Value {
    json!({"type":"tool_use","id":"toolu_programmatic","name":"lookup",
        "input":{"query":"latest records"},"caller":caller})
}

fn programmatic_tool() -> ToolSpec {
    ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "lookup".into(),
        description: "Look up records".into(),
        input_schema: json!({"type":"object","properties":{"query":{"type":"string"}}}),
        strict: false,
        defer_loading: false,
        native_options: vec![lingxi_llm_client::protocol::NativeExtension::from_typed(
            lingxi_llm_client::providers::anthropic::native::AnthropicToolOptions {
                allowed_callers: vec![AnthropicToolCaller::CodeExecution20260120],
            },
        )
        .unwrap()],
    }
}

fn response_body(container: Value) -> Value {
    json!({"id":"msg_1","model":MODEL,"role":"assistant","type":"message",
        "container":container, "content":[native_call(), native_result()],
        "stop_reason":"pause_turn", "usage":{"input_tokens":5,"output_tokens":7,
            "server_tool_use":{"code_execution_requests":1}}})
}

fn http(status: u16, body: Value) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![],
        body: body.to_string().into(),
    }
}

#[test]
fn exact_latest_tool_is_available_for_every_current_first_party_catalog_model() {
    let profiles = lingxi_llm_client::presets::builtin().unwrap();
    let profile = profiles
        .iter()
        .find(|profile| profile.provider_id.as_str() == "anthropic")
        .unwrap();
    assert!(profile.models.len() >= 12);
    for model in &profile.models {
        let mut request = request();
        request.model = model.request_model.clone();
        let context = CodecContext::for_model(profile, model, RequestMode::Complete);
        let wire = AnthropicMessagesCodec
            .encode_request(EncodeRequest::new(&request), &context)
            .unwrap_or_else(|error| panic!("catalog model {}: {error}", model.request_model));
        let body: Value = serde_json::from_slice(&wire.body).unwrap();
        assert_eq!(
            body["tools"],
            json!([{"type":"code_execution_20260521","name":"code_execution"}])
        );
        assert_eq!(body["tool_choice"], json!({"type":"auto"}));
        assert!(body.get("container").is_none());
        assert!(!wire
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("anthropic-beta")
                && value.contains("code-execution")));
    }
}

#[test]
fn execution_rejects_unverified_models_routes_and_other_provider_codecs() {
    let profile = profile();
    for model in [
        "claude-opus-4-1",
        "claude-opus-5-5-unverified",
        "unlisted-model",
    ] {
        assert!(matches!(
            encode(
                &request(),
                &CodecContext::new(&profile, model, RequestMode::Complete)
            ),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    for endpoint in [
        "https://proxy.example.test",
        "https://api.anthropic.com/custom",
        "https://api.anthropic.com?x=1",
        "https://api.anthropic.com:444",
        "https://user@api.anthropic.com",
    ] {
        let mut other = profile.clone();
        other.base_url = endpoint.into();
        assert!(
            matches!(
                encode(&request(), &context(&other)),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "{endpoint}"
        );
    }
    for (protocol, codec) in [
        (
            ProtocolFamily::OpenAiChat,
            &OpenAiChatCodec as &dyn WireCodec,
        ),
        (
            ProtocolFamily::OpenAiResponses,
            &OpenAiResponsesCodec as &dyn WireCodec,
        ),
        (
            ProtocolFamily::GeminiGenerateContent,
            &GeminiCodec as &dyn WireCodec,
        ),
    ] {
        let mut other = profile.clone();
        other.protocol = protocol;
        assert!(matches!(
            codec.encode_request(EncodeRequest::new(&request()), &context(&other)),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    for protocol in [
        ProtocolFamily::BedrockClaude,
        ProtocolFamily::VertexClaude,
        ProtocolFamily::FoundryClaude,
    ] {
        let mut other = profile.clone();
        other.protocol = protocol;
        assert!(matches!(
            encode(&request(), &context(&other)),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[test]
fn container_reuse_is_native_top_level_and_checks_scope_after_deserialization() {
    let normalized = AnthropicContainerScope::new(
        "anthropic",
        "https://API.ANTHROPIC.COM:443/",
        ACCOUNT,
        MODEL,
    )
    .unwrap();
    assert_eq!(normalized.endpoint(), ENDPOINT);
    let mut request = request();
    execution(&mut request).container =
        Some(AnthropicContainerRef::new("container_123", normalized).unwrap());
    let profile = profile();
    let body = encode(&request, &context(&profile)).unwrap();
    assert_eq!(body["container"], "container_123");
    assert!(!body.to_string().contains(ACCOUNT));
    assert!(body["tools"][0].get("container").is_none());
    assert!(encode(
        &request,
        &CodecContext::new(&profile, MODEL, RequestMode::Complete)
    )
    .is_err());
    assert!(encode(
        &request,
        &context(&profile).with_account_scope(Some("workspace-b"))
    )
    .is_err());
    assert!(encode(
        &request,
        &CodecContext::new(&profile, "claude-sonnet-4-6", RequestMode::Complete)
            .with_account_scope(Some(ACCOUNT))
    )
    .is_err());
    let mut other_profile = profile.clone();
    other_profile.profile_name = "sibling".into();
    assert!(encode(&request, &context(&other_profile)).is_err());
    for (field, value) in [
        ("profile_name", ""),
        ("account_scope", ""),
        ("request_model", ""),
        ("endpoint", "https://evil.test"),
    ] {
        let mut raw = serde_json::to_value(container()).unwrap();
        raw["scope"][field] = json!(value);
        execution(&mut request).container = Some(serde_json::from_value(raw).unwrap());
        assert!(
            matches!(
                encode(&request, &context(&profile)),
                Err(LlmError::InvalidRequest { .. })
            ),
            "{field}"
        );
    }
}

#[test]
fn skills_encode_in_the_scoped_container_with_current_version_shapes() {
    let mut request_with_skills = request();
    execution(&mut request_with_skills).skills = vec![
        AnthropicSkillRef::anthropic("pptx").with_version("20251013"),
        AnthropicSkillRef::custom("skill_01AbCdEfGhIjKlMnOpQrStUv", skill_scope())
            .with_version("skver_01AbCdEfGhIjKlMnOpQrStUv"),
    ];
    execution(&mut request_with_skills).container = Some(container());

    let profile = profile();
    let context = context(&profile);
    let body = encode(&request_with_skills, &context).unwrap();
    assert_eq!(
        body["container"],
        json!({
            "id":"container_123",
            "skills":[
                {"type":"anthropic","skill_id":"pptx","version":"20251013"},
                {"type":"custom","skill_id":"skill_01AbCdEfGhIjKlMnOpQrStUv","version":"skver_01AbCdEfGhIjKlMnOpQrStUv"}
            ]
        })
    );
    assert!(!body.to_string().contains(ACCOUNT));
    let wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request_with_skills), &context)
        .unwrap();
    assert!(!wire
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));

    let mut new_container = request();
    execution(&mut new_container).skills =
        vec![AnthropicSkillRef::anthropic("pdf").with_version("latest")];
    let body = encode(&new_container, &context).unwrap();
    assert_eq!(
        body["container"],
        json!({"skills":[{"type":"anthropic","skill_id":"pdf","version":"latest"}]})
    );

    let haiku_context = CodecContext::new(&profile, "claude-haiku-4-5", RequestMode::Complete)
        .with_account_scope(Some(ACCOUNT));
    assert!(encode(&new_container, &haiku_context).is_ok());
}

#[test]
fn skills_preflight_limits_versions_and_custom_scope_before_encoding() {
    let profile = profile();
    let context = context(&profile);

    let mut at_limit = request();
    execution(&mut at_limit).skills = (0..20)
        .map(|index| AnthropicSkillRef::anthropic(format!("skill-{index}")))
        .collect();
    assert!(encode(&at_limit, &context).is_ok());

    let mut too_many = request();
    execution(&mut too_many).skills = (0..21)
        .map(|index| AnthropicSkillRef::anthropic(format!("skill-{index}")))
        .collect();
    assert!(matches!(
        encode(&too_many, &context),
        Err(LlmError::InvalidRequest { .. })
    ));

    for version in ["", "not-a-version", "20251013"] {
        let mut invalid = request();
        execution(&mut invalid).skills =
            vec![
                AnthropicSkillRef::custom("skill_01AbCdEfGhIjKlMnOpQrStUv", skill_scope())
                    .with_version(version),
            ];
        let result = encode(&invalid, &context);
        assert!(
            matches!(result, Err(LlmError::InvalidRequest { .. })),
            "version {version:?}"
        );
    }

    let skill = AnthropicSkillRef::custom("skill_01AbCdEfGhIjKlMnOpQrStUv", skill_scope())
        .with_version("skill_version_legacy_id");
    let persisted: AnthropicSkillRef =
        serde_json::from_value(serde_json::to_value(&skill).unwrap()).unwrap();
    assert_eq!(persisted, skill);

    let mut account_mismatch = request();
    execution(&mut account_mismatch).skills = vec![AnthropicSkillRef::custom(
        "skill_01AbCdEfGhIjKlMnOpQrStUv",
        skill_scope(),
    )];
    assert!(matches!(
        encode(
            &account_mismatch,
            &CodecContext::new(&profile, MODEL, RequestMode::Complete)
                .with_account_scope(Some("another-workspace"))
        ),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut wrong_scope = request();
    execution(&mut wrong_scope).skills = vec![AnthropicSkillRef::custom(
        "skill_01AbCdEfGhIjKlMnOpQrStUv",
        AnthropicSkillScope::new("sibling-profile", ENDPOINT, ACCOUNT).unwrap(),
    )];
    assert!(matches!(
        encode(&wrong_scope, &context),
        Err(LlmError::InvalidRequest { .. })
    ));

    let unscoped_custom: AnthropicSkillRef = serde_json::from_value(json!({
        "type":"custom","skill_id":"skill_01AbCdEfGhIjKlMnOpQrStUv"
    }))
    .unwrap();
    execution(&mut wrong_scope).skills = vec![unscoped_custom];
    assert!(matches!(
        encode(&wrong_scope, &context),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn uploaded_files_preserve_user_order_and_accept_sandbox_file_types() {
    let mut request = request();
    request.messages[0]
        .content
        .push(ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({"type":"text","text":"Keep this native context","future":{"x":1}}),
        });
    execution(&mut request).files = vec![
        file("file_csv", "text/csv"),
        file(
            "file_sheet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        file("file_json", "application/json"),
    ];
    let body = encode(&request, &context(&profile())).unwrap();
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["text"], "Compute the mean using Python.");
    assert_eq!(content[1]["future"], json!({"x":1}));
    assert_eq!(
        &content[2..],
        &[
            json!({"type":"container_upload","file_id":"file_csv"}),
            json!({"type":"container_upload","file_id":"file_sheet"}),
            json!({"type":"container_upload","file_id":"file_json"}),
        ]
    );
    assert!(!body.to_string().contains(ACCOUNT));
}

#[test]
fn uploaded_file_scope_role_and_duplicate_validation_cannot_be_bypassed() {
    let mut request = request();
    execution(&mut request).files = vec![file("file_csv", "text/csv")];
    assert!(encode(&request, &context(&profile()).with_file_scope(None)).is_err());
    assert!(encode(
        &request,
        &context(&profile()).with_file_scope(Some("other-account"))
    )
    .is_err());
    for field in [
        "protocol",
        "provider_id",
        "profile_name",
        "endpoint_fingerprint",
        "account_scope",
    ] {
        let mut raw = serde_json::to_value(file("file_csv", "text/csv")).unwrap();
        raw[field] = if field == "protocol" {
            json!("open_ai_chat")
        } else {
            json!("wrong")
        };
        execution(&mut request).files = vec![serde_json::from_value(raw).unwrap()];
        assert!(encode(&request, &context(&profile())).is_err(), "{field}");
    }
    execution(&mut request).files =
        vec![file("file_csv", "text/csv"), file("file_csv", "text/csv")];
    assert!(matches!(
        encode(&request, &context(&profile())),
        Err(LlmError::InvalidRequest { .. })
    ));
    execution(&mut request).files.truncate(1);
    request.messages[0]
        .content
        .push(ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({"type":"container_upload","file_id":"file_csv"}),
        });
    assert!(matches!(
        encode(&request, &context(&profile())),
        Err(LlmError::InvalidRequest { .. })
    ));
    request.messages[0].content.pop();
    request.messages[0].role = MessageRole::Assistant;
    assert!(matches!(
        encode(&request, &context(&profile())),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn ambiguous_raw_controls_duplicate_tools_and_function_name_collisions_fail() {
    let mut request = request();
    let mut profile = profile();
    for raw in [
        json!({"container":"container_raw"}),
        json!({"tools":[{"type":"code_execution_20260521","name":"code_execution"}]}),
        json!({"tools":[]}),
    ] {
        profile.extra = json!({"body":raw});
        assert!(matches!(
            encode(&request, &context(&profile)),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    profile.extra = json!({});
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            AnthropicCodeExecutionConfig::default(),
        )
        .into(),
    );
    assert!(matches!(
        encode(&request, &context(&profile)),
        Err(LlmError::InvalidRequest { .. })
    ));
    request.hosted_tools.pop();
    request.tools.push(ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "code_execution".into(),
        description: "collision".into(),
        input_schema: json!({"type":"object"}),
        strict: false,
        defer_loading: false,
        native_options: Vec::new(),
    });
    assert!(matches!(
        encode(&request, &context(&profile)),
        Err(LlmError::InvalidRequest { .. })
    ));
    request.hosted_tools.clear();
    request.tools.clear();
    profile.extra = json!({"body":{"container":"container_raw"}});
    assert!(encode(&request, &context(&profile)).is_err());
}

#[test]
fn programmatic_callers_encode_and_preflight_documented_combinations() {
    let profile = profile();
    let mut request = request();
    request.tools.push(programmatic_tool());
    let body = encode(&request, &context(&profile)).unwrap();
    assert_eq!(
        body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "lookup")
            .unwrap()["allowed_callers"],
        json!(["code_execution_20260120"])
    );

    let mut direct_choice = request.clone();
    direct_choice.tools[0].set_anthropic_allowed_callers(vec![
        AnthropicToolCaller::Direct,
        AnthropicToolCaller::CodeExecution20260521,
    ]);
    direct_choice.tool_choice = ToolChoice::Tool {
        name: "lookup".into(),
    };
    let body = encode(&direct_choice, &context(&profile)).unwrap();
    assert_eq!(
        body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "lookup")
            .unwrap()["allowed_callers"],
        json!(["direct", "code_execution_20260521"])
    );

    let mut no_execution = request.clone();
    no_execution.hosted_tools.clear();
    assert!(matches!(
        encode(&no_execution, &context(&profile)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut haiku = request.clone();
    haiku.model = "claude-haiku-4-5-20251001".into();
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(
            EncodeRequest::new(&haiku),
            &CodecContext::new(&profile, "claude-haiku-4-5-20251001", RequestMode::Complete)
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut strict = request.clone();
    strict.tools[0].strict = true;
    assert!(matches!(
        encode(&strict, &context(&profile)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut recursive = request.clone();
    recursive.tools[0].input_schema = json!({
        "type":"object",
        "$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}}}},
        "properties":{"root":{"$ref":"#/$defs/Node"}}
    });
    assert!(matches!(
        encode(&recursive, &context(&profile)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut forced_programmatic = request.clone();
    forced_programmatic.tool_choice = ToolChoice::Tool {
        name: "lookup".into(),
    };
    assert!(matches!(
        encode(&forced_programmatic, &context(&profile)),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut parallel_disabled = profile.clone();
    parallel_disabled.extra = json!({
        "body":{"tool_choice":{"type":"auto","disable_parallel_tool_use":true}}
    });
    assert!(matches!(
        encode(&request, &context(&parallel_disabled)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut aliases_duplicated = request.clone();
    aliases_duplicated.tools[0].set_anthropic_allowed_callers(vec![
        AnthropicToolCaller::CodeExecution20260120,
        AnthropicToolCaller::CodeExecution20260521,
    ]);
    assert!(matches!(
        encode(&aliases_duplicated, &context(&profile)),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut openai_profile = profile.clone();
    openai_profile.provider_id = "openai".into();
    openai_profile.profile_name = "openai".into();
    openai_profile.base_url = "https://api.openai.com/v1".into();
    openai_profile.protocol = ProtocolFamily::OpenAiChat;
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(&openai_profile, MODEL, RequestMode::Complete)
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut caller_history: ChatRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role":"user","content":[{"type":"text","text":"resume"}]}]
    }))
    .unwrap();
    caller_history
        .messages
        .push(ConversationMessage::assistant(vec![
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: programmatic_server_call(),
            },
            ContentBlock::ToolUse {
                input_json: None,
                id: ToolUseId::new("toolu_programmatic"),
                name: "lookup".into(),
                input: json!({"query":"latest records"}),
                provider_id: None,
                caller: Some(json!({
                    "type":"code_execution_20260120",
                    "tool_id":"srvtoolu_programmatic"
                })),
                toolset_name: None,
                thought_signature: None,
            },
        ]));
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            EncodeRequest::new(&caller_history),
            &CodecContext::new(&openai_profile, MODEL, RequestMode::Complete)
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn programmatic_caller_metadata_replays_without_duplicate_client_tool_calls() {
    let caller = json!({
        "type":"code_execution_20260120",
        "tool_id":"srvtoolu_programmatic",
        "future":{"preserve":true}
    });
    let mut body = response_body(json!({"id":"container_123","expires_at":"later"}));
    body["content"] = json!([
        programmatic_server_call(),
        programmatic_tool_call(caller.clone())
    ]);
    let response = AnthropicMessagesCodec
        .decode_response(&http(200, body), &context(&profile()))
        .unwrap();
    assert_eq!(response.message.tool_uses().count(), 1);
    assert!(matches!(
        &response.message.content[0],
        ContentBlock::ProviderContent { value, .. }
            if value == &programmatic_server_call()
    ));
    assert!(matches!(
        &response.message.content[1],
        ContentBlock::ToolUse { caller: Some(actual), .. } if actual == &caller
    ));

    let mut replay_request = request();
    replay_request.tools.push(programmatic_tool());
    execution(&mut replay_request).container = Some(
        response
            .anthropic_container()
            .unwrap()
            .reference_for(scope())
            .unwrap(),
    );
    replay_request.messages.push(response.message);
    replay_request.messages.push(ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            cache_reference: None,
            output_json: None,
            tool_use_id: ToolUseId::new("toolu_programmatic"),
            content: "42".into(),
            is_error: Some(false),
            blocks: None,
            toolset_name: None,
        }],
    });
    let encoded = encode(&replay_request, &context(&profile())).unwrap();
    assert_eq!(
        encoded["messages"][1]["content"],
        json!([
            programmatic_server_call(),
            programmatic_tool_call(caller.clone())
        ])
    );
    assert_eq!(
        encoded["messages"][2]["content"][0]["tool_use_id"],
        "toolu_programmatic"
    );

    let malformed = json!({"type":"code_execution_20260120","tool_id":17});
    let mut malformed_body = response_body(json!({"id":"container_123"}));
    malformed_body["content"] = json!([
        programmatic_server_call(),
        programmatic_tool_call(malformed.clone())
    ]);
    let malformed_response = AnthropicMessagesCodec
        .decode_response(&http(200, malformed_body), &context(&profile()))
        .unwrap();
    assert!(matches!(
        &malformed_response.message.content[1],
        ContentBlock::ToolUse { caller: Some(actual), .. } if actual == &malformed
    ));
    let mut malformed_replay = request();
    malformed_replay.tools.push(programmatic_tool());
    malformed_replay.messages.push(malformed_response.message);
    execution(&mut malformed_replay).container = Some(container());
    assert!(matches!(
        encode(&malformed_replay, &context(&profile())),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut no_container = request();
    no_container.tools.push(programmatic_tool());
    no_container
        .messages
        .push(ConversationMessage::assistant(vec![
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: programmatic_server_call(),
            },
            ContentBlock::ToolUse {
                input_json: None,
                id: ToolUseId::new("toolu_programmatic"),
                name: "lookup".into(),
                input: json!({"query":"latest records"}),
                provider_id: None,
                caller: Some(caller),
                toolset_name: None,
                thought_signature: None,
            },
        ]));
    assert!(matches!(
        encode(&no_container, &context(&profile())),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn completed_programmatic_history_does_not_force_old_tools_or_container_on_later_turns() {
    let caller = json!({
        "type":"code_execution_20260120",
        "tool_id":"srvtoolu_programmatic"
    });
    let mut request = request();
    request.hosted_tools.clear();
    request.messages.push(ConversationMessage::assistant(vec![
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: programmatic_server_call(),
        },
        ContentBlock::ToolUse {
            input_json: None,
            id: ToolUseId::new("toolu_programmatic"),
            name: "lookup".into(),
            input: json!({"query":"latest records"}),
            provider_id: None,
            caller: Some(caller.clone()),
            toolset_name: None,
            thought_signature: None,
        },
    ]));
    request.messages.push(ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::ToolResult {
            cache_reference: None,
            output_json: None,
            tool_use_id: ToolUseId::new("toolu_programmatic"),
            content: "42".into(),
            is_error: Some(false),
            blocks: None,
            toolset_name: None,
        }],
    });
    request.messages.push(ConversationMessage::assistant(vec![
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"code_execution_tool_result",
                "tool_use_id":"srvtoolu_programmatic",
                "content":{"type":"code_execution_result","stdout":"42","stderr":"","return_code":0,"content":[]}
            }),
        },
    ]));
    request
        .messages
        .push(ConversationMessage::user_text("Start a new lookup."));

    let encoded = encode(&request, &context(&profile())).unwrap();
    assert_eq!(encoded["tools"], Value::Null);
    assert_eq!(encoded["messages"][1]["content"][1]["caller"], caller);
}

#[test]
fn synchronous_container_metadata_usage_and_paused_native_content_round_trip() {
    let envelope = json!({"id":"container_123", "expires_at":"2000-01-01T00:00:00Z", "skills":[], "unknown":{"preserved":true}});
    let response = AnthropicMessagesCodec
        .decode_response(
            &http(200, response_body(envelope.clone())),
            &context(&profile()),
        )
        .unwrap();
    let response: lingxi_llm_client::protocol::ChatResponse =
        serde_json::from_value(serde_json::to_value(&response).unwrap()).unwrap();
    assert_eq!(response.anthropic_container().unwrap().envelope, envelope);
    assert!(response.openrouter_container().is_none());
    assert_eq!(
        response.anthropic_usage().unwrap(),
        &json!({
            "input_tokens":5,"output_tokens":7,"server_tool_use":{"code_execution_requests":1}
        })
    );
    assert_eq!(response.stop_reason, StopReason::Other("pause_turn".into()));
    assert_eq!(response.message.tool_uses().count(), 0);
    assert_eq!(
        response
            .usage
            .usage
            .unwrap()
            .server_tool_usage
            .unwrap()
            .code_interpreter_requests,
        Some(1)
    );
    let mut request = request();
    execution(&mut request).container = Some(
        response
            .anthropic_container()
            .unwrap()
            .reference_for(scope())
            .unwrap(),
    );
    request.messages.push(response.message);
    let body = encode(&request, &context(&profile())).unwrap();
    // Rolling expiry is retained, not treated as a local rejection or creation request.
    assert_eq!(body["container"], "container_123");
    assert_eq!(
        body["messages"][1]["content"],
        json!([native_call(), native_result()])
    );
}

#[test]
fn metadata_import_rejects_missing_or_invalid_container_ids() {
    for envelope in [
        json!({}),
        json!({"id":null}),
        json!({"id":""}),
        json!({"id":"bad\nid"}),
    ] {
        assert!(AnthropicContainerMetadata { envelope }
            .reference_for(scope())
            .is_err());
    }
}

fn execution_frames(final_container: Value) -> Vec<Value> {
    vec![
        json!({"type":"message_start","message":{"model":MODEL,"container":{"id":"container_initial","expires_at":"2026-09-26T00:00:00Z"},"usage":{"input_tokens":5}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"bash_code_execution","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"pwd\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":native_result()}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"container":final_container,"stop_reason":"pause_turn"},"usage":{"output_tokens":7,"server_tool_use":{"code_execution_requests":1}}}),
        json!({"type":"message_stop"}),
    ]
}

#[test]
fn streaming_emits_native_blocks_and_both_container_envelopes_without_client_calls() {
    let final_container =
        json!({"id":"container_123","expires_at":"2026-09-27T00:00:00Z","native":{"keep":1}});
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&context(&profile()));
    let mut events = Vec::new();
    for frame in execution_frames(final_container.clone()) {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ProviderEvent { payload, .. } if payload["message"]["container"]["id"] == "container_initial")));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ProviderEvent { payload, .. } if payload["delta"]["container"] == final_container)));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ProviderContent { block:0, value, .. } if value["input"] == json!({"command":"pwd"}))));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ProviderContent { block:1, value, .. } if *value == native_result())));
    assert!(!events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { .. })));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::End { stop_reason:StopReason::Other(reason), usage, .. } if reason == "pause_turn" && usage.usage.unwrap().server_tool_usage.unwrap().code_interpreter_requests == Some(1))));
}

#[test]
fn streaming_programmatic_tool_call_exposes_caller_once_and_keeps_server_execution_native() {
    let caller = json!({
        "type":"code_execution_20260120",
        "tool_id":"srvtoolu_programmatic",
        "future":{"preserve":true}
    });
    let frames = vec![
        json!({"type":"message_start","message":{"model":MODEL,"usage":{"input_tokens":5}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_programmatic","name":"code_execution","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"code\":\"rows = await lookup({})\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_programmatic","name":"lookup","input":{},"caller":caller.clone()}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"latest records\"}"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":7}}),
        json!({"type":"message_stop"}),
    ];
    let mut decoder = AnthropicMessagesCodec.stream_decoder(&context(&profile()));
    let mut events = Vec::new();
    for frame in frames {
        events.extend(wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap());
    }
    events.extend(wire_api::finish(&mut *decoder).unwrap());

    let tool_deltas = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCallDelta {
                id,
                caller: Some(actual),
                ..
            } => Some((id.as_str(), actual)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_deltas.len(), 2);
    assert!(tool_deltas
        .iter()
        .all(|(id, actual)| { *id == "toolu_programmatic" && *actual == &caller }));
    let unique_calls = tool_deltas
        .iter()
        .map(|(id, _)| *id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique_calls.len(), 1);
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderContent { value, .. }
            if value == &programmatic_server_call()
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderContent { value, .. }
            if value["type"] == "tool_use"
    )));
    let encoded = serde_json::to_value(
        events
            .iter()
            .find(|event| {
                matches!(
                    event,
                    StreamEvent::ToolCallDelta {
                        caller: Some(_),
                        ..
                    }
                )
            })
            .unwrap(),
    )
    .unwrap();
    assert_eq!(encoded["caller"], caller);
}

enum Reply {
    Http(HttpResponse),
    Stream(Vec<Result<Bytes, LlmError>>),
    Error(LlmError),
}

#[derive(Default)]
struct MockTransport {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<Reply>>,
}
impl MockTransport {
    fn with(replies: Vec<Reply>) -> Self {
        Self {
            requests: Mutex::new(vec![]),
            replies: Mutex::new(replies.into()),
        }
    }
}
#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected second submission")
        {
            Reply::Http(response) => Ok(StreamResponse {
                status: response.status,
                headers: response.headers,
                body: stream::iter(vec![Ok(response.body)]).boxed(),
            }),
            Reply::Stream(chunks) => Ok(StreamResponse {
                status: 200,
                headers: vec![],
                body: stream::iter(chunks).boxed(),
            }),
            Reply::Error(error) => Err(error),
        }
    }
}

fn sse(frames: Vec<Value>) -> Vec<Result<Bytes, LlmError>> {
    frames
        .into_iter()
        .map(|frame| Ok(Bytes::from(format!("data: {frame}\n\n"))))
        .collect()
}

#[tokio::test]
async fn high_level_stream_observes_replaced_null_and_interrupted_container_metadata() {
    for latest in [
        json!({"id":"container_final","expires_at":"old","future":7}),
        Value::Null,
    ] {
        let transport = Arc::new(MockTransport::with(vec![Reply::Stream(sse(
            execution_frames(latest.clone()),
        ))]));
        let client = LlmClientBuilder::with_transport(transport, &[profile()])
            .with_region(Region::International)
            .build()
            .unwrap();
        let mut model_stream = client
            .chat()
            .stream(&request(), &RequestOptions::default())
            .await
            .unwrap();
        while let Some(event) = model_stream.next().await {
            event.unwrap();
        }
        assert_eq!(
            model_stream
                .anthropic_container()
                .map(|metadata| metadata.envelope.clone()),
            (!latest.is_null()).then_some(latest)
        );
        assert_eq!(
            model_stream.anthropic_usage(),
            Some(&json!({
                "input_tokens":5,"output_tokens":7,"server_tool_use":{"code_execution_requests":1}
            }))
        );
    }
    let mut chunks = sse(execution_frames(Value::Null).into_iter().take(1).collect());
    chunks.push(Err(LlmError::StreamInterrupted {
        message: "wire dropped".into(),
    }));
    let transport = Arc::new(MockTransport::with(vec![Reply::Stream(chunks)]));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut model_stream = client
        .chat()
        .stream(&request(), &RequestOptions::default())
        .await
        .unwrap();
    let mut saw_error = false;
    while let Some(event) = model_stream.next().await {
        if event.is_err() {
            saw_error = true;
        }
    }
    assert!(saw_error);
    assert_eq!(
        model_stream.anthropic_container().unwrap().envelope["id"],
        "container_initial"
    );
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn high_level_two_turn_loop_supplies_account_and_retains_original_context() {
    let envelope = json!({"id":"container_123","expires_at":"2026-09-27T00:00:00Z"});
    let transport = Arc::new(MockTransport::with(vec![
        Reply::Http(http(200, response_body(envelope.clone()))),
        Reply::Http(http(200, response_body(envelope))),
    ]));
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        account_scope: Some(ACCOUNT.into()),
        ..Default::default()
    };
    let mut request = request();
    let first = client.chat().complete(&request, &options).await.unwrap();
    execution(&mut request).container = Some(
        first
            .anthropic_container()
            .unwrap()
            .reference_for(scope())
            .unwrap(),
    );
    request.messages.push(first.message);
    let second = client.chat().complete(&request, &options).await.unwrap();
    assert_eq!(second.stop_reason, StopReason::Other("pause_turn".into()));
    let requests = transport.requests.lock().unwrap();
    let first: Value = serde_json::from_slice(&requests[0].body).unwrap();
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(first.get("container").is_none());
    assert_eq!(second["container"], "container_123");
    assert_eq!(second["messages"][0], first["messages"][0]);
    assert_eq!(
        second["messages"][1]["content"],
        json!([native_call(), native_result()])
    );
}

fn failover_profiles() -> Vec<ProviderProfile> {
    ["anthropic", "sibling"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let mut profile = profile();
            profile.profile_name = name.into();
            profile.connection = ConnectionSpec {
                group: Some("anthropic-execution".into()),
                connection_id: Some(name.into()),
                order: index as u32,
                hidden: false,
                failover: FailoverTriggers {
                    network: true,
                    server_error: true,
                    ..Default::default()
                },
            };
            profile
        })
        .collect()
}

#[tokio::test]
async fn uncertain_submission_and_server_errors_never_repeat_on_a_sibling() {
    for reply in [
        Reply::Error(LlmError::Transport {
            message: "unknown outcome after write".into(),
        }),
        Reply::Http(http(
            503,
            json!({"error":{"type":"overloaded_error","message":"busy"}}),
        )),
    ] {
        let transport = Arc::new(MockTransport::with(vec![reply]));
        let client = LlmClientBuilder::with_transport(transport.clone(), &failover_profiles())
            .with_region(Region::International)
            .build()
            .unwrap();
        assert!(client
            .chat()
            .complete(&request(), &RequestOptions::default())
            .await
            .is_err());
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn expired_container_errors_do_not_silently_create_replacement_containers() {
    let transport = Arc::new(MockTransport::with(vec![Reply::Http(http(
        400,
        json!({"error":{"type":"invalid_request_error","message":"container expired"}}),
    ))]));
    let client = LlmClientBuilder::with_transport(transport.clone(), &failover_profiles())
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    execution(&mut request).container = Some(container());
    let options = RequestOptions {
        account_scope: Some(ACCOUNT.into()),
        ..Default::default()
    };
    assert!(matches!(
        client.chat().complete(&request, &options).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["container"], "container_123");
}

#[derive(Default)]
struct Counts {
    resolver: AtomicUsize,
    auth: AtomicUsize,
}
struct Resolver(Arc<Counts>);
#[async_trait]
impl lingxi_llm_client::AttachmentResolver for Resolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.0.resolver.fetch_add(1, Ordering::SeqCst);
        Ok(Bytes::from_static(b"document"))
    }
}
struct Auth(Arc<Counts>);
#[async_trait]
impl Authenticator for Auth {
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

#[tokio::test]
async fn missing_or_wrong_account_fails_before_resolver_auth_or_http() {
    let counts = Arc::new(Counts::default());
    let transport = Arc::new(MockTransport::default());
    let mut profile = profile();
    profile.auth = AuthStrategy::ApiKey;
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(Resolver(counts.clone())));
    builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(Auth(counts.clone())));
    let client = builder.with_region(Region::International).build().unwrap();
    let mut request = request();
    execution(&mut request).container = Some(container());
    request.messages[0].content.push(ContentBlock::Document {
        source: DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "doc".into(),
                revision: "1".into(),
                filename: "a.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 8,
            },
        },
        title: None,
    });
    for account_scope in [None, Some("other-account".into())] {
        let options = RequestOptions {
            account_scope,
            ..Default::default()
        };
        assert!(matches!(
            client.chat().complete(&request, &options).await,
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    assert_eq!(counts.resolver.load(Ordering::SeqCst), 0);
    assert_eq!(counts.auth.load(Ordering::SeqCst), 0);
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[test]
fn custom_skill_workspace_binding_checks_the_effective_chat_header() {
    let mut configured_profile = profile();
    let workspace_scope = skill_scope().with_workspace_id("wrkspc_a").unwrap();
    let mut request = request();
    execution(&mut request).skills =
        vec![AnthropicSkillRef::custom("skill_alpha", workspace_scope)];
    assert!(encode(&request, &context(&configured_profile)).is_err());
    configured_profile.extra["headers"] = json!({"Anthropic-Workspace-Id":"wrkspc_a"});
    let wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request), &context(&configured_profile))
        .unwrap();
    assert!(wire.headers.iter().any(|(name, value)| name
        .eq_ignore_ascii_case("anthropic-workspace-id")
        && value == "wrkspc_a"));
    for headers in [
        json!({"anthropic-workspace-id":"wrkspc_b"}),
        json!({"anthropic-workspace-id":"wrkspc_a", "Anthropic-Workspace-Id":"wrkspc_a"}),
        json!({"anthropic-workspace-id":null}),
        json!({"anthropic-workspace-id":"wrkspc_a\n"}),
    ] {
        configured_profile.extra["headers"] = headers;
        assert!(encode(&request, &context(&configured_profile)).is_err());
    }
}
