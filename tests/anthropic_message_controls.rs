use lingxi_llm_client::protocol::*;
use lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestKind;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::*;
use serde_json::{json, Value};
use std::sync::Arc;

#[path = "support/json_payload_fixture.rs"]
mod json_payload_fixture;
#[path = "support/mod.rs"]
mod support;

const MODEL: &str = "claude-opus-5-5";
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"anthropic","profile_name":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","regions":["international"],"models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]})).unwrap()
}
fn req() -> ChatRequest {
    serde_json::from_value(json!({"model":MODEL,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]})).unwrap()
}
fn encode(
    r: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let wire = AnthropicMessagesCodec.encode_request(
        EncodeRequest::new(r),
        &CodecContext::new(p, model, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}
fn encode_with_native_endpoint_admission(
    r: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
    admission: &NativeAnthropicEndpointAdmission,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let context = CodecContext::new(p, model, RequestMode::Complete)
        .with_native_anthropic_endpoint_admission(Some(admission));
    let wire = AnthropicMessagesCodec.encode_request(EncodeRequest::new(r), &context)?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}
fn scoped() -> ConversationMessage {
    ConversationMessage::system_text("Use concise answers").with_anthropic_options(
        AnthropicMessageOptions {
            clear_at: Some(AnthropicClearAt::NextUserMessage),
            effort: None,
        },
    )
}
fn effort(level: AnthropicMessageEffort) -> ConversationMessage {
    let mut m =
        ConversationMessage::system_text("").with_anthropic_options(AnthropicMessageOptions {
            clear_at: None,
            effort: Some(level),
        });
    m.content.clear();
    m
}
fn has_beta(headers: &[(String, String)], beta: &str) -> bool {
    headers
        .iter()
        .any(|(n, v)| n.eq_ignore_ascii_case("anthropic-beta") && v.split(',').any(|t| t == beta))
}
#[test]
fn scoped_reminder_stays_verbatim_after_later_user_turn() {
    let mut r = req();
    r.messages.push(scoped());
    let (before, headers) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(before["messages"][1]["clear_at"], "next_user_message");
    assert!(before["messages"][1].get("anthropic").is_none());
    assert!(has_beta(
        &headers,
        "mid-conversation-system-clear-at-2026-08-21"
    ));
    r.messages
        .push(ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "Answer".into(),
            thought_signature: None,
            citations: None,
        }]));
    r.messages.push(ConversationMessage::user_text("Next"));
    let before_history = serde_json::to_value(&r.messages).unwrap();
    let (after, _) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(after["messages"][1], before["messages"][1]);
    assert_eq!(serde_json::to_value(&r.messages).unwrap(), before_history);
}
#[test]
fn never_is_preserved_and_can_carry_effort_and_text() {
    let mut r = req();
    r.messages.push(
        ConversationMessage::system_text("Reminder").with_anthropic_options(
            AnthropicMessageOptions {
                clear_at: Some(AnthropicClearAt::Never),
                effort: Some(AnthropicMessageEffort::Low),
            },
        ),
    );
    let (body, headers) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][1]["clear_at"], "never");
    assert_eq!(body["messages"][1]["output_config"]["effort"], "low");
    assert!(has_beta(
        &headers,
        "mid-conversation-system-clear-at-2026-08-21"
    ));
    assert!(has_beta(&headers, "per-turn-control-2026-07-01"));
}

#[test]
fn request_local_native_admission_is_bound_to_the_exact_first_party_profile() {
    let custom_endpoint: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"anthropic",
        "profile_name":"selected-native-loopback",
        "base_url":"http://127.0.0.1:43123",
        "protocol":"anthropic_messages",
        "auth":"none",
        "regions":["international"],
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap();
    let mut request = req();
    request.messages.push(effort(AnthropicMessageEffort::High));

    // The configured Anthropic Messages identity supports ordinary message
    // controls without Native endpoint admission. Native-only wire shaping
    // remains gated on the matching request-local token.
    let (ordinary_body, ordinary_headers) = encode(&request, &custom_endpoint, MODEL).unwrap();
    assert_eq!(
        ordinary_body["messages"][1]["output_config"]["effort"],
        "high"
    );
    assert!(ordinary_body.get("tools").is_none());
    assert!(has_beta(&ordinary_headers, "per-turn-control-2026-07-01"));

    let admission = NativeAnthropicEndpointAdmission::selected_first_party_messages_endpoint(
        &custom_endpoint.profile_name,
        &custom_endpoint.base_url,
    );
    let (body, headers) =
        encode_with_native_endpoint_admission(&request, &custom_endpoint, MODEL, &admission)
            .unwrap();
    assert_eq!(body["messages"][1]["output_config"]["effort"], "high");
    assert_eq!(body["tools"], json!([]));
    let admitted_context = CodecContext::new(&custom_endpoint, MODEL, RequestMode::Complete)
        .with_native_anthropic_endpoint_admission(Some(&admission));
    assert_eq!(
        AnthropicMessagesCodec
            .request_inference(&request, &admitted_context)
            .unwrap()
            .requested_effort,
        Some(ReasoningEffort::High)
    );
    let side_query_context = admitted_context.clone().with_anthropic_request_kind(
        lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestKind::SideQuery,
    );
    assert_eq!(
        AnthropicMessagesCodec
            .request_inference(&request, &side_query_context)
            .unwrap()
            .requested_effort,
        None
    );
    assert!(!has_beta(&headers, "per-turn-control-2026-07-01"));

    let wrong_base = NativeAnthropicEndpointAdmission::selected_first_party_messages_endpoint(
        &custom_endpoint.profile_name,
        "http://127.0.0.1:43124",
    );
    let (wrong_base_body, wrong_base_headers) =
        encode_with_native_endpoint_admission(&request, &custom_endpoint, MODEL, &wrong_base)
            .unwrap();
    assert_eq!(
        wrong_base_body["messages"][1]["output_config"]["effort"],
        "high"
    );
    assert!(wrong_base_body.get("tools").is_none());
    assert!(has_beta(&wrong_base_headers, "per-turn-control-2026-07-01"));

    let wrong_provider: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"compatible-anthropic",
        "profile_name":"selected-native-loopback",
        "base_url":"http://127.0.0.1:43123",
        "protocol":"anthropic_messages",
        "auth":"none",
        "regions":["international"],
        "models":[{"display_model":MODEL,"request_model":MODEL,"billing_model":MODEL}]
    }))
    .unwrap();
    assert!(
        encode_with_native_endpoint_admission(&request, &wrong_provider, MODEL, &admission)
            .is_err()
    );
}

#[tokio::test]
async fn client_prepare_dispatch_preserves_admitted_per_message_effort_to_wire() {
    let mut selected_profile = profile();
    selected_profile.profile_name = "selected-native-loopback".into();
    selected_profile.base_url = "http://127.0.0.1:43123".into();
    let client = LlmClientBuilder::with_transport(
        Arc::new(support::NoHttp),
        std::slice::from_ref(&selected_profile),
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    let admission = NativeAnthropicEndpointAdmission::selected_first_party_messages_endpoint(
        &selected_profile.profile_name,
        &selected_profile.base_url,
    );
    let options = RequestOptions {
        native_anthropic_endpoint_admission: Some(admission),
        ..RequestOptions::default()
    };
    let mut request = req();
    request.messages.push(effort(AnthropicMessageEffort::High));

    let draft = client
        .prepare_draft_on(
            &selected_profile.profile_name,
            &request,
            &options,
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let body = draft.semantic_body_source().unwrap();
    assert_eq!(
        body.string_at("/messages/1/output_config/effort")
            .as_deref(),
        Some("high")
    );
    assert_eq!(body.array_len("/tools"), Some(0));
    assert!(!draft.request().headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value
                .split(',')
                .any(|beta| beta == "per-turn-control-2026-07-01")
    }));
}

#[test]
fn native_main_wire_keeps_empty_tools_and_tool_result_shape() {
    let selected_profile = profile();
    let admission = NativeAnthropicEndpointAdmission::selected_first_party_messages_endpoint(
        &selected_profile.profile_name,
        &selected_profile.base_url,
    );
    let result = |is_error| ContentBlock::ToolResult {
        cache_reference: None,
        tool_use_id: ToolUseId::new("toolu_native"),
        content: JsonPayload::string("ok"),
        is_error,
        blocks: None,
        toolset_name: None,
    };
    let mut request = req();
    request.messages[0].content = vec![result(None)];

    let native = CodecContext::new(&selected_profile, MODEL, RequestMode::Complete)
        .with_native_anthropic_endpoint_admission(Some(&admission));
    let native_wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request), &native)
        .unwrap();
    let native_body = String::from_utf8(native_wire.body.to_vec()).unwrap();
    assert!(native_body.contains("\"tools\":[]"));
    assert!(native_body
        .contains("\"tool_use_id\":\"toolu_native\",\"type\":\"tool_result\",\"content\":\"ok\""));
    assert!(!native_body.contains("\"is_error\":false"));

    let compatible = CodecContext::new(&selected_profile, MODEL, RequestMode::Complete);
    let compatible_wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request), &compatible)
        .unwrap();
    let compatible_body = String::from_utf8(compatible_wire.body.to_vec()).unwrap();
    assert!(compatible_body
        .contains("\"type\":\"tool_result\",\"tool_use_id\":\"toolu_native\",\"content\":\"ok\""));
    assert!(!compatible_body.contains("\"is_error\":false"));
    assert!(!compatible_body.contains("\"tools\":[]"));

    let mut bash_request = req();
    bash_request.messages[0].content = vec![result(Some(false))];
    let bash_wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&bash_request), &native)
        .unwrap();
    let bash_body = String::from_utf8(bash_wire.body.to_vec()).unwrap();
    assert!(bash_body.contains(
        "\"tool_use_id\":\"toolu_native\",\"type\":\"tool_result\",\"content\":\"ok\",\"is_error\":false"
    ));

    let side_query = CodecContext::new(&selected_profile, MODEL, RequestMode::Complete)
        .with_native_anthropic_endpoint_admission(Some(&admission))
        .with_anthropic_request_kind(
            lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestKind::SideQuery,
        );
    let side_wire = AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&bash_request), &side_query)
        .unwrap();
    let side_body = String::from_utf8(side_wire.body.to_vec()).unwrap();
    assert!(!side_body.contains("\"tools\":[]"));
    assert!(side_body.contains(
        "\"type\":\"tool_result\",\"tool_use_id\":\"toolu_native\",\"content\":\"ok\",\"is_error\":false"
    ));
}

#[test]
fn scoped_messages_reject_cache_effort_tool_changes_and_wrong_roles() {
    let mut r = req();
    r.messages.push(scoped());
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.prompt_cache.breakpoints.clear();
    r.messages[1].content = vec![ContentBlock::ProviderContent {
        protocol: ProtocolFamily::AnthropicMessages,
        value: json_payload_fixture::generated(
            &json!({"type":"text","text":"reminder","cache_control":{"type":"ephemeral"}}),
        ),
    }];
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.messages[1] = scoped();
    r.messages[1].native_options[0]
        .edit::<AnthropicMessageOptions, _>(|options| {
            options.effort = Some(AnthropicMessageEffort::High)
        })
        .unwrap();
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.messages[1] = scoped();
    r.messages[1].content.push(
        AnthropicToolChange::add_reference(AnthropicToolReference::tool("lookup"))
            .into_content_block(),
    );
    assert!(encode(&r, &profile(), MODEL).is_err());
    for role in [MessageRole::User, MessageRole::Assistant] {
        r.messages[1] = scoped();
        r.messages[1].role = role;
        assert!(encode(&r, &profile(), MODEL).is_err());
    }
    r.messages[1] = scoped();
    r.messages[1].content.clear();
    assert!(encode(&r, &profile(), MODEL).is_err());
}
#[test]
fn automatic_cache_does_not_insert_breakpoint_into_scoped_message() {
    let mut r = req();
    r.messages.push(scoped());
    r.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    let (body, _) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["cache_control"], json!({"type":"ephemeral"}));
    assert!(body["messages"][1]["content"][0]
        .get("cache_control")
        .is_none());
}
#[test]
fn effort_only_groups_have_position_exception_but_mixed_groups_do_not() {
    let mut r = req();
    r.messages.insert(0, effort(AnthropicMessageEffort::Low));
    r.messages.push(ConversationMessage::assistant(vec![]));
    r.messages.push(effort(AnthropicMessageEffort::High));
    r.messages.push(ConversationMessage::user_text("Followup"));
    let (body, headers) = encode(&r, &profile(), MODEL).unwrap();
    assert_eq!(body["messages"][0]["content"], json!([]));
    assert_eq!(
        body["messages"][0]["output_config"],
        json!({"effort":"low"})
    );
    assert!(has_beta(&headers, "per-turn-control-2026-07-01"));
    r.messages
        .insert(1, ConversationMessage::system_text("Text in same group"));
    assert!(encode(&r, &profile(), MODEL).is_err());
    r.messages = vec![
        ConversationMessage::user_text("User"),
        effort(AnthropicMessageEffort::Low),
        ConversationMessage::system_text("Text"),
        ConversationMessage::assistant(vec![]),
    ];
    assert!(encode(&r, &profile(), MODEL).is_ok());
}
#[test]
fn effort_models_and_levels_are_exact() {
    let mut r = req();
    r.messages.insert(0, effort(AnthropicMessageEffort::Low));
    for model in [
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
    ] {
        assert!(encode(&r, &profile(), model).is_ok(), "{model}");
    }
    for model in [
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
    ] {
        assert!(encode(&r, &profile(), model).is_err(), "{model}");
    }
    for (level, wire) in [
        (AnthropicMessageEffort::Low, "low"),
        (AnthropicMessageEffort::Medium, "medium"),
        (AnthropicMessageEffort::High, "high"),
        (AnthropicMessageEffort::XHigh, "xhigh"),
        (AnthropicMessageEffort::Max, "max"),
    ] {
        r.messages[0] = effort(level);
        assert_eq!(
            encode(&r, &profile(), MODEL).unwrap().0["messages"][0]["output_config"]["effort"],
            wire
        );
    }
}
#[test]
fn effort_report_changes_only_at_next_user_without_rewriting_top_level() {
    let mut r = req();
    r.thinking = Some(ThinkingConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    r.messages.push(effort(AnthropicMessageEffort::Low));
    let p = profile();
    let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
    assert_eq!(
        AnthropicMessagesCodec
            .request_inference(&r, &ctx)
            .unwrap()
            .requested_effort,
        Some(ReasoningEffort::High)
    );
    r.messages.push(ConversationMessage::assistant(vec![]));
    r.messages.push(ConversationMessage::user_text("Next"));
    assert_eq!(
        AnthropicMessagesCodec
            .request_inference(&r, &ctx)
            .unwrap()
            .requested_effort,
        Some(ReasoningEffort::Low)
    );
    assert_eq!(
        encode(&r, &p, MODEL).unwrap().0["output_config"]["effort"],
        "high"
    );
    r.messages.push(effort(AnthropicMessageEffort::Max));
    assert_eq!(
        AnthropicMessagesCodec
            .request_inference(&r, &ctx)
            .unwrap()
            .requested_effort,
        Some(ReasoningEffort::Low)
    );
}
#[test]
fn unsupported_protocols_do_not_drop_anthropic_metadata() {
    let codecs: Vec<Box<dyn WireCodec>> = vec![
        Box::new(OpenAiChatCodec),
        Box::new(OpenAiResponsesCodec),
        Box::new(GeminiCodec),
        Box::new(AzureOpenAiCodec),
        Box::new(FoundryClaudeCodec),
        Box::new(VertexGeminiCodec),
        Box::new(BedrockClaudeCodec),
    ];
    let mut r = req();
    r.messages.push(scoped());
    for codec in codecs {
        let mut p = profile();
        p.protocol = codec.family();
        p.provider_id = "other".into();
        p.base_url = "https://other.example.test".into();
        let ctx = CodecContext::new(&p, MODEL, RequestMode::Complete);
        assert!(
            codec.validate_request(&r, &ctx).is_err(),
            "{:?}",
            codec.family()
        );
        assert!(
            codec.encode_request(EncodeRequest::new(&r), &ctx).is_err(),
            "{:?}",
            codec.family()
        );
    }
    let mut p = profile();
    p.base_url = "https://gateway.example.test".into();
    let (body, headers) = encode(&r, &p, MODEL).unwrap();
    assert_eq!(body["messages"][1]["clear_at"], "next_user_message");
    assert!(has_beta(
        &headers,
        "mid-conversation-system-clear-at-2026-08-21"
    ));
    assert!(body.get("tools").is_none());
}
#[tokio::test]
async fn invalid_message_controls_fail_before_transport() {
    struct NeverSend;
    #[async_trait::async_trait]
    impl Transport for NeverSend {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            panic!("invalid metadata must not reach transport")
        }
    }
    #[async_trait::async_trait]
    impl AttachmentResolver for NeverSend {
        async fn resolve(&self, _: &AttachmentRef) -> Result<bytes::Bytes, LlmError> {
            panic!("invalid message controls must fail before attachment resolution")
        }
    }
    let mut builder =
        LlmClientBuilder::with_transport(std::sync::Arc::new(NeverSend), &[profile()])
            .with_region(Region::International);
    builder.with_attachment_resolver(std::sync::Arc::new(NeverSend));
    let client = builder.build().unwrap();
    let mut r = req();
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
    r.messages[0] = r.messages[0]
        .clone()
        .with_anthropic_options(AnthropicMessageOptions {
            clear_at: Some(AnthropicClearAt::NextUserMessage),
            effort: None,
        });
    assert!(matches!(
        client.chat().complete(&r, &Default::default()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    r.messages[0].native_options.clear();
    r.messages.push(scoped());
    r.prompt_cache.breakpoints.push(CacheBreakpoint {
        scope: None,
        position: CachePosition::Message { index: 1, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    });
    assert!(matches!(
        client.chat().complete(&r, &Default::default()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
    r.prompt_cache.breakpoints.clear();
    r.messages.truncate(1);
    r.messages.insert(0, effort(AnthropicMessageEffort::Low));
    r.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Disabled),
        ..Default::default()
    });
    assert!(matches!(
        client.chat().complete(&r, &Default::default()).await,
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn explicit_never_keeps_effort_only_position_exception() {
    let mut r = req();
    let mut change = effort(AnthropicMessageEffort::Low);
    change.native_options[0]
        .edit::<AnthropicMessageOptions, _>(|options| {
            options.clear_at = Some(AnthropicClearAt::Never)
        })
        .unwrap();
    r.messages.insert(0, change);
    assert!(encode(&r, &profile(), MODEL).is_ok());
}
#[test]
fn active_effort_obeys_thinking_constraints_without_applying_future_changes_early() {
    let mut r = req();
    r.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Disabled),
        ..Default::default()
    });
    r.messages.push(effort(AnthropicMessageEffort::XHigh));
    assert!(encode(&r, &profile(), "claude-opus-5").is_ok());
    r.messages.push(ConversationMessage::assistant(vec![]));
    r.messages
        .push(ConversationMessage::user_text("Now use the new effort"));
    assert!(encode(&r, &profile(), "claude-opus-5").is_err());
}

fn foundry_profile() -> ProviderProfile {
    let mut profile = profile();
    profile.provider_id = ProviderId::new("foundry");
    profile.profile_name = "foundry".into();
    profile.base_url = "https://foundry.example.test/resource".into();
    profile.protocol = ProtocolFamily::FoundryClaude;
    profile.extra["betas"] = json!(["fixture-existing-beta"]);
    let model = profile.models.first_mut().expect("test profile model");
    model.info.features.effort.support = CapabilitySupport::Supported;
    model.info.features.effort.levels = Some(vec![ReasoningEffort::High, ReasoningEffort::Medium]);
    profile
}

fn encode_foundry(
    request: &ChatRequest,
    profile: &ProviderProfile,
    kind: AnthropicRequestKind,
) -> Result<(Value, Vec<(String, String)>), LlmError> {
    let context =
        CodecContext::new(profile, MODEL, RequestMode::Complete).with_anthropic_request_kind(kind);
    let wire = FoundryClaudeCodec.encode_request(EncodeRequest::new(request), &context)?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

#[test]
fn foundry_main_effort_only_marker_is_encoded_and_reported_without_extra_sdk_beta() {
    let profile = foundry_profile();
    let mut request = req();
    request.messages.push(effort(AnthropicMessageEffort::High));
    let context = CodecContext::new(&profile, MODEL, RequestMode::Complete)
        .with_anthropic_request_kind(AnthropicRequestKind::Main);

    let inference = FoundryClaudeCodec
        .request_inference(&request, &context)
        .expect("Foundry main marker inference");
    assert_eq!(inference.requested_effort, Some(ReasoningEffort::High));

    let (body, headers) = encode_foundry(&request, &profile, AnthropicRequestKind::Main).unwrap();
    assert_eq!(body["messages"][1]["content"], serde_json::json!([]));
    assert_eq!(body["messages"][1]["output_config"]["effort"], "high");
    assert_eq!(
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>(),
        vec!["fixture-existing-beta"],
        "per-turn effort preserves the existing beta header without adding SDK tokens"
    );
}

#[test]
fn foundry_per_message_controls_remain_restricted_to_main_effort_marker() {
    let profile = foundry_profile();
    let mut request = req();
    request.messages.push(effort(AnthropicMessageEffort::High));
    assert!(matches!(
        encode_foundry(&request, &profile, AnthropicRequestKind::SideQuery),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut request = req();
    request.messages.push(scoped());
    assert!(matches!(
        encode_foundry(&request, &profile, AnthropicRequestKind::Main),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut request = req();
    request.messages.push(
        ConversationMessage::system_text("text").with_anthropic_options(AnthropicMessageOptions {
            clear_at: None,
            effort: Some(AnthropicMessageEffort::High),
        }),
    );
    assert!(matches!(
        encode_foundry(&request, &profile, AnthropicRequestKind::Main),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn foundry_effort_history_keeps_multiple_rows_and_reports_the_latest_level() {
    let profile = foundry_profile();
    let mut request = req();
    request.messages.push(effort(AnthropicMessageEffort::High));
    request
        .messages
        .push(ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "First answer".into(),
            thought_signature: None,
            citations: None,
        }]));
    request
        .messages
        .push(ConversationMessage::user_text("Follow up"));
    request
        .messages
        .push(effort(AnthropicMessageEffort::Medium));
    let context = CodecContext::new(&profile, MODEL, RequestMode::Complete)
        .with_anthropic_request_kind(AnthropicRequestKind::Main);

    let inference = FoundryClaudeCodec
        .request_inference(&request, &context)
        .expect("Foundry effort history inference");
    assert_eq!(inference.requested_effort, Some(ReasoningEffort::Medium));

    let (body, headers) = encode_foundry(&request, &profile, AnthropicRequestKind::Main).unwrap();
    assert_eq!(body["messages"][1]["output_config"]["effort"], "high");
    assert_eq!(body["messages"][4]["output_config"]["effort"], "medium");
    assert_eq!(
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>(),
        vec!["fixture-existing-beta"],
        "per-turn effort preserves the existing beta header without adding SDK tokens"
    );
}
