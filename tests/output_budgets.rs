//! Selected-route output ceilings and provider-specific generation budget fields.
use lingxi_llm_client::client::{OutputTokenLimits, RoutingCatalog};
use lingxi_llm_client::protocol::{ChatRequest, ModelMetadata, ProviderProfile, Region};
use lingxi_llm_client::{
    AnthropicMessagesCodec, CodecContext, EncodeRequest, OpenAiChatCodec, RequestMode, WireCodec,
};
use serde_json::{json, Value};

fn profile(name: &str, limit: u64) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"acme", "profile_name":name, "base_url":"https://example.test/v1",
        "protocol":"open_ai_chat", "auth":"none", "regions":["international"],
        "models":[{"display_model":"Shared model", "request_model":"shared", "billing_model":"shared",
            "metadata":{"contextWindowTokens":1_000_000,"maxOutputTokens":limit}}]
    })).unwrap()
}

#[test]
fn identical_model_ids_keep_the_selected_routes_output_limit() {
    let catalog = RoutingCatalog::new(
        vec![profile("a", 65_536), profile("b", 262_144)],
        Region::International,
    );
    assert_eq!(
        catalog
            .output_token_limits_in("shared", Some("a"))
            .unwrap()
            .max_output_tokens,
        Some(65_536)
    );
    assert_eq!(
        catalog
            .output_token_limits_in("shared", Some("b"))
            .unwrap()
            .max_output_tokens,
        Some(262_144)
    );
    assert_eq!(
        catalog
            .output_token_limits_in("Shared model", Some("a"))
            .unwrap()
            .max_output_tokens,
        Some(65_536)
    );
}

#[test]
fn every_catalog_model_omits_optional_automatic_budget() {
    for profile in lingxi_llm_client::presets::builtin().unwrap() {
        for model in &profile.models {
            assert_eq!(
                OutputTokenLimits::from_metadata(&model.metadata).budget(None, 1000, 1024),
                None,
                "{}/{}",
                profile.profile_name,
                model.request_model
            );
        }
    }
}

fn body(profile: &ProviderProfile, model: &str) -> Value {
    let request: ChatRequest = serde_json::from_value(json!({
        "model":model, "messages":[], "max_tokens":65_536,
    }))
    .unwrap();
    serde_json::from_slice(
        &OpenAiChatCodec
            .encode_request(
                EncodeRequest::new(&request),
                &CodecContext::new(profile, model, RequestMode::Stream),
            )
            .unwrap()
            .body,
    )
    .unwrap()
}

#[test]
fn kimi_current_chat_routes_encode_total_completion_budget() {
    let profiles = lingxi_llm_client::presets::builtin().unwrap();
    for name in ["kimi", "kimi-intl", "kimi-code"] {
        let profile = profiles
            .iter()
            .find(|profile| profile.profile_name == name)
            .unwrap();
        for model in &profile.models {
            let encoded = body(profile, &model.request_model);
            assert_eq!(
                encoded["max_completion_tokens"], 65_536,
                "{name}/{}",
                model.request_model
            );
            assert!(encoded.get("max_tokens").is_none());
        }
    }
}

#[test]
fn qwen_total_budget_is_selected_only_for_documented_model_families() {
    let mut profile = profile("qwen", 65_536);
    profile.provider_id = "qwen".into();
    profile.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1".into();
    for model in [
        "qwen3.8-max",
        "qwen3.7-max-2026-05-20",
        "qwen3.5-plus",
        "qwen3.8-flash",
        "deepseek-v4-pro",
        "deepseek-v4.1-flash",
    ] {
        let encoded = body(&profile, model);
        assert_eq!(encoded["max_completion_tokens"], 65_536, "{model}");
        assert!(encoded.get("max_tokens").is_none());
    }
    for model in [
        "qwen-long",
        "qwen3-max",
        "qwen3.6-max",
        "qwen3-coder-plus",
        "qwen3.8-27b",
    ] {
        let encoded = body(&profile, model);
        assert_eq!(encoded["max_tokens"], 65_536, "{model}");
        assert!(encoded.get("max_completion_tokens").is_none());
    }
}

#[test]
fn missing_catalog_facts_stay_unknown() {
    let limits = OutputTokenLimits::from_metadata(&ModelMetadata::default());
    assert_eq!(limits.budget(None, 100, 1_024), None);
}

#[test]
fn messages_required_budget_uses_explicit_application_default() {
    let mut profile = profile("messages", 131_072);
    profile.models[0].metadata.application_output_tokens = Some(32_000);
    profile.protocol = lingxi_llm_client::protocol::ProtocolFamily::AnthropicMessages;
    let request: ChatRequest =
        serde_json::from_value(json!({"model":"shared","messages":[]})).unwrap();
    let encoded = AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(&profile, "shared", RequestMode::Stream),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["max_tokens"], 32_000);
}

#[test]
fn unknown_messages_model_requires_configuration() {
    let mut profile = profile("messages", 131072);
    profile.protocol = lingxi_llm_client::protocol::ProtocolFamily::AnthropicMessages;
    let request: ChatRequest =
        serde_json::from_value(json!({"model":"shared","messages":[]})).unwrap();
    let error = AnthropicMessagesCodec
        .encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(&profile, "shared", RequestMode::Stream),
        )
        .unwrap_err();
    assert!(error.to_string().contains("applicationOutputTokens"));
}

#[test]
fn recovery_budget_uses_complete_usage_actual_budget_and_exact_route() {
    use lingxi_llm_client::protocol::{Usage, UsageReport, UsageState};
    let mut selected = profile("deepseek", 393216);
    selected.provider_id = "deepseek".into();
    selected.base_url = "https://api.deepseek.com".into();
    let usage = UsageReport::measured(
        Usage {
            input_tokens: 22214,
            output_tokens: 65536,
            reasoning_tokens: 40000,
            ..Default::default()
        },
        UsageState::Complete,
    );
    assert_eq!(
        lingxi_llm_client::client::recovery_output_budget(
            &selected,
            "shared",
            &usage,
            Some(65536),
            None
        ),
        Some(131072)
    );
    assert_eq!(
        lingxi_llm_client::client::recovery_output_budget(
            &selected,
            "shared",
            &usage,
            Some(393216),
            None
        ),
        None
    );
    assert_eq!(
        lingxi_llm_client::client::recovery_output_budget(
            &selected,
            "shared",
            &usage,
            Some(65536),
            Some(90000)
        ),
        Some(66762)
    );
    let partial = UsageReport::measured(usage.usage.unwrap(), UsageState::Partial);
    assert_eq!(
        lingxi_llm_client::client::recovery_output_budget(
            &selected, "shared", &partial, None, None
        ),
        None
    );
    selected.base_url = "https://unverified-proxy.test".into();
    assert_eq!(
        lingxi_llm_client::client::recovery_output_budget(&selected, "shared", &usage, None, None),
        None
    );
}

#[test]
fn chat_length_keeps_valid_tools_and_reports_incomplete_arguments() {
    let selected = profile("chat", 65536);
    let response=lingxi_llm_client::HttpResponse { status:200,headers:vec![],body:serde_json::to_vec(&json!({
        "model":"shared","choices":[{"finish_reason":"length","message":{"role":"assistant","tool_calls":[
            {"id":"a","type":"function","function":{"name":"Read","arguments":"{}"}},
            {"id":"b","type":"function","function":{"name":"Write","arguments":"{\"content\":\"unfinished_secret_html"}}
        ]}}],"usage":{"prompt_tokens":20,"completion_tokens":32,"total_tokens":52}})).unwrap().into() };
    let decoded = OpenAiChatCodec
        .decode_response(
            &response,
            &CodecContext::new(&selected, "shared", RequestMode::Complete),
        )
        .unwrap();
    assert_eq!(
        decoded.stop_reason,
        lingxi_llm_client::protocol::StopReason::MaxTokens
    );
    assert_eq!(decoded.message.content.len(), 1);
    assert_eq!(decoded.tool_input_issues.len(), 1);
    assert_eq!(decoded.tool_input_issues[0].tool_name, "Write");
    assert_eq!(decoded.usage.complete().unwrap().output_tokens, 32);
    assert!(!serde_json::to_string(&decoded)
        .unwrap()
        .contains("unfinished_secret_html"));
}

#[test]
fn native_chatgpt_does_not_silently_ignore_explicit_hard_budget() {
    let mut selected = profile("chatgpt", 131072);
    selected.protocol = lingxi_llm_client::protocol::ProtocolFamily::OpenAiResponses;
    selected.auth = lingxi_llm_client::protocol::AuthStrategy::ChatGptOAuth;
    assert!(
        lingxi_llm_client::client::resolve_output_budget(&selected, "shared", Some(100)).is_err()
    );
    assert_eq!(
        lingxi_llm_client::client::resolve_output_budget(&selected, "shared", None)
            .unwrap()
            .value,
        None
    );
    selected.models[0].metadata.application_output_tokens = Some(4096);
    assert!(lingxi_llm_client::client::resolve_output_budget(&selected, "shared", None).is_err());
}

#[test]
fn every_builtin_mandatory_budget_model_has_an_explicit_application_policy() {
    use lingxi_llm_client::client::{resolve_output_budget, OutputBudgetSource};
    use lingxi_llm_client::protocol::ProtocolFamily;
    for profile in lingxi_llm_client::presets::builtin().unwrap() {
        if !matches!(
            profile.protocol,
            ProtocolFamily::AnthropicMessages
                | ProtocolFamily::VertexClaude
                | ProtocolFamily::BedrockClaude
                | ProtocolFamily::FoundryClaude
        ) {
            continue;
        }
        for model in &profile.models {
            let budget = resolve_output_budget(&profile, &model.request_model, None)
                .unwrap_or_else(|error| {
                    panic!("{}/{}: {error}", profile.profile_name, model.request_model)
                });
            assert_eq!(budget.source, OutputBudgetSource::ApplicationDefault);
            assert!(budget.value.is_some_and(|value| value > 0));
            assert!(budget
                .maximum
                .is_none_or(|cap| u64::from(budget.value.unwrap()) <= cap));
        }
    }
}

#[test]
fn application_defaults_are_capped_without_inventing_optional_defaults() {
    use lingxi_llm_client::client::{resolve_output_budget, OutputBudgetSource};
    use lingxi_llm_client::protocol::ProtocolFamily;
    for protocol in [
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::VertexClaude,
        ProtocolFamily::BedrockClaude,
        ProtocolFamily::FoundryClaude,
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::GeminiGenerateContent,
    ] {
        let mut row = profile("selected", 8192);
        row.protocol = protocol;
        row.models[0].metadata.application_output_tokens = Some(32768);
        let resolved = resolve_output_budget(&row, "shared", None).unwrap();
        assert_eq!(resolved.value, Some(8192));
        assert_eq!(
            resolved.source,
            OutputBudgetSource::ClampedApplicationDefault
        );
        let explicit = resolve_output_budget(&row, "shared", Some(4096)).unwrap();
        assert_eq!(explicit.value, Some(4096));
        assert_eq!(explicit.source, OutputBudgetSource::Explicit);
        row.models[0].metadata.application_output_tokens = Some(0);
        assert!(resolve_output_budget(&row, "shared", None).is_err());
        assert!(resolve_output_budget(&row, "shared", Some(0)).is_err());
    }
    for protocol in [
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::GeminiGenerateContent,
    ] {
        let mut row = profile("selected", 8192);
        row.protocol = protocol;
        row.models[0].request_model = "claude-sonnet-4-5".into();
        assert_eq!(
            resolve_output_budget(&row, "claude-sonnet-4-5", None)
                .unwrap()
                .value,
            None
        );
    }
    let mut row = profile("proxy", 8192);
    row.protocol = ProtocolFamily::AnthropicMessages;
    row.models[0].request_model = "claude-sonnet-4-5".into();
    assert_eq!(
        resolve_output_budget(&row, "claude-sonnet-4-5", None)
            .unwrap()
            .value,
        Some(8192)
    );
    row.models[0].metadata.max_output_tokens = None;
    assert_eq!(
        resolve_output_budget(&row, "claude-sonnet-4-5", None)
            .unwrap()
            .value,
        Some(32000)
    );
}

#[test]
fn duplicate_wire_ids_require_an_exact_model_row() {
    use lingxi_llm_client::client::resolve_output_budget;
    let mut row = profile("shared-wire", 8192);
    row.protocol = lingxi_llm_client::protocol::ProtocolFamily::AnthropicMessages;
    row.models[0].display_model = "Small".into();
    row.models[0].metadata.application_output_tokens = Some(32000);
    let mut large = row.models[0].clone();
    large.display_model = "Large".into();
    large.metadata.max_output_tokens = Some(64000);
    large.metadata.application_output_tokens = Some(48000);
    row.models.push(large);
    assert!(resolve_output_budget(&row, "shared", None).is_err());
    for default_protocol in [None, Some(row.protocol)] {
        let mut profile = row.clone();
        profile.connection.default_protocol = default_protocol;
        let catalog = RoutingCatalog::new(vec![profile], Region::International);
        for (display, cap) in [("Small", 8192), ("Large", 64000)] {
            assert_eq!(
                catalog
                    .resolve_in_prefer_native(display, Some("shared-wire"))
                    .unwrap()
                    .display_model,
                display
            );
            assert_eq!(
                catalog
                    .resolve_in_for_connection_failover(display, Some("shared-wire"))
                    .unwrap()
                    .display_model,
                display
            );
            assert_eq!(
                catalog
                    .output_token_limits_in(display, Some("shared-wire"))
                    .unwrap()
                    .max_output_tokens,
                Some(cap)
            );
        }
        assert!(catalog
            .resolve_in_for_connection_failover("shared", Some("shared-wire"))
            .is_err());
    }
    for (model, expected) in row.models.iter().zip([8192, 48000]) {
        let selected = CodecContext::for_model(&row, model, RequestMode::Complete);
        assert_eq!(
            resolve_output_budget(selected.profile(), "shared", None)
                .unwrap()
                .value,
            Some(expected)
        );
    }
}

#[test]
fn recovery_never_guesses_between_catalog_rows_with_shared_wire_ids() {
    use lingxi_llm_client::client::recovery_output_budget;
    use lingxi_llm_client::protocol::{Usage, UsageReport, UsageState};
    let mut row = profile("ambiguous", 8192);
    row.provider_id = "anthropic".into();
    row.base_url = "https://api.anthropic.com".into();
    row.models.push(row.models[0].clone());
    row.models[1].display_model = "Large".into();
    row.models[1].metadata.max_output_tokens = Some(65536);
    let usage = UsageReport::measured(
        Usage {
            input_tokens: 1,
            output_tokens: 4096,
            ..Default::default()
        },
        UsageState::Complete,
    );
    assert_eq!(
        recovery_output_budget(&row, "shared", &usage, Some(4096), None),
        None
    );
    let selected = CodecContext::for_model(&row, &row.models[1], RequestMode::Complete);
    assert_eq!(
        recovery_output_budget(selected.profile(), "shared", &usage, Some(4096), None),
        Some(8192)
    );
}
