use lingxi_llm_client::protocol::*;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::{CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec};
use serde_json::{json, Value};

const DEPLOYMENT: &str = "prod-eus-model-07";

fn profile(hosting: Option<FoundryHosting>, model_id: Option<&str>) -> ProviderProfile {
    let mut model = json!({
        "display_model": DEPLOYMENT,
        "request_model": DEPLOYMENT,
        "billing_model": DEPLOYMENT
    });
    if let Some(hosting) = hosting {
        model["foundry"] = json!({
            "hosting": hosting,
            "model_id": model_id.expect("a hosted model identity is required")
        });
    }
    serde_json::from_value(json!({
        "provider_id": "foundry-account",
        "profile_name": "foundry-test",
        "base_url": "https://example-resource.services.ai.azure.com/anthropic",
        "protocol": "foundry_claude",
        "auth": "none",
        "models": [model]
    }))
    .unwrap()
}

fn make_request(config: AnthropicWebFetchConfig) -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model": DEPLOYMENT,
        "messages": [{"role":"user","content":[{"type":"text","text":"Fetch https://example.com"}]}]
    }))
    .unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::WebFetch(config)
            .into(),
    );
    request
}

fn validate(request: &ChatRequest, profile: &ProviderProfile) -> Result<(), LlmError> {
    FoundryClaudeCodec.validate_request(
        request,
        &CodecContext::new(profile, DEPLOYMENT, RequestMode::Complete),
    )
}

type EncodedTools = (Value, Vec<(String, String)>);

fn encode(request: &ChatRequest, profile: &ProviderProfile) -> Result<EncodedTools, LlmError> {
    let wire = FoundryClaudeCodec.encode_request(
        EncodeRequest::new(request),
        &CodecContext::new(profile, DEPLOYMENT, RequestMode::Complete),
    )?;
    Ok((serde_json::from_slice(&wire.body).unwrap(), wire.headers))
}

#[test]
fn azure_hosting_accepts_only_basic_fetch() {
    let p = profile(Some(FoundryHosting::Azure), Some("claude-opus-5-5"));
    let request = make_request(AnthropicWebFetchConfig {
        version: AnthropicWebFetchVersion::V20250910,
        ..Default::default()
    });
    validate(&request, &p).unwrap();
    let (body, _) = encode(&request, &p).unwrap();
    assert_eq!(body["model"], DEPLOYMENT);
    assert_eq!(body["tools"][0]["type"], "web_fetch_20250910");

    let direct_only = make_request(AnthropicWebFetchConfig {
        version: AnthropicWebFetchVersion::V20250910,
        allowed_callers: vec![AnthropicFetchCaller::Direct],
        ..Default::default()
    });
    validate(&direct_only, &p).unwrap();
    assert!(encode(&direct_only, &p).is_ok());

    for allowed_callers in [
        vec![AnthropicFetchCaller::CodeExecution20250825],
        vec![AnthropicFetchCaller::CodeExecution20260120],
        vec![AnthropicFetchCaller::CodeExecution20260521],
        vec![
            AnthropicFetchCaller::Direct,
            AnthropicFetchCaller::CodeExecution20260120,
        ],
    ] {
        let request = make_request(AnthropicWebFetchConfig {
            version: AnthropicWebFetchVersion::V20250910,
            allowed_callers,
            ..Default::default()
        });
        assert!(validate(&request, &p).is_err());
        assert!(encode(&request, &p).is_err());
    }

    for version in [
        AnthropicWebFetchVersion::V20260209,
        AnthropicWebFetchVersion::V20260309,
        AnthropicWebFetchVersion::V20260318,
    ] {
        for allowed_callers in [vec![], vec![AnthropicFetchCaller::Direct]] {
            let request = make_request(AnthropicWebFetchConfig {
                version,
                allowed_callers,
                ..Default::default()
            });
            assert!(validate(&request, &p).is_err());
            assert!(encode(&request, &p).is_err());
        }
    }
}

#[test]
fn anthropic_hosting_accepts_all_four_versions_without_rewriting_deployment_name() {
    let p = profile(Some(FoundryHosting::Anthropic), Some("claude-opus-5-5"));
    for (version, wire_type) in [
        (AnthropicWebFetchVersion::V20250910, "web_fetch_20250910"),
        (AnthropicWebFetchVersion::V20260209, "web_fetch_20260209"),
        (AnthropicWebFetchVersion::V20260309, "web_fetch_20260309"),
        (AnthropicWebFetchVersion::V20260318, "web_fetch_20260318"),
    ] {
        let request = make_request(AnthropicWebFetchConfig {
            version,
            ..Default::default()
        });
        validate(&request, &p).unwrap();
        let (body, _) = encode(&request, &p).unwrap();
        assert_eq!(body["model"], DEPLOYMENT);
        assert_eq!(body["tools"][0]["type"], wire_type);
    }
}

#[test]
fn dynamic_filtering_uses_underlying_model_and_direct_only_disables_it() {
    let unsupported_model = profile(Some(FoundryHosting::Anthropic), Some("claude-haiku-4-5"));
    let dynamic = make_request(AnthropicWebFetchConfig {
        version: AnthropicWebFetchVersion::V20260209,
        ..Default::default()
    });
    assert!(validate(&dynamic, &unsupported_model).is_err());
    assert!(encode(&dynamic, &unsupported_model).is_err());

    let direct_only = make_request(AnthropicWebFetchConfig {
        version: AnthropicWebFetchVersion::V20260209,
        allowed_callers: vec![AnthropicFetchCaller::Direct],
        ..Default::default()
    });
    validate(&direct_only, &unsupported_model).unwrap();
    let (body, _) = encode(&direct_only, &unsupported_model).unwrap();
    assert_eq!(body["model"], DEPLOYMENT);
    assert_eq!(body["tools"][0]["allowed_callers"], json!(["direct"]));
}

#[test]
fn missing_foundry_identity_and_raw_tool_bypasses_are_rejected() {
    let request = make_request(AnthropicWebFetchConfig::default());
    let missing_identity = profile(None, None);
    assert!(validate(&request, &missing_identity).is_err());
    assert!(encode(&request, &missing_identity).is_err());

    let mut raw_profile = profile(Some(FoundryHosting::Azure), Some("claude-opus-5-5"));
    raw_profile.extra = json!({"body":{"tools":[{
        "type":"web_fetch_20260318",
        "name":"web_fetch"
    }]}});
    let no_typed_config = serde_json::from_value(json!({
        "model": DEPLOYMENT,
        "messages": [{"role":"user","content":[{"type":"text","text":"Hello"}]}]
    }))
    .unwrap();
    assert!(validate(&no_typed_config, &raw_profile).is_err());
    assert!(encode(&no_typed_config, &raw_profile).is_err());
}
