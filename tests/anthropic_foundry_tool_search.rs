use lingxi_llm_client::protocol::{ChatRequest, LlmError, ProviderProfile};
use lingxi_llm_client::{CodecContext, EncodeRequest, FoundryClaudeCodec, RequestMode, WireCodec};
use serde_json::{json, Value};

fn profile(hosting: Option<&str>, model_id: Option<&str>, deployment: &str) -> ProviderProfile {
    let mut model = json!({
        "display_model": "Friendly Claude label",
        "request_model": deployment,
        "billing_model": "claude-accounting-row"
    });
    if let (Some(hosting), Some(model_id)) = (hosting, model_id) {
        model["foundry"] = json!({"hosting": hosting, "model_id": model_id});
    }
    serde_json::from_value(json!({
        "provider_id": "my-foundry-project",
        "profile_name": "foundry-route",
        "base_url": "https://my-resource.services.ai.azure.com/anthropic",
        "protocol": "foundry_claude",
        "auth": "none",
        "models": [model],
        "extra": {}
    }))
    .unwrap()
}

fn request(deployment: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model": deployment,
        "messages": [{"role":"user","content":[{"type":"text","text":"Find a tool"}]}],
        "hosted_tools": [{
            "type": "anthropic_tool_search",
            "config": {"strategy":"bm25"}
        }]
    }))
    .unwrap()
}

type EncodedSearch = (String, Value, Vec<(String, String)>);

fn validate_and_encode(
    hosting: Option<&str>,
    model_id: Option<&str>,
    deployment: &str,
    mode: RequestMode,
) -> Result<EncodedSearch, LlmError> {
    let profile = profile(hosting, model_id, deployment);
    let context = CodecContext::new(&profile, deployment, mode);
    let request = request(deployment);
    FoundryClaudeCodec.validate_request(&request, &context)?;
    let http = FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?;
    let body = serde_json::from_slice(&http.body).unwrap();
    Ok((http.url, body, http.headers))
}

#[test]
fn documented_foundry_hosting_model_intersection_encodes_custom_deployments() {
    let cases = [
        ("azure", "claude-opus-5-5"),
        ("azure", "claude-opus-5"),
        ("azure", "claude-opus-4-8"),
        ("azure", "claude-haiku-4-5"),
        ("anthropic", "claude-fable-5-1"),
        ("anthropic", "claude-mythos-5-1"),
        ("anthropic", "claude-fable-5"),
        ("anthropic", "claude-mythos-5"),
        ("anthropic", "claude-opus-5-5"),
        ("anthropic", "claude-opus-5"),
        ("anthropic", "claude-opus-4-8"),
        ("anthropic", "claude-opus-4-7"),
        ("anthropic", "claude-opus-4-6"),
        ("anthropic", "claude-sonnet-4-6"),
        ("anthropic", "claude-opus-4-5"),
        ("anthropic", "claude-sonnet-4-5"),
        ("anthropic", "claude-haiku-4-5"),
    ];

    for (index, (hosting, model_id)) in cases.into_iter().enumerate() {
        for mode in [RequestMode::Complete, RequestMode::Stream] {
            let deployment = format!("custom-deployment-{index}");
            let (url, body, headers) =
                validate_and_encode(Some(hosting), Some(model_id), &deployment, mode)
                    .unwrap_or_else(|error| {
                        panic!("Foundry {hosting} model {model_id} was rejected: {error}")
                    });
            assert!(url.ends_with("/v1/messages"));
            assert_eq!(body["model"], deployment);
            assert_eq!(body["tools"][0]["type"], "tool_search_tool_bm25_20251119");
            assert_eq!(
                body["stream"].as_bool(),
                (mode == RequestMode::Stream).then_some(true)
            );
            assert!(!headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));
        }
    }
}

#[test]
fn foundry_tool_search_rejects_unsupported_models_and_hosting_pairs() {
    for (hosting, model_id) in [
        ("azure", "claude-fable-5-1"),
        ("azure", "claude-sonnet-5"),
        ("anthropic", "claude-sonnet-5"),
        ("anthropic", "claude-opus-4-5-20251101"),
        ("anthropic", "claude-not-a-model"),
    ] {
        assert!(
            matches!(
                validate_and_encode(
                    Some(hosting),
                    Some(model_id),
                    "arbitrary-foundry-deployment",
                    RequestMode::Complete
                ),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "unsupported Foundry host/model pair passed: {hosting} / {model_id}"
        );
    }
}

#[test]
fn foundry_tool_search_requires_explicit_deployment_identity() {
    for (hosting, model_id) in [
        (None, None),
        (Some("azure"), None),
        (None, Some("claude-opus-5")),
    ] {
        assert!(matches!(
            validate_and_encode(
                hosting,
                model_id,
                "renamed-deployment",
                RequestMode::Complete
            ),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[test]
fn request_deployment_name_is_never_used_to_guess_underlying_model() {
    let (_, body, _) = validate_and_encode(
        Some("anthropic"),
        Some("claude-opus-4-5"),
        "claude-sonnet-5-custom-deployment",
        RequestMode::Complete,
    )
    .unwrap();
    assert_eq!(body["model"], "claude-sonnet-5-custom-deployment");
}
