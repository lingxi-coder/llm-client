use lingxi_llm_client::client::context::{resolve_context, ContextRequest};
use lingxi_llm_client::protocol::*;
use serde_json::json;

fn profile(window: u64) -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"custom","profile_name":format!("connection-{window}"),"base_url":"https://example.test","protocol":"open_ai_chat","auth":"none",
        "models":[{"request_model":"same","display_model":"Same","billing_model":"same","metadata":{"contextWindowTokens":window,"maxInputTokens":window-100,"maxOutputTokens":8192,"context":{"sharedWindow":true}}}]})).unwrap()
}
fn resolved(
    p: &ProviderProfile,
    policy: ContextPolicy,
    output: Option<u32>,
) -> lingxi_llm_client::client::context::ResolvedContext {
    resolve_context(
        p,
        &p.models[0],
        &ContextRequest {
            policy: Some(policy),
            ..Default::default()
        },
        output,
        &PricingContext::default(),
    )
    .unwrap()
}
#[test]
fn same_name_connections_never_share_limits() {
    for n in [8192, 128000, 1000000] {
        let p = profile(n);
        let snapshot = resolved(&p, ContextPolicy::Long, Some(1024));
        assert_eq!(snapshot.context_window_tokens, Some(n));
        assert_eq!(snapshot.effective_input_budget, Some(n - 1024));
        assert_eq!(snapshot.profile_name, format!("connection-{n}"));
    }
}
#[test]
fn output_planning_does_not_become_a_hard_output_reservation() {
    let p = profile(8192);
    let snapshot = resolve_context(
        &p,
        &p.models[0],
        &ContextRequest {
            planning_output_tokens: Some(2048),
            safety_tokens: 100,
            ..Default::default()
        },
        None,
        &PricingContext::default(),
    )
    .unwrap();
    assert_eq!(snapshot.hard_input_limit, Some(8092));
    assert_eq!(snapshot.effective_input_budget, Some(6044));
    assert!(snapshot.output_reserve_estimated);
}
#[test]
fn independent_input_window_does_not_subtract_output() {
    let mut p = profile(8192);
    p.models[0].metadata.context.as_mut().unwrap().shared_window = Some(false);
    assert_eq!(
        resolved(&p, ContextPolicy::Long, Some(4096)).effective_input_budget,
        Some(8092)
    );
}
#[test]
fn thinking_limit_and_custom_policy_are_bounded_by_connection() {
    let mut p = profile(8192);
    p.models[0]
        .metadata
        .context
        .as_mut()
        .unwrap()
        .thinking_max_input_tokens = Some(6000);
    let snapshot = resolve_context(
        &p,
        &p.models[0],
        &ContextRequest {
            policy: Some(ContextPolicy::Custom {
                input_tokens: 1000000,
            }),
            thinking: Some(true),
            ..Default::default()
        },
        Some(1024),
        &PricingContext::default(),
    )
    .unwrap();
    assert_eq!(snapshot.effective_input_budget, Some(6000));
}
#[test]
fn unknown_is_not_a_claude_default() {
    let mut p = profile(8192);
    p.models[0].metadata = Default::default();
    let snapshot = resolved(&p, ContextPolicy::Auto, None);
    assert_eq!(snapshot.effective_input_budget, None);
    assert_eq!(snapshot.context_window_tokens, None);
    assert_eq!(snapshot.standard_price_input_tokens, None);
}
#[test]
fn explicit_policy_overrides_saved_policy_without_mutation() {
    let mut p = profile(8192);
    p.models[0].metadata.context.as_mut().unwrap().application =
        Some(ContextPolicy::Custom { input_tokens: 2000 });
    let default = resolve_context(
        &p,
        &p.models[0],
        &ContextRequest::default(),
        None,
        &PricingContext::default(),
    )
    .unwrap();
    assert_eq!(default.effective_input_budget, Some(2000));
    assert_eq!(default.policy_source, "model");
    assert_eq!(
        resolved(&p, ContextPolicy::Long, None).effective_input_budget,
        Some(8092)
    );
    assert_eq!(default.effective_input_budget, Some(2000));
}
#[test]
fn recommendation_requires_evidence_and_is_not_a_hard_limit() {
    let mut p = profile(8192);
    let metadata = p.models[0].metadata.context.as_mut().unwrap();
    metadata.recommendation = Some(ContextRecommendation {
        input_tokens: 3000,
        source: "https://vendor.test/spec".into(),
        verified_at: "2026-10-15".into(),
    });
    assert_eq!(
        resolved(&p, ContextPolicy::Auto, None).effective_input_budget,
        Some(3000)
    );
    assert_eq!(
        resolved(&p, ContextPolicy::Long, None).effective_input_budget,
        Some(8092)
    );
    p.models[0]
        .metadata
        .context
        .as_mut()
        .unwrap()
        .recommendation
        .as_mut()
        .unwrap()
        .source
        .clear();
    assert!(resolve_context(
        &p,
        &p.models[0],
        &ContextRequest::default(),
        None,
        &PricingContext::default()
    )
    .is_err());
}
#[test]
fn zero_custom_budget_rejected_and_saturation_never_wraps() {
    let p = profile(8192);
    assert!(resolve_context(
        &p,
        &p.models[0],
        &ContextRequest {
            policy: Some(ContextPolicy::Custom { input_tokens: 0 }),
            ..Default::default()
        },
        None,
        &PricingContext::default()
    )
    .is_err());
    assert_eq!(
        resolved(&p, ContextPolicy::Long, Some(u32::MAX)).effective_input_budget,
        Some(0)
    );
}
#[test]
fn official_catalog_price_boundaries_use_actual_inclusive_ranges() {
    let profiles = lingxi_llm_client::builtin_providers().unwrap();
    for (provider, model, expected) in [
        ("openai", "gpt-6-astra", 272000),
        ("gemini", "gemini-3.1-pro-preview", 200000),
        ("anthropic", "claude-haiku-5-5", 100000),
        ("grok", "grok-4.3", 199999),
    ] {
        let p = profiles
            .iter()
            .find(|p| p.profile_name == provider)
            .unwrap();
        let m = p.models.iter().find(|m| m.request_model == model).unwrap();
        let snapshot = resolve_context(
            p,
            m,
            &ContextRequest::default(),
            None,
            &PricingContext {
                unix_seconds: Some(1792108800),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            snapshot.standard_price_input_tokens,
            Some(expected),
            "{provider}/{model}"
        );
    }
}
#[test]
fn unpriced_and_subscription_models_do_not_acquire_price_ceiling() {
    let profiles = lingxi_llm_client::builtin_providers().unwrap();
    let p = profiles
        .iter()
        .find(|p| p.profile_name == "anthropic")
        .unwrap();
    let m = p
        .models
        .iter()
        .find(|m| m.request_model == "claude-opus-4-6")
        .unwrap();
    assert_eq!(
        resolve_context(
            p,
            m,
            &ContextRequest::default(),
            None,
            &PricingContext::default()
        )
        .unwrap()
        .standard_price_input_tokens,
        None
    );
}

#[tokio::test]
async fn prepared_requests_capture_each_connection_without_adding_wire_fields() {
    use async_trait::async_trait;
    use lingxi_llm_client::{
        HttpRequest, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse, Transport,
    };
    use std::sync::Arc;
    struct NoNetwork;
    #[async_trait]
    impl Transport for NoNetwork {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            panic!("preparation must not dispatch")
        }
    }
    let profiles = vec![profile(8192), profile(1000000)];
    let client = LlmClientBuilder::with_transport(Arc::new(NoNetwork), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let request: ChatRequest=serde_json::from_value(json!({"model":"Same","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]})).unwrap();
    let options = RequestOptions {
        context: ContextRequest {
            policy: Some(ContextPolicy::Custom {
                input_tokens: 200000,
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    for (name, expected) in [("connection-8192", 8092), ("connection-1000000", 200000)] {
        let call = client
            .prepare_on(name, &request, &options, RequestMode::Complete)
            .await
            .unwrap();
        assert_eq!(
            call.resolved_context().effective_input_budget,
            Some(expected)
        );
        let body: serde_json::Value = serde_json::from_slice(&call.request().body).unwrap();
        for key in [
            "context",
            "context_policy",
            "max_tokens",
            "max_completion_tokens",
        ] {
            assert!(body.get(key).is_none(), "{key}");
        }
    }
    assert_eq!(request.max_tokens, None);
}
