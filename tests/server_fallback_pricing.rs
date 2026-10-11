use lingxi_llm_client::protocol::{
    BillingMode, InferenceReport, ProtocolFamily, ProviderProfile, ServiceTier, Submission,
    TokenPricing,
};
use lingxi_llm_client::providers::anthropic::fallback_response::usage_iterations;
use lingxi_llm_client::{
    resolve_anthropic_server_fallback_summary_model, AnthropicFallbackCostCompleteness,
    AnthropicFallbackCostComponentKind, FrozenPricing,
};
use serde_json::{json, Value};

const ACCOUNTING_FIXTURE: &str = include_str!("fixtures/server_fallback_accounting_2_1_288.json");
const QUOTE_MATH_FIXTURE: &str = include_str!("fixtures/server_fallback_quote_math_2_1_288.json");

fn model(
    request_model: &str,
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
    web_search: Option<f64>,
) -> Value {
    json!({
        "display_model": request_model,
        "request_model": request_model,
        "billing_model": format!("bill-{request_model}"),
        "billing_mode": "per_token",
        "pricing": {
            "currency": "USD",
            "source": "synthetic test tariff only",
            "input_per_million": input,
            "output_per_million": output,
            "cache_read_per_million": cache_read,
            "cache_write_per_million": cache_write,
            "web_search_per_request": web_search,
            "rules": [{
                "service_tier": "fast",
                "submission": "interactive",
                "rates": {
                    "input_per_million": input * 2.0,
                    "output_per_million": output * 2.0,
                    "cache_read_per_million": cache_read * 2.0,
                    "cache_write_per_million": cache_write * 2.0,
                    "web_search_per_request": web_search.map(|price| price * 2.0),
                }
            }]
        }
    })
}

fn profile(protocol: ProtocolFamily) -> ProviderProfile {
    let protocol = serde_json::to_value(protocol).unwrap();
    serde_json::from_value(json!({
        "provider_id": "anthropic",
        "profile_name": "frozen-anthropic",
        "base_url": "https://api.example.test",
        "protocol": protocol,
        "auth": "none",
        "models": [
            model("primary-model", 1.0, 2.0, 0.5, 0.75, Some(0.1)),
            model("fallback-a", 2.0, 4.0, 1.0, 3.0, Some(0.25)),
            model("fallback-b", 5.0, 8.0, 2.0, 4.0, Some(0.5)),
            model("fallback-terminal", 60.0, 80.0, 9.0, 10.0, Some(1.1)),
            model("fallback-a[1m]", 99.0, 99.0, 99.0, 99.0, Some(99.0)),
        ]
    }))
    .unwrap()
}

fn frozen(profile: &ProviderProfile) -> FrozenPricing {
    FrozenPricing::capture(profile, "primary-model", "primary-model").unwrap()
}

fn native_refusal_case() -> (
    lingxi_llm_client::providers::anthropic::fallback_response::UsageIterations,
    String,
) {
    let fixture: Value = serde_json::from_str(ACCOUNTING_FIXTURE).unwrap();
    let case = &fixture["cases"][0];
    let mut iterations = usage_iterations(&case["input"]["usage"]);
    iterations.speed = Some("fast".into());
    iterations.inference_geo = Some("us-east-1".into());
    iterations.web_search_requests = Some(2.0);
    (iterations, case["stop_reason"].as_str().unwrap().into())
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn usage_decoder_captures_only_native_quote_context_and_preserves_search_presence() {
    let absent = usage_iterations(&json!({
        "speed":"fast",
        "inference_geo":"us",
        "server_tool_use":{"web_fetch_requests":9},
        "iterations":[{"type":"fallback_message","model":"fallback-a"}]
    }));
    assert_eq!(absent.speed.as_deref(), Some("fast"));
    assert_eq!(absent.inference_geo.as_deref(), Some("us"));
    assert_eq!(absent.web_search_requests, None);

    let zero = usage_iterations(&json!({
        "server_tool_use":{"web_search_requests":0,"web_fetch_requests":9},
        "iterations":[{"type":"fallback_message","model":"fallback-a"}]
    }));
    assert_eq!(zero.web_search_requests, Some(0.0));
}

#[test]
fn refusal_quote_prices_only_native_selected_iterations_and_first_model_search() {
    let native_math: Value = serde_json::from_str(QUOTE_MATH_FIXTURE).unwrap();
    let profile = profile(ProtocolFamily::AnthropicMessages);
    let frozen = frozen(&profile);
    let (iterations, stop_reason) = native_refusal_case();
    let quote = frozen
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();

    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Complete
    );
    assert_eq!(quote.currency.as_deref(), Some("USD"));
    assert_eq!(quote.served_fallback_model, "fallback-terminal");
    assert_eq!(quote.summary_model, "fallback-terminal");
    assert_eq!(quote.excluded_terminal_fallback_index, Some(4));
    assert_eq!(
        quote
            .components
            .iter()
            .filter(|component| component.kind == AnthropicFallbackCostComponentKind::Iteration)
            .map(|component| component.iteration_index.unwrap())
            .collect::<Vec<_>>(),
        [1, 2]
    );

    let first = &quote.components[1];
    assert_eq!(first.model.as_deref(), Some("fallback-a"));
    close(
        first.total_cost_usd.unwrap(),
        native_math["cases"][0]["native_yQe_usd_shaped_amount"]
            .as_f64()
            .unwrap(),
    );
    let second = &quote.components[2];
    assert_eq!(second.model.as_deref(), Some("fallback-b"));
    close(
        second.total_cost_usd.unwrap(),
        native_math["cases"][1]["native_yQe_usd_shaped_amount"]
            .as_f64()
            .unwrap(),
    );

    let tool = quote
        .components
        .iter()
        .find(|component| {
            component.kind == AnthropicFallbackCostComponentKind::FirstModelServerToolUse
        })
        .unwrap();
    assert_eq!(tool.model.as_deref(), Some("fallback-a"));
    assert_eq!(tool.usage.web_search_requests, Some(2.0));
    let native_tool = native_math["cases"][2]["native_yQe_usd_shaped_amount"]
        .as_f64()
        .unwrap();
    close(tool.total_cost_usd.unwrap(), native_tool);
    close(quote.server_tool_cost_usd.unwrap(), native_tool);
    close(quote.total_cost_usd.unwrap(), 1.000202);
    assert_eq!(quote.service_tier, ServiceTier::Fast);

    // Excluded, missing-model and non-fallback rows stay in the projection as
    // zero-valued observations but do not prevent a full CostEstimate.
    let excluded = quote
        .components
        .iter()
        .find(|component| {
            component.kind == AnthropicFallbackCostComponentKind::ExcludedRefusalTerminal
        })
        .unwrap();
    assert_eq!(excluded.total_cost_usd, Some(0.0));
    let estimate = quote.into_cost_estimate_if_complete().unwrap();
    close(estimate.total_cost, quote.total_cost_usd.unwrap());
    let serialized = serde_json::to_value(&quote).unwrap();
    assert_eq!(
        serialized["components"][5]["usage"]["webSearchRequests"],
        2.0
    );
}

#[test]
fn frozen_quote_keeps_native_summary_identity_and_does_not_reprice_by_it() {
    let profile = profile(ProtocolFamily::AnthropicMessages);
    let frozen = frozen(&profile);
    let mut iterations = usage_iterations(&json!({
        "iterations": [{"type":"fallback_message", "model":"fallback-a", "input_tokens":2}]
    }));
    assert_eq!(
        resolve_anthropic_server_fallback_summary_model(Some("fallback-a[1m]"), &iterations)
            .as_deref(),
        Some("fallback-a[1m]")
    );
    assert_eq!(
        resolve_anthropic_server_fallback_summary_model(Some("primary-model"), &Default::default()),
        None
    );
    iterations.inference_geo = Some("us-east-1".into());
    let quote = frozen
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("fallback-a[1m]"),
            Some("end_turn"),
            &InferenceReport {
                raw_speed: Some("fast".into()),
                ..InferenceReport::default()
            },
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();

    assert_eq!(quote.summary_model, "fallback-a[1m]");
    assert_eq!(
        quote.summary_pricing_model.as_ref().unwrap().request_model,
        "fallback-a[1m]"
    );
    assert_eq!(quote.service_tier, ServiceTier::Standard);
    // Native Mwe receives speed from the merged Anthropic usage object; an
    // unrelated inference report speed does not substitute for missing usage.
    // `us-east-1` is not the native exact geo sentinel `us`.
    close(
        quote.components[0].total_cost_usd.unwrap(),
        2.0 / 1_000_000.0 * 2.0,
    );
    // The first-model zero-token Mwe call has no server tool usage in this
    // body; absence remains absent while native arithmetic uses zero.
    assert_eq!(quote.components[1].usage.web_search_requests, None);
    assert_eq!(quote.server_tool_cost_usd, Some(0.0));
    let mut explicit_zero = iterations.clone();
    explicit_zero.web_search_requests = Some(0.0);
    let explicit_zero_quote = frozen
        .estimate_anthropic_server_fallback(
            &explicit_zero,
            Some("fallback-a[1m]"),
            Some("end_turn"),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        explicit_zero_quote.components[1].usage.web_search_requests,
        Some(0.0)
    );
    assert_eq!(explicit_zero_quote.server_tool_cost_usd, Some(0.0));

    // Native Mn strips case-insensitive marker spellings but preserves the
    // rest of the model string; it does not case-fold the model identity.
    let case_sensitive = frozen
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("Fallback-A[1M]"),
            Some("end_turn"),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(case_sensitive.summary_model, "fallback-a");

    let empty_model = usage_iterations(&json!({
        "iterations": [{"type":"fallback_message", "model":""}]
    }));
    assert_eq!(
        resolve_anthropic_server_fallback_summary_model(Some(""), &empty_model),
        Some(String::new())
    );
}

#[test]
fn exact_us_geo_scales_token_cost_but_leaves_first_model_search_unscaled() {
    let native_math: Value = serde_json::from_str(QUOTE_MATH_FIXTURE).unwrap();
    let frozen = frozen(&profile(ProtocolFamily::AnthropicMessages));
    let mut iterations = usage_iterations(&json!({
        "iterations": [{"type":"fallback_message", "model":"fallback-a", "input_tokens":1000000}]
    }));
    iterations.speed = Some("fast".into());
    iterations.inference_geo = Some("us".into());
    iterations.web_search_requests = Some(2.0);
    let quote = frozen
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("fallback-a"),
            Some("end_turn"),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    close(
        quote.total_cost_usd.unwrap(),
        native_math["cases"][3]["native_yQe_usd_shaped_amount"]
            .as_f64()
            .unwrap(),
    );
    close(
        quote.server_tool_cost_usd.unwrap(),
        native_math["cases"][2]["native_yQe_usd_shaped_amount"]
            .as_f64()
            .unwrap(),
    );
}

#[test]
fn unsupported_rows_rates_and_protocols_never_fall_back_to_another_tariff() {
    let (iterations, stop_reason) = native_refusal_case();

    let mut missing_row_profile = profile(ProtocolFamily::AnthropicMessages);
    missing_row_profile
        .models
        .retain(|model| model.request_model != "fallback-b");
    let quote = frozen(&missing_row_profile)
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Incomplete
    );
    assert!(quote.total_cost_usd.is_none());
    assert!(quote
        .incomplete_reasons
        .iter()
        .any(|reason| reason.contains("no exact frozen pricing model row for \"fallback-b\"")));

    let mut missing_search_profile = profile(ProtocolFamily::AnthropicMessages);
    missing_search_profile
        .models
        .iter_mut()
        .find(|model| model.request_model == "fallback-a")
        .unwrap()
        .pricing
        .as_mut()
        .unwrap()
        .web_search_per_request = None;
    let (mut standard_iterations, stop_reason) = native_refusal_case();
    standard_iterations.speed = None;
    let quote = frozen(&missing_search_profile)
        .estimate_anthropic_server_fallback(
            &standard_iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Incomplete
    );
    assert!(quote
        .incomplete_reasons
        .iter()
        .any(|reason| reason.contains("no web-search request rate")));

    let mut non_usd_profile = profile(ProtocolFamily::AnthropicMessages);
    non_usd_profile
        .models
        .iter_mut()
        .find(|model| model.request_model == "fallback-b")
        .unwrap()
        .pricing
        .as_mut()
        .unwrap()
        .currency = Some("EUR".into());
    let quote = frozen(&non_usd_profile)
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Incomplete
    );
    assert!(quote.total_cost_usd.is_none());

    let mut subscription_profile = profile(ProtocolFamily::AnthropicMessages);
    subscription_profile
        .models
        .iter_mut()
        .find(|model| model.request_model == "fallback-b")
        .unwrap()
        .billing_mode = Some(BillingMode::Subscription);
    let quote = frozen(&subscription_profile)
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Incomplete
    );
    assert!(quote
        .incomplete_reasons
        .iter()
        .any(|reason| reason.contains("not per-token")));

    let unsupported = frozen(&profile(ProtocolFamily::OpenAiChat))
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some(&stop_reason),
            &InferenceReport::default(),
            Submission::Interactive,
        );
    assert!(unsupported.is_err());
}

#[test]
fn claude_message_wrappers_use_the_same_frozen_multi_model_quote_path() {
    for protocol in [
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::BedrockClaude,
        ProtocolFamily::VertexClaude,
        ProtocolFamily::FoundryClaude,
    ] {
        let frozen = frozen(&profile(protocol));
        let (iterations, stop_reason) = native_refusal_case();
        let quote = frozen
            .estimate_anthropic_server_fallback(
                &iterations,
                Some("primary-model"),
                Some(&stop_reason),
                &InferenceReport::default(),
                Submission::Interactive,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            quote.completeness,
            AnthropicFallbackCostCompleteness::Complete
        );
    }
}

#[test]
fn explicit_frozen_override_is_used_for_its_model_without_repricing_fallback_rows() {
    let profile = profile(ProtocolFamily::AnthropicMessages);
    let frozen = frozen(&profile).with_token_pricing(TokenPricing {
        currency: Some("USD".into()),
        input_per_million: Some(10.0),
        output_per_million: Some(20.0),
        cache_read_per_million: Some(30.0),
        cache_write_per_million: Some(40.0),
        web_search_per_request: Some(9.0),
        source: Some("explicit frozen test override".into()),
        ..Default::default()
    });
    let mut iterations = usage_iterations(&json!({
        "iterations": [
            {"type":"fallback_message", "model":"primary-model", "input_tokens":2},
            {"type":"fallback_message", "model":"fallback-b", "output_tokens":3}
        ]
    }));
    iterations.web_search_requests = Some(1.0);
    let quote = frozen
        .estimate_anthropic_server_fallback(
            &iterations,
            Some("primary-model"),
            Some("end_turn"),
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .unwrap();

    assert_eq!(
        quote.completeness,
        AnthropicFallbackCostCompleteness::Complete
    );
    close(quote.components[0].total_cost_usd.unwrap(), 0.00002);
    close(quote.components[1].total_cost_usd.unwrap(), 0.000024);
    assert_eq!(
        quote.components[0].pricing_source.as_deref(),
        Some("explicit frozen test override")
    );
    assert_eq!(quote.components[2].model.as_deref(), Some("primary-model"));
    close(quote.server_tool_cost_usd.unwrap(), 9.0);
    close(quote.total_cost_usd.unwrap(), 9.000044);
}

#[test]
fn absence_of_a_served_fallback_returns_no_native_quote() {
    let frozen = frozen(&profile(ProtocolFamily::AnthropicMessages));
    assert!(frozen
        .estimate_anthropic_server_fallback(
            &Default::default(),
            Some("primary-model"),
            None,
            &InferenceReport::default(),
            Submission::Interactive,
        )
        .unwrap()
        .is_none());
}
