//! Regressions for inference capability, wire accounting, and price selection review.
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::Arc;
mod support;

fn preset(name: &str) -> ProviderProfile {
    builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == name)
        .unwrap()
}
fn client(p: &ProviderProfile) -> LlmClient {
    LlmClientBuilder::with_transport(Arc::new(support::NoHttp), std::slice::from_ref(p))
        .with_region(Region::International)
        .build()
        .unwrap()
}
fn request(model: &str, thinking: Value) -> CompletionRequest {
    serde_json::from_value(
        json!({"model":model,"messages":[],"max_tokens":8192,"thinking":thinking}),
    )
    .unwrap()
}

#[test]
fn gemini_effort_lists_match_model_specific_thinking_levels() {
    let p = preset("gemini");
    let c = client(&p);
    for (id, effort, allowed) in [
        ("gemini-3.7-flash", ReasoningEffort::Minimal, false),
        ("gemini-3.7-flash", ReasoningEffort::Low, true),
        ("gemini-3.7-flash", ReasoningEffort::Medium, true),
        ("gemini-3.7-flash", ReasoningEffort::High, true),
        ("gemini-3.1-pro-preview", ReasoningEffort::Medium, true),
        ("gemini-3.1-pro-preview", ReasoningEffort::Minimal, false),
        (
            "gemini-3.1-pro-preview-customtools",
            ReasoningEffort::Medium,
            true,
        ),
        ("gemini-3.6-flash", ReasoningEffort::Minimal, true),
    ] {
        let model = c
            .models()
            .into_iter()
            .find(|m| m.request_model == id)
            .unwrap();
        assert_eq!(
            model.info.features.supports_effort(None, effort),
            if allowed {
                CapabilitySupport::Supported
            } else {
                CapabilitySupport::Unsupported
            },
            "{id} {effort:?}"
        );
        let req = request(id, json!({"effort":effort}));
        for mode in [RequestMode::Complete, RequestMode::Stream] {
            let result = GeminiCodec
                .encode_request(EncodeRequest::new(&req), &CodecContext::new(&p, id, mode));
            if allowed {
                let body: Value = serde_json::from_slice(&result.unwrap().body).unwrap();
                assert_eq!(
                    body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
                    json!(effort)
                );
            } else {
                assert!(matches!(
                    result,
                    Err(LlmError::UnsupportedCapability { .. })
                ));
            }
        }
    }
}

#[test]
fn effort_mode_queries_and_validation_share_wire_constraints() {
    for (name, id, codec, disabled_effort) in [
        (
            "openai",
            "gpt-5.6",
            Box::new(OpenAiResponsesCodec) as Box<dyn WireCodec>,
            CapabilitySupport::Unsupported,
        ),
        (
            "openai",
            "gpt-5.6",
            Box::new(OpenAiChatCodec),
            CapabilitySupport::Unsupported,
        ),
        (
            "anthropic",
            "claude-opus-5",
            Box::new(AnthropicMessagesCodec),
            CapabilitySupport::Supported,
        ),
    ] {
        let mut p = preset(name);
        if codec.family() == ProtocolFamily::OpenAiChat {
            p.protocol = ProtocolFamily::OpenAiChat;
            p.inference.reasoning = Some(ReasoningWire::OpenAiChat);
        }
        p.models.retain(|m| m.request_model == id);
        let c = client(&p);
        let f = &c.models()[0].info.features;
        assert_eq!(f, &p.models[0].info.features);
        assert_eq!(f, &c.profiles()[0].models[0].info.features);
        assert_eq!(
            f,
            &c.configured_models(name).unwrap()[0].model.info.features
        );
        assert_eq!(f.effort.with_disabled_thinking, disabled_effort);
        assert_eq!(
            c.providers()[0].info.features.effort.with_disabled_thinking,
            disabled_effort
        );
        for mode in [
            ThinkingMode::Enabled,
            ThinkingMode::Adaptive,
            ThinkingMode::Disabled,
        ] {
            for effort in ReasoningEffort::ALL {
                let supported =
                    f.supports_effort(Some(mode), effort) == CapabilitySupport::Supported;
                let req = request(id, json!({"mode":mode,"effort":effort}));
                for request_mode in [RequestMode::Complete, RequestMode::Stream] {
                    // Use the original profile as well as runtime info, so direct codec
                    // callers get the same constraints without a client projection.
                    let context = CodecContext::new(&p, id, request_mode);
                    assert_eq!(
                        codec.validate_request(&req, &context).is_ok(),
                        supported,
                        "{name} {mode:?} {effort:?}"
                    );
                    assert_eq!(
                        codec
                            .encode_request(EncodeRequest::new(&req), &context)
                            .is_ok(),
                        supported,
                        "{name} {mode:?} {effort:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn explicit_modes_override_opposite_defaults_without_inventing_effort() {
    let mut p = preset("openrouter");
    p.models.retain(|m| m.request_model == "openai/gpt-5.4");
    let model = &mut p.models[0];
    model.info.features.modes = Some(vec![ThinkingMode::Enabled, ThinkingMode::Disabled]);
    model.info.features.effort.levels = Some(ReasoningEffort::ALL.to_vec());
    let id = model.request_model.clone();
    for (default_mode, default_effort, mode) in [
        (
            ThinkingMode::Enabled,
            ReasoningEffort::High,
            ThinkingMode::Disabled,
        ),
        (
            ThinkingMode::Disabled,
            ReasoningEffort::None,
            ThinkingMode::Enabled,
        ),
    ] {
        p.models[0].info.features.default_mode = Some(default_mode);
        p.models[0].info.features.effort.default = Some(default_effort);
        let req = request(&id, json!({"mode":mode}));
        let encoded = OpenAiChatCodec
            .encode_request(
                EncodeRequest::new(&req),
                &CodecContext::new(&p, &id, RequestMode::Complete),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&encoded.body).unwrap();
        assert_eq!(
            body["reasoning"]["enabled"],
            json!(mode == ThinkingMode::Enabled)
        );
        assert!(body["reasoning"].get("effort").is_none());
    }
}

fn pair(
    codec: &dyn WireCodec,
    p: &ProviderProfile,
    model: &str,
    body: Value,
    frames: Vec<Value>,
) -> (UsageReport, InferenceReport) {
    let context = CodecContext::new(p, model, RequestMode::Complete);
    let full = codec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&body).unwrap().into(),
            },
            &context,
        )
        .unwrap();
    let context = CodecContext::new(p, model, RequestMode::Stream);
    let mut decoder = codec.stream_decoder(&context);
    let mut events = Vec::new();
    for frame in frames {
        let encoded = format!("data: {frame}\n\n");
        for chunk in encoded.as_bytes().chunks(5) {
            events.extend(decoder.push_bytes(chunk));
        }
    }
    events.extend(decoder.finish());
    let end = events
        .into_iter()
        .map(Result::unwrap)
        .find_map(|event| match event {
            StreamEvent::End {
                usage, inference, ..
            } => Some((usage, inference)),
            _ => None,
        })
        .expect("terminal event");
    assert_eq!(end.0, full.usage);
    assert_eq!(end.1, full.inference);
    assert_eq!(decoder.usage_report(), full.usage);
    assert_eq!(decoder.inference_report(), full.inference);
    end
}
fn responses_body(usage: Value) -> Value {
    json!({"id":"resp-1","model":"gpt-5.6","status":"completed","output":[],"service_tier":"default","usage":usage})
}
#[test]
fn responses_cache_writes_are_disjoint_validated_and_priced_in_both_modes() {
    let mut p = preset("openai");
    p.models.retain(|m| m.request_model == "gpt-5.6");
    p.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(4.0),
        cache_read_per_million: Some(0.4),
        cache_write_per_million: Some(5.0),
        output_per_million: Some(20.0),
        ..Default::default()
    });
    let c = client(&p);
    let route = c.resolve("gpt-5.6").unwrap();
    for (writes, state) in [
        (json!(3000), UsageState::Complete),
        (json!(3001), UsageState::Invalid),
        (json!(-1), UsageState::Invalid),
        (json!(null), UsageState::Invalid),
        (json!("3000"), UsageState::Invalid),
    ] {
        let body = responses_body(
            json!({"input_tokens":15000,"output_tokens":0,"total_tokens":15000,"input_tokens_details":{"cached_tokens":12000,"cache_write_tokens":writes}}),
        );
        let (usage, inference) = pair(
            &OpenAiResponsesCodec,
            &p,
            "gpt-5.6",
            body.clone(),
            vec![json!({"type":"response.completed","response":body})],
        );
        assert_eq!(usage.state, state);
        let cost = c.estimate_cost_for_profile(
            &route,
            "openai",
            &usage,
            &inference,
            Submission::Interactive,
        );
        if state == UsageState::Complete {
            let u = usage.complete().unwrap();
            assert_eq!(
                (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens),
                (0, 12000, 3000)
            );
            assert_eq!(u.total(), 15000);
            assert!((cost.unwrap().total_cost - 0.0198).abs() < 1e-12);
        } else {
            assert!(matches!(cost, Err(LlmError::CostUnavailable { .. })));
        }
    }
}

#[test]
fn gemini_body_tiers_control_full_and_stream_costs_without_a_header() {
    let mut p = preset("gemini");
    p.models.retain(|m| m.request_model == "gemini-2.5-flash");
    p.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(1.0),
        output_per_million: Some(2.0),
        rules: vec![PriceRule {
            service_tier: ServiceTier::Fast,
            multiplier: Some(PriceMultiplier {
                factor: 2.0,
                buckets: vec![PriceBucket::Input, PriceBucket::Output],
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    let c = client(&p);
    let route = c.resolve("gemini-2.5-flash").unwrap();
    for (raw, tier, expected) in [
        ("priority", Some(ServiceTier::Fast), Some(6.0)),
        ("standard", Some(ServiceTier::Standard), Some(3.0)),
        ("future-tier", None, None),
    ] {
        let body = json!({"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1000000,"candidatesTokenCount":1000000,"totalTokenCount":2000000,"serviceTier":raw}});
        let (usage, mut inference) = pair(
            &GeminiCodec,
            &p,
            "gemini-2.5-flash",
            body.clone(),
            vec![body],
        );
        assert_eq!(inference.service_tier, tier);
        assert_eq!(inference.raw_service_tier.as_deref(), Some(raw));
        inference.requested_service_tier = Some(ServiceTier::Fast);
        let cost = c.estimate_cost_for_profile(
            &route,
            "gemini",
            &usage,
            &inference,
            Submission::Interactive,
        );
        if let Some(expected) = expected {
            assert_eq!(cost.unwrap().total_cost, expected);
        } else {
            assert!(matches!(cost, Err(LlmError::CostUnavailable { .. })));
        }
    }
}

#[test]
fn gemini_tier_aliases_are_canonicalized_and_conflicts_rejected_before_encoding() {
    for extra in [
        json!({"serviceTier":"priority"}),
        json!({"service_tier":"priority"}),
        json!({"serviceTier":"priority","service_tier":"priority"}),
    ] {
        let mut p = preset("gemini");
        p.extra["body"] = extra;
        let context = CodecContext::new(&p, "gemini-2.5-flash", RequestMode::Complete);
        let mut req = request("gemini-2.5-flash", Value::Null);
        let metadata = GeminiCodec.request_inference(&req, &context).unwrap();
        assert_eq!(metadata.requested_service_tier, Some(ServiceTier::Fast));
        assert_eq!(
            metadata.requested_raw_service_tier.as_deref(),
            Some("priority")
        );
        let http = GeminiCodec
            .encode_request(EncodeRequest::new(&req), &context)
            .unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["serviceTier"], "priority");
        assert!(body.get("service_tier").is_none());
        assert_eq!(
            GeminiCodec
                .encoded_body_len(EncodeRequest::new(&req), &context)
                .unwrap(),
            http.body.len()
        );
        req.service_tier = Some(ServiceTier::Standard);
        assert!(matches!(
            GeminiCodec.validate_request(&req, &context),
            Err(LlmError::InvalidRequest { .. })
        ));
        assert!(GeminiCodec
            .encode_request(EncodeRequest::new(&req), &context)
            .is_err());
    }
    let mut p = preset("gemini");
    p.extra["body"] = json!({"serviceTier":"priority","service_tier":"standard"});
    let req = request("gemini-2.5-flash", Value::Null);
    assert!(GeminiCodec
        .validate_request(
            &req,
            &CodecContext::new(&p, &req.model, RequestMode::Complete)
        )
        .is_err());
    p.extra["body"] = json!({"service_tier":"future-tier"});
    let ctx = CodecContext::new(&p, &req.model, RequestMode::Complete);
    let metadata = GeminiCodec.request_inference(&req, &ctx).unwrap();
    assert_eq!(metadata.requested_service_tier, None);
    assert_eq!(
        metadata.requested_raw_service_tier.as_deref(),
        Some("future-tier")
    );
    let body: Value = serde_json::from_slice(
        &GeminiCodec
            .encode_request(EncodeRequest::new(&req), &ctx)
            .unwrap()
            .body,
    )
    .unwrap();
    assert_eq!(body["serviceTier"], "future-tier");
    assert!(body.get("service_tier").is_none());
}

#[test]
fn anthropic_info_and_validation_enforce_model_specific_combinations() {
    let p = preset("anthropic");
    let c = client(&p);
    for name in ["claude-fable-5", "claude-fable-5-1"] {
        let model = c
            .models()
            .into_iter()
            .find(|m| m.request_model == name)
            .unwrap();
        assert_eq!(
            model.info.features.supports_mode(ThinkingMode::Disabled),
            CapabilitySupport::Unsupported
        );
        let req = request(name, json!({"mode":"disabled"}));
        assert!(matches!(
            AnthropicMessagesCodec.encode_request(
                EncodeRequest::new(&req),
                &CodecContext::new(&p, name, RequestMode::Complete)
            ),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    let model = c
        .models()
        .into_iter()
        .find(|m| m.request_model == "claude-opus-5")
        .unwrap();
    for effort in [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ] {
        let allowed = matches!(
            effort,
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        );
        assert_eq!(
            model
                .info
                .features
                .supports_effort(Some(ThinkingMode::Disabled), effort)
                != CapabilitySupport::Unsupported,
            allowed
        );
        let req = request("claude-opus-5", json!({"mode":"disabled","effort":effort}));
        assert_eq!(
            AnthropicMessagesCodec
                .encode_request(
                    EncodeRequest::new(&req),
                    &CodecContext::new(&p, &req.model, RequestMode::Complete)
                )
                .is_ok(),
            allowed
        );
    }
    for (name, thinking, allowed) in [
        (
            "claude-opus-5",
            json!({"mode":"adaptive","effort":"high"}),
            true,
        ),
        ("claude-opus-5", Value::Null, true),
        ("claude-fable-5", json!({"mode":"adaptive"}), true),
        ("claude-fable-5-1", json!({"mode":"adaptive"}), false),
        ("claude-fable-5-1", Value::Null, false),
        (
            "claude-opus-4-5-20251101",
            json!({"budget":{"tokens":2048}}),
            false,
        ),
        ("claude-opus-4-5-20251101", Value::Null, true),
    ] {
        let mut req = request(name, thinking);
        req.tools = serde_json::from_value(
            json!([{"name":"lookup","description":"test","input_schema":{"type":"object"}}]),
        )
        .unwrap();
        for choice in [
            ToolChoice::Any,
            ToolChoice::Tool {
                name: "lookup".into(),
            },
        ] {
            req.tool_choice = choice;
            let result = AnthropicMessagesCodec.encode_request(
                EncodeRequest::new(&req),
                &CodecContext::new(&p, name, RequestMode::Complete),
            );
            assert_eq!(result.is_ok(), allowed, "{name}: {result:?}");
        }
    }
}

fn priced_profile() -> ProviderProfile {
    let mut p = preset("grok");
    p.models.retain(|m| m.request_model == "grok-4.6");
    p.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(1.0),
        output_per_million: Some(2.0),
        rules: vec![PriceRule {
            service_tier: ServiceTier::Fast,
            multiplier: Some(PriceMultiplier {
                factor: 2.0,
                buckets: vec![PriceBucket::Input, PriceBucket::Output],
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    p
}
fn million_input() -> Usage {
    Usage {
        input_tokens: 1_000_000,
        ..Default::default()
    }
}
#[test]
fn preflight_price_uses_known_fast_default_without_overriding_an_explicit_tier() {
    for connection_default in [true, false] {
        let mut p = priced_profile();
        if connection_default {
            p.info.features.default_service_tier = Some(ServiceTier::Fast);
        } else {
            p.models[0].info.features.default_service_tier = Some(ServiceTier::Fast);
        }
        let c = client(&p);
        let route = c.resolve("grok-4.6").unwrap();
        for (requested, effective, expected) in [
            (None, ServiceTier::Fast, 2.0),
            (Some(ServiceTier::Fast), ServiceTier::Fast, 2.0),
            (Some(ServiceTier::Standard), ServiceTier::Standard, 1.0),
        ] {
            let context = PricingContext {
                service_tier: requested,
                ..Default::default()
            };
            let quote = c.price_quote_for_route(&route, &context).unwrap();
            assert_eq!(quote.context.service_tier, Some(effective));
            assert_eq!(quote.rates.unwrap().input_per_million, Some(expected));
            let estimate = c.estimate_cost(&route, &million_input(), &context).unwrap();
            assert_eq!(estimate.context.service_tier, Some(effective));
            assert_eq!(estimate.total_cost, expected);
        }
        let mut unpriced = p.clone();
        unpriced.models[0].pricing.as_mut().unwrap().rules.clear();
        let c = client(&unpriced);
        let route = c.resolve("grok-4.6").unwrap();
        let context = PricingContext::default();
        assert_eq!(
            c.price_quote_for_route(&route, &context).unwrap().status,
            PriceStatus::Unknown
        );
        assert!(matches!(
            c.estimate_cost(&route, &million_input(), &context),
            Err(LlmError::CostUnavailable { .. })
        ));
    }
}

#[test]
fn missing_actual_tier_cannot_use_standard_when_the_model_or_connection_defaults_to_fast() {
    for connection_default in [true, false] {
        let mut p = priced_profile();
        if connection_default {
            p.info.features.default_service_tier = Some(ServiceTier::Fast);
        } else {
            p.models[0].info.features.default_service_tier = Some(ServiceTier::Fast);
        }
        let c = client(&p);
        let route = c.resolve("grok-4.6").unwrap();
        assert_eq!(
            c.models()[0].info.features.default_service_tier,
            Some(ServiceTier::Fast)
        );
        let usage = UsageReport::measured(million_input(), UsageState::Complete);
        assert!(matches!(
            c.estimate_cost_for_profile(
                &route,
                "grok",
                &usage,
                &InferenceReport::default(),
                Submission::Interactive
            ),
            Err(LlmError::CostUnavailable { .. })
        ));
        for (tier, cost) in [(ServiceTier::Standard, 1.0), (ServiceTier::Fast, 2.0)] {
            assert_eq!(
                c.estimate_cost_for_profile(
                    &route,
                    "grok",
                    &usage,
                    &InferenceReport {
                        service_tier: Some(tier),
                        ..Default::default()
                    },
                    Submission::Interactive
                )
                .unwrap()
                .total_cost,
                cost
            );
        }
    }
}
struct ClockAt(u64);
impl Clock for ClockAt {
    fn now(&self) -> std::time::SystemTime {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(self.0)
    }
}
#[test]
fn fast_inherited_dated_prices_require_execution_time_and_keep_the_original_period() {
    let mut p = priced_profile();
    let boundary = PriceBoundary {
        local: "2027-01-01T00:00:00".into(),
        time_zone: Some("Asia/Shanghai".into()),
    };
    let cutoff = boundary.utc_bounds().unwrap().0;
    let prices = p.models[0].pricing.as_mut().unwrap();
    prices.rules.extend([
        PriceRule {
            valid_until: Some(boundary.clone()),
            rates: TokenRates {
                input_per_million: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        },
        PriceRule {
            valid_from: Some(boundary),
            rates: TokenRates {
                input_per_million: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        },
    ]);
    let mut builder = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[p])
        .with_region(Region::International);
    builder.with_clock(Arc::new(ClockAt(cutoff + 86400)));
    let c = builder.build().unwrap();
    let route = c.resolve("grok-4.6").unwrap();
    let usage = UsageReport::measured(million_input(), UsageState::Complete);
    for (time, expected) in [
        (None, None),
        (Some(cutoff - 1), Some(2.0)),
        (Some(cutoff), Some(20.0)),
    ] {
        let inference = InferenceReport {
            executed_at: time,
            service_tier: Some(ServiceTier::Fast),
            ..Default::default()
        };
        let cost = c.estimate_cost_for_profile(
            &route,
            "grok",
            &usage,
            &inference,
            Submission::Interactive,
        );
        if let Some(expected) = expected {
            assert_eq!(cost.unwrap().total_cost, expected);
        } else {
            assert!(matches!(cost, Err(LlmError::CostUnavailable { .. })));
        }
    }
}

#[test]
fn every_public_estimate_uses_the_same_context_band_as_its_quote() {
    let p = preset("gemini");
    let c = client(&p);
    let route = c.resolve("gemini-2.5-pro").unwrap();
    for submission in [Submission::Interactive, Submission::Batch] {
        let context = PricingContext {
            submission,
            input_tokens: Some(1_000_000),
            ..Default::default()
        };
        let quote = c.price_quote_for_route(&route, &context).unwrap();
        let preflight = c.estimate_cost(&route, &million_input(), &context).unwrap();
        let usage = UsageReport::measured(million_input(), UsageState::Complete);
        let actual = c
            .estimate_cost_for_profile(
                &route,
                "gemini",
                &usage,
                &InferenceReport {
                    service_tier: Some(ServiceTier::Standard),
                    ..Default::default()
                },
                submission,
            )
            .unwrap();
        assert_eq!(
            preflight.total_cost,
            quote.rates.unwrap().input_per_million.unwrap()
        );
        assert_eq!(actual.total_cost, preflight.total_cost);
        assert_eq!(
            actual.total_cost,
            if submission == Submission::Interactive {
                2.5
            } else {
                1.25
            }
        );
    }
}

#[test]
fn selected_rates_preserve_subsets_batch_tariffs_and_reject_missing_or_overflowing_costs() {
    let mut p = priced_profile();
    p.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(2.0),
        output_per_million: Some(4.0),
        cache_read_per_million: Some(0.25),
        cache_write_per_million: Some(3.0),
        cache_write_1h_per_million: Some(5.0),
        reasoning_per_million: Some(6.0),
        batch: Some(BatchPricing {
            input_per_million: Some(0.9),
            ..Default::default()
        }),
        ..Default::default()
    });
    let c = client(&p);
    let route = c.resolve("grok-4.6").unwrap();
    let usage = Usage {
        input_tokens: 1_000_000,
        output_tokens: 2_000_000,
        cache_read_tokens: 1_000_000,
        cache_write_tokens: 1_000_000,
        cache_write_1h_tokens: 400_000,
        reasoning_tokens: 1_000_000,
        ..Default::default()
    };
    let cost = c
        .estimate_cost(&route, &usage, &PricingContext::default())
        .unwrap();
    assert!((cost.total_cost - 16.05).abs() < 1e-12);
    assert_eq!(cost.output_cost, 4.0);
    assert_eq!(cost.reasoning_cost, 6.0);
    assert!(matches!(
        c.estimate_cost(
            &route,
            &usage,
            &PricingContext {
                submission: Submission::Batch,
                ..Default::default()
            }
        ),
        Err(LlmError::CostUnavailable { .. })
    ));
    for bad in [
        Usage {
            output_tokens: 1,
            reasoning_tokens: 2,
            ..Default::default()
        },
        Usage {
            cache_write_tokens: 1,
            cache_write_1h_tokens: 2,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            c.estimate_cost(&route, &bad, &PricingContext::default()),
            Err(LlmError::CostUnavailable { .. })
        ));
    }
    p.models[0].pricing.as_mut().unwrap().input_per_million = Some(f64::MAX);
    p.models[0].pricing.as_mut().unwrap().output_per_million = Some(f64::MAX);
    let c = client(&p);
    let route = c.resolve("grok-4.6").unwrap();
    assert!(matches!(
        c.estimate_cost(
            &route,
            &Usage {
                input_tokens: 1_000_000,
                output_tokens: 1_000_000,
                ..Default::default()
            },
            &PricingContext::default()
        ),
        Err(LlmError::CostUnavailable { .. })
    ));
}
