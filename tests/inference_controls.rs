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
fn request(model: &str, thinking: Value, tier: Value) -> CompletionRequest {
    serde_json::from_value(json!({"model":model,"messages":[],"max_tokens":8192,"thinking":thinking,"service_tier":tier})).unwrap()
}
fn codec(p: &ProviderProfile) -> Box<dyn WireCodec> {
    match p.protocol {
        ProtocolFamily::AnthropicMessages => Box::new(AnthropicMessagesCodec),
        ProtocolFamily::GeminiGenerateContent => Box::new(GeminiCodec),
        ProtocolFamily::OpenAiResponses => Box::new(OpenAiResponsesCodec),
        _ => Box::new(OpenAiChatCodec),
    }
}
fn encode(p: &ProviderProfile, req: &CompletionRequest) -> Result<HttpRequest, LlmError> {
    codec(p).encode_request(
        EncodeRequest::new(req),
        &CodecContext::new(p, &req.model, RequestMode::Complete),
    )
}
fn body(p: &ProviderProfile, req: &CompletionRequest) -> Value {
    serde_json::from_slice(&encode(p, req).unwrap().body).unwrap()
}
fn client(profiles: &[ProviderProfile]) -> LlmClient {
    LlmClientBuilder::with_transport(Arc::new(support::NoHttp), profiles)
        .with_region(Region::International)
        .build()
        .unwrap()
}

#[test]
fn supported_controls_are_encoded_in_each_endpoint_vocabulary() {
    let cases = [
        (
            "openai",
            "gpt-5.6",
            json!({"effort":"max"}),
            json!("fast"),
            "/reasoning/effort",
            json!("max"),
            "/service_tier",
            json!("fast"),
        ),
        (
            "anthropic",
            "claude-opus-5",
            json!({"mode":"adaptive","effort":"xhigh"}),
            json!("fast"),
            "/thinking/type",
            json!("adaptive"),
            "/speed",
            json!("fast"),
        ),
        (
            "gemini",
            "gemini-2.5-flash",
            json!({"budget":"dynamic"}),
            json!("fast"),
            "/generationConfig/thinkingConfig/thinkingBudget",
            json!(-1),
            "/serviceTier",
            json!("priority"),
        ),
        (
            "qwen",
            "qwen3.8-flash",
            json!({"budget":{"tokens":2048}}),
            Value::Null,
            "/thinking_budget",
            json!(2048),
            "/enable_thinking",
            json!(true),
        ),
        (
            "qwen-search",
            "qwen3.8-flash",
            json!({"effort":"high"}),
            Value::Null,
            "/reasoning/effort",
            json!("high"),
            "/max_output_tokens",
            json!(8192),
        ),
        (
            "deepseek",
            "deepseek-v4-pro",
            json!({"mode":"enabled","effort":"max"}),
            Value::Null,
            "/thinking/type",
            json!("enabled"),
            "/reasoning_effort",
            json!("max"),
        ),
        (
            "deepseek-search",
            "deepseek-v4-pro",
            json!({"mode":"enabled","effort":"max"}),
            Value::Null,
            "/thinking/type",
            json!("enabled"),
            "/output_config/effort",
            json!("max"),
        ),
        (
            "kimi",
            "kimi-k3",
            json!({"effort":"max"}),
            Value::Null,
            "/reasoning_effort",
            json!("max"),
            "/model",
            json!("kimi-k3"),
        ),
        (
            "zai",
            "glm-5.3",
            json!({"effort":"max"}),
            Value::Null,
            "/reasoning_effort",
            json!("max"),
            "/model",
            json!("glm-5.3"),
        ),
        (
            "minimax",
            "MiniMax-M3",
            json!({"mode":"adaptive"}),
            json!("fast"),
            "/thinking/type",
            json!("adaptive"),
            "/service_tier",
            json!("priority"),
        ),
        (
            "grok",
            "grok-4.6",
            json!({"effort":"xhigh"}),
            json!("fast"),
            "/reasoning_effort",
            json!("xhigh"),
            "/service_tier",
            json!("priority"),
        ),
        (
            "grok-responses",
            "grok-4.6",
            json!({"effort":"xhigh"}),
            json!("fast"),
            "/reasoning/effort",
            json!("xhigh"),
            "/service_tier",
            json!("priority"),
        ),
        (
            "openrouter",
            "anthropic/claude-opus-4.6",
            json!({"budget":{"tokens":2048}}),
            json!("fast"),
            "/reasoning/max_tokens",
            json!(2048),
            "/service_tier",
            json!("priority"),
        ),
    ];
    for (name, model, thinking, tier, path, want, tier_path, tier_want) in cases {
        let p = preset(name);
        let req = request(model, thinking, tier);
        let b = body(&p, &req);
        assert_eq!(b.pointer(path), Some(&want), "{name}");
        assert_eq!(b.pointer(tier_path), Some(&tier_want), "{name}");
        let context = CodecContext::new(&p, model, RequestMode::Complete);
        assert_eq!(
            codec(&p)
                .encoded_body_len(EncodeRequest::new(&req), &context)
                .unwrap(),
            encode(&p, &req).unwrap().body.len()
        );
    }
}

#[test]
fn known_unsupported_controls_fail_instead_of_being_dropped() {
    let cases = [
        (
            "openai",
            "gpt-5.6",
            json!({"budget":{"tokens":2048}}),
            Value::Null,
        ),
        (
            "deepseek-search",
            "deepseek-v4-pro",
            json!({"budget":{"tokens":2048}}),
            Value::Null,
        ),
        (
            "qwen-search",
            "qwen3.8-flash",
            json!({"budget":{"tokens":2048}}),
            Value::Null,
        ),
        (
            "anthropic",
            "claude-opus-4-7",
            json!({"budget":{"tokens":2048}}),
            Value::Null,
        ),
        ("anthropic", "claude-sonnet-5", Value::Null, json!("fast")),
        (
            "gemini",
            "gemini-2.5-pro",
            json!({"mode":"disabled"}),
            Value::Null,
        ),
        ("kimi", "kimi-k3", json!({"mode":"disabled"}), Value::Null),
        (
            "minimax",
            "MiniMax-M2.7",
            json!({"mode":"disabled"}),
            Value::Null,
        ),
        ("zai", "glm-5.3", Value::Null, json!("fast")),
        (
            "github-copilot",
            "gpt-5",
            json!({"effort":"high"}),
            Value::Null,
        ),
    ];
    for (name, model, thinking, tier) in cases {
        assert!(
            matches!(
                encode(&preset(name), &request(model, thinking, tier)),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "{name} {model}"
        );
    }
}

#[test]
fn budgets_and_control_combinations_are_validated() {
    for n in [0, 1, 1023, 8192] {
        assert!(matches!(
            encode(
                &preset("anthropic"),
                &request(
                    "claude-opus-4-5-20251101",
                    json!({"budget":{"tokens":n}}),
                    Value::Null
                )
            ),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    for cfg in [
        json!({"mode":"disabled","budget":{"tokens":1024}}),
        json!({"budget":{"tokens":1024},"effort":"high"}),
        json!({"mode":"enabled","effort":"none"}),
    ] {
        assert!(matches!(
            encode(
                &preset("gemini"),
                &request("gemini-3.1-pro-preview", cfg, Value::Null)
            ),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
    let b = body(
        &preset("gemini"),
        &request("gemini-2.5-flash", json!({"mode":"disabled"}), Value::Null),
    );
    assert_eq!(
        b.pointer("/generationConfig/thinkingConfig/thinkingBudget"),
        Some(&json!(0))
    );
}

#[test]
fn beta_headers_accumulate_and_nested_extras_survive() {
    let mut p = preset("anthropic");
    p.extra["betas"] = json!(["another-beta", "fast-mode-2026-02-01"]);
    p.extra["headers"] = json!({"Anthropic-Beta":"header-beta,another-beta"});
    p.extra["body"] =
        json!({"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}});
    let req = request(
        "claude-opus-5",
        json!({"mode":"adaptive","effort":"high"}),
        json!("fast"),
    );
    let http = encode(&p, &req).unwrap();
    let b: Value = serde_json::from_slice(&http.body).unwrap();
    assert_eq!(b["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        http.headers
            .iter()
            .find(|(k, _)| k == "anthropic-beta")
            .unwrap()
            .1,
        "another-beta,fast-mode-2026-02-01,header-beta"
    );
    p.extra["body"]["output_config"]["effort"] = json!("low");
    assert!(matches!(
        encode(&p, &req),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn public_model_info_exposes_effective_features_and_one_price_source() {
    let c = client(&[
        preset("anthropic"),
        preset("gemini"),
        preset("deepseek-search"),
    ]);
    let rows = c.models();
    let opus = rows
        .iter()
        .find(|m| m.request_model == "claude-opus-5")
        .unwrap();
    assert_eq!(opus.info.features.fast, CapabilitySupport::Supported);
    assert_eq!(
        opus.info.features.budget.support,
        CapabilitySupport::Unsupported
    );
    assert!(opus
        .info
        .features
        .effort
        .levels
        .as_ref()
        .unwrap()
        .contains(&ReasoningEffort::XHigh));
    assert_eq!(opus.info.pricing, opus.pricing);
    assert!(!c.providers()[0].info.pricing.sources.is_empty());
    let d = rows
        .iter()
        .find(|m| m.request_model == "deepseek-v4-pro")
        .unwrap();
    assert_eq!(d.info.features.fast, CapabilitySupport::Unsupported);
}

#[test]
fn effective_info_does_not_advertise_defaults_excluded_by_the_connection() {
    let model = InferenceFeatures {
        thinking: CapabilitySupport::Supported,
        modes: Some(vec![ThinkingMode::Enabled]),
        default_mode: Some(ThinkingMode::Enabled),
        effort: EffortSupport {
            support: CapabilitySupport::Supported,
            levels: Some(vec![ReasoningEffort::Low, ReasoningEffort::High]),
            default: Some(ReasoningEffort::Low),
            ..Default::default()
        },
        default_service_tier: Some(ServiceTier::Fast),
        ..Default::default()
    };
    let effective = model.on_connection(&InferenceFeatures {
        thinking: CapabilitySupport::Unsupported,
        effort: EffortSupport {
            levels: Some(vec![ReasoningEffort::High]),
            ..Default::default()
        },
        fast: CapabilitySupport::Unsupported,
        ..Default::default()
    });
    assert_eq!(effective.modes, Some(vec![]));
    assert_eq!(effective.default_mode, None);
    assert_eq!(effective.effort.default, None);
    assert_eq!(effective.default_service_tier, None);
}

#[test]
fn standard_fast_quotes_and_unknown_prices_do_not_depend_on_effort() {
    // Make the unknown-price case explicit rather than depending on which
    // service tiers the periodically refreshed catalog happens to publish.
    let mut openai = preset("openai");
    openai
        .models
        .iter_mut()
        .find(|model| model.request_model == "gpt-4.1")
        .unwrap()
        .pricing
        .as_mut()
        .unwrap()
        .rules
        .retain(|rule| rule.service_tier != ServiceTier::Fast);
    let c = client(&[
        preset("anthropic"),
        preset("minimax-intl"),
        openai,
        preset("kimi-code"),
    ]);
    let context = PricingContext {
        service_tier: Some(ServiceTier::Fast),
        input_tokens: Some(1000),
        ..Default::default()
    };
    let opus = c
        .price_quote("claude-opus-5", Some("anthropic"), &context)
        .unwrap();
    assert_eq!(opus.status, PriceStatus::Priced);
    assert_eq!(opus.rates.unwrap().input_per_million, Some(10.0));
    assert_eq!(opus.multiplier.unwrap().factor, 2.0);
    let mini = c
        .price_quote("MiniMax-M2.7", Some("minimax-intl"), &context)
        .unwrap();
    assert_eq!(mini.multiplier.unwrap().factor, 1.5);
    assert_eq!(
        c.price_quote("claude-sonnet-5", Some("anthropic"), &context)
            .unwrap()
            .status,
        PriceStatus::Unsupported
    );
    assert_eq!(
        c.price_quote("gpt-4.1", Some("openai"), &context)
            .unwrap()
            .status,
        PriceStatus::Unknown
    );
    let quota = c
        .price_quote(
            "kimi-for-coding-highspeed",
            Some("kimi-code"),
            &PricingContext::default(),
        )
        .unwrap();
    assert_eq!(quota.quota.unwrap().multiplier, 3.0);
    assert!(quota.rates.is_none());
    // There is deliberately no effort field on PricingContext.
    let route = c.resolve_in("claude-opus-5", Some("anthropic")).unwrap();
    let make = |effort| CompletionResponse {
        inference: InferenceReport {
            requested_effort: Some(effort),
            service_tier: Some(ServiceTier::Standard),
            ..Default::default()
        },
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content: vec![],
        },
        web_search: None,
        file_search: None,
        stop_reason: StopReason::EndTurn,
        usage: UsageReport::measured(
            Usage {
                input_tokens: 1000,
                output_tokens: 100,
                reasoning_tokens: 80,
                ..Default::default()
            },
            UsageState::Complete,
        ),
        model: route.request_model.clone(),
        response_id: None,
        executed_profile: Some("anthropic".into()),
    };
    let low = c
        .estimate_actual_cost(&route, &make(ReasoningEffort::Low), Submission::Interactive)
        .unwrap();
    let high = c
        .estimate_actual_cost(
            &route,
            &make(ReasoningEffort::High),
            Submission::Interactive,
        )
        .unwrap();
    assert_eq!(low.total_cost, high.total_cost);
    assert_eq!(low.currency, high.currency);
}

#[test]
fn context_bands_and_cache_write_ttl_are_priced_without_double_counting() {
    let c = client(&[preset("gemini"), preset("anthropic")]);
    let q = |n| {
        c.price_quote(
            "gemini-2.5-pro",
            Some("gemini"),
            &PricingContext {
                service_tier: Some(ServiceTier::Fast),
                input_tokens: n,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(
        q(Some(200000)).rates.unwrap().output_per_million,
        Some(18.0)
    );
    assert_eq!(
        q(Some(200001)).rates.unwrap().output_per_million,
        Some(27.0)
    );
    assert_eq!(q(None).status, PriceStatus::Unknown);
    let route = c.resolve_in("claude-opus-5", Some("anthropic")).unwrap();
    let usage = Usage {
        cache_write_tokens: 1000,
        cache_write_1h_tokens: 400,
        output_tokens: 100,
        reasoning_tokens: 80,
        ..Default::default()
    };
    let cost = c
        .estimate_cost(
            &route,
            &usage,
            &PricingContext {
                service_tier: Some(ServiceTier::Fast),
                ..Default::default()
            },
        )
        .unwrap();
    assert!((cost.cache_write_cost - (600.0 * 12.5 + 400.0 * 20.0) / 1e6).abs() < 1e-12);
    assert_eq!(cost.output_cost, 100.0 * 50.0 / 1e6);
}

#[test]
fn full_and_streamed_inference_reports_match_and_survive_unknown_values() {
    for (name, model, body, frames, headers) in [
        (
            "grok",
            "grok-4.6",
            json!({"model":"grok-4.6","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2},"service_tier":"default"}),
            vec![
                json!({"model":"grok-4.6","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2},"service_tier":"default"}),
            ],
            vec![],
        ),
        (
            "anthropic",
            "claude-opus-5",
            json!({"model":"claude-opus-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2,"speed":"fast","output_tokens_details":{"thinking_tokens":1}}}),
            vec![
                json!({"type":"message_start","message":{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":0,"speed":"fast"}}}),
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2,"output_tokens_details":{"thinking_tokens":1}}}),
                json!({"type":"message_stop"}),
            ],
            vec![],
        ),
        (
            "gemini",
            "gemini-2.5-flash",
            json!({"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}),
            vec![
                json!({"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}),
            ],
            vec![("X-Gemini-Service-Tier".into(), "priority".into())],
        ),
    ] {
        let p = preset(name);
        let ctx = CodecContext::new(&p, model, RequestMode::Stream);
        let codec = codec(&p);
        let response = codec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: headers.clone(),
                    body: serde_json::to_vec(&body).unwrap().into(),
                },
                &ctx,
            )
            .unwrap();
        let mut decoder = codec.stream_decoder(&ctx);
        decoder.set_response_headers(&headers);
        let mut events = vec![];
        for frame in frames {
            let bytes = format!("data: {frame}\n\n");
            for chunk in bytes.as_bytes().chunks(3) {
                events.extend(decoder.push_bytes(chunk));
            }
        }
        if name == "grok" {
            events.extend(decoder.push_bytes(b"data: [DONE]\n\n"));
        }
        events.extend(decoder.finish());
        assert!(events.iter().all(Result::is_ok), "{name}: {events:?}");
        assert_eq!(response.inference, decoder.inference_report(), "{name}");
        assert_eq!(response.usage, decoder.usage_report(), "{name}");
        let end = events
            .iter()
            .find_map(|e| match e {
                Ok(StreamEvent::End { inference, .. }) => Some(inference),
                _ => None,
            })
            .unwrap();
        assert_eq!(end, &response.inference);
    }
}

#[test]
fn directory_observations_keep_absent_controls_unknown() {
    use lingxi_llm_client::directory::{ModelDirectory, OpenAiChatDirectory};
    let response=HttpResponse {status:200,headers:vec![],body:serde_json::to_vec(&json!({"data":[{"id":"r","reasoning":{"supported_efforts":["low","high"],"mandatory":true}}]})).unwrap().into()};
    let page = OpenAiChatDirectory.decode_page(&response).unwrap();
    let f = page.models[0].inference_features.as_ref().unwrap();
    assert_eq!(
        f.effort.levels,
        Some(vec![ReasoningEffort::Low, ReasoningEffort::High])
    );
    assert_eq!(f.budget.support, CapabilitySupport::Unknown);
    assert_eq!(f.fast, CapabilitySupport::Unknown);
}

#[test]
fn thinking_config_accepts_only_the_current_schema() {
    for invalid in [
        json!({"budget_tokens": 2048}),
        json!({"budget": 2048}),
        json!({"effort": "x_high"}),
    ] {
        assert!(serde_json::from_value::<ThinkingConfig>(invalid).is_err());
    }
    let config = ThinkingConfig {
        budget: Some(ThinkingBudget::Tokens(2048)),
        ..Default::default()
    };
    assert_eq!(
        serde_json::from_value::<ThinkingConfig>(serde_json::to_value(&config).unwrap()).unwrap(),
        config
    );
}

#[test]
fn dated_prices_and_context_multipliers_match_exact_conditions() {
    let c = client(&[preset("gemini"), preset("minimax-intl")]);
    let quote = |date| {
        c.price_quote(
            "gemini-3.7-flash",
            Some("gemini"),
            &PricingContext {
                service_tier: Some(ServiceTier::Fast),
                input_tokens: Some(1000),
                unix_seconds: Some(date),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let before = quote(1_798_761_600 - 14 * 3600 - 1);
    let after = quote(1_798_761_600 + 12 * 3600);
    assert_eq!(before.rates.unwrap().input_per_million, Some(1.35));
    assert_eq!(after.rates.unwrap().input_per_million, Some(2.7));
    assert_eq!(
        before.rule.unwrap().valid_until.as_ref().unwrap().time_zone,
        None
    );
    assert_eq!(quote(1_798_761_600).status, PriceStatus::Unknown);
    assert_eq!(
        after.rule.unwrap().valid_from.as_ref().unwrap().local,
        "2027-01-01T00:00:00"
    );
    let long = c
        .price_quote(
            "MiniMax-M3",
            Some("minimax-intl"),
            &PricingContext {
                service_tier: Some(ServiceTier::Fast),
                input_tokens: Some(512_001),
                ..Default::default()
            },
        )
        .unwrap();
    assert!((long.rates.unwrap().input_per_million.unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(long.multiplier.unwrap().factor, 1.5);
}

#[test]
fn overlapping_rules_and_invalid_batch_rates_are_rejected_at_configuration_time() {
    let mut p = preset("anthropic");
    let prices = p
        .models
        .iter_mut()
        .find(|m| m.request_model == "claude-opus-5")
        .unwrap()
        .pricing
        .as_mut()
        .unwrap();
    prices.rules.push(prices.rules[0].clone());
    assert!(
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[p])
            .with_region(Region::International)
            .build()
            .is_err()
    );
    let mut p = preset("anthropic");
    p.models[0].pricing.as_mut().unwrap().batch = Some(BatchPricing {
        input_per_million: Some(-1.0),
        ..Default::default()
    });
    assert!(
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[p])
            .with_region(Region::International)
            .build()
            .is_err()
    );
}

struct TierTransport {
    tier: Option<&'static str>,
    fail_primary: bool,
    seen: std::sync::Mutex<Vec<(String, Value)>>,
}
struct PricingClock(std::sync::atomic::AtomicU64);
impl Clock for PricingClock {
    fn now(&self) -> std::time::SystemTime {
        std::time::UNIX_EPOCH
            + std::time::Duration::from_secs(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
}
#[async_trait::async_trait]
impl Transport for TierTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        self.seen
            .lock()
            .unwrap()
            .push((request.url.clone(), body.clone()));
        if self.fail_primary && request.url.contains("primary.test") {
            return Err(LlmError::Overloaded {
                message: "busy".into(),
            });
        }
        let mut response = json!({"id":"r","model":"grok-4.6","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1_000_000,"completion_tokens":800_000,"completion_tokens_details":{"reasoning_tokens":200_000}}});
        if let Some(tier) = self.tier {
            response["service_tier"] = json!(tier);
        }
        let bytes = if body["stream"] == true {
            response["choices"][0]["delta"] = json!({"content":"ok"});
            response["choices"][0]
                .as_object_mut()
                .unwrap()
                .remove("message");
            format!("data: {response}\n\ndata: [DONE]\n\n")
        } else {
            response.to_string()
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: bytes.into(),
        }
        .into())
    }
}
fn tier_profiles() -> Vec<ProviderProfile> {
    ["primary", "secondary"]
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let mut p = preset("grok");
            p.profile_name = name.into();
            p.auth = AuthStrategy::None;
            p.base_url = format!("https://{name}.test/v1");
            p.models.retain(|m| m.request_model == "grok-4.6");
            p.models[0].pricing = Some(TokenPricing {
                input_per_million: Some(if i == 0 { 1.0 } else { 2.0 }),
                output_per_million: Some(if i == 0 { 2.0 } else { 4.0 }),
                rules: vec![PriceRule {
                    service_tier: ServiceTier::Fast,
                    multiplier: Some(PriceMultiplier {
                        factor: 3.0,
                        buckets: vec![PriceBucket::Input, PriceBucket::Output],
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            });
            p.connection = ConnectionSpec {
                group: Some("tiers".into()),
                connection_id: Some(name.into()),
                order: i as u32,
                failover: FailoverTriggers {
                    overloaded: true,
                    ..Default::default()
                },
                ..Default::default()
            };
            p
        })
        .collect()
}

#[tokio::test]
async fn full_and_stream_costs_follow_actual_tier_and_failover_connection() {
    for (reported, expected_tier, expected_cost) in [
        (Some("default"), Some(ServiceTier::Standard), Some(6.0)),
        (Some("priority"), Some(ServiceTier::Fast), Some(18.0)),
        (None, None, None),
        (Some("future-tier"), None, None),
    ] {
        let http = Arc::new(TierTransport {
            tier: reported,
            fail_primary: true,
            seen: Default::default(),
        });
        let mut builder = LlmClientBuilder::with_transport(http.clone(), &tier_profiles())
            .with_region(Region::International);
        builder.with_clock(Arc::new(PricingClock(std::sync::atomic::AtomicU64::new(
            12 * 3600,
        ))));
        let c = builder.build().unwrap();
        let req = request("grok-4.6", json!({"effort":"high"}), json!("fast"));
        let route = c.resolve_in(&req.model, Some("primary")).unwrap();
        let response = c
            .complete_in("primary", &req, &RequestOptions::default())
            .await
            .unwrap();
        let mut stream = c
            .stream_in("primary", &req, &RequestOptions::default())
            .await
            .unwrap();
        while let Some(event) = stream.next().await {
            if let StreamEvent::End { inference, .. } = event.unwrap() {
                assert_eq!(inference, response.inference);
            }
        }
        assert_eq!(response.executed_profile.as_deref(), Some("secondary"));
        assert_eq!(stream.executed_profile(), "secondary");
        assert_eq!(stream.inference_report(), response.inference);
        assert_eq!(
            response.inference.requested_effort,
            Some(ReasoningEffort::High)
        );
        assert_eq!(response.inference.reported_effort, None);
        assert_eq!(
            response.inference.requested_service_tier,
            Some(ServiceTier::Fast)
        );
        assert_eq!(response.inference.service_tier, expected_tier);
        assert_eq!(response.inference.raw_service_tier.as_deref(), reported);
        for cost in [
            c.estimate_actual_cost(&route, &response, Submission::Interactive),
            c.estimate_stream_cost(&route, &stream, Submission::Interactive),
        ] {
            match expected_cost {
                Some(expected) => assert_eq!(cost.unwrap().total_cost, expected),
                None => assert!(matches!(cost, Err(LlmError::CostUnavailable { .. }))),
            }
        }
        let seen = http.seen.lock().unwrap();
        assert_eq!(seen.len(), 4);
        for (_, body) in seen.iter() {
            assert_eq!(body["service_tier"], "priority");
            assert_eq!(body["reasoning_effort"], "high");
            assert_eq!(body["model"], "grok-4.6");
        }
    }
}

#[tokio::test]
async fn actual_cost_keeps_dispatch_time_when_the_clock_moves_into_another_price_period() {
    let mut profiles = tier_profiles();
    profiles.truncate(1);
    profiles[0].pricing.peak = Some(PeakSchedule {
        weekdays_only: false,
        utc_windows: vec!["09:00-17:00".into()],
        off_peak_multiplier: 0.5,
    });
    let clock = Arc::new(PricingClock(std::sync::atomic::AtomicU64::new(12 * 3600)));
    let http = Arc::new(TierTransport {
        tier: Some("default"),
        fail_primary: false,
        seen: Default::default(),
    });
    let mut builder =
        LlmClientBuilder::with_transport(http, &profiles).with_region(Region::International);
    builder.with_clock(clock.clone());
    let c = builder.build().unwrap();
    let req = request("grok-4.6", Value::Null, Value::Null);
    let route = c.resolve(&req.model).unwrap();
    let mut response = c.complete(&req, &RequestOptions::default()).await.unwrap();
    clock
        .0
        .store(20 * 3600, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        c.estimate_actual_cost(&route, &response, Submission::Interactive)
            .unwrap()
            .total_cost,
        3.0
    );
    assert_eq!(
        c.estimate_cost(
            &route,
            response.usage.complete().unwrap(),
            &PricingContext::default()
        )
        .unwrap()
        .total_cost,
        1.5
    );
    response.inference.executed_at = None;
    assert!(matches!(
        c.estimate_actual_cost(&route, &response, Submission::Interactive),
        Err(LlmError::CostUnavailable { .. })
    ));
}

#[test]
fn regional_prices_keep_their_published_currency() {
    let c = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[preset("minimax")])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    let route = c.resolve("MiniMax-M2.7").unwrap();
    let cost = c
        .estimate_cost(
            &route,
            &Usage {
                input_tokens: 1_000_000,
                ..Default::default()
            },
            &PricingContext {
                service_tier: Some(ServiceTier::Fast),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(cost.currency, "CNY");
    assert!((cost.total_cost - 3.15).abs() < 1e-12);
}

#[test]
fn subscription_quota_is_not_a_batch_or_monetary_price() {
    let c = client(&[preset("kimi-code"), preset("zai-coding")]);
    let quote = c
        .price_quote("glm-4.7", Some("zai-coding"), &PricingContext::default())
        .unwrap();
    assert_eq!(quote.status, PriceStatus::Unknown);
    assert!(quote.currency.is_empty());
    assert_eq!(quote.unit, "subscription_quota");
    let quote = c
        .price_quote(
            "kimi-for-coding-highspeed",
            Some("kimi-code"),
            &PricingContext {
                submission: Submission::Batch,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(quote.status, PriceStatus::Unknown);
    assert!(quote.quota.is_none());
    assert!(quote.rates.is_none());
}

#[tokio::test]
async fn failover_revalidates_fast_before_contacting_an_unsupported_connection() {
    let mut profiles = tier_profiles();
    profiles[1].models[0].info.features.fast = CapabilitySupport::Unsupported;
    let http = Arc::new(TierTransport {
        tier: Some("default"),
        fail_primary: true,
        seen: Default::default(),
    });
    let c = LlmClientBuilder::with_transport(http.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let result = c
        .complete_in(
            "primary",
            &request("grok-4.6", Value::Null, json!("fast")),
            &RequestOptions::default(),
        )
        .await;
    assert!(matches!(
        result,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(http.seen.lock().unwrap().len(), 1);
}
