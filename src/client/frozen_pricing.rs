//! Prices captured with the exact model row before dispatch.
use super::{price_query, pricing, route::PricingModelRef};
use crate::protocol::*;
use crate::providers::anthropic::fallback_response::UsageIterations;
use serde::Serialize;

/// Native cNe line item and pricing input. Missing model rows or rates are
/// represented as incomplete rather than borrowed from another iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicFallbackCostComponentKind {
    Iteration,
    ExcludedRefusalTerminal,
    SkippedMissingModel,
    SkippedNonFallbackIteration,
    FirstModelServerToolUse,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicFallbackCostBuckets {
    pub input_cost_usd: Option<f64>,
    pub output_cost_usd: Option<f64>,
    pub cache_read_cost_usd: Option<f64>,
    pub cache_write_cost_usd: Option<f64>,
    pub web_search_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicFallbackCostComponent {
    pub kind: AnthropicFallbackCostComponentKind,
    pub iteration_index: Option<usize>,
    pub model: Option<String>,
    pub pricing_model: Option<PricingModelRef>,
    pub usage: AnthropicFallbackCostUsage,
    pub buckets: AnthropicFallbackCostBuckets,
    pub total_cost_usd: Option<f64>,
    pub pricing_source: Option<String>,
    /// Missing frozen catalog facts remain visible to host accounting.
    pub unmetered_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicFallbackCostUsage {
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cache_read_input_tokens: f64,
    /// Native iteration entries do not split one-hour cache writes, so these
    /// counts use the ordinary cache-write tariff like Mwe.
    pub cache_creation_input_tokens: f64,
    /// `None` preserves an absent `server_tool_use` field; native Mwe treats
    /// it as zero when calculating the helper result.
    pub web_search_requests: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicFallbackCostCompleteness {
    Complete,
    Incomplete,
}

/// A frozen, multi-model cost quote following native Anthropic cNe iteration
/// selection. `total_cost_usd` is absent unless every charged component has
/// an explicit frozen USD tariff and the selected summary identity resolves
/// to one exact model row in the captured provider snapshot.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicFallbackCostQuote {
    pub served_fallback_model: String,
    /// Ucn(configured lane model, served model), including native Mn marker
    /// normalization. This identity is separate from per-iteration pricing.
    pub summary_model: String,
    pub summary_pricing_model: Option<PricingModelRef>,
    pub stop_reason: Option<String>,
    pub excluded_terminal_fallback_index: Option<usize>,
    pub components: Vec<AnthropicFallbackCostComponent>,
    pub completeness: AnthropicFallbackCostCompleteness,
    pub currency: Option<String>,
    pub known_subtotal_usd: f64,
    pub server_tool_cost_usd: Option<f64>,
    pub total_cost_usd: Option<f64>,
    pub incomplete_reasons: Vec<String>,
    pub submission: Submission,
    pub service_tier: ServiceTier,
    pub inference_geo: Option<String>,
    pub executed_at: Option<u64>,
}

impl AnthropicFallbackCostQuote {
    /// Convert only a complete USD quote into the ordinary SDK estimate used
    /// by host accounting. The returned identity is Ucn's resolved summary
    /// model; detailed component identities remain on this quote.
    #[must_use]
    pub fn into_cost_estimate_if_complete(&self) -> Option<pricing::CostEstimate> {
        if self.completeness != AnthropicFallbackCostCompleteness::Complete
            || self.currency.as_deref() != Some("USD")
        {
            return None;
        }
        let pricing_model = self.summary_pricing_model.clone()?;
        let total_cost = self.total_cost_usd?;
        let charged = self
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component.kind,
                    AnthropicFallbackCostComponentKind::Iteration
                        | AnthropicFallbackCostComponentKind::FirstModelServerToolUse
                )
            })
            .collect::<Vec<_>>();
        let sum = |pick: fn(&AnthropicFallbackCostBuckets) -> Option<f64>| {
            charged.iter().try_fold(0.0, |total, component| {
                Some(total + pick(&component.buckets)?)
            })
        };
        let input_cost = sum(|buckets| buckets.input_cost_usd)?;
        let output_cost = sum(|buckets| buckets.output_cost_usd)?;
        let cache_read_cost = sum(|buckets| buckets.cache_read_cost_usd)?;
        let cache_write_cost = sum(|buckets| buckets.cache_write_cost_usd)?;
        let mut sources = self
            .components
            .iter()
            .filter_map(|component| component.pricing_source.as_deref());
        let source = sources
            .next()
            .map(str::to_owned)
            .filter(|first| sources.all(|candidate| candidate == first));
        Some(pricing::CostEstimate {
            currency: "USD".into(),
            context: PricingContext {
                service_tier: Some(self.service_tier),
                submission: self.submission,
                input_tokens: None,
                unix_seconds: self.executed_at,
            },
            pricing_model,
            submission: self.submission,
            input_cost,
            output_cost,
            cache_read_cost,
            cache_write_cost,
            reasoning_cost: 0.0,
            // The total includes the native first-model web-search call; its
            // separate amount is retained above because token buckets cannot
            // represent a per-request unit.
            total_cost,
            source,
        })
    }
}

/// Resolve the summary model using native Anthropic Ucn/Mn selection, without
/// requiring a pricing row. A served model is required; an empty served model
/// remains a valid `Some("")` identity.
#[must_use]
pub fn resolve_anthropic_server_fallback_summary_model(
    lane_model: Option<&str>,
    iterations: &UsageIterations,
) -> Option<String> {
    let served_model = iterations.served_fallback_model.as_deref()?;
    Some(
        match lane_model {
            Some(lane) if native_model_key(lane) == native_model_key(served_model) => lane,
            _ => served_model,
        }
        .into(),
    )
}

#[derive(Debug, Clone)]
pub struct FrozenPricing {
    pub(super) profile: ProviderProfile,
    pub(super) model: ModelProfile,
}
impl FrozenPricing {
    /// Capture one exact model row from a provider configuration.
    pub fn capture(
        profile: &ProviderProfile,
        display_model: &str,
        request_model: &str,
    ) -> Result<Self, LlmError> {
        let mut models = profile
            .models
            .iter()
            .filter(|m| m.display_model == display_model && m.request_model == request_model);
        let model = models.next().ok_or_else(|| LlmError::ModelUnavailable {
            message: "pricing row is missing".into(),
        })?;
        if models.next().is_some() {
            return Err(LlmError::InvalidRequest {
                message: "pricing row is ambiguous".into(),
            });
        }
        Ok(Self {
            profile: profile.clone(),
            model: model.clone(),
        })
    }
    /// Apply an explicit caller-provided price declaration before dispatch.
    pub fn with_token_pricing(mut self, pricing: TokenPricing) -> Self {
        self.model.pricing = Some(pricing);
        self.model.billing_mode = Some(BillingMode::PerToken);
        let updated_model = self.model.clone();
        let display_model = updated_model.display_model.clone();
        let request_model = updated_model.request_model.clone();
        if let Some(row) = self.profile.models.iter_mut().find(|model| {
            model.display_model == display_model && model.request_model == request_model
        }) {
            *row = updated_model;
        }
        self
    }
    /// Quote selected rates for budget policy without claiming actual usage.
    pub fn quote(&self, context: &PricingContext) -> Result<PriceQuote, LlmError> {
        let prices = self
            .model
            .pricing
            .as_ref()
            .ok_or_else(|| LlmError::CostUnavailable {
                message: "model prices are unpublished".into(),
            })?;
        Ok(price_query::quote(
            prices,
            self.model.billing_mode_on(&self.profile.pricing),
            self.profile.pricing.peak.as_ref(),
            context,
        ))
    }

    pub fn profile_name(&self) -> &str {
        &self.profile.profile_name
    }
    pub fn model(&self) -> &ModelProfile {
        &self.model
    }

    /// Estimate an observed attempt using only its frozen prices and confirmed
    /// execution facts. No configuration lookup or current-clock substitution.
    pub fn estimate(
        &self,
        report: &UsageReport,
        inference: &InferenceReport,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        let unavailable = |message: &str| LlmError::CostUnavailable {
            message: message.into(),
        };
        let usage = report
            .complete()
            .ok_or_else(|| unavailable("actual cost requires complete valid usage"))?;
        let features = self
            .model
            .info
            .features
            .on_connection(&self.profile.info.features);
        let tier = match inference.service_tier {
            Some(tier) => tier,
            None if inference.requested_service_tier == Some(ServiceTier::Fast)
                || inference.requested_raw_service_tier.is_some()
                || inference
                    .raw_service_tier
                    .as_deref()
                    .is_some_and(|s| s != "standard" && s != "default")
                || inference.raw_speed.is_some()
                || features.default_service_tier == Some(ServiceTier::Fast) =>
            {
                return Err(unavailable("the actual service tier is unknown"))
            }
            None => ServiceTier::Standard,
        };
        if tier == ServiceTier::Fast && features.fast == CapabilitySupport::Unsupported {
            return Err(unavailable(
                "the selected model does not support fast pricing",
            ));
        }
        let prices = self
            .model
            .pricing
            .as_ref()
            .ok_or_else(|| unavailable("model prices are unpublished"))?;
        if inference.executed_at.is_none()
            && (self.profile.pricing.peak.is_some()
                || price_query::requires_execution_time(prices, tier, submission))
        {
            return Err(unavailable(
                "execution time is required for time-dependent prices",
            ));
        }
        let context = PricingContext {
            service_tier: Some(tier),
            submission,
            unix_seconds: inference.executed_at,
            input_tokens: price_query::prompt_tokens(usage),
        };
        let quote = price_query::quote(
            prices,
            self.model.billing_mode_on(&self.profile.pricing),
            self.profile.pricing.peak.as_ref(),
            &context,
        );
        pricing::estimate(
            &quote,
            usage,
            &PricingModelRef {
                pricing_provider_id: self.profile.provider_id.clone(),
                billing_model: self.model.billing_model.clone(),
                request_model: self.model.request_model.clone(),
                display_model: self.model.display_model.clone(),
            },
        )
    }

    /// Price Anthropic's native per-iteration server-fallback accounting using
    /// only exact model rows in this frozen provider snapshot. Missing prices,
    /// unsupported billing modes, and non-USD rates remain an incomplete
    /// quote; this path never substitutes a default or the final model's rate
    /// for another iteration.
    pub fn estimate_anthropic_server_fallback(
        &self,
        iterations: &UsageIterations,
        lane_model: Option<&str>,
        stop_reason: Option<&str>,
        inference: &InferenceReport,
        submission: Submission,
    ) -> Result<Option<AnthropicFallbackCostQuote>, LlmError> {
        if !matches!(
            self.profile.protocol,
            ProtocolFamily::AnthropicMessages
                | ProtocolFamily::BedrockClaude
                | ProtocolFamily::VertexClaude
                | ProtocolFamily::FoundryClaude
        ) {
            return Err(LlmError::CostUnavailable {
                message: "Anthropic server-fallback pricing requires an Anthropic Claude Messages snapshot".into(),
            });
        }
        let Some(served_model) = iterations.served_fallback_model.as_deref() else {
            return Ok(None);
        };

        let Some(summary_model) =
            resolve_anthropic_server_fallback_summary_model(lane_model, iterations)
        else {
            return Ok(None);
        };
        let (summary_pricing_model, summary_error) =
            exact_pricing_model_ref(&self.profile, &summary_model);
        let service_tier = if iterations.speed.as_deref() == Some("fast") {
            ServiceTier::Fast
        } else {
            ServiceTier::Standard
        };
        let geo_multiplier = if iterations.inference_geo.as_deref() == Some("us") {
            1.1
        } else {
            1.0
        };
        let pricing_context = FallbackPricingContext {
            profile: &self.profile,
            service_tier,
            geo_multiplier,
            inference,
            submission,
        };
        let mut components = Vec::with_capacity(iterations.entries.len() + 1);
        let mut incomplete_reasons = Vec::new();
        if let Some(error) = summary_error {
            push_unique(&mut incomplete_reasons, error);
        }
        let terminal_fallback_index = iterations
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.r#type == "fallback_message")
            .map(|(index, _)| index)
            .next_back();
        let excluded_terminal_fallback_index = if stop_reason == Some("refusal") {
            terminal_fallback_index
        } else {
            None
        };
        let mut common_currency: Option<String> = None;
        let mut currency_conflict = false;
        let mut all_components_priced = true;

        for (index, entry) in iterations.entries.iter().enumerate() {
            let usage = AnthropicFallbackCostUsage {
                input_tokens: entry.input_tokens,
                output_tokens: entry.output_tokens,
                cache_read_input_tokens: entry.cache_read_input_tokens,
                cache_creation_input_tokens: entry.cache_creation_input_tokens,
                web_search_requests: None,
            };
            if Some(index) == excluded_terminal_fallback_index {
                components.push(unpriced_component(
                    AnthropicFallbackCostComponentKind::ExcludedRefusalTerminal,
                    Some(index),
                    entry.model.clone(),
                    usage,
                ));
                continue;
            }
            if entry.r#type != "fallback_message" {
                components.push(unpriced_component(
                    AnthropicFallbackCostComponentKind::SkippedNonFallbackIteration,
                    Some(index),
                    entry.model.clone(),
                    usage,
                ));
                continue;
            }
            let Some(model) = entry.model.as_deref() else {
                components.push(unpriced_component(
                    AnthropicFallbackCostComponentKind::SkippedMissingModel,
                    Some(index),
                    None,
                    usage,
                ));
                continue;
            };
            let result = quote_fallback_component(
                pricing_context,
                FallbackComponentRequest {
                    model_name: model,
                    kind: AnthropicFallbackCostComponentKind::Iteration,
                    iteration_index: Some(index),
                    usage,
                },
            );
            merge_component_result(
                &mut common_currency,
                &mut currency_conflict,
                &mut all_components_priced,
                &mut incomplete_reasons,
                &mut components,
                result,
            );
        }

        let first_fallback_model = iterations
            .entries
            .iter()
            .find(|entry| entry.r#type == "fallback_message" && entry.model.is_some())
            .and_then(|entry| entry.model.as_deref());
        let server_tool_cost_usd = if let Some(model) = first_fallback_model {
            let usage = AnthropicFallbackCostUsage {
                web_search_requests: iterations.web_search_requests,
                ..AnthropicFallbackCostUsage::default()
            };
            let result = quote_fallback_component(
                pricing_context,
                FallbackComponentRequest {
                    model_name: model,
                    kind: AnthropicFallbackCostComponentKind::FirstModelServerToolUse,
                    iteration_index: None,
                    usage,
                },
            );
            let component_total = result.component.total_cost_usd;
            merge_component_result(
                &mut common_currency,
                &mut currency_conflict,
                &mut all_components_priced,
                &mut incomplete_reasons,
                &mut components,
                result,
            );
            component_total
        } else {
            None
        };

        if currency_conflict {
            common_currency = None;
            push_unique(
                &mut incomplete_reasons,
                "iteration pricing rows use mixed currencies".into(),
            );
            all_components_priced = false;
        }
        if common_currency.as_deref() != Some("USD") {
            all_components_priced = false;
            if common_currency.is_none() {
                push_unique(
                    &mut incomplete_reasons,
                    "a common USD currency could not be established".into(),
                );
            } else {
                push_unique(
                    &mut incomplete_reasons,
                    format!(
                        "fallback pricing currency {} is not USD",
                        common_currency.as_deref().unwrap_or_default()
                    ),
                );
            }
        }
        if iterations.entries.iter().enumerate().any(|(index, entry)| {
            entry.r#type == "fallback_message"
                && Some(index) != excluded_terminal_fallback_index
                && (!valid_count(entry.input_tokens)
                    || !valid_count(entry.output_tokens)
                    || !valid_count(entry.cache_read_input_tokens)
                    || !valid_count(entry.cache_creation_input_tokens))
        }) {
            all_components_priced = false;
            push_unique(
                &mut incomplete_reasons,
                "iteration usage contains a non-finite or negative counter".into(),
            );
        }
        if first_fallback_model.is_some()
            && iterations
                .web_search_requests
                .is_some_and(|count| !valid_count(count))
        {
            all_components_priced = false;
            push_unique(
                &mut incomplete_reasons,
                "server tool usage contains a non-finite or negative web search count".into(),
            );
        }
        if first_fallback_model.is_none() {
            all_components_priced = false;
            push_unique(
                &mut incomplete_reasons,
                "served fallback usage has no model-bearing fallback iteration".into(),
            );
        }

        let complete_component_total = components.iter().try_fold(0.0, |total, component| {
            Some(total + component.total_cost_usd?)
        });
        let partial_bucket_subtotal = components
            .iter()
            .flat_map(|component| {
                [
                    component.buckets.input_cost_usd,
                    component.buckets.output_cost_usd,
                    component.buckets.cache_read_cost_usd,
                    component.buckets.cache_write_cost_usd,
                    component.buckets.web_search_cost_usd,
                ]
            })
            .flatten()
            .sum::<f64>();
        let known_subtotal_usd = complete_component_total.unwrap_or(partial_bucket_subtotal);
        let totals_finite = known_subtotal_usd.is_finite();
        if !totals_finite {
            all_components_priced = false;
            push_unique(
                &mut incomplete_reasons,
                "fallback cost arithmetic exceeded finite USD range".into(),
            );
        }
        let complete = all_components_priced
            && totals_finite
            && summary_pricing_model.is_some()
            && incomplete_reasons.is_empty();
        let total_cost_usd = complete.then_some(known_subtotal_usd);

        Ok(Some(AnthropicFallbackCostQuote {
            served_fallback_model: served_model.into(),
            summary_model,
            summary_pricing_model,
            stop_reason: stop_reason.map(str::to_owned),
            excluded_terminal_fallback_index,
            components,
            completeness: if complete {
                AnthropicFallbackCostCompleteness::Complete
            } else {
                AnthropicFallbackCostCompleteness::Incomplete
            },
            currency: common_currency,
            known_subtotal_usd,
            server_tool_cost_usd,
            total_cost_usd,
            incomplete_reasons,
            submission,
            service_tier,
            inference_geo: iterations.inference_geo.clone(),
            executed_at: inference.executed_at,
        }))
    }
}

struct ComponentQuoteResult {
    component: AnthropicFallbackCostComponent,
    currency: Option<String>,
    reasons: Vec<String>,
}

#[derive(Clone, Copy)]
struct FallbackPricingContext<'a> {
    profile: &'a ProviderProfile,
    service_tier: ServiceTier,
    geo_multiplier: f64,
    inference: &'a InferenceReport,
    submission: Submission,
}

#[derive(Clone, Copy)]
struct FallbackComponentRequest<'a> {
    model_name: &'a str,
    kind: AnthropicFallbackCostComponentKind,
    iteration_index: Option<usize>,
    usage: AnthropicFallbackCostUsage,
}

fn quote_fallback_component(
    pricing_context: FallbackPricingContext<'_>,
    component_request: FallbackComponentRequest<'_>,
) -> ComponentQuoteResult {
    let FallbackPricingContext {
        profile,
        service_tier,
        geo_multiplier,
        inference,
        submission,
    } = pricing_context;
    let FallbackComponentRequest {
        model_name,
        kind,
        iteration_index,
        usage,
    } = component_request;
    let (model, pricing_model) = match exact_model(profile, model_name) {
        Ok(model) => (
            Some(model),
            Some(PricingModelRef {
                pricing_provider_id: profile.provider_id.clone(),
                billing_model: model.billing_model.clone(),
                request_model: model.request_model.clone(),
                display_model: model.display_model.clone(),
            }),
        ),
        Err(reason) => {
            return ComponentQuoteResult {
                component: failed_component(
                    kind,
                    iteration_index,
                    Some(model_name),
                    usage,
                    None,
                    reason.clone(),
                ),
                currency: None,
                reasons: vec![reason],
            };
        }
    };
    let model = model.expect("exact model lookup returned a row");
    let Some(prices) = model.pricing.as_ref() else {
        let reason = format!("model {model_name:?} has no frozen pricing row");
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: None,
            reasons: vec![reason],
        };
    };
    let currency = prices.currency.clone().unwrap_or_else(|| "USD".into());
    if currency != "USD" {
        let reason = format!("model {model_name:?} uses non-USD currency {currency}");
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: Some(currency),
            reasons: vec![reason],
        };
    }
    let billing_mode = model.billing_mode_on(&profile.pricing);
    if billing_mode != BillingMode::PerToken {
        let reason =
            format!("model {model_name:?} billing mode is {billing_mode:?}, not per-token");
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: Some(currency),
            reasons: vec![reason],
        };
    }
    let features = model.info.features.on_connection(&profile.info.features);
    if service_tier == ServiceTier::Fast && features.fast == CapabilitySupport::Unsupported {
        let reason = format!("model {model_name:?} does not support fast pricing");
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: Some(currency),
            reasons: vec![reason],
        };
    }
    if !valid_count(usage.input_tokens)
        || !valid_count(usage.output_tokens)
        || !valid_count(usage.cache_read_input_tokens)
        || !valid_count(usage.cache_creation_input_tokens)
        || usage
            .web_search_requests
            .is_some_and(|count| !valid_count(count))
    {
        let reason = format!("model {model_name:?} usage contains an invalid counter");
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: Some(currency),
            reasons: vec![reason],
        };
    }
    let context_input_tokens = prompt_token_count(usage);
    let context = PricingContext {
        service_tier: Some(service_tier),
        submission,
        input_tokens: context_input_tokens,
        unix_seconds: inference.executed_at,
    };
    let selected = price_query::quote(
        prices,
        billing_mode,
        profile.pricing.peak.as_ref(),
        &context,
    );
    let Some(rates) = selected.rates.as_ref() else {
        let reason = selected
            .reason
            .unwrap_or_else(|| format!("model {model_name:?} has no selected frozen rates"));
        return ComponentQuoteResult {
            component: failed_component(
                kind,
                iteration_index,
                Some(model_name),
                usage,
                pricing_model,
                reason.clone(),
            ),
            currency: Some(currency),
            reasons: vec![reason],
        };
    };
    let search_count = usage.web_search_requests.unwrap_or(0.0);
    let input_raw = token_cost(usage.input_tokens, rates.input_per_million, 1.0);
    let output_raw = token_cost(usage.output_tokens, rates.output_per_million, 1.0);
    let cache_read_raw = token_cost(
        usage.cache_read_input_tokens,
        rates.cache_read_per_million,
        1.0,
    );
    let cache_write_raw = token_cost(
        usage.cache_creation_input_tokens,
        rates.cache_write_per_million,
        1.0,
    );
    let buckets = AnthropicFallbackCostBuckets {
        input_cost_usd: input_raw
            .map(|value| value * geo_multiplier)
            .filter(|value| value.is_finite()),
        output_cost_usd: output_raw
            .map(|value| value * geo_multiplier)
            .filter(|value| value.is_finite()),
        cache_read_cost_usd: cache_read_raw
            .map(|value| value * geo_multiplier)
            .filter(|value| value.is_finite()),
        cache_write_cost_usd: cache_write_raw
            .map(|value| value * geo_multiplier)
            .filter(|value| value.is_finite()),
        web_search_cost_usd: per_request_cost(search_count, rates.web_search_per_request),
    };
    let mut reasons = Vec::new();
    for (field, count, rate) in [
        ("input", usage.input_tokens, rates.input_per_million),
        ("output", usage.output_tokens, rates.output_per_million),
        (
            "cache read",
            usage.cache_read_input_tokens,
            rates.cache_read_per_million,
        ),
        (
            "cache write",
            usage.cache_creation_input_tokens,
            rates.cache_write_per_million,
        ),
    ] {
        if count > 0.0 && rate.is_none() {
            reasons.push(format!("model {model_name:?} has no {field} rate"));
        }
    }
    if search_count > 0.0 && rates.web_search_per_request.is_none() {
        reasons.push(format!(
            "model {model_name:?} has no web-search request rate"
        ));
    }
    if (buckets.input_cost_usd.is_none()
        || buckets.output_cost_usd.is_none()
        || buckets.cache_read_cost_usd.is_none()
        || buckets.cache_write_cost_usd.is_none()
        || buckets.web_search_cost_usd.is_none())
        && reasons.is_empty()
    {
        reasons.push(format!(
            "model {model_name:?} cost arithmetic is not finite"
        ));
    }
    let total_cost_usd = if reasons.is_empty() {
        Some(
            (input_raw.unwrap_or(0.0)
                + output_raw.unwrap_or(0.0)
                + cache_read_raw.unwrap_or(0.0)
                + cache_write_raw.unwrap_or(0.0))
                * geo_multiplier
                + buckets.web_search_cost_usd.unwrap_or(0.0),
        )
    } else {
        None
    };
    if total_cost_usd.is_some_and(|total| !total.is_finite()) {
        reasons.push(format!(
            "model {model_name:?} cost arithmetic is not finite"
        ));
    }
    let total_cost_usd = total_cost_usd.filter(|total| total.is_finite());
    if total_cost_usd.is_none() {
        // Preserve individually known buckets for an incomplete line item.
        if reasons.is_empty() {
            reasons.push(format!("model {model_name:?} cost is incomplete"));
        }
    }
    let unmetered_reason = (!reasons.is_empty()).then(|| reasons.join("; "));
    ComponentQuoteResult {
        component: AnthropicFallbackCostComponent {
            kind,
            iteration_index,
            model: Some(model_name.into()),
            pricing_model,
            usage,
            buckets,
            total_cost_usd,
            pricing_source: selected.source,
            unmetered_reason,
        },
        currency: Some(currency),
        reasons,
    }
}

fn merge_component_result(
    common_currency: &mut Option<String>,
    currency_conflict: &mut bool,
    all_components_priced: &mut bool,
    incomplete_reasons: &mut Vec<String>,
    components: &mut Vec<AnthropicFallbackCostComponent>,
    result: ComponentQuoteResult,
) {
    if let Some(currency) = result.currency {
        match common_currency.as_deref() {
            None => *common_currency = Some(currency),
            Some(current) if current != currency => *currency_conflict = true,
            _ => {}
        }
    }
    if !result.reasons.is_empty() || result.component.total_cost_usd.is_none() {
        *all_components_priced = false;
    }
    for reason in result.reasons {
        push_unique(incomplete_reasons, reason);
    }
    components.push(result.component);
}

fn exact_model<'a>(
    profile: &'a ProviderProfile,
    request_model: &str,
) -> Result<&'a ModelProfile, String> {
    let mut rows = profile
        .models
        .iter()
        .filter(|model| model.request_model == request_model);
    let Some(model) = rows.next() else {
        return Err(format!(
            "no exact frozen pricing model row for {request_model:?}"
        ));
    };
    if rows.next().is_some() {
        return Err(format!(
            "multiple frozen pricing model rows match {request_model:?}"
        ));
    }
    Ok(model)
}

fn exact_pricing_model_ref(
    profile: &ProviderProfile,
    request_model: &str,
) -> (Option<PricingModelRef>, Option<String>) {
    match exact_model(profile, request_model) {
        Ok(model) => (
            Some(PricingModelRef {
                pricing_provider_id: profile.provider_id.clone(),
                billing_model: model.billing_model.clone(),
                request_model: model.request_model.clone(),
                display_model: model.display_model.clone(),
            }),
            None,
        ),
        Err(reason) => (None, Some(reason)),
    }
}

fn native_model_key(model: &str) -> String {
    let lowered = model.to_ascii_lowercase();
    let mut normalized = String::with_capacity(model.len());
    let mut offset = 0;
    while offset < model.len() {
        let marker = lowered
            .get(offset..offset.saturating_add(4))
            .is_some_and(|marker| marker == "[1m]" || marker == "[2m]");
        if marker {
            offset += 4;
            continue;
        }
        let character = model[offset..]
            .chars()
            .next()
            .expect("offset is kept on a UTF-8 boundary");
        normalized.push(character);
        offset += character.len_utf8();
    }
    normalized
}

fn prompt_token_count(usage: AnthropicFallbackCostUsage) -> Option<u64> {
    let count =
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
    if !count.is_finite()
        || count.fract() != 0.0
        || !(0.0..18_446_744_073_709_551_616.0).contains(&count)
    {
        return None;
    }
    Some(count as u64)
}

fn token_cost(count: f64, rate: Option<f64>, geo_multiplier: f64) -> Option<f64> {
    let cost = if count == 0.0 {
        0.0
    } else {
        count / 1_000_000.0 * rate? * geo_multiplier
    };
    cost.is_finite().then_some(cost)
}

fn per_request_cost(count: f64, rate: Option<f64>) -> Option<f64> {
    let cost = if count == 0.0 { 0.0 } else { count * rate? };
    cost.is_finite().then_some(cost)
}

fn valid_count(count: f64) -> bool {
    count.is_finite() && count >= 0.0
}

fn unpriced_component(
    kind: AnthropicFallbackCostComponentKind,
    iteration_index: Option<usize>,
    model: Option<String>,
    usage: AnthropicFallbackCostUsage,
) -> AnthropicFallbackCostComponent {
    AnthropicFallbackCostComponent {
        kind,
        iteration_index,
        model,
        pricing_model: None,
        usage,
        buckets: AnthropicFallbackCostBuckets {
            input_cost_usd: Some(0.0),
            output_cost_usd: Some(0.0),
            cache_read_cost_usd: Some(0.0),
            cache_write_cost_usd: Some(0.0),
            web_search_cost_usd: Some(0.0),
        },
        total_cost_usd: Some(0.0),
        pricing_source: None,
        unmetered_reason: None,
    }
}

fn failed_component(
    kind: AnthropicFallbackCostComponentKind,
    iteration_index: Option<usize>,
    model: Option<&str>,
    usage: AnthropicFallbackCostUsage,
    pricing_model: Option<PricingModelRef>,
    reason: String,
) -> AnthropicFallbackCostComponent {
    AnthropicFallbackCostComponent {
        kind,
        iteration_index,
        model: model.map(str::to_owned),
        pricing_model,
        usage,
        buckets: AnthropicFallbackCostBuckets {
            input_cost_usd: None,
            output_cost_usd: None,
            cache_read_cost_usd: None,
            cache_write_cost_usd: None,
            web_search_cost_usd: None,
        },
        total_cost_usd: None,
        pricing_source: None,
        unmetered_reason: Some(reason),
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}
