//! Resolve context against the selected attempt. Never consult a global model-name cache.
use crate::protocol::{
    BillingMode, ContextPolicy, ContextRecommendation, ContextWindowSource, LlmError, ModelProfile,
    PricingContext, ProviderProfile, TokenRates,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextRequest {
    pub policy: Option<ContextPolicy>,
    /// Planning only. Never serialized as an output limit or used as an exact count.
    pub planning_output_tokens: Option<u64>,
    pub safety_tokens: u64,
    /// None conservatively retains a published thinking-mode input ceiling.
    pub thinking: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedContext {
    pub profile_name: String,
    pub request_model: String,
    pub display_model: String,
    pub billing_model: String,
    pub context_window_tokens: Option<u64>,
    /// Provenance of the selected model's raw context-window value.
    pub context_window_source: ContextWindowSource,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub shared_window: Option<bool>,
    pub recommendation: Option<ContextRecommendation>,
    pub policy: ContextPolicy,
    pub policy_source: String,
    /// Last input count before the first confirmed price increase.
    pub standard_price_input_tokens: Option<u64>,
    pub output_reserve_tokens: Option<u64>,
    pub output_reserve_estimated: bool,
    pub hard_input_limit: Option<u64>,
    pub effective_input_budget: Option<u64>,
    pub safety_tokens: u64,
}
fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn positive(v: Option<u64>) -> Option<u64> {
    v.filter(|v| *v > 0)
}
fn minimum(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
fn increased(a: &TokenRates, b: &TokenRates) -> bool {
    [
        (a.input_per_million, b.input_per_million),
        (a.output_per_million, b.output_per_million),
        (a.cache_read_per_million, b.cache_read_per_million),
        (a.cache_write_per_million, b.cache_write_per_million),
        (a.cache_write_1h_per_million, b.cache_write_1h_per_million),
        // Billing includes reasoning at the output rate when no separate rate exists.
        (
            a.reasoning_per_million.or(a.output_per_million),
            b.reasoning_per_million.or(b.output_per_million),
        ),
    ]
    .into_iter()
    .any(|(a, b)| matches!((a,b),(Some(a),Some(b)) if b > a))
}
/// Uses the same selector as billing, including service tier and dated rules.
/// Missing prices never imply a known free/cheap range.
pub fn standard_price_input_limit(
    profile: &ProviderProfile,
    model: &ModelProfile,
    context: &PricingContext,
) -> Option<u64> {
    let billing = model.billing_mode_on(&profile.pricing);
    if billing != BillingMode::PerToken {
        return None;
    }
    let prices = model.pricing.as_ref()?;
    let mut query = context.clone();
    query.service_tier = query.service_tier.or(model
        .info
        .features
        .on_connection(&profile.info.features)
        .default_service_tier);
    query.input_tokens = Some(1);
    let baseline =
        super::price_query::quote(prices, billing, profile.pricing.peak.as_ref(), &query).rates?;
    let mut starts: Vec<_> = prices
        .rules
        .iter()
        .flat_map(|r| {
            [
                r.min_input_tokens,
                r.max_input_tokens.and_then(|n| n.checked_add(1)),
            ]
        })
        .flatten()
        .filter(|n| *n > 1)
        .collect();
    starts.sort_unstable();
    starts.dedup();
    for start in starts {
        query.input_tokens = Some(start);
        let quote =
            super::price_query::quote(prices, billing, profile.pricing.peak.as_ref(), &query);
        let rates = quote.rates?;
        if increased(&baseline, &rates) {
            return Some(start - 1);
        }
    }
    None
}

pub fn resolve_context(
    profile: &ProviderProfile,
    model: &ModelProfile,
    request: &ContextRequest,
    output_tokens: Option<u32>,
    pricing: &PricingContext,
) -> Result<ResolvedContext, LlmError> {
    let metadata = &model.metadata;
    let context = metadata.context.as_ref();
    if let Some(context) = context {
        context.validate().map_err(invalid)?;
    }
    let configured = context.and_then(|c| c.application.as_ref());
    let policy = request
        .policy
        .as_ref()
        .or(configured)
        .cloned()
        .unwrap_or_default();
    if matches!(policy, ContextPolicy::Custom { input_tokens: 0 }) {
        return Err(invalid("custom context input budget must be positive"));
    }
    let window = positive(metadata.context_window_tokens);
    let context_window_source = match metadata.context_window_source {
        ContextWindowSource::Unknown if window.is_some() => ContextWindowSource::Explicit,
        source => source,
    };
    let mut input = positive(metadata.max_input_tokens);
    if request.thinking != Some(false) {
        input = minimum(
            input,
            context.and_then(|c| positive(c.thinking_max_input_tokens)),
        );
    }
    let shared = context.and_then(|c| c.shared_window);
    let output_reserve = output_tokens
        .map(u64::from)
        .or(request.planning_output_tokens);
    // An unknown output reservation cannot establish a stricter *hard* limit.
    let hard = minimum(
        input,
        window.map(|n| {
            if shared == Some(true) {
                n.saturating_sub(output_tokens.map(u64::from).unwrap_or(0))
            } else {
                n
            }
        }),
    );
    let planning = minimum(
        input,
        window.map(|n| {
            if shared == Some(true) {
                n.saturating_sub(output_reserve.unwrap_or(0))
            } else {
                n
            }
        }),
    );
    let standard = standard_price_input_limit(profile, model, pricing);
    let recommendation = context.and_then(|c| c.recommendation.clone());
    let target = match policy {
        ContextPolicy::Auto => minimum(standard, recommendation.as_ref().map(|r| r.input_tokens)),
        ContextPolicy::Long => None,
        ContextPolicy::Custom { input_tokens } => Some(input_tokens),
    };
    let budget = minimum(planning, target).map(|n| n.saturating_sub(request.safety_tokens));
    Ok(ResolvedContext {
        profile_name: profile.profile_name.clone(),
        request_model: model.request_model.clone(),
        display_model: model.display_model.clone(),
        billing_model: model.billing_model.clone(),
        context_window_tokens: window,
        context_window_source,
        max_input_tokens: input,
        max_output_tokens: positive(metadata.max_output_tokens),
        shared_window: shared,
        recommendation,
        policy,
        policy_source: if request.policy.is_some() {
            "request"
        } else if configured.is_some() {
            "model"
        } else {
            "default"
        }
        .into(),
        standard_price_input_tokens: standard,
        output_reserve_tokens: output_reserve,
        output_reserve_estimated: output_tokens.is_none(),
        hard_input_limit: hard,
        effective_input_budget: budget,
        safety_tokens: request.safety_tokens,
    })
}

impl super::ClientSnapshot {
    /// Resolve on this immutable catalog snapshot, preserving the exact row identity.
    pub fn resolve_context_for_route(
        &self,
        route: &super::ResolvedRoute,
        request: &ContextRequest,
        output_tokens: Option<u32>,
        pricing: &PricingContext,
    ) -> Result<ResolvedContext, LlmError> {
        let (profile, model) = self.pricing_selection(route)?;
        let mut pricing = pricing.clone();
        pricing.unix_seconds = pricing.unix_seconds.or_else(|| {
            self.runtime
                .clock
                .now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
        });
        resolve_context(profile, model, request, output_tokens, &pricing)
    }
}
