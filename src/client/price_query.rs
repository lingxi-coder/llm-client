//! Published prices selected by model and service tier, never by effort.
use super::{pricing, LlmClient, ResolvedRoute};
use crate::protocol::*;

impl LlmClient {
    /// Query published unit rates. Effort changes consumption, not unit prices.
    pub fn price_quote(
        &self,
        model: &str,
        profile: Option<&str>,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        let route = self.resolve_in(model, profile)?;
        self.price_quote_for_route(&route, context)
    }
    pub fn price_quote_for_route(
        &self,
        route: &ResolvedRoute,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        let (profile, model) = self.pricing_selection(route)?;
        let mut context = context.clone();
        if context.unix_seconds.is_none() {
            context.unix_seconds = Some(
                self.clock
                    .now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs()),
            );
        }
        let features = model.info.features.on_connection(&profile.info.features);
        // An omitted tier follows the model/connection default when known.
        // Keep an explicit tier as the caller's pricing assumption.
        context.service_tier = context.service_tier.or(features.default_service_tier);
        if context.service_tier == Some(ServiceTier::Fast)
            && features.fast == CapabilitySupport::Unsupported
        {
            return Ok(empty_quote(
                &context,
                PriceStatus::Unsupported,
                "this model does not support fast mode",
            ));
        }
        let Some(prices) = &model.pricing else {
            let mut quote = empty_quote(
                &context,
                PriceStatus::Unknown,
                "model prices are unpublished",
            );
            if model.billing_mode_on(&profile.pricing) == BillingMode::Subscription {
                quote.unit = "subscription_quota".into();
            }
            return Ok(quote);
        };
        Ok(quote(
            prices,
            model.billing_mode_on(&profile.pricing),
            profile.pricing.peak.as_ref(),
            &context,
        ))
    }
    /// Preflight estimate under explicit pricing assumptions.
    pub fn estimate_cost(
        &self,
        route: &ResolvedRoute,
        usage: &Usage,
        context: &PricingContext,
    ) -> Result<pricing::CostEstimate, LlmError> {
        let mut context = context.clone();
        context.input_tokens = context.input_tokens.or_else(|| prompt_tokens(usage));
        let quote = self.price_quote_for_route(route, &context)?;
        pricing::estimate(&quote, usage, &route.pricing_model)
    }
    pub(super) fn pricing_selection<'a>(
        &'a self,
        route: &ResolvedRoute,
    ) -> Result<(&'a ProviderProfile, &'a ModelProfile), LlmError> {
        let fail = || LlmError::CostUnavailable {
            message: "price route does not identify exactly one current model".into(),
        };
        let p = self.profile(&route.profile_name).ok_or_else(fail)?;
        if p.provider_id != route.provider_id
            || p.provider_id != route.pricing_model.pricing_provider_id
            || route.request_model != route.pricing_model.request_model
            || route.display_model != route.pricing_model.display_model
        {
            return Err(fail());
        }
        let mut rows = p.models.iter().filter(|m| {
            m.request_model == route.request_model
                && m.display_model == route.display_model
                && m.billing_model == route.pricing_model.billing_model
        });
        let model = rows.next().ok_or_else(fail)?;
        if rows.next().is_some() {
            return Err(fail());
        }
        Ok((p, model))
    }
}

pub(super) fn prompt_tokens(usage: &Usage) -> Option<u64> {
    usage
        .input_tokens
        .checked_add(usage.cache_read_tokens)?
        .checked_add(usage.cache_write_tokens)
}

pub(super) fn requires_execution_time(
    prices: &TokenPricing,
    tier: ServiceTier,
    submission: Submission,
) -> bool {
    let inherits_standard = tier == ServiceTier::Fast
        && prices.rules.iter().any(|rule| {
            rule.service_tier == tier && rule.submission == submission && rule.multiplier.is_some()
        });
    prices.rules.iter().any(|rule| {
        rule.submission == submission
            && (rule.service_tier == tier
                || (inherits_standard && rule.service_tier == ServiceTier::Standard))
            && (rule.valid_from.is_some() || rule.valid_until.is_some())
    })
}

pub(crate) fn empty_quote(
    context: &PricingContext,
    status: PriceStatus,
    reason: &str,
) -> PriceQuote {
    PriceQuote {
        rule: None,
        status,
        context: context.clone(),
        rates: None,
        quota: None,
        currency: String::new(),
        unit: "per_million_tokens".into(),
        source: None,
        verified_at: None,
        reason: Some(reason.into()),
        multiplier: None,
    }
}
fn base_rates(p: &TokenPricing) -> TokenRates {
    TokenRates {
        input_per_million: p.input_per_million,
        output_per_million: p.output_per_million,
        cache_read_per_million: p.cache_read_per_million,
        cache_write_per_million: p.cache_write_per_million,
        cache_write_1h_per_million: p.cache_write_1h_per_million,
        reasoning_per_million: p.reasoning_per_million,
    }
}
fn specificity(r: &PriceRule) -> usize {
    usize::from(r.min_input_tokens.is_some())
        + usize::from(r.max_input_tokens.is_some())
        + usize::from(r.valid_from.is_some())
        + usize::from(r.valid_until.is_some())
}
fn contains(r: &PriceRule, n: u64) -> bool {
    r.min_input_tokens.is_none_or(|min| n >= min) && r.max_input_tokens.is_none_or(|max| n <= max)
}
type TimeBounds = (u64, u64);
fn temporal_bounds(r: &PriceRule) -> Result<(TimeBounds, TimeBounds), String> {
    Ok((
        r.valid_from
            .as_ref()
            .map(PriceBoundary::utc_bounds)
            .transpose()?
            .unwrap_or((0, 0)),
        r.valid_until
            .as_ref()
            .map(PriceBoundary::utc_bounds)
            .transpose()?
            .unwrap_or((u64::MAX, u64::MAX)),
    ))
}
fn temporal_match(r: &PriceRule, now: u64) -> Result<Option<bool>, String> {
    let (from, until) = temporal_bounds(r)?;
    if now < from.0 || now >= until.1 {
        return Ok(Some(false));
    }
    if now < from.1 || now >= until.0 {
        return Ok(None);
    }
    Ok(Some(true))
}
fn scaled(rates: &mut TokenRates, multiplier: &PriceMultiplier) {
    for bucket in &multiplier.buckets {
        let value = match bucket {
            PriceBucket::Input => &mut rates.input_per_million,
            PriceBucket::Output => &mut rates.output_per_million,
            PriceBucket::CacheRead => &mut rates.cache_read_per_million,
            PriceBucket::CacheWrite => &mut rates.cache_write_per_million,
            PriceBucket::CacheWrite1h => &mut rates.cache_write_1h_per_million,
            PriceBucket::Reasoning => &mut rates.reasoning_per_million,
        };
        *value = value.map(|v| v * multiplier.factor);
    }
}
pub(crate) fn quote(
    p: &TokenPricing,
    billing: BillingMode,
    schedule: Option<&PeakSchedule>,
    context: &PricingContext,
) -> PriceQuote {
    let mut quote = empty_quote(
        context,
        PriceStatus::Unknown,
        "prices for this combination are unpublished",
    );
    quote.currency = p.currency.clone().unwrap_or_else(|| "USD".into());
    quote.source = p.source.clone();
    quote.verified_at = p.verified_at.clone();
    let tier = context.service_tier.unwrap_or(ServiceTier::Standard);
    if billing == BillingMode::Subscription {
        quote.currency.clear();
        quote.unit = "subscription_quota".into();
        if context.submission != Submission::Interactive {
            quote.reason = Some("subscription batch quota pricing is unpublished".into());
            return quote;
        }
        if let Some(q) = p.quota.iter().find(|q| q.service_tier == tier) {
            quote.status = PriceStatus::Priced;
            quote.quota = Some(q.clone());
            quote.currency = String::new();
            quote.unit = q.unit.clone();
            quote.source = Some(q.source.clone());
            quote.verified_at = q.verified_at.clone();
            quote.reason = None;
        } else {
            quote.reason=Some("subscription quota pricing is unpublished; catalog token prices are not a subscription bill".into());
        }
        return quote;
    }
    let candidates = p
        .rules
        .iter()
        .filter(|r| r.service_tier == tier && r.submission == context.submission)
        .collect::<Vec<_>>();
    if context.input_tokens.is_none()
        && candidates
            .iter()
            .any(|r| r.min_input_tokens.is_some() || r.max_input_tokens.is_some())
    {
        quote.reason = Some("prompt token count is required for context-dependent rates".into());
        return quote;
    }
    if context.unix_seconds.is_none()
        && candidates
            .iter()
            .any(|r| r.valid_from.is_some() || r.valid_until.is_some())
    {
        quote.reason = Some("time is required for dated rates".into());
        return quote;
    }
    let mut selected: Option<&PriceRule> = None;
    let mut uncertain: Option<&PriceRule> = None;
    for rule in candidates
        .iter()
        .copied()
        .filter(|r| context.input_tokens.is_none_or(|n| contains(r, n)))
    {
        match temporal_match(rule, context.unix_seconds.unwrap_or(0)) {
            Ok(Some(true)) if selected.is_none_or(|old| specificity(rule) > specificity(old)) => {
                selected = Some(rule)
            }
            Ok(None) if uncertain.is_none_or(|old| specificity(rule) > specificity(old)) => {
                uncertain = Some(rule)
            }
            Err(reason) => {
                quote.reason = Some(reason);
                return quote;
            }
            _ => {}
        }
    }
    if let Some(rule) = uncertain
        .filter(|candidate| selected.is_none_or(|rule| specificity(candidate) >= specificity(rule)))
    {
        quote.rule = Some(rule.clone());
        quote.source = rule.source.clone().or(quote.source);
        quote.verified_at = rule.verified_at.clone().or(quote.verified_at);
        quote.reason =
            Some("the published price boundary has no confirmed billing time zone".into());
        return quote;
    }
    let (mut rates, apply_schedule) = if let Some(rule) = selected {
        quote.rule = Some(rule.clone());
        quote.source = rule.source.clone().or_else(|| p.source.clone());
        quote.verified_at = rule.verified_at.clone().or_else(|| p.verified_at.clone());
        quote.multiplier = rule.multiplier.clone();
        let mut rates = if rule.multiplier.is_some() && tier == ServiceTier::Fast {
            let standard = self::quote(
                p,
                billing,
                None,
                &PricingContext {
                    service_tier: Some(ServiceTier::Standard),
                    ..context.clone()
                },
            );
            let Some(rates) = standard.rates else {
                return quote;
            };
            rates
        } else if rule.multiplier.is_some() {
            base_rates(p)
        } else {
            rule.rates.clone()
        };
        if let Some(multiplier) = &rule.multiplier {
            scaled(&mut rates, multiplier);
        }
        (rates, rule.apply_peak_schedule)
    } else {
        if tier == ServiceTier::Fast || !candidates.is_empty() {
            return quote;
        }
        quote.source = p.source.clone();
        quote.verified_at = p.verified_at.clone();
        match context.submission {
            Submission::Interactive => (base_rates(p), true),
            Submission::Batch => {
                let Some(batch) = &p.batch else {
                    return quote;
                };
                (
                    TokenRates {
                        input_per_million: batch.input_per_million,
                        output_per_million: batch.output_per_million,
                        cache_read_per_million: batch.cache_read_per_million,
                        cache_write_per_million: batch.cache_write_per_million,
                        cache_write_1h_per_million: batch.cache_write_1h_per_million,
                        reasoning_per_million: batch.reasoning_per_million,
                    },
                    true,
                )
            }
        }
    };
    if apply_schedule {
        if let Some(schedule) = schedule {
            let Some(now) = context.unix_seconds else {
                quote.reason = Some("time is required for scheduled rates".into());
                return quote;
            };
            if !schedule.is_peak(now) {
                scaled(
                    &mut rates,
                    &PriceMultiplier {
                        factor: schedule.off_peak_multiplier,
                        buckets: vec![
                            PriceBucket::Input,
                            PriceBucket::Output,
                            PriceBucket::CacheRead,
                            PriceBucket::CacheWrite,
                            PriceBucket::CacheWrite1h,
                            PriceBucket::Reasoning,
                        ],
                    },
                );
            }
        }
    }
    if rates == TokenRates::default() || validate_rates(&rates).is_err() {
        return quote;
    }
    quote.rates = Some(rates);
    quote.status = PriceStatus::Priced;
    quote.reason = None;
    quote
}
fn validate_rates(r: &TokenRates) -> Result<(), String> {
    for v in [
        r.input_per_million,
        r.output_per_million,
        r.cache_read_per_million,
        r.cache_write_per_million,
        r.cache_write_1h_per_million,
        r.reasoning_per_million,
    ]
    .into_iter()
    .flatten()
    {
        if !v.is_finite() || v < 0.0 {
            return Err("token rates must be finite and non-negative".into());
        }
    }
    Ok(())
}
pub(crate) fn validate_prices(p: &TokenPricing) -> Result<(), String> {
    if p.currency
        .as_ref()
        .is_some_and(|c| c.len() != 3 || !c.bytes().all(|b| b.is_ascii_uppercase()))
    {
        return Err("currency must be a three-letter ISO code".into());
    }
    validate_rates(&base_rates(p))?;
    if let Some(batch) = &p.batch {
        validate_rates(&TokenRates {
            input_per_million: batch.input_per_million,
            output_per_million: batch.output_per_million,
            cache_read_per_million: batch.cache_read_per_million,
            cache_write_per_million: batch.cache_write_per_million,
            cache_write_1h_per_million: batch.cache_write_1h_per_million,
            reasoning_per_million: batch.reasoning_per_million,
        })?;
    }
    for (i, r) in p.rules.iter().enumerate() {
        validate_rates(&r.rates)?;
        if r.min_input_tokens.unwrap_or(0) > r.max_input_tokens.unwrap_or(u64::MAX) {
            return Err("invalid context pricing interval".into());
        }
        let (from, until) = temporal_bounds(r)?;
        if from.0 >= until.1 || (r.valid_from.is_some() && r.valid_from == r.valid_until) {
            return Err("invalid price validity interval".into());
        }
        if let Some(m) = &r.multiplier {
            if !m.factor.is_finite()
                || m.factor <= 0.0
                || m.buckets.is_empty()
                || m.buckets
                    .iter()
                    .enumerate()
                    .any(|(i, v)| m.buckets[..i].contains(v))
                || r.rates != TokenRates::default()
            {
                return Err("a multiplier needs finite positive factor, unique buckets and no conflicting fixed rates".into());
            }
        }
        for other in &p.rules[..i] {
            let (other_from, other_until) = temporal_bounds(other)?;
            // Identical published boundaries are adjacent, even if their zone is unpublished.
            let adjacent = (r.valid_from.is_some() && r.valid_from == other.valid_until)
                || (r.valid_until.is_some() && r.valid_until == other.valid_from);
            let overlap = r.min_input_tokens.unwrap_or(0)
                <= other.max_input_tokens.unwrap_or(u64::MAX)
                && other.min_input_tokens.unwrap_or(0) <= r.max_input_tokens.unwrap_or(u64::MAX)
                && from.0 < other_until.1
                && other_from.0 < until.1
                && !adjacent;
            if other.service_tier == r.service_tier
                && other.submission == r.submission
                && specificity(other) == specificity(r)
                && overlap
            {
                return Err("overlapping price rules have equal precedence".into());
            }
        }
    }
    for (i, q) in p.quota.iter().enumerate() {
        if !q.multiplier.is_finite()
            || q.multiplier <= 0.0
            || q.unit.is_empty()
            || p.quota[..i]
                .iter()
                .any(|o| o.service_tier == q.service_tier)
        {
            return Err("invalid or duplicate subscription quota price".into());
        }
    }
    Ok(())
}
