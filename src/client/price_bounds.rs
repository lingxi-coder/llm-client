//! Conservative published token-rate bounds for a frozen model row.
//!
//! This samples the same quote selector used for final settlement. It does not
//! estimate usage or choose the time at which an attempt will execute.

use std::collections::BTreeSet;

use super::FrozenPricing;
use crate::protocol::{
    BillingMode, LlmError, PriceStatus, PricingContext, ServiceTier, Submission, TokenRates,
};

/// Per-bucket ceilings over every published interactive price context.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceBounds {
    /// Whether an omitted request tier defaults to Fast for this connection.
    pub default_fast: bool,
    pub standard: TokenRates,
    /// `None` means no Fast quote can be selected from the published row.
    pub fast: Option<TokenRates>,
}

fn unavailable(message: impl Into<String>) -> LlmError {
    LlmError::CostUnavailable {
        message: message.into(),
    }
}

impl FrozenPricing {
    /// Bound all published interactive rate rules without assuming a prompt
    /// length or dispatch time. Missing buckets remain missing in the bound.
    pub fn interactive_price_bounds(&self) -> Result<Option<PriceBounds>, LlmError> {
        let model = self.model();
        let profile = &self.profile;
        let default_fast = model
            .info
            .features
            .on_connection(&profile.info.features)
            .default_service_tier
            == Some(ServiceTier::Fast);
        let dynamic = default_fast
            || profile.pricing.peak.is_some()
            || model
                .pricing
                .as_ref()
                .is_some_and(|prices| !prices.rules.is_empty());
        if !dynamic {
            return Ok(None);
        }
        let prices = model
            .pricing
            .as_ref()
            .ok_or_else(|| unavailable("dynamic attempt prices are unpublished"))?;
        if model.billing_mode_on(&profile.pricing) != BillingMode::PerToken {
            return Err(unavailable(
                "attempt price bounds require published per-token rates",
            ));
        }

        // A quote can change only at a rule's input/time boundary. Sampling
        // both sides also covers gaps and open-ended bands.
        let mut inputs = BTreeSet::from([0, u64::MAX]);
        // The quote selector uses u64::MAX as its exclusive open-ended end.
        let mut times = BTreeSet::from([0, u64::MAX - 1]);
        for rule in &prices.rules {
            for boundary in [rule.min_input_tokens, rule.max_input_tokens]
                .into_iter()
                .flatten()
            {
                insert_boundary(&mut inputs, boundary);
            }
            for boundary in [rule.valid_from.as_ref(), rule.valid_until.as_ref()]
                .into_iter()
                .flatten()
            {
                let (earliest, latest) = boundary.utc_bounds().map_err(unavailable)?;
                if earliest != latest {
                    return Err(unavailable(
                        "attempt price boundaries require a confirmed billing time zone",
                    ));
                }
                insert_boundary(&mut times, earliest);
                insert_boundary(&mut times, latest);
            }
        }

        // A recurring peak schedule can be bounded by its largest factor;
        // applying it even to an opting-out rule is conservative.
        let schedule_factor = if let Some(schedule) = &profile.pricing.peak {
            schedule.validate().map_err(unavailable)?;
            schedule.off_peak_multiplier.max(1.0)
        } else {
            1.0
        };
        let mut unscheduled = self.clone();
        unscheduled.profile.pricing.peak = None;
        let standard = tier_bounds(
            &unscheduled,
            ServiceTier::Standard,
            &inputs,
            &times,
            schedule_factor,
        )?
        .ok_or_else(|| unavailable("no published standard rates can bound this attempt"))?;
        let fast = tier_bounds(
            &unscheduled,
            ServiceTier::Fast,
            &inputs,
            &times,
            schedule_factor,
        )?;
        Ok(Some(PriceBounds {
            default_fast,
            standard,
            fast,
        }))
    }
}

fn insert_boundary(points: &mut BTreeSet<u64>, boundary: u64) {
    points.extend([
        boundary.saturating_sub(1),
        boundary,
        boundary.saturating_add(1),
    ]);
}

fn tier_bounds(
    snapshot: &FrozenPricing,
    tier: ServiceTier,
    inputs: &BTreeSet<u64>,
    times: &BTreeSet<u64>,
    factor: f64,
) -> Result<Option<TokenRates>, LlmError> {
    let mut upper = None;
    let mut unpriced = None;
    for input_tokens in inputs {
        for unix_seconds in times {
            let quote = snapshot.quote(&PricingContext {
                service_tier: Some(tier),
                submission: Submission::Interactive,
                input_tokens: Some(*input_tokens),
                unix_seconds: Some(*unix_seconds),
            })?;
            if quote.status != PriceStatus::Priced {
                unpriced.get_or_insert((*input_tokens, *unix_seconds, quote.reason));
                continue;
            }
            let Some(mut rates) = quote.rates else {
                unpriced.get_or_insert((*input_tokens, *unix_seconds, quote.reason));
                continue;
            };
            rates.reasoning_per_million = rates.reasoning_per_million.or(rates.output_per_million);
            for value in [
                &mut rates.input_per_million,
                &mut rates.output_per_million,
                &mut rates.cache_read_per_million,
                &mut rates.cache_write_per_million,
                &mut rates.cache_write_1h_per_million,
                &mut rates.reasoning_per_million,
            ]
            .into_iter()
            .flatten()
            {
                *value *= factor;
                if !value.is_finite() || *value < 0.0 {
                    return Err(unavailable("attempt rate bound exceeds the numeric range"));
                }
            }
            upper = Some(match upper {
                None => rates,
                Some(previous) => merge_bounds(previous, rates),
            });
        }
    }
    if upper.is_some() {
        if let Some((input, time, reason)) = unpriced {
            return Err(unavailable(format!(
                "published {tier:?} price rules leave an unpriced context at {input} input tokens and {time} UTC seconds: {}",
                reason.unwrap_or_else(|| "rates unavailable".into())
            )));
        }
    }
    Ok(upper)
}

fn merge_bounds(a: TokenRates, b: TokenRates) -> TokenRates {
    // A published rate in one band cannot bound an unknown rate in another.
    let max = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| a.max(b));
    TokenRates {
        input_per_million: max(a.input_per_million, b.input_per_million),
        output_per_million: max(a.output_per_million, b.output_per_million),
        cache_read_per_million: max(a.cache_read_per_million, b.cache_read_per_million),
        cache_write_per_million: max(a.cache_write_per_million, b.cache_write_per_million),
        cache_write_1h_per_million: max(a.cache_write_1h_per_million, b.cache_write_1h_per_million),
        reasoning_per_million: max(a.reasoning_per_million, b.reasoning_per_million),
        web_search_per_request: max(a.web_search_per_request, b.web_search_per_request),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{PriceBoundary, PriceBucket, PriceMultiplier, PriceRule, TokenPricing};

    fn fixture() -> crate::protocol::ProviderProfile {
        let mut profile = crate::builtin_providers()
            .unwrap()
            .into_iter()
            .find(|profile| profile.profile_name == "deepseek")
            .unwrap();
        profile
            .models
            .retain(|model| model.request_model == "deepseek-flash");
        profile
    }

    fn bounds(profile: &crate::protocol::ProviderProfile) -> Result<Option<PriceBounds>, LlmError> {
        FrozenPricing::capture(
            profile,
            &profile.models[0].display_model,
            &profile.models[0].request_model,
        )?
        .interactive_price_bounds()
    }

    fn rates(input: f64, output: f64, cache: Option<f64>) -> TokenRates {
        TokenRates {
            input_per_million: Some(input),
            output_per_million: Some(output),
            cache_read_per_million: cache,
            ..Default::default()
        }
    }

    #[test]
    fn peak_bound_retains_unknown_cache_bucket() {
        let upper = bounds(&fixture()).unwrap().unwrap();
        assert_eq!(upper.standard.input_per_million, Some(0.3));
        assert_eq!(upper.standard.output_per_million, Some(1.2));
        assert_eq!(upper.standard.cache_write_per_million, None);
    }

    #[test]
    fn token_bands_and_fast_multiplier_use_selected_quotes() {
        let mut profile = fixture();
        profile.pricing.peak = None;
        profile.models[0].pricing = Some(TokenPricing {
            rules: vec![
                PriceRule {
                    max_input_tokens: Some(99),
                    rates: rates(2.0, 5.0, Some(0.2)),
                    ..Default::default()
                },
                PriceRule {
                    min_input_tokens: Some(100),
                    rates: rates(4.0, 10.0, None),
                    ..Default::default()
                },
                PriceRule {
                    service_tier: ServiceTier::Fast,
                    multiplier: Some(PriceMultiplier {
                        factor: 2.0,
                        buckets: vec![PriceBucket::Input, PriceBucket::Output],
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let upper = bounds(&profile).unwrap().unwrap();
        assert_eq!(upper.standard.input_per_million, Some(4.0));
        assert_eq!(upper.fast.unwrap().output_per_million, Some(20.0));
        assert_eq!(upper.standard.cache_read_per_million, None);
    }

    #[test]
    fn uncovered_token_band_cannot_authorize_a_bound() {
        let mut profile = fixture();
        profile.pricing.peak = None;
        profile.models[0].pricing = Some(TokenPricing {
            rules: vec![PriceRule {
                max_input_tokens: Some(99),
                rates: rates(2.0, 5.0, Some(0.2)),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert!(matches!(
            bounds(&profile),
            Err(LlmError::CostUnavailable { .. })
        ));
    }

    #[test]
    fn dated_rules_cover_both_sides_and_require_a_time_zone() {
        let mut profile = fixture();
        profile.pricing.peak = None;
        let boundary = PriceBoundary {
            local: "2027-01-01T00:00:00".into(),
            time_zone: Some("UTC".into()),
        };
        profile.models[0].pricing = Some(TokenPricing {
            rules: vec![
                PriceRule {
                    valid_until: Some(boundary.clone()),
                    rates: rates(1.0, 2.0, Some(0.1)),
                    ..Default::default()
                },
                PriceRule {
                    valid_from: Some(boundary),
                    rates: rates(3.0, 4.0, Some(0.3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            bounds(&profile)
                .unwrap()
                .unwrap()
                .standard
                .input_per_million,
            Some(3.0)
        );
        profile.models[0].pricing.as_mut().unwrap().rules[0]
            .valid_until
            .as_mut()
            .unwrap()
            .time_zone = None;
        assert!(bounds(&profile).is_err());
    }

    #[test]
    fn fixed_rows_need_no_bound_and_connection_default_fast_is_reported() {
        let mut profile = fixture();
        profile.pricing.peak = None;
        assert_eq!(bounds(&profile).unwrap(), None);
        profile.info.features.fast = crate::protocol::CapabilitySupport::Supported;
        profile.models[0].info.features.fast = crate::protocol::CapabilitySupport::Supported;
        profile.models[0].info.features.default_service_tier = None;
        profile.info.features.default_service_tier = Some(ServiceTier::Fast);
        assert!(bounds(&profile).unwrap().unwrap().default_fast);
        profile.models[0].info.features.default_service_tier = Some(ServiceTier::Standard);
        assert_eq!(bounds(&profile).unwrap(), None);
    }

    #[test]
    fn fixed_override_replaces_peak_and_does_not_publish_fast() {
        let mut profile = fixture();
        profile.models[0].pricing =
            Some(TokenPricing::input_output(2.5, 8.0).with_fixed_standard_override());
        let upper = bounds(&profile).unwrap().unwrap();
        assert_eq!(upper.standard.input_per_million, Some(2.5));
        assert_eq!(upper.fast, None);
    }
}
