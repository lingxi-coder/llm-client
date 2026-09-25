//! Prices captured with the exact model row before dispatch.
use super::{price_query, pricing, route::PricingModelRef};
use crate::protocol::*;

#[derive(Debug, Clone)]
pub struct FrozenPricing {
    pub(super) profile: ProviderProfile,
    pub(super) model: ModelProfile,
}
impl FrozenPricing {
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
}
