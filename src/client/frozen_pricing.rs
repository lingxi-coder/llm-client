//! Prices captured with the exact model row before dispatch.
use super::{price_query, pricing, route::PricingModelRef};
use crate::protocol::*;

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
}
