//! Live entry points capture exactly one published view per operation.
use super::*;
use crate::protocol::*;

impl LlmClient {
    /// Capture one immutable configuration and account-binding revision.
    /// The publication lock is released before returning this view.
    pub fn snapshot(&self) -> ClientSnapshot {
        let state = self
            .published
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        ClientSnapshot {
            runtime: self.runtime.clone(),
            state,
            bound_profile: None,
        }
    }
    pub fn chat(&self) -> ChatService<'_> {
        ChatService::new(ClientSource::live(&self.runtime, &self.published))
    }
    pub fn decisions(&self) -> super::DecisionService<'_> {
        super::DecisionService::new(ClientSource::live(&self.runtime, &self.published))
    }
    pub fn images(&self) -> crate::images::ImageService<'_> {
        crate::images::ImageService::new(ClientSource::live(&self.runtime, &self.published))
    }
    pub fn embeddings(&self) -> crate::embeddings::EmbeddingService<'_> {
        crate::embeddings::EmbeddingService::new(ClientSource::live(&self.runtime, &self.published))
    }
    /// Bind one exact profile to its provider-specific client.
    pub fn provider<T: crate::providers::ProviderClient>(
        &self,
        profile_name: &str,
    ) -> Result<T, crate::providers::ProviderBindingError> {
        crate::providers::binding::bind(
            crate::runtime::OwnedClientSource::live(&self.runtime, &self.published),
            profile_name,
        )
    }

    pub async fn account_usage(
        &self,
        profile_name: &str,
        query: &account::AccountQuery,
    ) -> Result<account::AccountSnapshot, account::AccountUsageError> {
        self.snapshot().account_usage(profile_name, query).await
    }

    pub async fn accounts_usage(
        &self,
        queries: &BTreeMap<String, account::AccountQuery>,
    ) -> Vec<(
        String,
        Result<account::AccountSnapshot, account::AccountUsageError>,
    )> {
        self.snapshot().accounts_usage(queries).await
    }

    pub fn models(&self) -> Vec<ModelListing> {
        self.snapshot().models()
    }

    pub fn providers(&self) -> Vec<ProviderListing> {
        self.snapshot().providers()
    }

    pub fn region(&self) -> Region {
        self.snapshot().region()
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.snapshot().codec_families()
    }

    pub fn directory_for(&self, profile: &ProviderProfile) -> Option<Arc<dyn ModelDirectory>> {
        self.snapshot().directory_for(profile)
    }

    pub fn directory_shapes(&self) -> Vec<ProtocolFamily> {
        self.snapshot().directory_shapes()
    }

    pub fn estimate_actual_cost(
        &self,
        route: &ResolvedRoute,
        response: &crate::protocol::ChatResponse,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_actual_cost(route, response, submission)
    }

    pub fn estimate_stream_cost(
        &self,
        route: &ResolvedRoute,
        stream: &ModelStream,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_stream_cost(route, stream, submission)
    }

    pub fn estimate_cost_for_profile(
        &self,
        route: &ResolvedRoute,
        name: &str,
        report: &crate::protocol::UsageReport,
        inference: &crate::protocol::InferenceReport,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_cost_for_profile(route, name, report, inference, submission)
    }

    pub fn resolve(&self, model: &str) -> Result<ResolvedRoute, ResolveError> {
        self.snapshot().resolve(model)
    }

    pub fn resolve_in(
        &self,
        model: &str,
        profile: Option<&str>,
    ) -> Result<ResolvedRoute, ResolveError> {
        self.snapshot().resolve_in(model, profile)
    }

    pub fn price_quote(
        &self,
        model: &str,
        profile: Option<&str>,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        self.snapshot().price_quote(model, profile, context)
    }

    pub fn price_quote_for_route(
        &self,
        route: &ResolvedRoute,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        self.snapshot().price_quote_for_route(route, context)
    }

    pub fn estimate_cost(
        &self,
        route: &ResolvedRoute,
        usage: &Usage,
        context: &PricingContext,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot().estimate_cost(route, usage, context)
    }

    pub fn estimate_local_tokens(
        &self,
        request: &ChatRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        self.snapshot().estimate_local_tokens(request)
    }

    pub fn estimate_local_tokens_in(
        &self,
        profile_or_group: &str,
        request: &ChatRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        self.snapshot()
            .estimate_local_tokens_in(profile_or_group, request)
    }
}
