//! openrouter client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct OpenRouterClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for OpenRouterClient {}
impl ProviderClient for OpenRouterClient {
    const PROVIDER_ID: &'static str = "openrouter";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl OpenRouterClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn batch(
        &self,
        scope: crate::providers::openrouter::batch::OpenRouterBatchScope,
    ) -> Result<
        crate::providers::openrouter::batch::OpenRouterBatchService<'_>,
        crate::providers::openrouter::batch::OpenRouterBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::openrouter::batch::OpenRouterBatchService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn audio(&self) -> crate::providers::openrouter::audio::OpenRouterAudioService<'_> {
        crate::providers::openrouter::audio::OpenRouterAudioService::new(self.binding.transport())
            .with_binding(&self.binding)
    }

    pub fn rerank(
        &self,
        scope: crate::providers::openrouter::rerank::OpenRouterRerankScope,
    ) -> Result<
        crate::providers::openrouter::rerank::OpenRouterRerankService<'_>,
        crate::providers::openrouter::rerank::OpenRouterRerankError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::openrouter::rerank::OpenRouterRerankService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn embeddings(&self) -> crate::providers::openrouter::embeddings::Embeddings<'_> {
        crate::providers::openrouter::embeddings::Embeddings::new(
            self.binding.source(),
            self.profile_name(),
        )
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
