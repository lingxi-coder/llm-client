//! google client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct GoogleClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for GoogleClient {}
impl ProviderClient for GoogleClient {
    const PROVIDER_ID: &'static str = "google";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl GoogleClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn batch(
        &self,
        scope: crate::providers::google::batch::GeminiBatchScope,
    ) -> Result<
        crate::providers::google::batch::GeminiBatchService<'_>,
        crate::providers::google::batch::GeminiBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::google::batch::GeminiBatchService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn speech(
        &self,
        scope: crate::providers::google::speech::GeminiSpeechScope,
    ) -> Result<
        crate::providers::google::speech::GeminiSpeechService<'_>,
        crate::providers::google::speech::GeminiSpeechError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::google::speech::GeminiSpeechService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn file_search(
        &self,
    ) -> crate::providers::google::file_search::GeminiFileSearchService<'_> {
        crate::providers::google::file_search::GeminiFileSearchService::new(self.binding.source())
    }

    pub fn context_cache(
        &self,
        scope: crate::providers::google::context_cache::GeminiContextCacheScope,
    ) -> Result<
        crate::providers::google::context_cache::GeminiContextCacheService<'_>,
        crate::providers::google::context_cache::GeminiContextCacheError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::google::context_cache::GeminiContextCacheService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn interactions(&self) -> crate::providers::google::interactions::InteractionService<'_> {
        crate::providers::google::interactions::InteractionService::new(self.binding.source())
    }

    pub fn embeddings(&self) -> crate::providers::google::embeddings::Embeddings<'_> {
        crate::providers::google::embeddings::Embeddings::new(
            self.binding.source(),
            self.profile_name(),
        )
    }

    /// Speech generation on an explicit Vertex AI project and location.
    pub fn vertex_speech(
        &self,
        scope: crate::hosting::vertex::speech::VertexSpeechScope,
    ) -> Result<
        crate::hosting::vertex::speech::VertexSpeechService<'_>,
        crate::hosting::vertex::speech::VertexSpeechError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;
        crate::hosting::vertex::speech::VertexSpeechService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}

impl GoogleClient {
    /// Google's stored and stateless voice resources on an explicit account route.
    pub fn voices(
        &self,
        scope: crate::providers::google::speech::GeminiVoicesScope,
    ) -> Result<
        crate::providers::google::speech::GeminiVoicesService<'_>,
        crate::providers::google::speech::GeminiVoicesError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;
        crate::providers::google::speech::GeminiVoicesService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }
}
