//! qwen client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct QwenClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for QwenClient {}
impl ProviderClient for QwenClient {
    const PROVIDER_ID: &'static str = "qwen";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl QwenClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn rerank(
        &self,
        scope: crate::providers::qwen::rerank::QwenRerankScope,
    ) -> Result<
        crate::providers::qwen::rerank::QwenRerankService<'_>,
        crate::providers::qwen::rerank::QwenRerankError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::rerank::QwenRerankService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn audio_generation(
        &self,
        scope: crate::providers::qwen::audio_generation::QwenAudioGenerationScope,
    ) -> Result<
        crate::providers::qwen::audio_generation::QwenAudioGenerationService<'_>,
        crate::providers::qwen::audio_generation::QwenAudioGenerationError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::audio_generation::QwenAudioGenerationService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn batch(
        &self,
        scope: crate::providers::qwen::batch::QwenBatchScope,
    ) -> Result<
        crate::providers::qwen::batch::QwenBatchService<'_>,
        crate::providers::qwen::batch::QwenBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::batch::QwenBatchService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn tts(
        &self,
        scope: crate::providers::qwen::tts::QwenTtsScope,
    ) -> Result<
        crate::providers::qwen::tts::QwenTtsService<'_>,
        crate::providers::qwen::tts::QwenTtsError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::tts::QwenTtsService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn asr(
        &self,
        scope: crate::providers::qwen::asr::QwenAsrScope,
    ) -> Result<
        crate::providers::qwen::asr::QwenAsrService<'_>,
        crate::providers::qwen::asr::QwenAsrError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::asr::QwenAsrService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn knowledge(
        &self,
        scope: crate::providers::qwen::knowledge::QwenKnowledgeScope,
    ) -> Result<
        crate::providers::qwen::knowledge::QwenKnowledgeService<'_>,
        crate::providers::qwen::knowledge::QwenKnowledgeError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::qwen::knowledge::QwenKnowledgeService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn embeddings(&self) -> crate::providers::qwen::embeddings::Embeddings<'_> {
        crate::providers::qwen::embeddings::Embeddings::new(
            self.binding.source(),
            self.profile_name(),
        )
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
