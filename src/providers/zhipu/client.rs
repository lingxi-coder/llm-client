//! zhipu client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct ZhipuClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for ZhipuClient {}
impl ProviderClient for ZhipuClient {
    const PROVIDER_ID: &'static str = "zhipu";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl ZhipuClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn async_tasks(
        &self,
        config: crate::providers::zhipu::async_tasks::GlmAsyncConfig,
    ) -> Result<
        crate::providers::zhipu::async_tasks::GlmAsyncService<'_>,
        crate::providers::zhipu::async_tasks::GlmAsyncError,
    > {
        self.binding
            .validate_scope(config.profile_name(), config.account_scope())?;

        crate::providers::zhipu::async_tasks::GlmAsyncService::new(self.binding.transport(), config)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn cloud_audio(
        &self,
        scope: crate::providers::zhipu::cloud_audio::GlmCloudAudioScope,
    ) -> Result<
        crate::providers::zhipu::cloud_audio::GlmCloudAudioService<'_>,
        crate::providers::zhipu::cloud_audio::GlmCloudAudioError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::zhipu::cloud_audio::GlmCloudAudioService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn batch(
        &self,
        scope: crate::providers::zhipu::batch::GlmBatchScope,
    ) -> Result<
        crate::providers::zhipu::batch::GlmBatchService<'_>,
        crate::providers::zhipu::batch::GlmBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::zhipu::batch::GlmBatchService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn knowledge(&self) -> crate::providers::zhipu::knowledge::GlmKnowledgeService<'_> {
        crate::providers::zhipu::knowledge::GlmKnowledgeService::new(self.binding.source())
    }

    pub fn embeddings(&self) -> crate::providers::zhipu::embeddings::Embeddings<'_> {
        crate::providers::zhipu::embeddings::Embeddings::new(
            self.binding.source(),
            self.profile_name(),
        )
    }

    pub fn asr(
        &self,
        route: crate::providers::zhipu::audio::GlmAsrRoute,
        scope: crate::providers::zhipu::audio::GlmAsrScope,
    ) -> Result<
        crate::providers::zhipu::audio::GlmAsrService<'_>,
        crate::providers::zhipu::audio::GlmAsrError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;
        crate::providers::zhipu::audio::GlmAsrService::new(self.binding.transport(), route, scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
