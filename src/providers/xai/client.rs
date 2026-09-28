//! xai client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct XaiClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for XaiClient {}
impl ProviderClient for XaiClient {
    const PROVIDER_ID: &'static str = "xai";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl XaiClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn collections(
        &self,
        config: crate::providers::xai::collections::XaiCollectionsConfig,
    ) -> Result<
        crate::providers::xai::collections::XaiCollectionsClient<'_>,
        crate::providers::xai::collections::XaiCollectionsError,
    > {
        self.binding
            .validate_scope(&config.profile_name, &config.account_scope)?;

        crate::providers::xai::collections::XaiCollectionsClient::new(
            self.binding.transport(),
            config,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn audio(
        &self,
        config: crate::providers::xai::audio::XaiAudioConfig,
    ) -> Result<
        crate::providers::xai::audio::XaiAudioService<'_>,
        crate::providers::xai::audio::XaiAudioError,
    > {
        self.binding
            .validate_scope(&config.profile_name, &config.account_scope)?;

        crate::providers::xai::audio::XaiAudioService::new(self.binding.transport(), config)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn batch(
        &self,
        scope: crate::providers::xai::batch::XaiBatchScope,
    ) -> Result<
        crate::providers::xai::batch::XaiBatchService<'_>,
        crate::providers::xai::batch::XaiBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::xai::batch::XaiBatchService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn deferred(&self) -> crate::providers::xai::deferred::DeferredService<'_> {
        crate::providers::xai::deferred::DeferredService::new(self.binding.source())
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
