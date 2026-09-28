//! minimax client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct MiniMaxClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for MiniMaxClient {}
impl ProviderClient for MiniMaxClient {
    const PROVIDER_ID: &'static str = "minimax";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl MiniMaxClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn async_tts(
        &self,
        config: crate::providers::minimax::async_tts::MiniMaxAsyncTtsConfig,
    ) -> Result<
        crate::providers::minimax::async_tts::MiniMaxAsyncTtsService<'_>,
        crate::providers::minimax::async_tts::MiniMaxAsyncTtsError,
    > {
        self.binding
            .validate_scope(&config.profile_name, &config.account_scope)?;

        crate::providers::minimax::async_tts::MiniMaxAsyncTtsService::new(
            self.binding.transport(),
            config,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn tts(
        &self,
        config: crate::providers::minimax::tts::MiniMaxTtsConfig,
    ) -> Result<
        crate::providers::minimax::tts::MiniMaxTtsService<'_>,
        crate::providers::minimax::tts::MiniMaxTtsError,
    > {
        self.binding
            .validate_scope(&config.profile_name, &config.account_scope)?;

        crate::providers::minimax::tts::MiniMaxTtsService::new(self.binding.transport(), config)
            .map(|service| service.with_binding(&self.binding))
    }

    pub fn audio(
        &self,
        endpoint: impl Into<String>,
    ) -> Result<
        crate::providers::minimax::audio::MiniMaxAudioService<'_>,
        crate::providers::minimax::audio::MiniMaxAudioError,
    > {
        self.binding.pin()?;

        crate::providers::minimax::audio::MiniMaxAudioService::new(
            self.binding.transport(),
            endpoint,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn voices(
        &self,
        config: crate::providers::minimax::voices::MiniMaxVoicesConfig,
    ) -> Result<
        crate::providers::minimax::voices::MiniMaxVoicesService<'_>,
        crate::providers::minimax::voices::MiniMaxVoicesError,
    > {
        self.binding
            .validate_scope(&config.profile_name, &config.account_scope)?;
        crate::providers::minimax::voices::MiniMaxVoicesService::new(
            self.binding.transport(),
            config,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
