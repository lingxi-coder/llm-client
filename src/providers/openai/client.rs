//! openai client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct OpenAiClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for OpenAiClient {}
impl ProviderClient for OpenAiClient {
    const PROVIDER_ID: &'static str = "openai";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl OpenAiClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    /// List models visible to the selected ChatGPT account and its plan grant.
    /// The caller supplies a current OAuth token for this exact account.
    pub async fn chatgpt_plan_models(
        &self,
        access_token: &crate::protocol::Secret<String>,
    ) -> Result<
        Vec<crate::providers::openai::chatgpt_plan::ChatGptPlanModel>,
        crate::providers::openai::chatgpt_plan::ChatGptPlanModelError,
    > {
        crate::providers::openai::chatgpt_plan::list_models(&self.binding, access_token).await
    }

    pub fn containers(
        &self,
        scope: crate::providers::openai::containers::OpenAiContainerScope,
    ) -> Result<
        crate::providers::openai::containers::OpenAiContainersService<'_>,
        crate::providers::openai::containers::OpenAiContainersError,
    > {
        let snapshot = self.binding.pin()?;
        if snapshot
            .profile(self.profile_name())
            .is_some_and(|profile| profile.auth == crate::protocol::AuthStrategy::ChatGptPlan)
        {
            return Err(crate::protocol::LlmError::UnsupportedCapability {
                message: "ChatGPT plan usage does not support OpenAI Containers".into(),
            }
            .into());
        }
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::openai::containers::OpenAiContainersService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn audio(&self) -> crate::providers::openai::audio::AudioService<'_> {
        crate::providers::openai::audio::AudioService::new(self.binding.source())
    }

    pub fn batches(&self) -> crate::providers::openai::batches::BatchService<'_> {
        crate::providers::openai::batches::BatchService::new(self.binding.source())
    }

    pub fn background(&self) -> crate::providers::openai::background::BackgroundService<'_> {
        crate::providers::openai::background::BackgroundService::new(self.binding.source())
    }

    pub fn retrieval(&self) -> crate::providers::openai::retrieval::RetrievalService<'_> {
        crate::providers::openai::retrieval::RetrievalService::new(self.binding.source())
    }

    pub fn embeddings(&self) -> crate::providers::openai::embeddings::Embeddings<'_> {
        crate::providers::openai::embeddings::Embeddings::new(
            self.binding.source(),
            self.profile_name(),
        )
    }

    pub fn images(&self) -> crate::images::ProviderImageService<'_> {
        crate::images::ProviderImageService::new(self.binding.source(), self.profile_name())
    }
}
