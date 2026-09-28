//! anthropic client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct AnthropicClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for AnthropicClient {}
impl ProviderClient for AnthropicClient {
    const PROVIDER_ID: &'static str = "anthropic";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl AnthropicClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn batch(
        &self,
        scope: crate::providers::anthropic::batch::AnthropicBatchScope,
    ) -> Result<
        crate::providers::anthropic::batch::AnthropicBatchService<'_>,
        crate::providers::anthropic::batch::AnthropicBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::anthropic::batch::AnthropicBatchService::new(
            self.binding.transport(),
            scope,
        )
        .map(|service| service.with_binding(&self.binding))
    }

    pub fn skills(
        &self,
        scope: crate::providers::anthropic::types::AnthropicSkillScope,
    ) -> Result<
        crate::providers::anthropic::skills::AnthropicSkillsService<'_>,
        crate::providers::anthropic::skills::AnthropicSkillsError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;
        crate::providers::anthropic::skills::AnthropicSkillsService::bound(&self.binding, scope)
    }
}
