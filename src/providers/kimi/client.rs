//! kimi client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct KimiClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for KimiClient {}
impl ProviderClient for KimiClient {
    const PROVIDER_ID: &'static str = "kimi";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}

impl KimiClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }

    pub fn batch(
        &self,
        scope: crate::providers::kimi::batch::KimiBatchScope,
    ) -> Result<
        crate::providers::kimi::batch::KimiBatchService<'_>,
        crate::providers::kimi::batch::KimiBatchError,
    > {
        self.binding
            .validate_scope(scope.profile_name(), scope.account_scope())?;

        crate::providers::kimi::batch::KimiBatchService::new(self.binding.transport(), scope)
            .map(|service| service.with_binding(&self.binding))
    }
}
