//! github-copilot client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct GithubCopilotClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for GithubCopilotClient {}
impl ProviderClient for GithubCopilotClient {
    const PROVIDER_ID: &'static str = "github-copilot";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}
impl GithubCopilotClient {
    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }
}
