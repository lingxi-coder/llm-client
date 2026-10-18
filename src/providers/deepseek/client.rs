//! deepseek client and its provider-native resource entry points.
use crate::providers::binding::{private, ProviderBinding, ProviderClient};

#[derive(Clone)]
pub struct DeepSeekClient {
    pub(crate) binding: ProviderBinding,
}
impl private::Sealed for DeepSeekClient {}
impl ProviderClient for DeepSeekClient {
    const PROVIDER_ID: &'static str = "deepseek";
    fn from_binding(binding: ProviderBinding) -> Self {
        Self { binding }
    }
}
impl DeepSeekClient {
    /// Independent Search/Fetch operations using this exact account binding.
    pub fn web(&self) -> super::web::DeepSeekWeb<'_> {
        super::web::DeepSeekWeb {
            binding: &self.binding,
        }
    }

    pub fn profile_name(&self) -> &str {
        self.binding.profile_name()
    }
}
