//! Account queries share the same registry and execution budgets as unified queries.
use crate::account::{AccountQuery, AccountSnapshot, AccountUsageError};

macro_rules! account_queries {
    ($($provider:ident),+ $(,)?) => {$ (
        impl super::$provider {
            /// Query this exact profile using its per-profile source and caller credentials.
            pub async fn account_usage(&self, query: &AccountQuery) -> Result<AccountSnapshot, AccountUsageError> {
                let snapshot = self.binding.pin().map_err(|error| AccountUsageError::ProviderBinding(error.to_string()))?;
                snapshot.account_usage(self.profile_name(), query).await
            }
        }
    )+};
}
account_queries!(
    OpenAiClient,
    AnthropicClient,
    GoogleClient,
    QwenClient,
    MiniMaxClient,
    ZhipuClient,
    KimiClient,
    XaiClient,
    OpenRouterClient,
    DeepSeekClient,
    GithubCopilotClient,
);
