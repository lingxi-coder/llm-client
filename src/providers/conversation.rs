//! Conversation operations on an exact provider binding.
//!
//! These operations enter the same snapshot execution kernel as the unified
//! conversation service. Provider policies, registered codecs, admission and
//! transport are selected once by that kernel.
use super::binding::ProviderBinding;
use crate::client::{ModelStream, PreparedCall, RequestOptions};
use crate::codecs::RequestMode;
use crate::protocol::{ChatRequest, ChatResponse, LlmError, WebSearchConfig};

#[derive(Clone, Copy)]
pub struct ProviderChat<'a> {
    binding: &'a ProviderBinding,
}

impl ProviderChat<'_> {
    pub async fn complete(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        self.binding
            .pin()?
            .complete_bound_in(
                self.binding.profile_name(),
                self.binding.provider_id(),
                request,
                options,
            )
            .await
    }

    pub async fn stream(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.binding
            .pin()?
            .stream_bound_in(
                self.binding.profile_name(),
                self.binding.provider_id(),
                request,
                options,
            )
            .await
    }

    pub async fn web_search(
        &self,
        request: &ChatRequest,
        search: WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let mut request = request.clone();
        request.set_hosted_web_search(Some(search));
        self.complete(&request, options).await
    }

    pub async fn web_search_stream(
        &self,
        request: &ChatRequest,
        search: WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut request = request.clone();
        request.set_hosted_web_search(Some(search));
        self.stream(&request, options).await
    }

    /// Prepare a single-use call on this exact connection, without generation.
    pub async fn prepare(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<PreparedCall, LlmError> {
        self.binding
            .pin()?
            .prepare_on(self.binding.profile_name(), request, options, mode)
            .await
    }
}

macro_rules! conversations {
    ($($provider:ident),+ $(,)?) => {$ (
        impl super::$provider {
            pub fn chat(&self) -> ProviderChat<'_> {
                ProviderChat { binding: &self.binding }
            }
        }
    )+};
}
conversations!(
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
