//! Provider clients and provider-native resource APIs.
pub mod anthropic;
pub(crate) mod binding;
pub mod deepseek;
pub(crate) mod dispatch;
pub mod github_copilot;
pub mod google;
pub mod kimi;
pub mod minimax;
pub mod openai;
pub mod openrouter;
pub mod qwen;
pub mod response_headers;
pub mod xai;
pub mod zhipu;
pub use anthropic::AnthropicClient;
pub use binding::{ProviderBindingError, ProviderClient};
pub use deepseek::DeepSeekClient;
pub use github_copilot::GithubCopilotClient;
pub use google::GoogleClient;
pub use kimi::KimiClient;
pub use minimax::MiniMaxClient;
pub use openai::OpenAiClient;
pub use openrouter::OpenRouterClient;
pub use qwen::QwenClient;
pub use xai::XaiClient;
pub use zhipu::ZhipuClient;
pub mod conversation;
pub mod file_resources;
pub(crate) mod native;

mod account_resources;

pub(crate) mod attachment_policy;
