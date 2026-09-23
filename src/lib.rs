//! Provider-neutral LLM client with built-in wire codecs and provider presets.
//!
//! A protocol family is a [`WireCodec`]; a credential scheme is an
//! [`Authenticator`]. Use [`LlmClientBuilder::new`] for built-in
//! HTTP/HTTPS or inject a custom HTTP client through [`Transport`].
//! [`LlmClientBuilder`] registers the built-in codecs and model directories;
//! the built-in HTTP constructor also registers API-key and bearer authentication.
//!
//! It does not hold, fetch or store credentials. A secret arrives per request
//! on `RequestOptions::credential`, from whoever owns it; key and token
//! storage, expiry and refresh all stay outside a crate whose job is HTTP.
//!
//! Shared request, response and provider types live in
//! [`lingxi_agent_api::protocol`]. See [`api`] for the detailed API guide.
#![forbid(unsafe_code)]

#[doc = include_str!("../docs/api.md")]
pub mod api {}

#[doc = include_str!("../docs/web-search.md")]
pub mod web_search {}

pub mod auth;
pub mod client;
pub mod codecs;
pub mod directory;
pub mod framing;
pub mod presets;
pub mod transport;

pub use auth::{ApiKeyAuthenticator, Authenticator, BearerAuthenticator};
pub use client::route::{ConnectionHop, PricingModelRef, ResolvedRoute};
pub use client::{
    BuildError, LlmClient, LlmClientBuilder, ModelStream, ProviderStoreError, RequestOptions,
    ResolveError,
};
pub use codecs::anthropic::AnthropicMessagesCodec;
pub use codecs::gemini::GeminiCodec;
pub use codecs::hosted::{
    AzureOpenAiCodec, BedrockClaudeCodec, FoundryClaudeCodec, VertexClaudeCodec, VertexGeminiCodec,
};
pub use codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
pub use codecs::{FrameStream, StreamDecoder, WireCodec};
pub use directory::{
    AnthropicMessagesDirectory, GeminiDirectory, LiveModel, ModelDirectory, ModelPage,
    OpenAiChatDirectory,
};
pub use framing::sse::SseFrameSplitter;
pub use presets::{builtin as builtin_providers, merge as merge_providers, PresetError};
pub use transport::{
    Clock, HttpRequest, HttpResponse, HttpTransport, StreamResponse, SystemClock, Transport,
    UrlOpener, WebSocketSession, WsMessage,
};
