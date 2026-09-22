//! LLM layer (§7.2): the traits and the provider-neutral client. A protocol
//! family is a `WireCodec`; a credential scheme is an `Authenticator`; the HTTP
//! client is the platform's `Transport`. This crate depends on none of them
//! (gate 28) — `LlmClientBuilder` receives them and refuses to build if a
//! profile's protocol has no codec (gate 33).
//!
//! It does not hold, fetch or store credentials. A secret arrives per request
//! on `RequestOptions::credential`, from whoever owns it; key and token
//! storage, expiry and refresh all stay outside a crate whose job is HTTP.
//!
//! `ProviderProfile` is not here: it is serializable configuration that the
//! codec capabilities, this crate and `config` all read, so it lives once in
//! `agent_api::protocol` (review C2).
#![forbid(unsafe_code)]

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
    BuildError, LlmClient, LlmClientBuilder, ModelStream, RequestOptions, ResolveError,
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
    Clock, HttpRequest, HttpResponse, LlmServices, StreamResponse, Transport, UrlOpener,
    WebSocketSession, WsMessage,
};
