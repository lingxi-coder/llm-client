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
//! Request, response and provider types are defined by this crate in
//! [`protocol`]. See [`api`] for the detailed API guide.
#![forbid(unsafe_code)]

#[doc = include_str!("../docs/api.md")]
pub mod api {}

#[doc = include_str!("../docs/inference.md")]
pub mod inference {}
#[doc = include_str!("../docs/images.md")]
pub mod image_generation {}

#[doc = include_str!("../docs/web-search.md")]
pub mod web_search {}

pub mod account;
pub mod anthropic_batch;
pub mod anthropic_skills;
pub mod auth;
pub mod batches;
pub mod client;
pub mod codecs;
pub mod configuration;
pub mod directory;
pub mod exact_json;
pub mod files;
pub mod framing;
pub mod gemini_batch;
pub mod gemini_context_cache;
pub mod gemini_file_search;
pub mod gemini_speech;
pub mod glm_async;
pub mod glm_audio;
pub mod glm_batch;
pub mod glm_cloud_audio;
pub mod glm_knowledge;
pub mod images;
pub mod interactions;
pub mod kimi_batch;
pub mod minimax_async_tts;
pub mod minimax_audio;
pub mod minimax_bidi_tts;
pub mod minimax_streaming_tts;
pub mod minimax_tts;
pub mod minimax_voices;
pub mod openai_containers;
pub mod openrouter_audio;
pub mod openrouter_batch;
pub mod openrouter_rerank;
pub mod presets;
pub mod protocol;
pub mod qwen_asr;
pub mod qwen_asr_realtime;
pub mod qwen_audio_generation;
pub mod qwen_batch;
pub mod qwen_knowledge;
pub mod qwen_rerank;
pub mod qwen_tts;
pub mod qwen_tts_realtime;
pub mod realtime;
pub mod retrieval;
mod runtime;
pub mod token_count;
pub mod transport;
pub mod vertex_speech;
pub mod websocket;
mod wire_options;
pub mod xai_audio;
pub mod xai_batch;
pub mod xai_collections;
pub mod xai_streaming_tts;
pub mod xai_stt;

pub use auth::{ApiKeyAuthenticator, Authenticator, BearerAuthenticator};
pub use client::account::{
    AccountRpc, CodexAccountSource, CopilotAccountSource, KimiCodeAccountSource,
};
pub use client::route::{ConnectionHop, PricingModelRef, ResolvedRoute};
pub use client::{
    AccountBalance, AccountCostBucket, AccountCostUsage, AccountExecutionOptions, AccountFailure,
    AccountFetchContext, AccountIdentity, AccountMetric, AccountQuery, AccountQuotaWindow,
    AccountReport, AccountScope, AccountScopeKind, AccountSelector, AccountSnapshot,
    AccountSubscription, AccountTokenBucket, AccountTokenUsage, AccountUsageError,
    AccountUsageSource, AlibabaAccessKey, AttachmentResolver, BuildError, ChatService,
    ClientConfigManager, ClientSnapshot, CollectedResponse, FrozenPricing, LlmClient,
    LlmClientBuilder, LocalTokenCountError, LocalTokenEstimate, LocalTokenEstimateOmission,
    ModelStream, OpenRouterResponseCache, PreparedCall, ProviderStoreError, ProviderSyncOperation,
    ProviderSyncResult, ReceivedCall, RequestDraft, RequestOptions, ResolveError, ResponsesSession,
    RoutingCatalog, StreamBatch, StructuredStreamError, StructuredStreamResult, SubscriptionStatus,
    MAX_ATTACHMENT_BYTES,
};
pub use codecs::anthropic::AnthropicMessagesCodec;
pub use codecs::gemini::GeminiCodec;
pub use codecs::hosted::{
    AzureOpenAiCodec, BedrockClaudeCodec, FoundryClaudeCodec, VertexClaudeCodec, VertexGeminiCodec,
};
pub use codecs::openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec};
pub use codecs::{
    CodecContext, ContentBinding, EncodeRequest, PreparedMedia, RequestMode, StreamDecoder,
    WireCodec,
};
pub use directory::{
    AnthropicMessagesDirectory, DecodedModelPage, GeminiDirectory, LiveModel, ModelDirectory,
    ModelPage, OpenAiChatDirectory,
};
pub use files::{
    capabilities as provider_file_capabilities,
    capabilities_for_purpose as provider_file_capabilities_for_purpose, DownloadSupport,
    FileCapabilities, FileOperation, FilePurpose, FileService, ModelFileReference,
    ProviderFileContent, ProviderFileMetadata, ProviderFilePage, ProviderFileRef, UploadFile,
    MAX_PROVIDER_FILE_DOWNLOAD_BYTES,
};
pub use framing::sse::SseFrameSplitter;
pub use images::{ImageAdapter, ImageAuthenticator, ImageDispatch, ImageError, ImageService};
pub use presets::{
    builtin as builtin_providers, builtin_catalog, merge as merge_providers, PresetError,
};
pub use transport::{
    Clock, HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest, HttpTransport,
    StreamResponse, SystemClock, Transport,
};

pub mod audio;
pub mod background;
pub mod deferred;
pub mod embeddings;
