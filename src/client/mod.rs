//! `LlmClientBuilder` / `LlmClient`: the provider-neutral machine.
//!
//! `resolve` turns a model id into a route and its failover chain; `complete`
//! and `stream` walk that chain (gate 32). Nothing here knows a provider by
//! name — a new OpenAI-compatible provider is a `ProviderProfile` in settings
//! and no code change (gate 30). The builder refuses to build a client whose
//! profile names a protocol with no codec (gate 33).

pub use crate::account;
mod executor;
mod live;
mod requests;
pub use crate::files;
mod attachments;
mod builder;
pub(crate) use attachments::*;
use builder::validate_profiles;
pub use builder::{BuildError, LlmClientBuilder};
pub mod chat;
pub mod options;
mod price_query;
pub mod pricing;
mod resolve;
mod response_cache;
pub mod route;
mod snapshot;
mod store;
mod stream;
pub use crate::token_count;

pub use account::{
    AccountBalance, AccountCostBucket, AccountCostUsage, AccountExecutionOptions, AccountFailure,
    AccountFetchContext, AccountIdentity, AccountMetric, AccountQuery, AccountQuotaWindow,
    AccountReport, AccountScope, AccountScopeKind, AccountSelector, AccountSnapshot,
    AccountSubscription, AccountTokenBucket, AccountTokenUsage, AccountUsageError,
    AccountUsageSource, AlibabaAccessKey, SubscriptionStatus,
};
pub use chat::ChatService;
pub(crate) use options::apply_openrouter_response_cache;
pub use options::{OpenRouterResponseCache, RequestOptions};
pub use resolve::ResolveError;
pub(crate) use response_cache::openrouter_observation;
pub use store::{
    ClientConfigManager, ProviderStoreError, ProviderSyncOperation, ProviderSyncResult,
};
pub use stream::{ModelStream, StructuredStreamError, StructuredStreamResult};
pub use token_count::{LocalTokenCountError, LocalTokenEstimate, LocalTokenEstimateOmission};

use crate::auth::Authenticator;
use crate::codecs::WireCodec;
use crate::directory::ModelDirectory;
use crate::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, CredentialConfig, LlmError, ModelListing,
    ProtocolFamily, ProviderListing, ProviderProfile, Region, Submission,
};
use crate::transport::{Clock, HttpTransport, SystemClock, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use route::ResolvedRoute;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Maximum app attachment content resolved into one in-memory model request.
pub const MAX_ATTACHMENT_BYTES: u64 = 64 * 1024 * 1024;

/// Application-owned source of attachment bytes. A resolver can read from a
/// local object store, a remote task service, or another authenticated host
/// application. It must return the exact immutable bytes identified by the
/// attachment id and revision.
#[async_trait]
pub trait AttachmentResolver: Send + Sync + 'static {
    async fn resolve(&self, attachment: &AttachmentRef) -> Result<Bytes, LlmError>;

    /// Opt in to reuse of this immutable revision's bytes across requests.
    ///
    /// This is called for every unique attachment in every request, including
    /// cache misses. Return `true` only after checking that the current caller
    /// may access the attachment, that it remains available, and that the full
    /// reference identifies the same immutable content. Errors are propagated
    /// even when bytes are cached. A `true` result permits the client to reuse
    /// bounded cached content and combine concurrent reads of this reference.
    ///
    /// The default returns `false`, so `resolve` remains the per-request
    /// authority and availability check. Returning `false` never reuses bytes.
    async fn validate_content_reuse(&self, _attachment: &AttachmentRef) -> Result<bool, LlmError> {
        Ok(false)
    }
}

/// A cheap, cloneable handle to shared services and the latest published configuration.
/// Model, inference controls, credentials and deadlines remain request-scoped.
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) runtime: Arc<RuntimeResources>,
    pub(crate) published: Arc<RwLock<Arc<PublishedState>>>,
}

/// An owned, immutable view of one configuration revision.
/// Keep this view for related routing, estimation, execution and pricing operations.
#[derive(Clone)]
pub struct ClientSnapshot {
    pub(crate) runtime: Arc<RuntimeResources>,
    pub(crate) state: Arc<PublishedState>,
}

pub(crate) struct RuntimeResources {
    pub(crate) region: Region,
    pub(crate) http: Arc<dyn Transport>,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    pub(crate) image_adapters:
        BTreeMap<crate::protocol::ImageApi, Arc<dyn crate::images::ImageAdapter>>,
    pub(crate) image_authenticators: BTreeMap<String, Arc<dyn crate::images::ImageAuthenticator>>,
    pub(crate) directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    pub(crate) authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    pub(crate) accounts: account::Service,
    pub(crate) attachments: AttachmentManager,
}

pub(crate) struct PublishedState {
    pub(crate) revision: u64,
    pub(crate) config: Arc<snapshot::RuntimeSnapshot>,
    pub(crate) account_registry: Arc<account::Registry>,
    pub(crate) cache_namespace: u64,
    pub(crate) cache_generations: BTreeMap<String, u64>,
}

#[derive(Clone, Copy)]
pub(crate) enum ClientSource<'a> {
    Live(&'a LlmClient),
    Snapshot(&'a ClientSnapshot),
}
impl ClientSource<'_> {
    pub(crate) fn snapshot(self) -> ClientSnapshot {
        match self {
            Self::Live(client) => client.snapshot(),
            Self::Snapshot(snapshot) => snapshot.clone(),
        }
    }
}

impl ClientSnapshot {
    /// The conversation service. All chat requests enter through this facade.
    pub fn chat(&self) -> ChatService<'_> {
        ChatService::new(ClientSource::Snapshot(self))
    }

    pub fn embeddings(&self) -> crate::embeddings::EmbeddingService<'_> {
        crate::embeddings::EmbeddingService::new(ClientSource::Snapshot(self))
    }

    pub fn retrieval(&self) -> crate::retrieval::RetrievalService<'_> {
        crate::retrieval::RetrievalService::new(ClientSource::Snapshot(self))
    }

    pub fn batches(&self) -> crate::batches::BatchService<'_> {
        crate::batches::BatchService::new(ClientSource::Snapshot(self))
    }

    pub fn deferred(&self) -> crate::deferred::DeferredService<'_> {
        crate::deferred::DeferredService::new(ClientSource::Snapshot(self))
    }
    pub fn background(&self) -> crate::background::BackgroundService<'_> {
        crate::background::BackgroundService::new(ClientSource::Snapshot(self))
    }
    pub fn audio(&self) -> crate::audio::AudioService<'_> {
        crate::audio::AudioService::new(ClientSource::Snapshot(self))
    }
    pub fn interactions(&self) -> crate::interactions::InteractionService<'_> {
        crate::interactions::InteractionService::new(ClientSource::Snapshot(self))
    }
    /// Gemini File Search store and document lifecycle.
    pub fn gemini_file_search(&self) -> crate::gemini_file_search::GeminiFileSearchService<'_> {
        crate::gemini_file_search::GeminiFileSearchService::new(ClientSource::Snapshot(self))
    }
    /// Gemini Developer API inline Batch lifecycle on an explicit account route.
    pub fn gemini_batch(
        &self,
        scope: crate::gemini_batch::GeminiBatchScope,
    ) -> Result<crate::gemini_batch::GeminiBatchService<'_>, crate::gemini_batch::GeminiBatchError>
    {
        crate::gemini_batch::GeminiBatchService::new(self.runtime.http.as_ref(), scope)
    }
    /// Gemini Developer API explicit cached-content resources on a selected route.
    pub fn gemini_context_cache(
        &self,
        scope: crate::gemini_context_cache::GeminiContextCacheScope,
    ) -> Result<
        crate::gemini_context_cache::GeminiContextCacheService<'_>,
        crate::gemini_context_cache::GeminiContextCacheError,
    > {
        crate::gemini_context_cache::GeminiContextCacheService::new(
            self.runtime.http.as_ref(),
            scope,
        )
    }
    /// Gemini Interactions unary speech synthesis on an explicit account route.
    pub fn gemini_speech(
        &self,
        scope: crate::gemini_speech::GeminiSpeechScope,
    ) -> Result<
        crate::gemini_speech::GeminiSpeechService<'_>,
        crate::gemini_speech::GeminiSpeechError,
    > {
        crate::gemini_speech::GeminiSpeechService::new(self.runtime.http.as_ref(), scope)
    }
    /// OpenAI explicit Containers and files on an account-bound route.
    pub fn openai_containers(
        &self,
        scope: crate::openai_containers::OpenAiContainerScope,
    ) -> Result<
        crate::openai_containers::OpenAiContainersService<'_>,
        crate::openai_containers::OpenAiContainersError,
    > {
        crate::openai_containers::OpenAiContainersService::new(self.runtime.http.as_ref(), scope)
    }
    /// GLM Knowledge Base managed retrieval and ingestion.
    pub fn glm_knowledge(&self) -> crate::glm_knowledge::GlmKnowledgeService<'_> {
        crate::glm_knowledge::GlmKnowledgeService::new(ClientSource::Snapshot(self))
    }
    /// Hosted GLM speech transcription and mainland synthesis on an explicit account route.
    pub fn glm_cloud_audio(
        &self,
        scope: crate::glm_cloud_audio::GlmCloudAudioScope,
    ) -> Result<
        crate::glm_cloud_audio::GlmCloudAudioService<'_>,
        crate::glm_cloud_audio::GlmCloudAudioError,
    > {
        crate::glm_cloud_audio::GlmCloudAudioService::new(self.runtime.http.as_ref(), scope)
    }
    /// MiniMax file transcription at one explicit official regional endpoint.
    pub fn minimax_audio(
        &self,
        endpoint: impl Into<String>,
    ) -> Result<
        crate::minimax_audio::MiniMaxAudioService<'_>,
        crate::minimax_audio::MiniMaxAudioError,
    > {
        crate::minimax_audio::MiniMaxAudioService::new(self.runtime.http.as_ref(), endpoint)
    }
    /// MiniMax native synchronous text-to-speech on a selected region.
    pub fn minimax_tts(
        &self,
        config: crate::minimax_tts::MiniMaxTtsConfig,
    ) -> Result<crate::minimax_tts::MiniMaxTtsService<'_>, crate::minimax_tts::MiniMaxTtsError>
    {
        crate::minimax_tts::MiniMaxTtsService::new(self.runtime.http.as_ref(), config)
    }
    /// MiniMax native asynchronous long-text speech jobs on a selected region.
    pub fn minimax_async_tts(
        &self,
        config: crate::minimax_async_tts::MiniMaxAsyncTtsConfig,
    ) -> Result<
        crate::minimax_async_tts::MiniMaxAsyncTtsService<'_>,
        crate::minimax_async_tts::MiniMaxAsyncTtsError,
    > {
        crate::minimax_async_tts::MiniMaxAsyncTtsService::new(self.runtime.http.as_ref(), config)
    }
    /// OpenRouter speech transcription and synthesis with request-scoped credentials.
    pub fn openrouter_audio(&self) -> crate::openrouter_audio::OpenRouterAudioService<'_> {
        crate::openrouter_audio::OpenRouterAudioService::new(self.runtime.http.as_ref())
    }
    /// GLM's provider-native asynchronous Chat task lifecycle.
    pub fn glm_async(
        &self,
        config: crate::glm_async::GlmAsyncConfig,
    ) -> Result<crate::glm_async::GlmAsyncService<'_>, crate::glm_async::GlmAsyncError> {
        crate::glm_async::GlmAsyncService::new(self.runtime.http.as_ref(), config)
    }
    /// Self-hosted GLM-ASR on an explicit SGLang route and account scope.
    pub fn glm_asr(
        &self,
        credential: crate::protocol::Secret<String>,
        route: crate::glm_audio::GlmAsrRoute,
        scope: crate::glm_audio::GlmAsrScope,
    ) -> Result<crate::glm_audio::GlmAsrService<'_>, crate::glm_audio::GlmAsrError> {
        crate::glm_audio::GlmAsrService::new(self.runtime.http.as_ref(), credential, route, scope)
    }
    /// GLM native Batch upload and task lifecycle in a documented region.
    pub fn glm_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::glm_batch::GlmBatchScope,
    ) -> Result<crate::glm_batch::GlmBatchService<'_>, crate::glm_batch::GlmBatchError> {
        crate::glm_batch::GlmBatchService::new(self.runtime.http.as_ref(), credential, scope)
    }
    /// Qwen Batch at an explicitly selected account, region, and workspace scope.
    pub fn qwen_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::qwen_batch::QwenBatchScope,
    ) -> Result<crate::qwen_batch::QwenBatchService<'_>, crate::qwen_batch::QwenBatchError> {
        crate::qwen_batch::QwenBatchService::new(self.runtime.http.as_ref(), credential, scope)
    }
    /// Qwen asynchronous file transcription with an explicit regional scope.
    pub fn qwen_asr(
        &self,
        scope: crate::qwen_asr::QwenAsrScope,
    ) -> Result<crate::qwen_asr::QwenAsrService<'_>, crate::qwen_asr::QwenAsrError> {
        crate::qwen_asr::QwenAsrService::new(self.runtime.http.as_ref(), scope)
    }
    /// Qwen's native knowledge retrieval with an explicit regional workspace.
    pub fn qwen_knowledge(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::qwen_knowledge::QwenKnowledgeScope,
    ) -> Result<
        crate::qwen_knowledge::QwenKnowledgeService<'_>,
        crate::qwen_knowledge::QwenKnowledgeError,
    > {
        crate::qwen_knowledge::QwenKnowledgeService::new(
            self.runtime.http.as_ref(),
            credential,
            scope,
        )
    }
    /// Qwen non-realtime text-to-speech on an explicit regional account scope.
    pub fn qwen_tts(
        &self,
        scope: crate::qwen_tts::QwenTtsScope,
    ) -> Result<crate::qwen_tts::QwenTtsService<'_>, crate::qwen_tts::QwenTtsError> {
        crate::qwen_tts::QwenTtsService::new(self.runtime.http.as_ref(), scope)
    }
    /// Qwen Audio Generation on an explicit Beijing workspace scope.
    pub fn qwen_audio_generation(
        &self,
        scope: crate::qwen_audio_generation::QwenAudioGenerationScope,
    ) -> Result<
        crate::qwen_audio_generation::QwenAudioGenerationService<'_>,
        crate::qwen_audio_generation::QwenAudioGenerationError,
    > {
        crate::qwen_audio_generation::QwenAudioGenerationService::new(
            self.runtime.http.as_ref(),
            scope,
        )
    }
    /// Qwen native text reranking in a documented regional workspace.
    pub fn qwen_rerank(
        &self,
        scope: crate::qwen_rerank::QwenRerankScope,
    ) -> Result<crate::qwen_rerank::QwenRerankService<'_>, crate::qwen_rerank::QwenRerankError>
    {
        crate::qwen_rerank::QwenRerankService::new(self.runtime.http.as_ref(), scope)
    }
    /// Anthropic Messages Batch with an explicit account and route scope.
    pub fn anthropic_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::anthropic_batch::AnthropicBatchScope,
    ) -> Result<
        crate::anthropic_batch::AnthropicBatchService<'_>,
        crate::anthropic_batch::AnthropicBatchError,
    > {
        crate::anthropic_batch::AnthropicBatchService::new(
            self.runtime.http.as_ref(),
            credential,
            scope,
        )
    }
    /// Kimi Batch at an explicitly bound account and region.
    pub fn kimi_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::kimi_batch::KimiBatchScope,
    ) -> Result<crate::kimi_batch::KimiBatchService<'_>, crate::kimi_batch::KimiBatchError> {
        crate::kimi_batch::KimiBatchService::new(self.runtime.http.as_ref(), credential, scope)
    }
    /// OpenRouter inline Batch on an explicitly bound account and route.
    pub fn openrouter_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::openrouter_batch::OpenRouterBatchScope,
    ) -> Result<
        crate::openrouter_batch::OpenRouterBatchService<'_>,
        crate::openrouter_batch::OpenRouterBatchError,
    > {
        crate::openrouter_batch::OpenRouterBatchService::new(
            self.runtime.http.as_ref(),
            credential,
            scope,
        )
    }
    /// OpenRouter native text reranking with an explicit account scope.
    pub fn openrouter_rerank(
        &self,
        scope: crate::openrouter_rerank::OpenRouterRerankScope,
    ) -> Result<
        crate::openrouter_rerank::OpenRouterRerankService<'_>,
        crate::openrouter_rerank::OpenRouterRerankError,
    > {
        crate::openrouter_rerank::OpenRouterRerankService::new(self.runtime.http.as_ref(), scope)
    }
    /// xAI Collections with separate per-operation API and management keys.
    pub fn xai_collections(
        &self,
        config: crate::xai_collections::XaiCollectionsConfig,
    ) -> Result<
        crate::xai_collections::XaiCollectionsClient<'_>,
        crate::xai_collections::XaiCollectionsError,
    > {
        crate::xai_collections::XaiCollectionsClient::new(self.runtime.http.as_ref(), config)
    }
    /// xAI speech transcription and synthesis over an explicit account route.
    pub fn xai_audio(
        &self,
        config: crate::xai_audio::XaiAudioConfig,
    ) -> Result<crate::xai_audio::XaiAudioService<'_>, crate::xai_audio::XaiAudioError> {
        crate::xai_audio::XaiAudioService::new(self.runtime.http.as_ref(), config)
    }
    /// xAI's native create-and-add Batch lifecycle on an explicit API route.
    pub fn xai_batch(
        &self,
        credential: crate::protocol::Secret<String>,
        scope: crate::xai_batch::XaiBatchScope,
    ) -> Result<crate::xai_batch::XaiBatchService<'_>, crate::xai_batch::XaiBatchError> {
        crate::xai_batch::XaiBatchService::new(self.runtime.http.as_ref(), credential, scope)
    }

    /// The independent image generation service.
    pub fn images(&self) -> crate::images::ImageService<'_> {
        crate::images::ImageService::new(ClientSource::Snapshot(self))
    }
    /// Query one account with its configured execution budgets.
    pub async fn account_usage(
        &self,
        profile_name: &str,
        query: &account::AccountQuery,
    ) -> Result<account::AccountSnapshot, account::AccountUsageError> {
        self.runtime
            .accounts
            .query(
                &self.state.account_registry,
                &self.state.config.profiles,
                profile_name,
                query,
            )
            .await
    }
    /// Independently schedule accounts and return results in profile order.
    pub async fn accounts_usage(
        &self,
        queries: &BTreeMap<String, account::AccountQuery>,
    ) -> Vec<(
        String,
        Result<account::AccountSnapshot, account::AccountUsageError>,
    )> {
        self.runtime
            .accounts
            .query_all(
                &self.state.account_registry,
                &self.state.config.profiles,
                queries,
            )
            .await
    }

    /// Every model a picker may offer in the selected region. A hidden connection is skipped: those
    /// exist only to be failed over onto, so offering them would let a user
    /// pick the spare key directly.
    ///
    /// `id` is the display model, which is the ref a session stores and
    /// `resolve` accepts back.
    pub fn models(&self) -> Vec<ModelListing> {
        self.models_matching(|_| true)
    }

    fn models_matching(
        &self,
        include: impl Fn(&crate::protocol::ModelProfile) -> bool,
    ) -> Vec<ModelListing> {
        self.state
            .config
            .profiles
            .iter()
            .filter(|p| p.supports_region(self.runtime.region) && !p.connection.hidden)
            .flat_map(|p| {
                let include = &include;
                p.models
                    .iter()
                    .filter(move |m| !m.hidden && self.tracks(p, m) && include(m))
                    .map(move |m| ModelListing {
                        info: {
                            let mut info = m.info.clone();
                            info.features = info.features.on_connection(&p.info.features);
                            info.pricing = m.pricing.clone();
                            info
                        },
                        regions: p.regions.clone(),
                        id: m.display_model.clone(),
                        profile_name: p.profile_name.clone(),
                        request_model: m.request_model.clone(),
                        billing_mode: m.billing_mode_on(&p.pricing),
                        pricing: m.pricing.clone(),
                        // The catalog's display name, not its description: the
                        // latter is a paragraph of vendor prose and was never a
                        // name. It keeps its own field below.
                        display_name: m.display_model.clone(),
                        description: m.description.clone(),
                        provider_id: p.provider_id.clone(),
                        context_window: m.metadata.context_window_tokens,
                        max_output_tokens: m.metadata.max_output_tokens,
                        capability_support: m.capability_support.unwrap_or_default(),
                    })
            })
            .collect()
    }

    /// Every provider in the selected region, including those with no credential and the
    /// spare connections a picker should not offer.
    ///
    /// Within the region, visibility is unfiltered: an app needs
    /// the unconfigured ones to offer "set this up", and it needs to see the
    /// spares to explain a group. `hidden` says which is which.
    pub fn providers(&self) -> Vec<ProviderListing> {
        self.state
            .config
            .profiles
            .iter()
            .filter(|p| p.supports_region(self.runtime.region))
            .map(|p| ProviderListing {
                regions: p.regions.clone(),
                provider_id: p.provider_id.clone(),
                profile_name: p.profile_name.clone(),
                group: p.group().to_owned(),
                info: p.info.clone(),
                protocol: p.protocol,
                auth: p.auth,
                credential_env: match &p.credential {
                    CredentialConfig::Env { var } => Some(var.clone()),
                    _ => None,
                },
                billing_mode: p.pricing.billing_mode,
                model_count: p.models.iter().filter(|m| self.tracks(p, m)).count(),
                hidden: p.connection.hidden,
            })
            .collect()
    }

    /// Selected usage region. Configuration management retains every region.
    #[must_use]
    pub fn region(&self) -> Region {
        self.runtime.region
    }

    /// All configured profiles, including those unavailable in the selected region.
    pub fn profiles(&self) -> &[ProviderProfile] {
        &self.state.config.profiles
    }

    fn tracks(&self, profile: &ProviderProfile, model: &crate::protocol::ModelProfile) -> bool {
        self.state.config.tracks(profile, model)
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.runtime.codecs.keys().copied().collect()
    }

    /// How to ask this connection what it serves, when it publishes that at
    /// all and this crate can read the shape it publishes it in.
    ///
    /// `None` is an ordinary answer, not a failure: a connection that declares
    /// no directory, and one whose directory speaks a shape with no reader,
    /// both keep working from the shipped catalog.
    #[must_use]
    pub fn directory_for(&self, profile: &ProviderProfile) -> Option<Arc<dyn ModelDirectory>> {
        if !profile.chat_enabled {
            return None;
        }
        let shape = profile.model_list.shape(profile.protocol)?;
        self.runtime.directories.get(&shape).cloned()
    }

    pub fn directory_shapes(&self) -> Vec<ProtocolFamily> {
        self.runtime.directories.keys().copied().collect()
    }

    /// One provider from this fixed configuration view.
    pub fn provider(&self, name: &str) -> Option<&ProviderProfile> {
        self.state.config.profile(name)
    }

    pub(super) fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.provider(name)
    }

    /// Monotonically increasing within this client's publication history.
    pub fn revision(&self) -> u64 {
        self.state.revision
    }

    /// Price usage against the connection that actually served a response.
    pub fn estimate_actual_cost(
        &self,
        route: &ResolvedRoute,
        response: &crate::protocol::ChatResponse,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        let name =
            response
                .executed_profile
                .as_deref()
                .ok_or_else(|| LlmError::CostUnavailable {
                    message: "response has no executed profile".into(),
                })?;
        self.estimate_cost_for_profile(
            route,
            name,
            &response.usage,
            &response.inference,
            submission,
        )
    }

    /// Price a stream including its actual service tier and executed connection.
    pub fn estimate_stream_cost(
        &self,
        route: &ResolvedRoute,
        stream: &ModelStream,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.estimate_cost_for_profile(
            route,
            stream.executed_profile(),
            &stream.usage_report(),
            &stream.inference_report(),
            submission,
        )
    }

    /// Price complete usage with explicit execution and inference metadata.
    pub fn estimate_cost_for_profile(
        &self,
        route: &ResolvedRoute,
        name: &str,
        report: &crate::protocol::UsageReport,
        inference: &crate::protocol::InferenceReport,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        use crate::protocol::{PricingContext, ServiceTier};
        let profile = self
            .profile(name)
            .ok_or_else(|| LlmError::CostUnavailable {
                message: format!("executed profile {name:?} is unavailable"),
            })?;
        // A connection or model default is not a server confirmation. It does
        // rule out assuming standard rates when the provider reports no tier.
        let default_fast = profile
            .models
            .iter()
            .filter(|m| m.request_model == route.request_model)
            .any(|m| {
                m.info
                    .features
                    .on_connection(&profile.info.features)
                    .default_service_tier
                    == Some(ServiceTier::Fast)
            });
        let tier = match inference.service_tier {
            Some(tier) => tier,
            None if inference.requested_service_tier == Some(ServiceTier::Fast)
                || inference.requested_raw_service_tier.is_some()
                || inference
                    .raw_service_tier
                    .as_deref()
                    .is_some_and(|s| s != "standard" && s != "default")
                || inference.raw_speed.is_some()
                || default_fast =>
            {
                return Err(LlmError::CostUnavailable {
                    message: "the actual service tier is unknown".into(),
                });
            }
            None => ServiceTier::Standard,
        };
        let context = PricingContext {
            service_tier: Some(tier),
            submission,
            unix_seconds: inference.executed_at,
            ..Default::default()
        };
        let usage = report.complete().ok_or_else(|| LlmError::CostUnavailable {
            message: "actual cost requires a complete, valid usage report".into(),
        })?;
        if name != route.profile_name
            && !route
                .connection_chain
                .iter()
                .any(|hop| hop.profile_name == name)
        {
            return Err(LlmError::CostUnavailable {
                message: format!("profile {name:?} is not on this route"),
            });
        }
        if inference.executed_at.is_none()
            && (profile.pricing.peak.is_some()
                || profile
                    .models
                    .iter()
                    .filter(|m| m.request_model == route.request_model)
                    .filter_map(|m| m.pricing.as_ref())
                    .any(|prices| price_query::requires_execution_time(prices, tier, submission)))
        {
            return Err(LlmError::CostUnavailable {
                message: "execution time is required for time-dependent prices".into(),
            });
        }
        if name == route.profile_name {
            return self.estimate_cost(route, usage, &context);
        }
        let mut candidates = profile
            .models
            .iter()
            .filter(|m| m.request_model == route.request_model);
        let model = match (candidates.next(), candidates.next()) {
            (Some(model), None) => model,
            (None, _) => {
                return Err(LlmError::CostUnavailable {
                    message: format!(
                        "executed profile {name:?} does not serve {:?}",
                        route.request_model
                    ),
                });
            }
            (Some(_), Some(_)) => {
                return Err(LlmError::CostUnavailable {
                    message: format!(
                        "executed profile {name:?} ambiguously serves {:?}",
                        route.request_model
                    ),
                });
            }
        };
        let mut actual_route = route.clone();
        actual_route.profile_name = name.to_owned();
        actual_route.provider_id = profile.provider_id.clone();
        actual_route.display_model = model.display_model.clone();
        actual_route.pricing_model.pricing_provider_id = profile.provider_id.clone();
        actual_route.pricing_model.billing_model = model.billing_model.clone();
        actual_route.pricing_model.display_model = model.display_model.clone();
        self.estimate_cost(&actual_route, usage, &context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::LlmError;
    use crate::protocol::{PricingContext, Usage};
    use crate::transport::{HttpRequest, HttpResponse, StreamResponse, Transport};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MemoryAttachments {
        bytes: Bytes,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AttachmentResolver for MemoryAttachments {
        async fn resolve(&self, _attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.bytes.clone())
        }
    }

    struct MissingAttachments;

    #[async_trait]
    impl AttachmentResolver for MissingAttachments {
        async fn resolve(&self, attachment: &AttachmentRef) -> Result<Bytes, LlmError> {
            Err(LlmError::ModelUnavailable {
                message: format!(
                    "application attachment {:?} revision {:?} is unavailable",
                    attachment.attachment_id, attachment.revision
                ),
            })
        }
    }

    fn request_with_attachment_blocks() -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m1",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "attachment", "attachment": {
                        "attachment_id": "att-1", "revision": "rev-1", "filename": "image.png",
                        "media_type": "image/png", "size_bytes": 3
                    }}},
                    {"type": "document", "source": {"type": "attachment", "attachment": {
                        "attachment_id": "att-1", "revision": "rev-1", "filename": "image.png",
                        "media_type": "image/png", "size_bytes": 3
                    }}}
                ]
            }]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn attachment_refs_are_resolved_once_and_only_in_a_request_copy() {
        let resolver = Arc::new(MemoryAttachments {
            bytes: Bytes::from_static(b"png"),
            calls: AtomicUsize::new(0),
        });
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(resolver.clone());
        let client = builder
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        let original = request_with_attachment_blocks();

        let resolved = client
            .runtime
            .attachments
            .resolve_attachments(&original)
            .await
            .unwrap();

        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
        assert_eq!(resolved.attachments.len(), 2);
        assert_eq!(original, request_with_attachment_blocks());
        assert!(std::ptr::eq(resolved.request, &original));
        assert_eq!(resolved.attachments[0].bytes, Bytes::from_static(b"png"));
        assert_eq!(resolved.attachments[1].bytes, Bytes::from_static(b"png"));
    }

    #[tokio::test]
    async fn attachment_refs_without_a_resolver_fail_clearly() {
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[])
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        assert!(matches!(
            client.runtime.attachments.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::UnsupportedCapability { message })
                if message.contains("no AttachmentResolver")
        ));
    }

    #[tokio::test]
    async fn resolver_byte_count_must_match_attachment_metadata() {
        let resolver = Arc::new(MemoryAttachments {
            bytes: Bytes::from_static(b"different"),
            calls: AtomicUsize::new(0),
        });
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(resolver);
        let client = builder
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        assert!(matches!(
            client.runtime.attachments.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::InvalidRequest { message }) if message.contains("expected 3")
        ));
    }

    #[tokio::test]
    async fn a_missing_remote_attachment_revision_returns_the_resolver_error() {
        let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[]);
        builder.with_attachment_resolver(Arc::new(MissingAttachments));
        let client = builder
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();

        assert!(matches!(
            client.runtime.attachments.resolve_attachments(&request_with_attachment_blocks()).await,
            Err(LlmError::ModelUnavailable { message })
                if message.contains("attachment \"att-1\" revision \"rev-1\" is unavailable")
        ));
    }

    struct NoHttp;
    #[async_trait]
    impl Transport for NoHttp {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.response(request).await.map(Into::into)
        }
    }
    impl NoHttp {
        async fn response(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
    }
    fn profile(name: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(serde_json::json!({
            "provider_id": "acme", "profile_name": name, "base_url": "https://x.test",
            "protocol": protocol, "auth": "none",
            "models": [{"display_model": "m1", "request_model": "m1", "billing_model": "m1"}]
        }))
        .unwrap()
    }

    #[test]
    fn handles_share_resources_and_release_them_without_an_ownership_cycle() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<LlmClient>();
        send_sync::<ClientSnapshot>();
        send_sync::<ClientConfigManager>();

        let (client, manager) =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("p1", "open_ai_chat")])
                .with_region(crate::protocol::Region::International)
                .build_managed()
                .unwrap();
        let cloned = client.clone();
        let snapshot = client.snapshot();
        let resources = Arc::downgrade(&client.runtime);
        let publication = Arc::downgrade(&client.published);
        assert!(Arc::ptr_eq(&client.runtime, &cloned.runtime));
        assert!(Arc::ptr_eq(&client.runtime, &snapshot.runtime));
        assert!(Arc::ptr_eq(&snapshot.state, &cloned.snapshot().state));
        drop(manager);
        drop(client);
        assert!(cloned.resolve("m1").is_ok());
        drop(cloned);
        assert!(publication.upgrade().is_none());
        assert!(snapshot.resolve("m1").is_ok());
        drop(snapshot);
        assert!(resources.upgrade().is_none());
    }

    #[tokio::test]
    async fn replaced_views_live_until_the_last_snapshot_is_released() {
        let (client, manager) =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("p1", "open_ai_chat")])
                .with_region(crate::protocol::Region::International)
                .build_managed()
                .unwrap();
        let snapshot = client.snapshot();
        let previous = Arc::downgrade(&snapshot.state);
        let path = std::env::temp_dir().join(format!(
            "llm-snapshot-lifetime-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        manager.set_config_dir(&path).await.unwrap();
        assert!(client.snapshot().revision() > snapshot.revision());
        assert!(previous.upgrade().is_some());
        assert_eq!(snapshot.provider("p1").unwrap().profile_name, "p1");
        drop(snapshot);
        assert!(previous.upgrade().is_none());
        std::fs::remove_dir_all(path).unwrap();
    }

    /// Gate 33's error path. Every family in the closed set now ships a codec,
    /// so this is only reachable by removing one or by adding a tenth family —
    /// which is exactly what it is here to catch. `tests/families.rs` asserts
    /// the covering invariant that keeps it unreachable in practice.
    #[test]
    fn build_names_the_profile_and_the_family_when_a_codec_is_missing() {
        let mut b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "gemini_generate_content")],
        );
        b.codecs.remove(&ProtocolFamily::GeminiGenerateContent);
        assert_eq!(
            b.with_region(crate::protocol::Region::International)
                .build()
                .err()
                .expect("a profile with no codec cannot build"),
            BuildError::MissingCodec {
                profile_name: "p1".into(),
                family: ProtocolFamily::GeminiGenerateContent
            }
        );
    }

    #[test]
    fn a_family_this_crate_speaks_needs_no_capability_entry() {
        let b =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("p1", "open_ai_chat")]);
        assert!(
            b.with_region(crate::protocol::Region::International)
                .build()
                .is_ok(),
            "an OpenAI-compatible provider is a settings entry and nothing else (gate 30)"
        );
    }

    #[test]
    fn build_rejects_duplicate_profile_names() {
        let b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "open_ai_chat"), profile("p1", "open_ai_chat")],
        );
        assert_eq!(
            b.with_region(crate::protocol::Region::International)
                .build()
                .err()
                .unwrap(),
            BuildError::DuplicateProfile {
                profile_name: "p1".into()
            }
        );
    }

    #[test]
    fn empty_builder_builds_and_lists_nothing() {
        let c = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[])
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        assert!(c.models().is_empty());
    }

    #[test]
    fn build_rejects_empty_and_malformed_peak_windows() {
        use crate::protocol::PeakSchedule;

        for windows in [vec![], vec!["25:00-26:00".to_owned()]] {
            let mut profile = profile("p1", "open_ai_chat");
            profile.pricing.peak = Some(PeakSchedule {
                utc_windows: windows,
                weekdays_only: false,
                off_peak_multiplier: 0.5,
            });
            assert!(matches!(
                LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile])
                    .with_region(crate::protocol::Region::International)
                    .build(),
                Err(BuildError::InvalidPeakSchedule { .. })
            ));
        }
    }

    #[test]
    fn clock_override_controls_time_based_pricing() {
        use crate::protocol::{PeakSchedule, TokenPricing};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        struct FixedClock(u64);
        impl Clock for FixedClock {
            fn now(&self) -> SystemTime {
                UNIX_EPOCH + Duration::from_secs(self.0)
            }
        }

        let mut p = profile("p1", "open_ai_chat");
        p.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            ..Default::default()
        });
        p.pricing.peak = Some(PeakSchedule {
            weekdays_only: false,
            utc_windows: vec!["09:00-17:00".into()],
            off_peak_multiplier: 0.5,
        });
        for (hour, expected) in [(12, 2.0), (20, 1.0)] {
            let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p.clone()]);
            builder.with_clock(Arc::new(FixedClock(hour * 3600)));
            let client = builder
                .with_region(crate::protocol::Region::International)
                .build()
                .unwrap();
            let route = client.resolve("m1").unwrap();
            let cost = client
                .estimate_cost(
                    &route,
                    &Usage {
                        input_tokens: 1_000_000,
                        ..Default::default()
                    },
                    &PricingContext::default(),
                )
                .unwrap();
            assert_eq!(cost.total_cost, expected);
        }
    }

    #[test]
    fn duplicate_display_models_use_the_resolved_model_for_pricing() {
        use crate::protocol::{ModelProfile, TokenPricing};

        let mut p = profile("p1", "open_ai_chat");
        p.models = vec![
            serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-a",
                "billing_model": "bill-a",
                "aliases": ["first"]
            }))
            .unwrap(),
            serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-b",
                "billing_model": "bill-b",
                "aliases": ["second"]
            }))
            .unwrap(),
        ];
        p.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            ..Default::default()
        });
        p.models[1].pricing = Some(TokenPricing {
            input_per_million: Some(7.0),
            ..Default::default()
        });

        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p])
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();
        assert_eq!(route.display_model, "shared");
        assert_eq!(route.request_model, "wire-b");
        assert_eq!(route.pricing_model.billing_model, "bill-b");
        let estimate = client
            .estimate_cost(
                &route,
                &Usage {
                    input_tokens: 1_000_000,
                    ..Usage::default()
                },
                &PricingContext::default(),
            )
            .unwrap();
        assert_eq!(estimate.input_cost, 7.0);
    }

    #[test]
    fn identical_resolved_model_identities_do_not_guess_between_prices() {
        use crate::protocol::{ModelProfile, TokenPricing};

        let model = |alias: &str, rate: f64| {
            let mut model = serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": "shared",
                "request_model": "wire-shared",
                "billing_model": "bill-shared",
                "aliases": [alias]
            }))
            .unwrap();
            model.pricing = Some(TokenPricing {
                input_per_million: Some(rate),
                ..Default::default()
            });
            model
        };
        let mut p = profile("p1", "open_ai_chat");
        p.models = vec![model("first", 2.0), model("second", 7.0)];
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p])
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();

        assert!(matches!(
            client.estimate_cost(
                &route,
                &Usage {
                    input_tokens: 1_000_000,
                    ..Usage::default()
                },
                &PricingContext::default(),
            ),
            Err(LlmError::CostUnavailable { .. })
        ));
    }

    #[test]
    fn actual_cost_estimation_uses_the_resolved_identity_or_rejects_ambiguous_failover() {
        use crate::protocol::{ModelProfile, TokenPricing};

        let model = |display: &str, billing: &str, alias: &str| {
            let mut model = serde_json::from_value::<ModelProfile>(serde_json::json!({
                "display_model": display,
                "request_model": "wire-shared",
                "billing_model": billing,
                "aliases": [alias]
            }))
            .unwrap();
            model.pricing = Some(TokenPricing {
                input_per_million: Some(if billing == "bill-second" { 7.0 } else { 2.0 }),
                ..Default::default()
            });
            model
        };
        let mut primary = profile("p1", "open_ai_chat");
        primary.models = vec![
            model("first-label", "bill-first", "first"),
            model("second-label", "bill-second", "second"),
        ];
        let mut sibling = profile("p2", "open_ai_chat");
        sibling.models = vec![
            model("other-first", "other-bill-first", "other-first"),
            model("other-second", "other-bill-second", "other-second"),
        ];
        let client = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[primary, sibling])
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        let route = client.resolve("second").unwrap();
        let usage = Usage {
            input_tokens: 1_000_000,
            ..Usage::default()
        };

        let cost = client
            .estimate_cost_for_profile(
                &route,
                "p1",
                &crate::protocol::UsageReport::measured(
                    usage,
                    crate::protocol::UsageState::Complete,
                ),
                &crate::protocol::InferenceReport::default(),
                Submission::Interactive,
            )
            .unwrap();
        assert_eq!(cost.input_cost, 7.0);

        let mut ambiguous_route = route;
        ambiguous_route.connection_chain.push(route::ConnectionHop {
            profile_name: "p2".into(),
            request_model: "wire-shared".into(),
        });
        assert!(matches!(
            client.estimate_cost_for_profile(
                &ambiguous_route,
                "p2",
                &crate::protocol::UsageReport::measured(usage, crate::protocol::UsageState::Complete),
                &crate::protocol::InferenceReport::default(),
                Submission::Interactive
            ),
            Err(LlmError::CostUnavailable { message }) if message.contains("ambiguously")
        ));
    }
}
