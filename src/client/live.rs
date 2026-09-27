//! Live entry points capture exactly one published view per operation.
use super::*;
use crate::protocol::*;

impl LlmClient {
    /// Capture one immutable configuration and account-binding revision.
    /// The publication lock is released before returning this view.
    pub fn snapshot(&self) -> ClientSnapshot {
        let state = self
            .published
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        ClientSnapshot {
            runtime: self.runtime.clone(),
            state,
        }
    }
    pub fn chat(&self) -> ChatService<'_> {
        ChatService::new(ClientSource::Live(self))
    }
    pub fn images(&self) -> crate::images::ImageService<'_> {
        crate::images::ImageService::new(ClientSource::Live(self))
    }
    pub fn embeddings(&self) -> crate::embeddings::EmbeddingService<'_> {
        crate::embeddings::EmbeddingService::new(ClientSource::Live(self))
    }
    pub fn retrieval(&self) -> crate::retrieval::RetrievalService<'_> {
        crate::retrieval::RetrievalService::new(ClientSource::Live(self))
    }
    pub fn batches(&self) -> crate::batches::BatchService<'_> {
        crate::batches::BatchService::new(ClientSource::Live(self))
    }

    pub fn deferred(&self) -> crate::deferred::DeferredService<'_> {
        crate::deferred::DeferredService::new(ClientSource::Live(self))
    }
    pub fn background(&self) -> crate::background::BackgroundService<'_> {
        crate::background::BackgroundService::new(ClientSource::Live(self))
    }
    pub fn audio(&self) -> crate::audio::AudioService<'_> {
        crate::audio::AudioService::new(ClientSource::Live(self))
    }
    pub fn interactions(&self) -> crate::interactions::InteractionService<'_> {
        crate::interactions::InteractionService::new(ClientSource::Live(self))
    }
    /// Gemini File Search store and document lifecycle.
    pub fn gemini_file_search(&self) -> crate::gemini_file_search::GeminiFileSearchService<'_> {
        crate::gemini_file_search::GeminiFileSearchService::new(ClientSource::Live(self))
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
        crate::glm_knowledge::GlmKnowledgeService::new(ClientSource::Live(self))
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

    pub async fn account_usage(
        &self,
        profile_name: &str,
        query: &account::AccountQuery,
    ) -> Result<account::AccountSnapshot, account::AccountUsageError> {
        self.snapshot().account_usage(profile_name, query).await
    }

    pub async fn accounts_usage(
        &self,
        queries: &BTreeMap<String, account::AccountQuery>,
    ) -> Vec<(
        String,
        Result<account::AccountSnapshot, account::AccountUsageError>,
    )> {
        self.snapshot().accounts_usage(queries).await
    }

    pub fn models(&self) -> Vec<ModelListing> {
        self.snapshot().models()
    }

    pub fn providers(&self) -> Vec<ProviderListing> {
        self.snapshot().providers()
    }

    pub fn region(&self) -> Region {
        self.snapshot().region()
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.snapshot().codec_families()
    }

    pub fn directory_for(&self, profile: &ProviderProfile) -> Option<Arc<dyn ModelDirectory>> {
        self.snapshot().directory_for(profile)
    }

    pub fn directory_shapes(&self) -> Vec<ProtocolFamily> {
        self.snapshot().directory_shapes()
    }

    pub fn estimate_actual_cost(
        &self,
        route: &ResolvedRoute,
        response: &crate::protocol::ChatResponse,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_actual_cost(route, response, submission)
    }

    pub fn estimate_stream_cost(
        &self,
        route: &ResolvedRoute,
        stream: &ModelStream,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_stream_cost(route, stream, submission)
    }

    pub fn estimate_cost_for_profile(
        &self,
        route: &ResolvedRoute,
        name: &str,
        report: &crate::protocol::UsageReport,
        inference: &crate::protocol::InferenceReport,
        submission: Submission,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot()
            .estimate_cost_for_profile(route, name, report, inference, submission)
    }

    pub fn resolve(&self, model: &str) -> Result<ResolvedRoute, ResolveError> {
        self.snapshot().resolve(model)
    }

    pub fn resolve_in(
        &self,
        model: &str,
        profile: Option<&str>,
    ) -> Result<ResolvedRoute, ResolveError> {
        self.snapshot().resolve_in(model, profile)
    }

    pub fn price_quote(
        &self,
        model: &str,
        profile: Option<&str>,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        self.snapshot().price_quote(model, profile, context)
    }

    pub fn price_quote_for_route(
        &self,
        route: &ResolvedRoute,
        context: &PricingContext,
    ) -> Result<PriceQuote, LlmError> {
        self.snapshot().price_quote_for_route(route, context)
    }

    pub fn estimate_cost(
        &self,
        route: &ResolvedRoute,
        usage: &Usage,
        context: &PricingContext,
    ) -> Result<pricing::CostEstimate, LlmError> {
        self.snapshot().estimate_cost(route, usage, context)
    }

    pub fn estimate_local_tokens(
        &self,
        request: &ChatRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        self.snapshot().estimate_local_tokens(request)
    }

    pub fn estimate_local_tokens_in(
        &self,
        profile_or_group: &str,
        request: &ChatRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        self.snapshot()
            .estimate_local_tokens_in(profile_or_group, request)
    }
}
