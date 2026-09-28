//! Typed client entry points for explicitly scoped realtime connections.
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::RealtimeTransport;
use std::sync::Arc;

impl super::MiniMaxClient {
    /// Connect using the explicit regional resource configuration and per-call credentials.
    pub async fn connect_streaming_tts(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::streaming_tts::MiniMaxStreamingTtsConfig,
        credential: &crate::protocol::Secret<String>,
        request: &super::streaming_tts::MiniMaxStreamingTtsRequest,
        limits: super::streaming_tts::MiniMaxStreamingTtsLimits,
    ) -> Result<
        super::streaming_tts::MiniMaxStreamingTtsSession,
        super::streaming_tts::MiniMaxStreamingTtsError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::streaming_tts::MiniMaxStreamingTtsService::new(transport, config)?
            .connect(credential, request, limits)
            .await
    }

    /// Connect with a provider voice reference whose account and region are verified.
    pub async fn connect_streaming_tts_with_voice(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::streaming_tts::MiniMaxStreamingTtsConfig,
        credential: &crate::protocol::Secret<String>,
        request: &super::streaming_tts::MiniMaxStreamingTtsRequest,
        voice: &super::voices::MiniMaxVoiceRef,
        limits: super::streaming_tts::MiniMaxStreamingTtsLimits,
    ) -> Result<
        super::streaming_tts::MiniMaxStreamingTtsSession,
        super::streaming_tts::MiniMaxStreamingTtsError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::streaming_tts::MiniMaxStreamingTtsService::new(transport, config)?
            .connect_with_voice(credential, request, voice, limits)
            .await
    }
}
impl super::MiniMaxClient {
    /// Connect using the explicit regional resource configuration and per-call credentials.
    pub async fn connect_bidi_tts(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::bidi_tts::MiniMaxBidiTtsConfig,
        credential: &super::bidi_tts::MiniMaxBidiTtsCredentials,
        request: &super::bidi_tts::MiniMaxBidiTtsRequest,
        limits: super::bidi_tts::MiniMaxBidiTtsLimits,
    ) -> Result<super::bidi_tts::MiniMaxBidiTtsSession, super::bidi_tts::MiniMaxBidiTtsError> {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::bidi_tts::MiniMaxBidiTtsService::new(transport, config)?
            .connect(credential, request, limits)
            .await
    }

    /// Connect with a provider voice reference whose account and region are verified.
    pub async fn connect_bidi_tts_with_voice(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::bidi_tts::MiniMaxBidiTtsConfig,
        credential: &super::bidi_tts::MiniMaxBidiTtsCredentials,
        request: &super::bidi_tts::MiniMaxBidiTtsRequest,
        voice: &super::voices::MiniMaxVoiceRef,
        limits: super::bidi_tts::MiniMaxBidiTtsLimits,
    ) -> Result<super::bidi_tts::MiniMaxBidiTtsSession, super::bidi_tts::MiniMaxBidiTtsError> {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::bidi_tts::MiniMaxBidiTtsService::new(transport, config)?
            .connect_with_voice(credential, request, voice, limits)
            .await
    }
}
