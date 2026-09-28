//! Typed client entry points for explicitly scoped realtime connections.
use crate::protocol::Secret;
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::{RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport};
use std::sync::Arc;

impl super::QwenClient {
    /// Open one session with a fresh configuration snapshot and a per-call credential.
    pub async fn connect_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        route: super::realtime::QwenRealtimeRoute,
        scope: super::realtime::QwenRealtimeScope,
        credential: Secret<String>,
        config: super::realtime::QwenRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(super::realtime::QwenRealtimeSession, RealtimeDriver), RealtimeError> {
        let _pinned =
            pin_realtime_scope(&self.binding, scope.profile_name(), scope.account_scope())?;
        super::realtime::QwenRealtimeSession::connect(
            transport, route, scope, credential, config, limits,
        )
        .await
    }
}

impl super::QwenClient {
    /// Open one session with a fresh configuration snapshot and a per-call credential.
    pub async fn connect_live_translate(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        route: super::live_translate::QwenLiveTranslateRoute,
        scope: super::live_translate::QwenLiveTranslateScope,
        credential: Secret<String>,
        config: super::live_translate::QwenLiveTranslateConfig,
        limits: RealtimeLimits,
    ) -> Result<
        (
            super::live_translate::QwenLiveTranslateSession,
            RealtimeDriver,
        ),
        RealtimeError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, scope.profile_name(), scope.account_scope())?;
        super::live_translate::QwenLiveTranslateSession::connect(
            transport, route, scope, credential, config, limits,
        )
        .await
    }
}

impl super::QwenClient {
    /// Connect using the configured resource scope and request-scoped credentials.
    pub async fn connect_asr_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::asr_realtime::QwenAsrRealtimeConfig,
        options: &crate::RequestOptions,
        request: &super::asr_realtime::QwenAsrRealtimeRequest,
        limits: super::asr_realtime::QwenAsrRealtimeLimits,
    ) -> Result<
        super::asr_realtime::QwenAsrRealtimeSession,
        super::asr_realtime::QwenAsrRealtimeError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::asr_realtime::QwenAsrRealtimeService::new(transport, config)?
            .connect(options, request, limits)
            .await
    }
}
impl super::QwenClient {
    /// Connect using the configured resource scope and request-scoped credentials.
    pub async fn connect_tts_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        config: super::tts_realtime::QwenTtsRealtimeConfig,
        options: &crate::RequestOptions,
        request: &super::tts_realtime::QwenTtsRealtimeRequest,
        limits: super::tts_realtime::QwenTtsRealtimeLimits,
    ) -> Result<
        super::tts_realtime::QwenTtsRealtimeSession,
        super::tts_realtime::QwenTtsRealtimeError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, &config.profile_name, &config.account_scope)?;
        super::tts_realtime::QwenTtsRealtimeService::new(transport, config)?
            .connect(options, request, limits)
            .await
    }
}
