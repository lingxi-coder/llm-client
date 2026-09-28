//! Typed client entry points for explicitly scoped realtime connections.
use crate::protocol::Secret;
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::{
    RealtimeConnectRequest, RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport,
};
use std::sync::Arc;

impl super::XaiClient {
    /// Open one Voice connection, retaining its explicit account scope for resumption.
    pub async fn connect_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        request: RealtimeConnectRequest,
        config: super::realtime::XaiRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(super::realtime::XaiRealtimeSession, RealtimeDriver), RealtimeError> {
        let _pinned =
            pin_realtime_scope(&self.binding, self.profile_name(), &config.credential_scope)?;
        super::realtime::XaiRealtimeSession::connect(transport, request, config, limits).await
    }

    /// Open one streaming transcription connection with a per-call key.
    pub async fn connect_stt(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        account_scope: &str,
        credential: Secret<String>,
        config: super::stt::XaiSttConfig,
        limits: RealtimeLimits,
    ) -> Result<(super::stt::XaiSttSession, super::stt::XaiSttDriver), super::stt::XaiSttError>
    {
        let _pinned = pin_realtime_scope(&self.binding, self.profile_name(), account_scope)?;
        super::stt::XaiSttSession::connect(transport, credential, config, limits).await
    }

    /// Open one streaming speech connection with a per-call key.
    pub async fn connect_streaming_tts(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        account_scope: &str,
        credential: Secret<String>,
        config: super::streaming_tts::XaiStreamingTtsConfig,
        limits: RealtimeLimits,
    ) -> Result<
        (
            super::streaming_tts::XaiStreamingTtsSession,
            super::streaming_tts::XaiStreamingTtsDriver,
        ),
        super::streaming_tts::XaiStreamingTtsError,
    > {
        let _pinned = pin_realtime_scope(&self.binding, self.profile_name(), account_scope)?;
        super::streaming_tts::XaiStreamingTtsSession::connect(transport, credential, config, limits)
            .await
    }
}
