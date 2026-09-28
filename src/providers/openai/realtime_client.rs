//! Typed client entry points for explicitly scoped realtime connections.
use crate::protocol::Secret;
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::{
    RealtimeConnectRequest, RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport,
};
use std::sync::Arc;

impl super::OpenAiClient {
    /// Open the explicitly selected Realtime endpoint with request-scoped headers.
    pub async fn connect_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        request: RealtimeConnectRequest,
        account_scope: &str,
        config: super::realtime::OpenAiRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(crate::realtime::RealtimeSession, RealtimeDriver), RealtimeError> {
        let _pinned = pin_realtime_scope(&self.binding, self.profile_name(), account_scope)?;
        crate::realtime::RealtimeSession::connect(
            transport.as_ref(),
            request,
            Arc::new(super::realtime::OpenAiRealtimeCodec::new(config)),
            limits,
        )
        .await
    }

    /// Open one GPT-Live connection on an explicit account and route.
    #[allow(clippy::too_many_arguments)]
    pub async fn connect_live(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        route: super::live::OpenAiLiveRoute,
        scope: super::live::OpenAiLiveScope,
        credential: Secret<String>,
        safety_identifier: Option<String>,
        config: super::live::OpenAiLiveConfig,
        limits: RealtimeLimits,
    ) -> Result<
        (
            super::live::OpenAiLiveSession,
            super::live::OpenAiLiveDriver,
        ),
        RealtimeError,
    > {
        let _pinned =
            pin_realtime_scope(&self.binding, scope.profile_name(), scope.account_scope())?;
        super::live::OpenAiLiveSession::connect(
            transport,
            route,
            scope,
            credential,
            safety_identifier,
            config,
            limits,
        )
        .await
    }
}
