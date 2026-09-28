//! Typed client entry points for explicitly scoped realtime connections.
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::{
    RealtimeConnectRequest, RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport,
};
use std::sync::Arc;

impl super::GoogleClient {
    /// Open a scoped Live session using explicitly supplied endpoint and headers.
    pub async fn connect_live(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        request: RealtimeConnectRequest,
        config: super::live::GeminiLiveConfig,
        limits: RealtimeLimits,
    ) -> Result<(super::live::GeminiLiveSession, RealtimeDriver), RealtimeError> {
        let _pinned =
            pin_realtime_scope(&self.binding, self.profile_name(), &config.credential_scope)?;
        super::live::GeminiLiveSession::connect(transport, request, config, limits).await
    }
}
