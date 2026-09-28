//! Typed client entry points for explicitly scoped realtime connections.
use crate::protocol::Secret;
use crate::providers::binding::pin_realtime_scope;
use crate::realtime::{RealtimeDriver, RealtimeError, RealtimeLimits, RealtimeTransport};
use std::sync::Arc;

impl super::ZhipuClient {
    /// Open one session with a fresh configuration snapshot and a per-call credential.
    pub async fn connect_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        route: super::realtime::GlmRealtimeRoute,
        scope: super::realtime::GlmRealtimeScope,
        credential: Secret<String>,
        config: super::realtime::GlmRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(super::realtime::GlmRealtimeSession, RealtimeDriver), RealtimeError> {
        let _pinned =
            pin_realtime_scope(&self.binding, scope.profile_name(), scope.account_scope())?;
        super::realtime::GlmRealtimeSession::connect(
            transport, route, scope, credential, config, limits,
        )
        .await
    }
}
