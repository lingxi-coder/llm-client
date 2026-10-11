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

impl super::OpenAiClient {
    /// Connect the exact profile's standard Realtime API and seed host history.
    /// The SDK constructs the endpoint and authentication frame; credentials
    /// remain request scoped. History never starts an unsolicited response.
    #[allow(clippy::too_many_arguments)]
    pub async fn connect_agent_realtime(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        account_scope: &str,
        credential: Secret<String>,
        model: &str,
        mut config: super::realtime::OpenAiRealtimeConfig,
        history: Vec<crate::realtime::RealtimeHistoryItem>,
        limits: RealtimeLimits,
    ) -> Result<(crate::realtime::RealtimeSession, RealtimeDriver), RealtimeError> {
        let pinned = pin_realtime_scope(&self.binding, self.profile_name(), account_scope)?;
        let snapshot = pinned.pin().map_err(|error| RealtimeError::InvalidConfig {
            message: error.to_string(),
        })?;
        let profile =
            snapshot
                .profile(self.profile_name())
                .ok_or_else(|| RealtimeError::InvalidConfig {
                    message: "realtime profile disappeared".into(),
                })?;
        let request = crate::realtime::authenticated_request(
            &profile.base_url,
            "realtime",
            model,
            "Authorization",
            credential,
            limits,
        )?;
        config.manual_turns = true;
        if config.input_transcription_model.is_none() {
            return Err(RealtimeError::InvalidConfig {
                message: "Agent realtime requires input transcription".into(),
            });
        }
        let codec = Arc::new(super::realtime::OpenAiRealtimeCodec::new(config));
        crate::realtime::preflight_input(
            &crate::realtime::RealtimeInput::ImportHistory {
                items: history.clone(),
            },
            limits.max_frame_bytes,
        )?;
        let history_frames = if history.is_empty() {
            Vec::new()
        } else {
            crate::realtime::RealtimeCodec::encode(
                codec.as_ref(),
                &crate::realtime::RealtimeInput::ImportHistory { items: history },
            )?
        };
        let history_bytes = history_frames
            .iter()
            .fold(0usize, |total, frame| total.saturating_add(frame.len()));
        if history_bytes > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: history_bytes,
                max: limits.max_frame_bytes,
            });
        }
        crate::realtime::RealtimeSession::connect(
            transport.as_ref(),
            request,
            Arc::new(crate::realtime::SeededCodec {
                inner: codec,
                history: history_frames,
            }),
            limits,
        )
        .await
    }
}
