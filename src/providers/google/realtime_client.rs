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

impl super::GoogleClient {
    /// Open the Gemini Developer Live API with native host history and tools.
    /// Vertex AI has a distinct route/authentication contract and is rejected.
    pub async fn connect_agent_live(
        &self,
        transport: Arc<dyn RealtimeTransport>,
        credential: crate::protocol::Secret<String>,
        config: super::live::GeminiLiveConfig,
        history: Vec<crate::realtime::RealtimeHistoryItem>,
        limits: RealtimeLimits,
    ) -> Result<
        (
            crate::realtime::RealtimeControl,
            crate::realtime::RealtimeEvents,
            RealtimeDriver,
        ),
        RealtimeError,
    > {
        let pinned =
            pin_realtime_scope(&self.binding, self.profile_name(), &config.credential_scope)?;
        let snapshot = pinned.pin().map_err(|error| RealtimeError::InvalidConfig {
            message: error.to_string(),
        })?;
        let profile =
            snapshot
                .profile(self.profile_name())
                .ok_or_else(|| RealtimeError::InvalidConfig {
                    message: "Live profile disappeared".into(),
                })?;
        let parsed =
            url::Url::parse(&profile.base_url).map_err(|_| RealtimeError::InvalidConfig {
                message: "invalid Gemini Developer base URL".into(),
            })?;
        if parsed.host_str() != Some("generativelanguage.googleapis.com") {
            return Err(RealtimeError::InvalidConfig { message: "Agent Live connector currently supports Gemini Developer API; Vertex and custom gateways require their own verified adapter".into() });
        }
        let request = crate::realtime::authenticated_request(
            "https://generativelanguage.googleapis.com",
            "ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
            "",
            "x-goog-api-key",
            credential,
            limits,
        )?;
        // Validate all history and its actual wire size before remote dispatch.
        let turns = super::live::encode_gemini_history(&history)?;
        crate::realtime::preflight_input(
            &crate::realtime::RealtimeInput::ImportHistory {
                items: history.clone(),
            },
            limits.max_frame_bytes,
        )?;
        let wire = serde_json::to_vec(
            &serde_json::json!({"clientContent":{"turns":turns,"turnComplete":true}}),
        )
        .map_err(|error| RealtimeError::Codec {
            message: error.to_string(),
        })?;
        if wire.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: wire.len(),
                max: limits.max_frame_bytes,
            });
        }
        let (session, driver) = self
            .connect_live(transport, request, config.with_agent_history(), limits)
            .await?;
        let (control, events) = session.into_realtime_parts();
        // Seed while the driver is still unstarted, so no live input precedes it.
        control.import_history(history)?;
        Ok((control, events, driver))
    }
}
