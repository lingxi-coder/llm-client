//! Reusable Responses transport state; retry policy remains in the caller.
use super::{PreparedCall, ReceivedCall, RequestDraft};
use crate::{
    protocol::LlmError,
    transport::{StreamResponse, WebSocketConnection},
    websocket::*,
};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

pub struct ResponsesSession {
    connection: Option<Box<dyn WebSocketConnection>>,
    // A digest avoids retaining authentication headers or signed URLs.
    identity: Option<[u8; 32]>,
    prepared_binding: Option<Arc<()>>,
    state: Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical_body: serde_json::Value,
    prewarm: bool,
}
impl Default for ResponsesSession {
    fn default() -> Self {
        Self::new()
    }
}
impl ResponsesSession {
    pub fn new() -> Self {
        Self {
            connection: None,
            identity: None,
            prepared_binding: None,
            state: Arc::new(Mutex::new(ResponsesWebSocketSessionState {
                connection_healthy: true,
                ..Default::default()
            })),
            logical_body: serde_json::Value::Null,
            prewarm: false,
        }
    }
    pub fn fallback_to_http(&self) -> bool {
        self.state.lock().expect("session").fallback_to_http
    }
    pub fn last_response_id(&self) -> Option<String> {
        self.state.lock().expect("session").last_response_id.clone()
    }
    pub fn last_added_response_items(&self) -> Vec<serde_json::Value> {
        self.state
            .lock()
            .expect("session")
            .last_added_response_items
            .clone()
    }
    pub fn last_response_from_prewarm(&self) -> bool {
        self.state
            .lock()
            .expect("session")
            .last_response_from_prewarm
    }
    pub fn last_request_snapshot(&self) -> ResponsesWebSocketRequestSnapshot {
        let s = self.state.lock().expect("session");
        ResponsesWebSocketRequestSnapshot {
            logical_request_body: s.last_logical_request_body.clone(),
            wire_request_body: s.last_wire_request_body.clone(),
            wire_used_previous_response_id: s.last_wire_used_previous_response_id,
            wire_used_prewarm_response_id: s.last_wire_used_prewarm_response_id,
        }
    }
    pub async fn close(&mut self) -> Result<(), LlmError> {
        let connection = self.connection.take();
        let fallback = self.fallback_to_http();
        self.reset_state(fallback);
        if let Some(mut connection) = connection {
            connection.close().await?;
        }
        Ok(())
    }
    fn reset_state(&mut self, fallback_to_http: bool) {
        // Old response streams retain their own observer state and cannot
        // repopulate a new account's continuation after the session is reset.
        self.state = Arc::new(Mutex::new(ResponsesWebSocketSessionState {
            fallback_to_http,
            connection_healthy: true,
            ..Default::default()
        }));
        self.logical_body = serde_json::Value::Null;
        self.prewarm = false;
        self.prepared_binding = None;
    }
    async fn reset_identity(&mut self) -> Result<(), LlmError> {
        let connection = self.connection.take();
        self.identity = None;
        self.reset_state(false);
        if let Some(mut connection) = connection {
            connection.close().await?;
        }
        Ok(())
    }
    fn fail_to_http(&mut self) {
        self.connection = None;
        clear_continuation(&self.state, true);
        self.state.lock().expect("session").fallback_to_http = true;
    }
    /// Connect and compute the incremental wire request before signing and
    /// host admission. A handshake 426 may select HTTP without any generation.
    /// Reuse is bound to the account, provider connection and authenticated
    /// handshake. Changing that identity closes the old connection and clears
    /// its continuation. Only the latest successfully prepared draft may be
    /// sealed and dispatched; mutating it again requires another preparation.
    pub async fn prepare(
        &mut self,
        draft: &mut RequestDraft,
        prewarm: bool,
        allow_http: bool,
    ) -> Result<(), LlmError> {
        self.prepare_using(draft, prewarm, allow_http, None).await
    }
    pub async fn prepare_using(
        &mut self,
        draft: &mut RequestDraft,
        prewarm: bool,
        allow_http: bool,
        transport: Option<&dyn crate::Transport>,
    ) -> Result<(), LlmError> {
        self.prepared_binding = None;
        // Authenticate the handshake once per preparation so custom
        // authenticators and rotated credentials participate in the binding.
        // If a connection is needed, these exact bytes open it without another
        // authentication pass.
        let handshake = match draft.websocket_handshake().await {
            Ok(handshake) => handshake,
            Err(error) => {
                let _ = self.reset_identity().await;
                return Err(error);
            }
        };
        let identity = session_identity(draft, &handshake);
        if self.identity.as_ref() != Some(&identity) {
            self.reset_identity().await?;
            self.identity = Some(identity);
        }
        if self.fallback_to_http() {
            if !allow_http {
                return Err(LlmError::InvalidRequest {
                    message: "Responses WebSocket is unavailable; HTTP fallback is disabled".into(),
                });
            }
            self.bind_draft(draft);
            return Ok(());
        }
        if !self.state.lock().expect("session").connection_healthy {
            self.connection = None;
        }
        if self.connection.is_none() {
            let opened = draft
                .connect_websocket_handshake_using(handshake, transport)
                .await;
            match opened {
                Ok(connection) => {
                    self.connection = Some(connection);
                    self.state.lock().expect("session").connection_healthy = true;
                }
                Err(error) if upgrade_required(&error) => {
                    self.fail_to_http();
                    return if allow_http {
                        self.bind_draft(draft);
                        Ok(())
                    } else {
                        Err(error)
                    };
                }
                Err(error) => return Err(error),
            }
        }
        self.logical_body = serde_json::from_slice(&draft.request().body).map_err(|e| {
            LlmError::InvalidRequest {
                message: e.to_string(),
            }
        })?;
        self.prewarm = prewarm;
        let mut body = if prewarm {
            self.logical_body.clone()
        } else {
            incremental_responses_body(&self.state, &self.logical_body)
                .unwrap_or_else(|| self.logical_body.clone())
        };
        if prewarm {
            set_responses_generate(&mut body, false)?;
        }
        record_responses_wire_request(&self.state, &self.logical_body, &body);
        draft.request_mut().body = serde_json::to_vec(&body)
            .map_err(|e| LlmError::InvalidRequest {
                message: e.to_string(),
            })?
            .into();
        self.bind_draft(draft);
        Ok(())
    }
    fn bind_draft(&mut self, draft: &mut RequestDraft) {
        let binding = Arc::new(());
        draft.bind_session(binding.clone());
        self.prepared_binding = Some(binding);
    }
    pub async fn dispatch(
        &mut self,
        call: PreparedCall,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
    ) -> Result<ReceivedCall, LlmError> {
        self.dispatch_using(call, on_dispatch, None).await
    }
    pub async fn dispatch_using(
        &mut self,
        call: PreparedCall,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
        transport: Option<&dyn crate::Transport>,
    ) -> Result<ReceivedCall, LlmError> {
        let binding = self
            .prepared_binding
            .as_ref()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "session must be prepared before dispatch".into(),
            })?;
        if !call.is_bound_to_session(binding) {
            return Err(LlmError::InvalidRequest {
                message: "prepared call does not match the session's current preparation".into(),
            });
        }
        self.prepared_binding = None;
        if self.fallback_to_http() {
            return match transport {
                Some(transport) => call.dispatch_once_using(transport, on_dispatch).await,
                None => call.dispatch_once_with(on_dispatch).await,
            };
        }
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "session must be prepared before dispatch".into(),
            })?;
        let mut tracking = TrackingConnection {
            inner: connection.as_mut(),
            state: self.state.clone(),
            logical: self.logical_body.clone(),
            prewarm: self.prewarm,
        };
        let result = call
            .dispatch_websocket_once_with(&mut tracking, on_dispatch)
            .await;
        if let Err(error) = &result {
            if upgrade_required(error) {
                self.fail_to_http();
            } else {
                self.connection = None;
                clear_continuation(&self.state, true);
            }
        }
        result
    }
}
fn session_identity(draft: &RequestDraft, handshake: &crate::HttpRequest) -> [u8; 32] {
    let profile = draft.profile();
    let mut headers: Vec<_> = handshake
        .headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect();
    headers.sort_unstable();
    let identity = serde_json::json!([
        draft.account_scope(),
        profile.provider_id,
        profile.profile_name,
        profile.protocol,
        profile.auth,
        profile.connection.group,
        profile.connection.connection_id,
        profile.supports_websockets,
        profile.supports_websocket_compression,
        handshake.url,
        headers,
    ]);
    Sha256::digest(serde_json::to_vec(&identity).expect("session identity JSON")).into()
}
fn upgrade_required(error: &LlmError) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("426") || message.contains("upgrade required")
}
struct TrackingConnection<'a> {
    inner: &'a mut dyn WebSocketConnection,
    state: Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical: serde_json::Value,
    prewarm: bool,
}
#[async_trait::async_trait]
impl WebSocketConnection for TrackingConnection<'_> {
    async fn send(&mut self, payload: bytes::Bytes) -> Result<StreamResponse, LlmError> {
        let mut response = self.inner.send(payload).await?;
        let state = self.state.clone();
        let logical = self.logical.clone();
        let prewarm = self.prewarm;
        let observation = SessionObservation {
            state,
            logical,
            prewarm,
            items: Vec::new(),
            terminal: false,
        };
        response.body = futures::stream::unfold(
            (response.body, observation),
            move |(mut body, mut observation)| async move {
                match body.next().await {
                    Some(frame) => {
                        match &frame {
                            Ok(bytes) => observe_response_frame(
                                &observation.state,
                                &observation.logical,
                                observation.prewarm,
                                &mut observation.items,
                                &mut observation.terminal,
                                bytes,
                            ),
                            Err(_) => clear_continuation(&observation.state, true),
                        }
                        Some((frame, (body, observation)))
                    }
                    None => None,
                }
            },
        )
        .boxed();
        Ok(response)
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        self.inner.close().await
    }
}

struct SessionObservation {
    state: Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical: serde_json::Value,
    prewarm: bool,
    items: Vec<serde_json::Value>,
    terminal: bool,
}
impl Drop for SessionObservation {
    fn drop(&mut self) {
        if !self.terminal {
            clear_continuation(&self.state, true);
        }
    }
}
