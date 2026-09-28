//! Reusable Responses transport state; retry policy remains in the caller.
use super::{PreparedCall, ReceivedCall, RequestDraft};
use crate::{
    protocol::LlmError,
    transport::{StreamResponse, WebSocketConnection},
    websocket::*,
};
use futures::StreamExt;
use std::sync::{Arc, Mutex};

pub struct ResponsesSession {
    connection: Option<Box<dyn WebSocketConnection>>,
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
        if let Some(mut connection) = self.connection.take() {
            connection.close().await?;
        }
        let fallback = self.fallback_to_http();
        *self.state.lock().expect("session") = ResponsesWebSocketSessionState {
            fallback_to_http: fallback,
            connection_healthy: true,
            ..Default::default()
        };
        Ok(())
    }
    fn fail_to_http(&mut self) {
        self.connection = None;
        clear_continuation(&self.state, true);
        self.state.lock().expect("session").fallback_to_http = true;
    }
    /// Connect and compute the incremental wire request before signing and
    /// host admission. A handshake 426 may select HTTP without any generation.
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
        if self.fallback_to_http() {
            return Ok(());
        }
        if !self.state.lock().expect("session").connection_healthy {
            self.connection = None;
        }
        if self.connection.is_none() {
            let opened = match transport {
                Some(transport) => draft.connect_websocket_using(transport).await,
                None => draft.connect_websocket().await,
            };
            match opened {
                Ok(connection) => {
                    self.connection = Some(connection);
                    self.state.lock().expect("session").connection_healthy = true;
                }
                Err(error) if upgrade_required(&error) => {
                    self.fail_to_http();
                    return if allow_http { Ok(()) } else { Err(error) };
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
        Ok(())
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
