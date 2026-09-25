//! Single-dispatch execution for applications which own retries and accounting.
use super::executor::{PreparedAttempt, RequestExecutor};
use super::{LlmClient, ModelStream, RequestOptions};
use crate::codecs::RequestMode;
use crate::protocol::{
    CompletionRequest, CompletionResponse, InferenceReport, LlmError, ModelProfile,
    ProviderProfile, UsageReport,
};
use crate::transport::{HttpExecutor, HttpRequest, HttpResponse, StreamResponse, Transport};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

/// An authenticated request pinned to one model row and one connection.
/// Preparing may upload attachments, but never sends a generation request.
/// This object deliberately cannot be cloned: dispatch consumes it.
pub struct PreparedCall {
    pub(super) prepared: PreparedAttempt,
    http: Arc<dyn Transport>,
    clock: Arc<dyn crate::transport::Clock>,
    deadline: Option<Instant>,
    profile: ProviderProfile,
    model: ModelProfile,
    mode: RequestMode,
}

impl LlmClient {
    /// Count the exact prepared prompt using the provider's counting endpoint.
    /// None means unsupported; network, authentication and invalid responses are
    /// errors, never replaced with an approximation or a generation request.
    pub async fn count_tokens_exact_in(
        &self,
        profile: &str,
        request: &CompletionRequest,
        options: &RequestOptions,
    ) -> Result<Option<u64>, LlmError> {
        let resolved = self
            .snapshot
            .resolve_request(&request.model, Some(profile))?;
        if resolved.connections[0].profile.protocol
            != crate::protocol::ProtocolFamily::AnthropicMessages
        {
            return Ok(None);
        }
        let collected = self
            .prepare_on(profile, request, options, RequestMode::CountTokens)
            .await?
            .dispatch_once()
            .await?
            .collect()
            .await?;
        let result = if !(200..300).contains(&collected.response.status) {
            Err(collected
                .decode()
                .err()
                .unwrap_or_else(|| LlmError::ProviderInternal {
                    message: "token count failed".into(),
                }))
        } else {
            serde_json::from_slice::<serde_json::Value>(&collected.response.body)
                .ok()
                .and_then(|v| v.get("input_tokens").and_then(serde_json::Value::as_u64))
                .map(Some)
                .ok_or_else(|| LlmError::ProviderInternal {
                    message: "token count response has no numeric input_tokens".into(),
                })
        };
        collected.finish().await;
        result
    }

    /// Prepare one exact connection. Group names are not accepted here: resolve
    /// a group first and pass its selected profile name. Failover is the caller's
    /// decision and each further generation requires a new prepared call.
    pub async fn prepare_on(
        &self,
        profile: &str,
        request: &CompletionRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<PreparedCall, LlmError> {
        let resolved = self
            .snapshot
            .resolve_request(&request.model, Some(profile))?;
        let selected = resolved
            .connections
            .first()
            .ok_or_else(|| LlmError::ModelUnavailable {
                message: "no connection selected".into(),
            })?;
        if selected.profile.profile_name != profile {
            return Err(LlmError::InvalidRequest {
                message: "prepare_on requires an exact connection name, not a group".into(),
            });
        }
        let options = super::executor::execution_options(request, options, mode);
        let started = Instant::now();
        let executor = RequestExecutor::new(self);
        let request = executor
            .resolve_before_deadline(request, &options, started)
            .await?;
        let prepared = executor
            .prepare_before_deadline(&resolved.route, selected, &request, &options, started, mode)
            .await?;
        Ok(PreparedCall {
            prepared,
            http: self.http.clone(),
            clock: self.clock.clone(),
            deadline: options
                .total_timeout
                .and_then(|timeout| started.checked_add(timeout)),
            profile: selected.profile.clone(),
            model: selected.model.clone(),
            mode,
        })
    }
}

impl PreparedCall {
    /// Establish the connection during preparation, before host admission and
    /// the synchronous dispatch marker. No generation is sent by this method.
    pub async fn connect_websocket(
        &self,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        if self.profile.protocol != crate::protocol::ProtocolFamily::OpenAiResponses
            || !self.profile.supports_websockets
        {
            return Err(LlmError::UnsupportedCapability {
                message: "the selected connection does not support Responses WebSocket".into(),
            });
        }
        if self.profile.supports_websocket_compression {
            return Err(LlmError::UnsupportedCapability {
                message: "WebSocket compression is not supported".into(),
            });
        }
        let mut handshake = self.prepared.http.clone();
        handshake.method = "GET".into();
        handshake.body = Default::default();
        if let Some(milliseconds) = self.profile.websocket_connect_timeout_ms {
            handshake.timeout = Some(std::time::Duration::from_millis(milliseconds));
        }
        crate::runtime::Deadline::at(self.deadline)
            .cap(handshake.timeout)
            .run(self.http.connect_websocket(handshake))
            .await?
    }

    /// Send one request on an already-open connection. Connection failures are
    /// returned to the host; this never reconnects or falls back to HTTP.
    pub async fn dispatch_websocket_once(
        mut self,
        connection: &mut dyn crate::transport::WebSocketConnection,
    ) -> Result<ReceivedCall, LlmError> {
        use futures::StreamExt;
        if self.mode != RequestMode::Stream
            || self.profile.protocol != crate::protocol::ProtocolFamily::OpenAiResponses
        {
            return Err(LlmError::UnsupportedCapability {
                message: "WebSocket dispatch requires a streaming Responses request".into(),
            });
        }
        let body = crate::websocket::request_payload(&self.prepared.http.body)?;
        self.prepared.inference.executed_at = self
            .clock
            .now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|t| t.as_secs());
        let mut response = crate::runtime::Deadline::at(self.deadline)
            .cap(self.prepared.http.timeout)
            .run(connection.send(body))
            .await??;
        response.body = response
            .body
            .map(|frame| {
                frame.map(|bytes| {
                    let mut sse = Vec::with_capacity(bytes.len() + 8);
                    sse.extend_from_slice(b"data: ");
                    sse.extend_from_slice(&bytes);
                    sse.extend_from_slice(b"\n\n");
                    sse.into()
                })
            })
            .boxed();
        Ok(ReceivedCall {
            call: self,
            response,
        })
    }
    pub(super) fn new(
        prepared: PreparedAttempt,
        http: Arc<dyn Transport>,
        clock: Arc<dyn crate::transport::Clock>,
        deadline: Option<Instant>,
        profile: ProviderProfile,
        model: ModelProfile,
        mode: RequestMode,
    ) -> Self {
        Self {
            prepared,
            http,
            clock,
            deadline,
            profile,
            model,
            mode,
        }
    }

    /// Capture prices before dispatch for cancellation-safe host accounting.
    pub fn pricing_snapshot(&self) -> super::FrozenPricing {
        super::FrozenPricing {
            profile: self.profile.clone(),
            model: self.model.clone(),
        }
    }
    /// Final authenticated wire request. Its Debug implementation redacts secrets.
    pub fn request(&self) -> &HttpRequest {
        &self.prepared.http
    }
    /// Frozen connection and pricing configuration, unaffected by later edits.
    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }
    /// The exact selected row, including its billing mode and prices.
    pub fn model(&self) -> &ModelProfile {
        &self.model
    }

    /// Send at most one generation request. No authentication, upload, retry or
    /// connection fallback occurs between entry and invoking the transport.
    pub async fn dispatch_once(mut self) -> Result<ReceivedCall, LlmError> {
        self.prepared.inference.executed_at = self
            .clock
            .now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|t| t.as_secs());
        let response = HttpExecutor::new(self.http.as_ref())
            .with_deadline(crate::runtime::Deadline::at(self.deadline))
            .send(self.prepared.http.clone())
            .await?;
        Ok(ReceivedCall {
            call: self,
            response,
        })
    }
}

/// Status and headers from one physical request, including unsuccessful status.
/// The host must observe usage before interpreting content or deciding to retry.
pub struct ReceivedCall {
    call: PreparedCall,
    response: StreamResponse,
}
impl ReceivedCall {
    pub fn status(&self) -> u16 {
        self.response.status
    }
    pub fn headers(&self) -> &[(String, String)] {
        &self.response.headers
    }

    /// Collect an ordinary response without decoding its semantic content.
    /// Resource cleanup is retained until after the observation is delivered.
    pub async fn collect(self) -> Result<CollectedResponse, LlmError> {
        let response = if (200..300).contains(&self.response.status) {
            HttpExecutor::collect_response(self.response, None).await?
        } else {
            HttpResponse {
                status: self.response.status,
                headers: self.response.headers,
                body: crate::transport::collect_error_body(self.response.body).await,
            }
        };
        let usage = self
            .call
            .prepared
            .codec
            .response_usage(&response, &self.call.prepared.context);
        let mut inference = self
            .call
            .prepared
            .codec
            .response_inference(&response, &self.call.prepared.context);
        inference.executed_at = self.call.prepared.inference.executed_at;
        inference.requested_effort = self.call.prepared.inference.requested_effort;
        inference.requested_service_tier = self.call.prepared.inference.requested_service_tier;
        inference.requested_raw_service_tier = self
            .call
            .prepared
            .inference
            .requested_raw_service_tier
            .clone();
        Ok(CollectedResponse {
            call: self.call,
            response,
            usage,
            inference,
        })
    }

    /// Enter streaming only after checking the response status. Error responses
    /// remain collectable, preserving any usage carried by their JSON envelope.
    pub fn into_stream(self) -> Result<ModelStream, Box<Self>> {
        if self.call.mode != RequestMode::Stream || !(200..300).contains(&self.response.status) {
            return Err(Box::new(self));
        }
        let pricing = self.call.pricing_snapshot();
        let mut stream = ModelStream::new(
            self.response,
            self.call
                .prepared
                .codec
                .stream_decoder(&self.call.prepared.context),
            self.call.profile.profile_name,
            self.call.prepared.cleanup,
            self.call.deadline,
            self.call.prepared.inference,
        );
        stream.pricing = Some(pricing);
        Ok(stream)
    }
}

/// Raw response plus accounting facts independent of tool/content validation.
pub struct CollectedResponse {
    call: PreparedCall,
    response: HttpResponse,
    usage: UsageReport,
    inference: InferenceReport,
}
impl CollectedResponse {
    pub fn pricing_snapshot(&self) -> super::FrozenPricing {
        self.call.pricing_snapshot()
    }
    pub fn response(&self) -> &HttpResponse {
        &self.response
    }
    pub fn usage_report(&self) -> &UsageReport {
        &self.usage
    }
    pub fn inference_report(&self) -> &InferenceReport {
        &self.inference
    }
    pub fn profile(&self) -> &ProviderProfile {
        &self.call.profile
    }
    pub fn model(&self) -> &ModelProfile {
        &self.call.model
    }
    /// Decode only after the caller has retained usage_report(). This is
    /// synchronous; malformed tool JSON cannot erase the retained observation.
    pub fn decode(&self) -> Result<CompletionResponse, LlmError> {
        let mut response = self
            .call
            .prepared
            .codec
            .decode_response(&self.response, &self.call.prepared.context)?;
        response.executed_profile = Some(self.call.profile.profile_name.clone());
        response.inference.executed_at = self.call.prepared.inference.executed_at;
        response.inference.requested_effort = self.call.prepared.inference.requested_effort;
        response.inference.requested_service_tier =
            self.call.prepared.inference.requested_service_tier;
        response.inference.requested_raw_service_tier = self
            .call
            .prepared
            .inference
            .requested_raw_service_tier
            .clone();
        Ok(response)
    }
    /// Invalidate referenced files after a provider reports that they expired.
    /// This does not resend a generation request; the caller must prepare again.
    pub async fn invalidate_missing_files(&self, client: &LlmClient) -> bool {
        let missing =
            super::executor::missing_provider_file_uses(&self.response, &self.call.prepared.files);
        if missing.is_empty() {
            return false;
        }
        client
            .attachments
            .invalidate_provider_file_cache(&missing)
            .await;
        true
    }
    /// Complete provider file cleanup after recording usage. Dropping this
    /// object retains the existing automatic cleanup lease's cancellation path.
    pub async fn finish(self) {
        if let Some(cleanup) = self.call.prepared.cleanup {
            cleanup.finish(self.call.deadline).await;
        }
    }
}
