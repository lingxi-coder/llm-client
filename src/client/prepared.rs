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
        let result = collected.decode_token_count().map(Some);
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
        let options = super::executor::execution_options(request, options, mode);
        self.prepare_draft_on(profile, request, &options, mode)
            .await?
            .seal()
            .await
    }

    /// Resolve, encode and prepare attachments without signing the generation
    /// request. The host can finish request policy before sealing the draft.
    pub async fn prepare_draft_on(
        &self,
        profile: &str,
        request: &CompletionRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<RequestDraft, LlmError> {
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
        let options = options.clone();
        let started = Instant::now();
        let executor = RequestExecutor::new(self);
        let request = executor
            .resolve_before_deadline(request, &options, started)
            .await?;
        let prepared = executor
            .prepare_before_deadline(
                &resolved.route,
                selected,
                &request,
                &options,
                started,
                (mode, false),
            )
            .await?;
        Ok(RequestDraft {
            semantic_body: None,
            authenticator: self.authenticators.get(&selected.profile.auth).cloned(),
            credential: options.credential.clone(),
            call: PreparedCall {
                prepared,
                http: self.http.clone(),
                clock: self.clock.clone(),
                deadline: options
                    .total_timeout
                    .and_then(|timeout| started.checked_add(timeout)),
                profile: selected.profile.clone(),
                model: selected.model.clone(),
                mode,
            },
        })
    }
}

/// Mutable generation request before its final authentication step. It cannot
/// dispatch. Sealing consumes the draft and produces an immutable single-use call.
pub struct RequestDraft {
    semantic_body: Option<(bytes::Bytes, serde_json::Value)>,
    call: PreparedCall,
    authenticator: Option<Arc<dyn crate::Authenticator>>,
    credential: Option<crate::protocol::Secret<String>>,
}
impl RequestDraft {
    pub async fn connect_websocket(
        &self,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        self.connect_websocket_using(self.call.http.as_ref()).await
    }
    pub async fn connect_websocket_using(
        &self,
        transport: &dyn crate::Transport,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        let mut handshake = self.call.prepared.http.clone();
        handshake.method = "GET".into();
        handshake.body = Default::default();
        if self.call.profile.auth != crate::protocol::AuthStrategy::None {
            if let Some(auth) = &self.authenticator {
                crate::runtime::Deadline::at(self.call.deadline)
                    .run(auth.apply(&mut handshake, &self.call.profile, self.credential.as_ref()))
                    .await??;
            }
        }
        self.call
            .connect_websocket_request(handshake, transport)
            .await
    }
    pub fn request(&self) -> &HttpRequest {
        &self.call.prepared.http
    }
    pub fn request_mut(&mut self) -> &mut HttpRequest {
        &mut self.call.prepared.http
    }
    /// Set exact JSON bytes while retaining a safe semantic view for final
    /// inference facts, including when strings contain lone UTF-16 surrogates.
    pub fn set_json_body(
        &mut self,
        value: serde_json::Value,
        overrides: &std::collections::BTreeMap<String, Vec<u16>>,
    ) -> Result<(), LlmError> {
        let bytes: bytes::Bytes = crate::exact_json::serialize(&value, overrides)?.into();
        self.call.prepared.http.body = bytes.clone();
        self.semantic_body = Some((bytes, value));
        Ok(())
    }
    pub fn profile(&self) -> &ProviderProfile {
        &self.call.profile
    }
    pub fn model(&self) -> &ModelProfile {
        &self.call.model
    }
    pub async fn seal(mut self) -> Result<PreparedCall, LlmError> {
        if self.call.profile.protocol == crate::protocol::ProtocolFamily::AnthropicMessages
            && self.call.prepared.http.body.len() > 32_000_000
        {
            return Err(LlmError::RequestTooLarge {
                message: "final Anthropic body exceeds 32000000 bytes".into(),
            });
        }
        if self.call.profile.auth != crate::protocol::AuthStrategy::None {
            if let Some(auth) = &self.authenticator {
                crate::runtime::Deadline::at(self.call.deadline)
                    .run(auth.apply(
                        &mut self.call.prepared.http,
                        &self.call.profile,
                        self.credential.as_ref(),
                    ))
                    .await??;
            }
        }
        let body = serde_json::from_slice::<serde_json::Value>(&self.call.prepared.http.body)
            .ok()
            .or_else(|| {
                self.semantic_body
                    .as_ref()
                    .filter(|(bytes, _)| bytes == &self.call.prepared.http.body)
                    .map(|(_, value)| value.clone())
            });
        if let Some(body) = body {
            use crate::protocol::ServiceTier;
            let raw = body
                .get("service_tier")
                .or_else(|| body.get("speed"))
                .and_then(serde_json::Value::as_str);
            self.call.prepared.inference.requested_service_tier = match raw {
                Some("fast" | "priority") => Some(ServiceTier::Fast),
                Some("standard" | "default") => Some(ServiceTier::Standard),
                _ => None,
            };
            self.call.prepared.inference.requested_raw_service_tier = raw
                .filter(|v| !matches!(*v, "fast" | "priority" | "standard" | "default"))
                .map(str::to_owned);
        } else {
            // Unknown final wire controls cannot silently use Standard prices.
            self.call.prepared.inference.requested_raw_service_tier =
                Some("unavailable-final-request-controls".into());
        }
        crate::runtime::Deadline::at(self.call.deadline).remaining()?;
        Ok(self.call)
    }
}

impl PreparedCall {
    /// Establish the connection during preparation, before host admission and
    /// the synchronous dispatch marker. No generation is sent by this method.
    pub async fn connect_websocket(
        &self,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        self.connect_websocket_request(self.prepared.http.clone(), self.http.as_ref())
            .await
    }

    async fn connect_websocket_request(
        &self,
        mut handshake: HttpRequest,
        transport: &dyn crate::Transport,
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
        handshake.method = "GET".into();
        handshake.body = Default::default();
        if let Some(milliseconds) = self.profile.websocket_connect_timeout_ms {
            handshake.timeout = Some(std::time::Duration::from_millis(milliseconds));
        }
        crate::runtime::Deadline::at(self.deadline)
            .cap(handshake.timeout)
            .run(transport.connect_websocket(handshake))
            .await?
    }

    /// Send one request on an already-open connection. Connection failures are
    /// returned to the host; this never reconnects or falls back to HTTP.
    pub async fn dispatch_websocket_once(
        self,
        connection: &mut dyn crate::transport::WebSocketConnection,
    ) -> Result<ReceivedCall, LlmError> {
        self.dispatch_websocket_once_with(connection, || Ok(()))
            .await
    }

    /// WebSocket counterpart of `dispatch_once_with`; no reconnect or fallback.
    pub async fn dispatch_websocket_once_with(
        mut self,
        connection: &mut dyn crate::transport::WebSocketConnection,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
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
        let deadline = crate::runtime::Deadline::at(self.deadline).cap(self.prepared.http.timeout);
        deadline.remaining()?;
        on_dispatch()?;
        let response = deadline.run(connection.send(body)).await??;
        let mut response = HttpExecutor::bound_response(response, deadline);
        // Some native stacks retain the successful upgrade status on frames.
        if response.status == 101 {
            response.status = 200;
        }
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
    pub async fn dispatch_once(self) -> Result<ReceivedCall, LlmError> {
        self.dispatch_once_with(|| Ok(())).await
    }

    /// Run a synchronous host admission marker immediately before invoking
    /// the transport. A rejected marker causes no send; this never retries.
    pub async fn dispatch_once_with(
        self,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
    ) -> Result<ReceivedCall, LlmError> {
        let http = self.http.clone();
        self.dispatch_once_using(http.as_ref(), on_dispatch).await
    }

    /// Execute through an explicitly supplied host transport. The request is
    /// still consumed once; response streams own their resources independently.
    pub async fn dispatch_once_using(
        mut self,
        transport: &dyn crate::Transport,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
    ) -> Result<ReceivedCall, LlmError> {
        self.prepared.inference.executed_at = self
            .clock
            .now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|t| t.as_secs());
        // Validate the deadline before marking a physical attempt dispatched.
        crate::runtime::Deadline::at(self.deadline).remaining()?;
        on_dispatch()?;
        let response = HttpExecutor::new(transport)
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
    /// Decode an exact count response without interpreting it as a completion.
    pub fn decode_token_count(&self) -> Result<u64, LlmError> {
        if self.call.mode != RequestMode::CountTokens {
            return Err(LlmError::InvalidRequest {
                message: "response is not a token-count request".into(),
            });
        }
        if !(200..300).contains(&self.response.status) {
            return Err(self
                .decode()
                .err()
                .unwrap_or_else(|| LlmError::ProviderInternal {
                    message: "token count failed".into(),
                }));
        }
        serde_json::from_slice::<serde_json::Value>(&self.response.body)
            .ok()
            .and_then(|v| v.get("input_tokens").and_then(serde_json::Value::as_u64))
            .ok_or_else(|| LlmError::ProviderInternal {
                message: "token count response has no numeric input_tokens".into(),
            })
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
