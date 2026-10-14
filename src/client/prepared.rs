//! Single-dispatch execution for applications which own retries and accounting.
use super::executor::{PreparedAttempt, RequestExecutor};
use super::{ClientSnapshot, LlmClient, ModelStream, RequestOptions};
use crate::codecs::RequestMode;
use crate::protocol::{
    ChatRequest, ChatResponse, InferenceReport, LlmError, ModelProfile, ProtocolFamily,
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
    model_source: Option<crate::protocol::JsonPayload>,
    mode: RequestMode,
    continuation: Option<crate::protocol::ContinuationRef>,
    file_expirations: Vec<String>,
    session_binding: Option<Arc<()>>,
    native_anthropic_request_header_facts:
        Option<crate::providers::response_headers::NativeAnthropicRequestHeaderFacts>,
}

impl ClientSnapshot {
    /// Count the exact prepared prompt using the provider's counting endpoint.
    /// None means unsupported; network, authentication and invalid responses are
    /// errors, never replaced with an approximation or a generation request.
    pub async fn count_tokens_exact_in(
        &self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<Option<u64>, LlmError> {
        let resolved = self
            .state
            .config
            .resolve_request(&request.model, Some(profile))?;
        let selected = resolved.connections[0].profile;
        if !crate::providers::dispatch::chat(selected).supports_exact_count(selected) {
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
        request: &ChatRequest,
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
        request: &ChatRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<RequestDraft, LlmError> {
        let resolved = self
            .state
            .config
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
        executor.validate_host_request(std::slice::from_ref(selected), request, &options, mode)?;
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
                (mode, false, false),
            )
            .await?;
        let continuation =
            super::executor::continuation_template(request.request, selected, &options);
        let mut file_expirations = super::executor::request_file_expirations(request.request);
        file_expirations.extend(
            prepared
                .files
                .iter()
                .filter_map(|file| file.expires_at.clone()),
        );
        let has_carried_text_units = request.request.messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, crate::protocol::ContentBlock::TextJsUtf16 { .. }))
        });
        let message_json_string_overrides =
            if options.message_text_utf16_overrides.is_empty() && !has_carried_text_units {
                std::collections::BTreeMap::new()
            } else {
                let body = json_body_source(
                    &prepared.http.body,
                    "encoded request body is not JSON for UTF-16 text mapping",
                )?;
                crate::exact_json::map_message_text_overrides(
                    request.request,
                    selected.profile.protocol,
                    &body,
                    &options.message_text_utf16_overrides,
                )?
            };
        let native_anthropic_request_header_facts = prepared.native_anthropic_request_header_facts;
        let pricing_identity = super::FrozenPricing::for_request(
            selected.profile,
            selected.model,
            request.request.model_source.as_ref(),
        );
        Ok(RequestDraft {
            native_traceparent: None,
            semantic_body: None,
            request_json_string_overrides: message_json_string_overrides.clone(),
            message_json_string_overrides,
            raw_body_set: false,
            body_auth_strategy: None,
            auth_body_policy_applied: false,
            mutable_body_baseline: None,
            authenticator: options
                .authenticator
                .as_ref()
                .map(|auth| auth.0.clone())
                .or_else(|| {
                    self.runtime
                        .authenticators
                        .get(&selected.profile.auth)
                        .cloned()
                }),
            credential: options.credential.clone(),
            account_scope: options.account_scope.clone(),
            call: PreparedCall {
                prepared,
                continuation,
                file_expirations,
                session_binding: None,
                native_anthropic_request_header_facts,
                http: self.runtime.http.clone(),
                clock: self.runtime.clock.clone(),
                deadline: options
                    .total_timeout
                    .and_then(|timeout| started.checked_add(timeout)),
                profile: selected.profile.clone(),
                model: pricing_identity.model.clone(),
                model_source: pricing_identity.model_source.clone(),
                mode,
            },
        })
    }
}

/// Mutable generation request before its final authentication step. It cannot
/// dispatch. Sealing consumes the draft and produces an immutable single-use call.
pub struct RequestDraft {
    native_traceparent: Option<String>,
    semantic_body: Option<(bytes::Bytes, crate::protocol::JsonPayload)>,
    request_json_string_overrides: std::collections::BTreeMap<String, Vec<u16>>,
    message_json_string_overrides: std::collections::BTreeMap<String, Vec<u16>>,
    raw_body_set: bool,
    body_auth_strategy: Option<crate::protocol::AuthStrategy>,
    auth_body_policy_applied: bool,
    mutable_body_baseline: Option<bytes::Bytes>,
    call: PreparedCall,
    authenticator: Option<Arc<dyn crate::Authenticator>>,
    credential: Option<crate::protocol::Secret<String>>,
    account_scope: Option<String>,
}

pub(super) fn chatgpt_body_policy_applies(
    strategy: crate::protocol::AuthStrategy,
    protocol: ProtocolFamily,
    mode: RequestMode,
) -> bool {
    strategy == crate::protocol::AuthStrategy::ChatGptOAuth
        && protocol == ProtocolFamily::OpenAiResponses
        && matches!(mode, RequestMode::Complete | RequestMode::Stream)
}

impl RequestDraft {
    /// Exact source-message strings reindexed onto this selected codec's JSON
    /// body. This source-message projection stays separate from exact mappings
    /// added to later request fields through `set_json_body`.
    pub fn message_json_string_overrides(&self) -> &std::collections::BTreeMap<String, Vec<u16>> {
        &self.message_json_string_overrides
    }
    /// The complete retained exact-string map, including host fields added
    /// through `set_json_body` after the source-message projection.
    pub fn request_json_string_overrides(&self) -> &std::collections::BTreeMap<String, Vec<u16>> {
        &self.request_json_string_overrides
    }
    /// Return the authoritative source JSON for the current byte image.
    /// Exact value/key units and deep fields remain source-backed for hosts.
    pub fn semantic_body_source(&self) -> Result<crate::protocol::JsonPayload, LlmError> {
        if let Some((bytes, source)) = &self.semantic_body {
            if bytes == &self.call.prepared.http.body {
                return Ok(source.clone());
            }
        }
        json_body_source(
            &self.call.prepared.http.body,
            "prepared request body is invalid JSON",
        )
    }

    /// Select a provider auth strategy whose generation-body policy must be
    /// applied by the SDK before exact JSON serialization and signing. Hosts
    /// that replace `ProviderProfile.auth` while deferring credentials pass the
    /// selected route's original strategy here before exposing the body.
    pub fn apply_request_body_auth_policy(
        &mut self,
        strategy: crate::protocol::AuthStrategy,
    ) -> Result<(), LlmError> {
        self.body_auth_strategy = Some(strategy);
        self.apply_current_request_body_auth_policy()
    }

    fn apply_current_request_body_auth_policy(&mut self) -> Result<(), LlmError> {
        let strategy = self.body_auth_strategy.unwrap_or(self.call.profile.auth);
        if !chatgpt_body_policy_applies(strategy, self.call.profile.protocol, self.call.mode)
            || self.auth_body_policy_applied
        {
            return Ok(());
        }
        let body = self.semantic_body_source()?;
        if body.kind_at("/model").is_none() || body.kind_at("/input").is_none() {
            self.auth_body_policy_applied = true;
            return Ok(());
        }
        let body = crate::auth::header_policy::chatgpt_source_body(&body)?;
        let mut overrides = self.request_json_string_overrides.clone();
        let scanner = body.scanner()?;
        overrides.retain(|pointer, _| {
            scanner.kind_at(pointer) == Some(crate::response_json::FlatJsonKind::String)
        });
        self.set_json_body(body, &overrides)?;
        self.auth_body_policy_applied = true;
        Ok(())
    }
    pub(super) fn transport(&self) -> Arc<dyn Transport> {
        self.call.http.clone()
    }
    pub async fn connect_websocket(
        &self,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        self.connect_websocket_using(self.call.http.as_ref()).await
    }
    pub async fn connect_websocket_using(
        &self,
        transport: &dyn crate::Transport,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        let handshake = self.websocket_handshake().await?;
        self.connect_websocket_handshake_using(handshake, Some(transport))
            .await
    }
    pub(super) fn account_scope(&self) -> Option<&str> {
        self.account_scope.as_deref()
    }
    pub(super) async fn websocket_handshake(&self) -> Result<HttpRequest, LlmError> {
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
        Ok(handshake)
    }
    pub(super) async fn connect_websocket_handshake_using(
        &self,
        handshake: HttpRequest,
        transport: Option<&dyn crate::Transport>,
    ) -> Result<Box<dyn crate::transport::WebSocketConnection>, LlmError> {
        self.call
            .connect_websocket_request(handshake, transport.unwrap_or(self.call.http.as_ref()))
            .await
    }
    /// Select HTTP/1 field serialization explicitly. The transport resolves the
    /// layout against the final authenticated headers at dispatch time.
    pub fn set_http1_header_layout(&mut self, layout: Option<crate::Http1HeaderLayout>) {
        self.call.prepared.http.http1_header_layout = layout;
    }

    pub fn request(&self) -> &HttpRequest {
        &self.call.prepared.http
    }
    pub fn request_mut(&mut self) -> &mut HttpRequest {
        self.call.session_binding = None;
        if self.mutable_body_baseline.is_none() {
            self.mutable_body_baseline = Some(self.call.prepared.http.body.clone());
        }
        &mut self.call.prepared.http
    }
    pub(super) fn bind_session(&mut self, binding: Arc<()>) {
        self.call.session_binding = Some(binding);
    }
    /// Replace the current request using exact source JSON and string overrides.
    /// This creates a draft image; Native sanitation occurs only when sealing.
    pub fn set_json_body(
        &mut self,
        value: crate::protocol::JsonPayload,
        overrides: &std::collections::BTreeMap<String, Vec<u16>>,
    ) -> Result<(), LlmError> {
        let bytes: bytes::Bytes = crate::exact_json::serialize_json_payload_with_overrides(
            &value,
            overrides,
            crate::exact_json::JsonEncoding::for_protocol(self.call.profile.protocol),
        )?
        .into();
        let source = json_body_source(&bytes, "encoded draft request body is invalid JSON")?;
        self.call.session_binding = None;
        self.call.prepared.http.body = bytes.clone();
        self.semantic_body = Some((bytes, source));
        self.request_json_string_overrides = overrides.clone();
        self.raw_body_set = false;
        self.auth_body_policy_applied = false;
        self.mutable_body_baseline = None;
        Ok(())
    }
    /// Replace the raw draft byte image. Its validated source is retained when
    /// available; changed raw images do not inherit stale string overrides.
    pub fn set_body_bytes(&mut self, body: bytes::Bytes) {
        self.call.session_binding = None;
        let unchanged = body == self.call.prepared.http.body;
        let source = json_body_source(&body, "raw draft request body is invalid JSON").ok();
        self.call.prepared.http.body = body.clone();
        self.semantic_body = source.map(|source| (body, source));
        if !unchanged {
            self.request_json_string_overrides.clear();
            self.message_json_string_overrides.clear();
        }
        self.raw_body_set = true;
        self.auth_body_policy_applied &= unchanged;
        self.mutable_body_baseline = None;
    }
    pub fn profile(&self) -> &ProviderProfile {
        &self.call.profile
    }
    pub fn model(&self) -> &ModelProfile {
        &self.call.model
    }
    /// Record exact membership in Native's planned request `betas` list. The
    /// Native refusal latch uses this input fact rather than the final signed
    /// HTTP header collection.
    pub fn set_native_anthropic_thinking_resumption_beta_requested(
        &mut self,
        requested: Option<bool>,
    ) {
        self.call
            .prepared
            .context
            .set_native_anthropic_thinking_resumption_beta_requested(requested);
    }

    /// Record Native's planned context-1m beta membership for the SDK context
    /// budget calculation. This remains independent of final HTTP headers.
    pub fn set_native_anthropic_context_1m_beta_requested(&mut self, requested: Option<bool>) {
        self.call
            .prepared
            .context
            .set_native_anthropic_context_1m_beta_requested(requested);
    }
    /// Record exact Native beta admission computed after initial draft
    /// encoding. The value is retained for late header emission and response
    /// error facts; it is not reconstructed from wire headers.
    pub fn set_native_anthropic_thinking_binding_controls_beta_requested(
        &mut self,
        requested: Option<bool>,
    ) {
        self.call
            .prepared
            .context
            .set_native_thinking_binding_controls_beta_requested(requested);
    }
    /// Provide trusted selected-route facts for Native Anthropic request-id
    /// policy. The SDK applies the per-attempt stage while sealing and the
    /// fetch fallback immediately before dispatch.
    pub fn set_native_anthropic_request_header_facts(
        &mut self,
        facts: crate::providers::response_headers::NativeAnthropicRequestHeaderFacts,
    ) -> Result<(), LlmError> {
        if facts.protocol != self.call.profile.protocol {
            return Err(LlmError::InvalidRequest {
                message: "Native request header facts do not match the selected protocol".into(),
            });
        }
        self.call.native_anthropic_request_header_facts = Some(facts);
        self.call.prepared.native_anthropic_request_header_facts = Some(facts);
        Ok(())
    }

    /// Supply the host's current trace context. The SDK decides whether Native's
    /// route and opt-in gates permit it on the outgoing Anthropic request.
    pub fn set_native_traceparent(&mut self, traceparent: String) -> Result<(), LlmError> {
        if !traceparent.bytes().all(|byte| byte.is_ascii_graphic()) {
            return Err(LlmError::InvalidRequest {
                message: "Native traceparent must be a visible ASCII header value".into(),
            });
        }
        self.native_traceparent = (!traceparent.is_empty()).then_some(traceparent);
        Ok(())
    }
    /// Settle the request body: reconcile a raw `request_mut` edit, re-encode
    /// retained exact string mappings, then apply the selected auth strategy's
    /// generation-body policy. Idempotent, and sealing runs it again. A session
    /// preparation runs it first: the session derives its continuation from this
    /// body and binds the draft to it, so a rewrite after that point would
    /// unbind the call.
    pub(super) fn settle_body(&mut self) -> Result<(), LlmError> {
        if self
            .mutable_body_baseline
            .as_ref()
            .is_some_and(|baseline| baseline != &self.call.prepared.http.body)
        {
            // `request_mut` exposes the raw byte image. If the caller changed
            // it, the old semantic value and UTF-16 map no longer describe the
            // request. Re-parse the new JSON body at policy application time,
            // and require callers that need exact lone units to use the typed
            // `set_json_body` API with a fresh complete override map.
            self.semantic_body = None;
            self.request_json_string_overrides.clear();
            self.message_json_string_overrides.clear();
            self.raw_body_set = true;
            self.auth_body_policy_applied = false;
        }
        self.mutable_body_baseline = None;
        if !self.raw_body_set
            && self.semantic_body.is_none()
            && !self.request_json_string_overrides.is_empty()
        {
            let body = json_body_source(
                &self.call.prepared.http.body,
                "prepared request body is not JSON for UTF-16 text encoding",
            )?;
            let overrides = self.request_json_string_overrides.clone();
            self.set_json_body(body, &overrides)?;
        }
        self.apply_current_request_body_auth_policy()
    }
    /// Install a session's wire form of the already settled request, such as
    /// its incremental continuation. Unlike `request_mut` this is not a caller
    /// edit: it leaves the settled state alone and the session binds afterwards.
    pub(super) fn set_session_wire_body(&mut self, body: bytes::Bytes) {
        self.call.prepared.http.body = body;
    }
    pub async fn seal(mut self) -> Result<PreparedCall, LlmError> {
        self.settle_body()?;
        if matches!(self.call.mode, RequestMode::Complete | RequestMode::Stream) {
            if let Some(body) = crate::exact_json::normalize_native_generation_json_body(
                &self.call.prepared.http.body,
                self.call.profile.protocol,
            )? {
                self.call.prepared.http.body = body.into();
                self.semantic_body = None;
            }
        }
        if matches!(self.call.mode, RequestMode::Complete | RequestMode::Stream) {
            if let Some(facts) = self.call.native_anthropic_request_header_facts {
                crate::providers::response_headers::ensure_native_client_request_id(
                    &mut self.call.prepared.http.headers,
                    facts,
                    crate::providers::response_headers::NativeRequestIdStage::PerAttempt,
                )?;
                if let Some(traceparent) = self.native_traceparent.as_deref() {
                    let _ = crate::providers::response_headers::ensure_native_traceparent(
                        &mut self.call.prepared.http.headers,
                        traceparent,
                        facts,
                        crate::providers::response_headers::native_traceparent_opt_in_from_environment(),
                    );
                }
            }
        }
        if matches!(self.call.mode, RequestMode::Complete | RequestMode::Stream) {
            super::executor::apply_native_thinking_binding_controls_beta(
                &self.call.prepared.context,
                &mut self.call.prepared.http.headers,
            );
        }
        // Context-1m suppression runs after the final TBC header append, while
        // preserving all unrelated final body and header source bytes.
        super::executor::apply_native_context_1m_suppression(
            &self.call.prepared.context,
            &self.call.profile,
            &mut self.call.prepared.http,
        )?;
        self.call
            .prepared
            .backend
            .validate_sealed_body(&self.call.profile, self.call.prepared.http.body.len())?;
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
        if self.call.profile.auth == crate::protocol::AuthStrategy::ChatGptPlan {
            crate::auth::chatgpt_plan::validate_request(
                &self.call.prepared.http,
                &self.call.profile,
                Some(&self.call.model.request_model),
            )?;
        }
        if let Ok(body) = json_body_source(
            &self.call.prepared.http.body,
            "sealed request body is invalid JSON",
        ) {
            use crate::protocol::ServiceTier;
            let scanner = body.scanner()?;
            let raw = scanner.string_at("/service_tier").or_else(|| {
                if scanner.kind_at("/service_tier").is_none() {
                    scanner.string_at("/speed")
                } else {
                    None
                }
            });
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

fn json_body_source(body: &[u8], message: &str) -> Result<crate::protocol::JsonPayload, LlmError> {
    let source = std::str::from_utf8(body).map_err(|error| LlmError::InvalidRequest {
        message: format!("{message}: {error}"),
    })?;
    crate::protocol::JsonPayload::parse(source).map_err(|error| LlmError::InvalidRequest {
        message: format!("{message}: {error}"),
    })
}

fn request_client_id(headers: &[(String, String)]) -> Option<&str> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
        .map(|(_, value)| value.as_str())
}

impl PreparedCall {
    pub(super) fn is_bound_to_session(&self, binding: &Arc<()>) -> bool {
        self.session_binding
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, binding))
    }
    fn validate_file_expirations(&self) -> Result<(), LlmError> {
        let now = self.clock.now();
        for expiry in &self.file_expirations {
            crate::files::validate_file_expiration_at(Some(expiry), now)?;
        }
        Ok(())
    }
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
        self.validate_websocket_profile()?;
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

    fn validate_websocket_profile(&self) -> Result<(), LlmError> {
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
        Ok(())
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
        self.validate_websocket_profile()?;
        self.validate_file_expirations()?;
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
                    // A WebSocket message is one JSON event, even when its
                    // representation spans lines. Prefix every line so the
                    // SSE splitter retains the entire event rather than only
                    // the opening line. CRLF and blank lines are valid JSON
                    // whitespace and must not terminate the synthetic event.
                    for line in bytes.split(|byte| *byte == b'\n') {
                        sse.extend_from_slice(b"data: ");
                        sse.extend_from_slice(line.strip_suffix(b"\r").unwrap_or(line));
                        sse.push(b'\n');
                    }
                    sse.push(b'\n');
                    sse.into()
                })
            })
            .boxed();
        Ok(ReceivedCall {
            call: self,
            response,
            proxy_auth_helper_challenge: None,
        })
    }
    /// Capture prices before dispatch for cancellation-safe host accounting.
    pub fn pricing_snapshot(&self) -> super::FrozenPricing {
        super::FrozenPricing {
            profile: self.profile.clone(),
            model: self.model.clone(),
            model_source: self.model_source.clone(),
        }
    }
    /// Final authenticated wire request. Its Debug implementation redacts secrets.
    pub fn request(&self) -> &HttpRequest {
        &self.prepared.http
    }
    /// The client request ID on this exact selected wire request, if one was
    /// supplied or generated by the SDK's native route policy.
    pub fn client_request_id(&self) -> Option<&str> {
        request_client_id(&self.prepared.http.headers)
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
        self.validate_file_expirations()?;
        // Validate the deadline before marking a physical attempt dispatched.
        crate::runtime::Deadline::at(self.deadline).remaining()?;
        on_dispatch()?;
        if let Some(facts) = self.native_anthropic_request_header_facts {
            crate::providers::response_headers::ensure_native_client_request_id(
                &mut self.prepared.http.headers,
                facts,
                crate::providers::response_headers::NativeRequestIdStage::FetchFallback,
            )?;
        }
        let http_executor =
            HttpExecutor::new(transport).with_deadline(crate::runtime::Deadline::at(self.deadline));
        let (response, proxy_auth_helper_challenge) =
            if super::options::uses_native_proxy_route(&self.profile) {
                let response = http_executor
                    .send_native_proxy_request(self.prepared.http.clone())
                    .await?;
                (response.response, response.proxy_auth_helper_challenge)
            } else {
                (http_executor.send(self.prepared.http.clone()).await?, None)
            };
        Ok(ReceivedCall {
            call: self,
            response,
            proxy_auth_helper_challenge,
        })
    }
}

/// Status and headers from one physical request, including unsuccessful status.
/// The host must observe usage before interpreting content or deciding to retry.
pub struct ReceivedCall {
    call: PreparedCall,
    response: StreamResponse,
    proxy_auth_helper_challenge: Option<crate::protocol::ProxyAuthHelperChallenge>,
}
impl ReceivedCall {
    /// A configured Native proxy-auth helper handled a real HTTP 407 before
    /// this response arrived. The payload itself remains unmodified.
    #[must_use]
    pub fn proxy_auth_helper_challenge(
        &self,
    ) -> Option<&crate::protocol::ProxyAuthHelperChallenge> {
        self.proxy_auth_helper_challenge.as_ref()
    }

    /// Client ID associated with the outgoing request after both SDK stages.
    pub fn client_request_id(&self) -> Option<&str> {
        self.call.client_request_id()
    }
    pub fn status(&self) -> u16 {
        self.response.status
    }
    pub fn headers(&self) -> &[(String, String)] {
        &self.response.headers
    }

    /// Collect an ordinary response without decoding its semantic content.
    /// Resource cleanup is retained until after the observation is delivered.
    pub async fn collect(self) -> Result<CollectedResponse, LlmError> {
        let mut response = if (200..300).contains(&self.response.status) {
            HttpExecutor::collect_response(self.response, None).await?
        } else {
            HttpResponse {
                status: self.response.status,
                headers: self.response.headers,
                body: crate::transport::collect_error_body(self.response.body).await,
            }
        };
        if self.call.mode == RequestMode::Complete {
            super::executor::normalize_codex_completion(
                &mut response,
                self.call.prepared.codec.as_ref(),
                &self.call.prepared.context,
            )?;
        }
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
            proxy_auth_helper_challenge: self.proxy_auth_helper_challenge,
        })
    }

    /// Enter streaming only after checking the response status. Error responses
    /// remain collectable, preserving any usage carried by their JSON envelope.
    pub fn into_stream(self) -> Result<ModelStream, Box<Self>> {
        if self.call.mode != RequestMode::Stream || !(200..300).contains(&self.response.status) {
            return Err(Box::new(self));
        }
        let pricing = self.call.pricing_snapshot();
        let observation = self
            .call
            .prepared
            .backend
            .stream_observation(&self.call.prepared.context);
        let response_cache = self.call.prepared.backend.response_cache(
            &self.call.profile,
            &self.call.prepared.http.url,
            &self.response.headers,
        );
        let mut stream = ModelStream::new(
            self.response,
            self.call
                .prepared
                .codec
                .stream_decoder(&self.call.prepared.context),
            self.call.profile.profile_name.clone(),
            response_cache,
            self.call
                .prepared
                .cleanup
                .map(|cleanup| (cleanup, self.call.deadline)),
            self.call.prepared.inference,
            self.call.continuation,
        )
        .with_provider_observation(observation);
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
    proxy_auth_helper_challenge: Option<crate::protocol::ProxyAuthHelperChallenge>,
}
impl CollectedResponse {
    /// Real HTTP 407 challenge for which the enabled Native helper gate ran.
    #[must_use]
    pub fn proxy_auth_helper_challenge(
        &self,
    ) -> Option<&crate::protocol::ProxyAuthHelperChallenge> {
        self.proxy_auth_helper_challenge.as_ref()
    }

    /// Client ID associated with the outgoing request after both SDK stages.
    pub fn client_request_id(&self) -> Option<&str> {
        self.call.client_request_id()
    }
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
    /// Anthropic server-fallback facts extracted independently of assistant
    /// content decoding. Usage iterations remain available if malformed
    /// content makes [`Self::decode`] fail.
    pub fn anthropic_fallback(
        &self,
    ) -> Option<crate::providers::anthropic::fallback_response::FallbackResponse> {
        if self.call.profile.protocol != ProtocolFamily::AnthropicMessages
            || !(200..300).contains(&self.response.status)
        {
            return None;
        }
        let source = std::str::from_utf8(&self.response.body).ok()?;
        let body = crate::protocol::JsonPayload::parse(source).ok()?;
        crate::providers::anthropic::fallback_response::from_nonstream_body(&body)
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
    pub fn decode(&self) -> Result<ChatResponse, LlmError> {
        let request_authority = super::executor::request_authority(&self.call.prepared.http.url);
        let decoded = self
            .call
            .prepared
            .codec
            .decode_response(&self.response, &self.call.prepared.context)
            .map_err(|error| error.with_native_anthropic_request_authority(request_authority));
        let mut response = match decoded {
            Ok(response) => response,
            Err(error) => {
                return Err(match self.proxy_auth_helper_challenge.clone() {
                    Some(challenge) => error.with_proxy_auth_helper_challenge(challenge),
                    None => error,
                });
            }
        };
        response.executed_profile = Some(self.call.profile.profile_name.clone());
        response.response_cache = self.call.prepared.backend.response_cache(
            &self.call.profile,
            &self.call.prepared.http.url,
            &self.response.headers,
        );
        if let (Some(mut reference), Some(id)) =
            (self.call.continuation.clone(), response.response_id.clone())
        {
            reference.response_id = id;
            response.continuation = Some(reference);
        }
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
            .runtime
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

impl LlmClient {
    pub async fn count_tokens_exact_in(
        &self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<Option<u64>, LlmError> {
        self.snapshot()
            .count_tokens_exact_in(profile, request, options)
            .await
    }
    pub async fn prepare_on(
        &self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<PreparedCall, LlmError> {
        self.snapshot()
            .prepare_on(profile, request, options, mode)
            .await
    }
    pub async fn prepare_draft_on(
        &self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
        mode: RequestMode,
    ) -> Result<RequestDraft, LlmError> {
        self.snapshot()
            .prepare_draft_on(profile, request, options, mode)
            .await
    }
}

#[cfg(test)]
mod request_body_policy_tests {
    use super::*;

    type CapturedRequest = (Vec<(String, String)>, Vec<u8>);

    #[derive(Debug, Default)]
    struct CaptureAuthenticator(std::sync::Mutex<Option<CapturedRequest>>);

    #[async_trait::async_trait]
    impl crate::Authenticator for CaptureAuthenticator {
        async fn apply(
            &self,
            request: &mut HttpRequest,
            _profile: &ProviderProfile,
            _credential: Option<&crate::protocol::Secret<String>>,
        ) -> Result<(), LlmError> {
            *self.0.lock().expect("capture lock") =
                Some((request.headers.clone(), request.body.to_vec()));
            request
                .headers
                .push(("x-test-authenticated".into(), "yes".into()));
            Ok(())
        }
    }

    #[derive(Debug)]
    struct UnusedTransport;

    #[async_trait::async_trait]
    impl Transport for UnusedTransport {
        async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
            Err(LlmError::Transport {
                message: "the TBC draft test must not dispatch".into(),
            })
        }
    }

    #[test]
    fn chatgpt_body_policy_is_generation_only() {
        use crate::protocol::{AuthStrategy, ProtocolFamily};

        assert!(chatgpt_body_policy_applies(
            AuthStrategy::ChatGptOAuth,
            ProtocolFamily::OpenAiResponses,
            RequestMode::Complete
        ));
        assert!(chatgpt_body_policy_applies(
            AuthStrategy::ChatGptOAuth,
            ProtocolFamily::OpenAiResponses,
            RequestMode::Stream
        ));
        assert!(!chatgpt_body_policy_applies(
            AuthStrategy::ChatGptOAuth,
            ProtocolFamily::OpenAiResponses,
            RequestMode::CountTokens
        ));
        assert!(!chatgpt_body_policy_applies(
            AuthStrategy::Bearer,
            ProtocolFamily::OpenAiResponses,
            RequestMode::Complete
        ));
        assert!(!chatgpt_body_policy_applies(
            AuthStrategy::ChatGptOAuth,
            ProtocolFamily::OpenAiChat,
            RequestMode::Complete
        ));
    }

    #[tokio::test]
    async fn post_encode_tbc_and_context_1m_facts_finalize_before_signing() {
        use crate::client::options::RequestAuthenticator;
        use crate::client::{LlmClientBuilder, NativeAnthropicContextInput};
        use crate::protocol::{ConversationMessage, ProtocolFamily, Region};

        let beta = crate::providers::anthropic::beta_repair::Beta::ThinkingBindingControls.header();

        let profile: ProviderProfile = serde_json::from_value(serde_json::json!({
            "provider_id": "bedrock",
            "profile_name": "bedrock-tbc-draft-test",
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "protocol": "bedrock_claude",
            "auth": "api_key",
            "models": [{
                "display_model": "wire-model",
                "request_model": "wire-model",
                "billing_model": "wire-model"
            }],
            "extra": {
                "betas": ["configured-beta", beta, "context-1m-2025-08-07"],
                "body": {"anthropic_beta": ["configured-body-beta", "context-1m-2025-08-07"]}
            }
        }))
        .expect("Bedrock test profile");
        assert_eq!(profile.protocol, ProtocolFamily::BedrockClaude);

        let client = LlmClientBuilder::with_transport(
            std::sync::Arc::new(UnusedTransport),
            std::slice::from_ref(&profile),
        )
        .with_region(Region::International)
        .build()
        .expect("client with hosted codec");
        let authenticator = std::sync::Arc::new(CaptureAuthenticator::default());
        let options = RequestOptions {
            native_anthropic_context: NativeAnthropicContextInput {
                context_1m_beta_suppressed: true,
                ..Default::default()
            },
            authenticator: Some(RequestAuthenticator(authenticator.clone())),
            ..Default::default()
        };

        let mut request = ChatRequest::new("wire-model");
        request
            .messages
            .push(ConversationMessage::user_text("hello"));
        let mut draft = client
            .prepare_draft_on(
                &profile.profile_name,
                &request,
                &options,
                RequestMode::Complete,
            )
            .await
            .expect("prepared hosted draft");

        assert!(draft.request().headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("anthropic-beta")
                && value.split(',').any(|part| part.trim() == beta)
        }));
        // This is the late Host header producer, after codec encoding and all
        // configured/inference beta sources have already been assembled.
        draft.request_mut().headers.push((
            "anthropic-beta".into(),
            "host-late-beta,context-1m-2025-08-07".into(),
        ));
        draft.set_native_anthropic_thinking_binding_controls_beta_requested(Some(true));
        let call = draft.seal().await.expect("seal and authenticate request");

        let captured = authenticator
            .0
            .lock()
            .expect("capture lock")
            .clone()
            .expect("authenticator saw request");
        let beta_headers = captured
            .0
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            beta_headers,
            [
                "configured-beta",
                "host-late-beta,thinking-binding-controls-2026-08-01"
            ]
        );
        let body = std::str::from_utf8(&captured.1).expect("request JSON");
        assert!(body.contains("configured-body-beta"), "{body}");
        assert!(!body.contains(beta), "{body}");
        assert!(!body.contains("context-1m-2025-08-07"), "{body}");
        assert!(call
            .request()
            .headers
            .iter()
            .any(|(name, value)| name == "x-test-authenticated" && value == "yes"));
    }
}
