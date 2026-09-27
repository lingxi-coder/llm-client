//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::options::DEFAULT_REQUEST_TIMEOUT;
use super::resolve::{RequestRoute, ResolvedConnection as Attempt};
use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{
    files, AttachmentKind, ClientSnapshot, FirstPartyEndpoint, PreparedProviderFileUse,
    ProviderFilePreparation, RequestOptions, ResolvedRequest, INLINE_IMAGE_PREFERENCE_LIMIT,
};
use crate::codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec};
use crate::protocol::{
    AuthStrategy, ChatRequest, ChatResponse, ContentBlock, ContinuationRef, LlmError,
    ProtocolFamily, ProviderFileSource, ProviderProfile, ResponseId,
};
use crate::transport::{collect_error_body, HttpExecutor, HttpRequest, HttpResponse, Transport};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{future::Future, time::Duration};

const ANTHROPIC_MAX_REQUEST_BODY_BYTES: usize = 32_000_000;

/// Check caller-supplied file lifetimes independently of route preparation so
/// an already-expired reference cannot trigger attachment reads or uploads.
fn validate_request_file_expirations(req: &ChatRequest, now: SystemTime) -> Result<(), LlmError> {
    use crate::protocol::{DocumentSource, ImageSource, VideoSource};
    for file in req
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            ContentBlock::Image {
                source: ImageSource::ProviderFile { file },
            }
            | ContentBlock::Document {
                source: DocumentSource::ProviderFile { file },
                ..
            }
            | ContentBlock::Video {
                source: VideoSource::ProviderFile { file },
            } => Some(file),
            _ => None,
        })
        .chain(
            req.hosted_anthropic_code_execution()
                .into_iter()
                .flat_map(|config| &config.files),
        )
    {
        files::validate_file_expiration_at(file.expires_at.as_deref(), now)?;
    }
    Ok(())
}

fn validate_prepared_file_expirations(
    req: &ChatRequest,
    prepared: &[PreparedProviderFileUse],
    now: SystemTime,
) -> Result<(), LlmError> {
    validate_request_file_expirations(req, now)?;
    for file in prepared {
        files::validate_file_expiration_at(file.expires_at.as_deref(), now)?;
    }
    Ok(())
}

fn projected_anthropic_file_request<'a>(
    req: &ResolvedRequest<'a>,
    profile: &ProviderProfile,
    model: &str,
    opts: &RequestOptions,
    promote_small_images: bool,
) -> Result<Vec<crate::codecs::ContentBinding<'a>>, LlmError> {
    let mut projected = Vec::new();
    for payload in &req.attachments {
        let capabilities = files::capabilities(profile, model, &payload.attachment.media_type);
        if !capabilities.upload
            || capabilities.model_input == files::ModelFileReference::Unsupported
        {
            continue;
        }
        if payload.kind == AttachmentKind::Image
            && !promote_small_images
            && payload.bytes.len() <= INLINE_IMAGE_PREFERENCE_LIMIT
        {
            continue;
        }
        if payload.kind == AttachmentKind::Video {
            continue;
        }
        let file = ProviderFileSource {
            protocol: profile.protocol,
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: files::provider_file_endpoint_fingerprint(&profile.base_url),
            account_scope: opts.file_account_scope.clone(),
            // The two JSON quote bytes are included in the preflight bound.
            expires_at: None,
            processing_status: None,
            file_id: "x".repeat(files::MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES - 2),
            uri: None,
            media_type: Some(payload.attachment.media_type.clone()),
            purpose: None,
        };
        projected.push(super::provider_file_binding(req.request, payload, file)?);
    }
    Ok(projected)
}

/// One connection to try: the head of the route, then each sibling in order.
/// The chain holds only the siblings, so the head is prepended here rather than
/// special-cased inside the walk.
struct PreparedAttempt {
    inference: crate::protocol::InferenceReport,
    http: HttpRequest,
    context: CodecContext,
    codec: Arc<dyn WireCodec>,
    files: Vec<PreparedProviderFileUse>,
    cleanup: Option<Arc<files::AutomaticFileCleanup>>,
}

fn deadline_elapsed() -> LlmError {
    crate::runtime::timeout_error()
}

async fn within_deadline<F: Future>(
    future: F,
    timeout: Option<Duration>,
) -> Result<F::Output, LlmError> {
    crate::runtime::Deadline::after(timeout).run(future).await
}

fn is_attachment_capability_fallback(
    attempt_index: usize,
    has_attachments: bool,
    error: &LlmError,
) -> bool {
    attempt_index > 0 && has_attachments && matches!(error, LlmError::UnsupportedCapability { .. })
}

fn ephemeral_file_scope() -> String {
    static NEXT_SCOPE: AtomicU64 = AtomicU64::new(1);
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!(
        "request:{}:{clock}:{}",
        std::process::id(),
        NEXT_SCOPE.fetch_add(1, Ordering::Relaxed)
    )
}

fn missing_provider_file_uses(
    resp: &HttpResponse,
    prepared_file_uses: &[PreparedProviderFileUse],
) -> Vec<PreparedProviderFileUse> {
    if resp.status != 404 {
        return Vec::new();
    }
    let body = String::from_utf8_lossy(&resp.body).to_ascii_lowercase();
    let missing = body.contains("not found")
        || body.contains("no such")
        || body.contains("does not exist")
        || body.contains("invalid file")
        || body.contains("unknown file");
    if !missing {
        return Vec::new();
    }
    prepared_file_uses
        .iter()
        .filter(|used| {
            let file_id = used.file_id.to_ascii_lowercase();
            let id_suffix = file_id.rsplit('/').next().unwrap_or(&file_id);
            body.contains(&file_id)
                || (id_suffix.len() >= 4 && body.contains(id_suffix))
                || used
                    .uri
                    .as_deref()
                    .is_some_and(|uri| body.contains(&uri.to_ascii_lowercase()))
        })
        .cloned()
        .collect()
}

fn remaining_timeout(
    started: Instant,
    total: Option<std::time::Duration>,
) -> Result<Option<std::time::Duration>, LlmError> {
    total
        .map(|total| {
            total
                .checked_sub(started.elapsed())
                .filter(|remaining| !remaining.is_zero())
                .ok_or_else(deadline_elapsed)
        })
        .transpose()
}

fn attempts<'a>(connections: &[Attempt<'a>], continuation: bool) -> Vec<Attempt<'a>> {
    connections
        .iter()
        .copied()
        .take(if continuation { 1 } else { connections.len() })
        .collect()
}

fn supports_continuation(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::OpenAiResponses
        && profile
            .extra
            .get("supports_previous_response_id")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
}

fn continuation_template(
    req: &ChatRequest,
    attempt: &Attempt<'_>,
    opts: &RequestOptions,
) -> Option<ContinuationRef> {
    if !supports_continuation(attempt.profile) {
        return None;
    }
    let scope = opts.account_scope.as_deref()?;
    Some(ContinuationRef::scoped(
        ResponseId::new(""),
        attempt.profile,
        &attempt.model.request_model,
        scope,
        req.hosted_file_search()
            .as_ref()
            .map(|search| search.workspace_id.as_str()),
    ))
}

pub(super) struct RequestExecutor<'a> {
    clock: &'a dyn crate::transport::Clock,
    http: &'a Arc<dyn Transport>,
    codecs: &'a std::collections::BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    authenticators:
        &'a std::collections::BTreeMap<AuthStrategy, Arc<dyn crate::auth::Authenticator>>,
    attachments: &'a super::AttachmentManager,
    state: &'a super::PublishedState,
}
pub(super) enum RequestOutput {
    Complete(Box<ChatResponse>),
    Stream(Box<ModelStream>),
}
impl<'client> RequestExecutor<'client> {
    pub(super) fn new(client: &'client ClientSnapshot) -> Self {
        Self {
            clock: client.runtime.clock.as_ref(),
            http: &client.runtime.http,
            codecs: &client.runtime.codecs,
            authenticators: &client.runtime.authenticators,
            attachments: &client.runtime.attachments,
            state: &client.state,
        }
    }

    async fn prepare_before_deadline(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt<'_>,
        req: &ResolvedRequest<'_>,
        opts: &RequestOptions,
        started: Instant,
        mode: RequestMode,
    ) -> Result<PreparedAttempt, LlmError> {
        let remaining = remaining_timeout(started, opts.total_timeout)?;
        let mut attempt_opts = opts.clone();
        attempt_opts.total_timeout = remaining;
        let request_deadline = opts.total_timeout.and_then(|timeout| {
            started.checked_add(timeout).map(|deadline| {
                // Leave time to return an unfinished file reference before the
                // outer deadline, without consuming a short request's budget.
                let reserve = Duration::from_secs(1)
                    .min(deadline.saturating_duration_since(Instant::now()) / 10);
                deadline.checked_sub(reserve).unwrap_or(deadline)
            })
        });
        let preparation = self.prepare(route, attempt, req, &attempt_opts, request_deadline, mode);
        let mut prepared = within_deadline(preparation, remaining).await??;
        // Authentication may have awaited a token refresh. The transport
        // receives only the time still available after that work.
        prepared.http.timeout = remaining_timeout(started, opts.total_timeout)?;
        Ok(prepared)
    }

    async fn resolve_before_deadline<'a>(
        &self,
        req: &'a ChatRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<ResolvedRequest<'a>, LlmError> {
        let remaining = remaining_timeout(started, opts.total_timeout)?;
        let resolution = self.attachments.resolve_attachments(req);
        within_deadline(resolution, remaining).await?
    }
    async fn prepare(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt<'_>,
        req: &ResolvedRequest<'_>,
        opts: &RequestOptions,
        request_deadline: Option<Instant>,
        mode: RequestMode,
    ) -> Result<PreparedAttempt, LlmError> {
        let profile = attempt.profile;
        validate_request_file_expirations(req.request, self.clock.now())?;
        super::options::validate_openrouter_response_cache(
            opts.openrouter_response_cache,
            profile,
        )?;
        super::options::validate_mcp_authorizations(
            &opts.mcp_authorizations,
            req.request,
            profile,
        )?;
        if req.request.continuation.is_some() && profile.protocol != ProtocolFamily::OpenAiResponses
        {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "profile {:?} uses {:?}, which cannot continue a Responses id",
                    profile.profile_name, profile.protocol
                ),
            });
        }
        let codec = self.codecs.get(&profile.protocol).cloned().ok_or_else(|| {
            LlmError::UnsupportedCapability {
                message: format!("no codec for {:?}", profile.protocol),
            }
        })?;

        // Failing over re-points the endpoint and the credential, never the
        // model — so the codec encodes against this connection's profile and
        // wire model, and the rest of the route is untouched.

        if opts
            .file_account_scope
            .as_deref()
            .is_some_and(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "file_account_scope must be a non-empty, non-secret account identifier"
                    .into(),
            });
        }
        let mut attempt_opts = opts.clone();
        if !req.attachments.is_empty() && attempt_opts.file_account_scope.is_none() {
            // The request-local scope binds uploaded IDs during this attempt;
            // it is deliberately not retained in the cross-request cache.
            attempt_opts.file_account_scope = Some(ephemeral_file_scope());
        }

        if crate::codecs::gemini::encode::has_gemini_hosted_tools(req.request) {
            crate::codecs::gemini::encode::validate_hosted_tool_request(
                req.request,
                profile,
                &attempt.model.request_model,
            )?;
        }
        crate::codecs::structured::validate_qwen_output_contract(
            req.request,
            profile,
            &attempt.model.request_model,
        )?;
        crate::codecs::anthropic_code_execution::validate(
            req.request,
            &CodecContext::for_model(profile, attempt.model, mode)
                .with_account_scope(opts.account_scope.as_deref())
                .with_file_scope(opts.file_account_scope.as_deref())
                .with_file_validation_time(self.clock.now()),
        )?;
        if crate::codecs::openrouter_server_tools::has_tools(req.request) {
            crate::codecs::openrouter_server_tools::validate(
                req.request,
                profile,
                opts.account_scope.as_deref(),
                true,
            )?;
        }

        let credential = if attempt.profile.profile_name == route.profile_name {
            opts.credential.as_ref()
        } else {
            opts.fallback_credentials.get(&attempt.profile.profile_name)
        };
        if attempt.profile.profile_name != route.profile_name
            && profile.auth != AuthStrategy::None
            && credential.is_none()
        {
            return Err(LlmError::Authentication {
                message: format!(
                    "profile {:?} needs its own credential",
                    attempt.profile.profile_name
                ),
            });
        }
        let authenticator = if profile.auth == AuthStrategy::None {
            None
        } else {
            self.authenticators.get(&profile.auth).map(Arc::as_ref)
        };
        let automatic_cleanup = self.attachments.cleanup_lease(
            profile,
            self.authenticators.get(&profile.auth).cloned(),
            credential,
            attempt_opts.file_account_scope.clone(),
            opts.file_account_scope.as_deref(),
        );
        let mut context = CodecContext::for_model(profile, attempt.model, mode)
            .with_file_scope(attempt_opts.file_account_scope.as_deref())
            .with_account_scope(opts.account_scope.as_deref())
            .with_file_validation_time(self.clock.now());
        crate::codecs::structured::validate(req.request, &context)?;
        codec.validate_request(req.request, &context)?;
        let inference = codec.request_inference(req.request, &context)?;
        let media = req
            .attachments
            .iter()
            .map(|payload| crate::codecs::PreparedMedia {
                attachment: &payload.attachment,
                bytes: &payload.bytes,
            })
            .collect::<Vec<_>>();
        if matches!(
            profile.protocol,
            ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
        ) && req
            .request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| matches!(block, ContentBlock::Audio { .. }))
        {
            // Bound the complete inline representation before automatic file
            // uploads. Callers with larger mixed-media requests can explicitly
            // upload files first and supply scoped references instead.
            codec.encoded_body_len(EncodeRequest::new(req.request).with_media(&media), &context)?;
        }
        let endpoint = self
            .state
            .config
            .first_party_endpoint(&profile.profile_name);
        let anthropic_request_limit = endpoint == FirstPartyEndpoint::Anthropic;
        let anthropic_mcp_auth_overhead = if anthropic_request_limit {
            super::options::anthropic_mcp_authorization_body_overhead(
                req.request,
                &opts.mcp_authorizations,
            )?
        } else {
            0
        };
        let inline_image_data_budget_bytes = if anthropic_request_limit
            && !req.attachments.is_empty()
        {
            // Simulate the exact attachment choices before uploading. If the
            // inline body fits, only large images and supported documents use
            // files. Otherwise, every eligible app image is promoted.
            let inline = codec
                .encoded_body_len(EncodeRequest::new(req.request).with_media(&media), &context)?;
            let inline_with_auth = inline.saturating_add(anthropic_mcp_auth_overhead);
            let promote_small_images = inline_with_auth > ANTHROPIC_MAX_REQUEST_BODY_BYTES;
            let projected = projected_anthropic_file_request(
                req,
                context.profile(),
                &attempt.model.request_model,
                &attempt_opts,
                promote_small_images,
            )?;
            let projected = codec.encoded_body_len(
                EncodeRequest::new(req.request)
                    .with_media(&media)
                    .with_bindings(&projected),
                &context,
            )?;
            let projected = projected.saturating_add(anthropic_mcp_auth_overhead);
            if projected > ANTHROPIC_MAX_REQUEST_BODY_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "Anthropic request body projects to {} bytes with bounded file IDs; the limit is {} bytes",
                        projected,
                        ANTHROPIC_MAX_REQUEST_BODY_BYTES
                    ),
                });
            }
            Some(if promote_small_images { 0 } else { usize::MAX })
        } else {
            None
        };
        let prepared = self
            .attachments
            .prepare_provider_file_inputs(
                profile,
                &attempt.model.request_model,
                req.request,
                &req.attachments,
                ProviderFilePreparation {
                    file_validation_time: self.clock.now(),
                    endpoint,
                    cache_namespace: self.state.cache_namespace,
                    cache_generation: self.state.cache_generations[&profile.profile_name],
                    planning_profile: context.profile(),
                    opts: &attempt_opts,
                    request_deadline,
                    stable_account_scope: opts.file_account_scope.as_deref(),
                    inline_image_data_budget_bytes,
                    authenticator,
                    credential,
                    automatic_cleanup: automatic_cleanup.clone(),
                },
                |bindings| {
                    // First-party Anthropic's bounded-ID projection above
                    // already encoded the complete request before uploads.
                    // Planning still validates every attachment, and the final
                    // body is checked below after substituting real file IDs.
                    if inline_image_data_budget_bytes.is_some() {
                        return Ok(());
                    }
                    codec
                        .encoded_body_len(
                            EncodeRequest::new(req.request)
                                .with_media(&media)
                                .with_bindings(bindings),
                            &context,
                        )
                        .map(|_| ())
                },
            )
            .await?;

        // Uploads and processing may have consumed a file's remaining lifetime.
        context = context.with_file_validation_time(self.clock.now());
        validate_prepared_file_expirations(
            req.request,
            &prepared.uses,
            context.file_validation_time(),
        )?;
        let mut http = codec.encode_request(
            EncodeRequest::new(req.request)
                .with_media(&media)
                .with_bindings(&prepared.bindings),
            &context,
        )?;
        http.timeout = opts.total_timeout;
        super::options::apply_openrouter_response_cache(
            opts.openrouter_response_cache,
            profile,
            &mut http,
        )?;
        super::options::apply_mcp_authorizations(&opts.mcp_authorizations, profile, &mut http)?;
        if anthropic_request_limit && http.body.len() > ANTHROPIC_MAX_REQUEST_BODY_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: format!(
                    "Anthropic request body is {} bytes; the limit is {} bytes",
                    http.body.len(),
                    ANTHROPIC_MAX_REQUEST_BODY_BYTES
                ),
            });
        }

        validate_prepared_file_expirations(req.request, &prepared.uses, self.clock.now())?;
        if let Some(auth) = authenticator {
            auth.apply(&mut http, profile, credential).await?;
        }
        Ok(PreparedAttempt {
            inference,
            http,
            context,
            codec,
            files: prepared.uses,
            cleanup: prepared.cleanup,
        })
    }
    async fn execute_attempt(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt<'_>,
        req: &ResolvedRequest<'_>,
        opts: &RequestOptions,
        started: Instant,
        mode: RequestMode,
    ) -> Result<RequestOutput, (LlmError, Option<HttpResponse>, Vec<PreparedProviderFileUse>)> {
        let PreparedAttempt {
            inference: mut requested,
            http,
            context,
            codec,
            files,
            cleanup,
        } = self
            .prepare_before_deadline(route, attempt, req, opts, started, mode)
            .await
            .map_err(|e| (e, None, Vec::new()))?;
        let deadline = opts
            .total_timeout
            .and_then(|timeout| started.checked_add(timeout));
        requested.executed_at = self
            .clock
            .now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|time| time.as_secs());
        let request_url = http.url.clone();
        // The authenticator may await a refresh. Re-check just before the
        // model write, including uploaded references no longer in req.messages.
        validate_prepared_file_expirations(req.request, &files, self.clock.now())
            .map_err(|error| (error, None, files.clone()))?;
        let response = HttpExecutor::new(self.http.as_ref())
            .with_deadline(crate::runtime::Deadline::at(deadline))
            .send(http)
            .await
            .map_err(|e| (e, None, files.clone()))?;
        if !(200..300).contains(&response.status) {
            let response = HttpResponse {
                status: response.status,
                headers: response.headers,
                body: collect_error_body(response.body).await,
            };
            let error = codec
                .decode_response(&response, &context)
                .err()
                .unwrap_or_else(|| LlmError::ProviderInternal {
                    message: format!("request failed with HTTP {}", response.status),
                });
            return Err((error, Some(response), files));
        }
        if mode == RequestMode::Stream {
            let continuation = continuation_template(req.request, attempt, opts);
            let response_cache = super::response_cache::openrouter_observation(
                attempt.profile,
                &request_url,
                &response.headers,
            );
            return Ok(RequestOutput::Stream(Box::new(
                ModelStream::new(
                    response,
                    codec.stream_decoder(&context),
                    attempt.profile.profile_name.clone(),
                    response_cache,
                    cleanup.map(|cleanup| (cleanup, deadline)),
                    requested,
                    continuation,
                )
                .with_anthropic_container_observation(
                    crate::codecs::anthropic_code_execution::is_official_profile(attempt.profile)
                        || crate::codecs::anthropic_code_execution::supports_execution(&context),
                ),
            )));
        }
        let response = HttpExecutor::collect_response(response, None)
            .await
            .map_err(|e| (e, None, files.clone()))?;
        let mut decoded = match codec.decode_response(&response, &context) {
            Ok(decoded) => decoded,
            Err(error) => return Err((error, Some(response), files)),
        };
        decoded.executed_profile = Some(attempt.profile.profile_name.clone());
        decoded.response_cache = super::response_cache::openrouter_observation(
            attempt.profile,
            &request_url,
            &response.headers,
        );
        if let (Some(mut reference), Some(id)) = (
            continuation_template(req.request, attempt, opts),
            decoded.response_id.clone(),
        ) {
            reference.response_id = id;
            decoded.continuation = Some(reference);
        }
        decoded.inference.executed_at = requested.executed_at;
        decoded.inference.requested_effort = requested.requested_effort;
        decoded.inference.requested_service_tier = requested.requested_service_tier;
        decoded.inference.requested_raw_service_tier = requested.requested_raw_service_tier;
        if let Some(cleanup) = cleanup {
            cleanup.finish(deadline).await;
        }
        Ok(RequestOutput::Complete(Box::new(decoded)))
    }
    pub(super) async fn run(
        &self,
        resolved: RequestRoute<'_>,
        req: &ChatRequest,
        opts: &RequestOptions,
        mode: RequestMode,
    ) -> Result<RequestOutput, LlmError> {
        let RequestRoute { route, connections } = resolved;
        let default_timeout = if req
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .any(|b| matches!(b, ContentBlock::Video { .. }))
        {
            files::GEMINI_VIDEO_FILE_TIMEOUT
        } else {
            DEFAULT_REQUEST_TIMEOUT
        };
        let opts = RequestOptions {
            total_timeout: if mode == RequestMode::Complete {
                Some(opts.total_timeout.unwrap_or(default_timeout))
            } else {
                opts.total_timeout
            },
            ..opts.clone()
        };
        let started = Instant::now();
        req.validate_hosted_tools()?;
        validate_request_file_expirations(req, self.clock.now())?;
        if crate::codecs::gemini::encode::has_gemini_hosted_tools(req) {
            let head = connections
                .first()
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("no connection served {:?}", req.model),
                })?;
            crate::codecs::gemini::encode::validate_hosted_tool_request(
                req,
                head.profile,
                &head.model.request_model,
            )?;
        }
        if let Some(head) = connections.first() {
            let context = CodecContext::for_model(head.profile, head.model, mode)
                .with_account_scope(opts.account_scope.as_deref())
                .with_file_scope(opts.file_account_scope.as_deref())
                .with_file_validation_time(self.clock.now());
            if req
                .messages
                .iter()
                .any(|message| message.anthropic.is_some())
                || crate::codecs::anthropic_conversation::has_tool_changes(req)
                || !req.anthropic_client_toolsets.is_empty()
                || (crate::codecs::anthropic_conversation::supports_profile(head.profile)
                    && req
                        .messages
                        .iter()
                        .any(|message| message.role == crate::protocol::MessageRole::System))
            {
                crate::codecs::anthropic_conversation::validate(req, &context)?;
                crate::codecs::cache::validate(req, &context)?;
            }
            crate::codecs::inference::validate(req, &context.profile, &context.request_model)?;
            crate::codecs::anthropic_web_fetch::validate(req, &context)?;
            crate::codecs::anthropic_tool_search::validate(req, &context)?;
            crate::codecs::anthropic_client_toolsets::validate(req, &context)?;
            if crate::codecs::anthropic_web_fetch::has_fetch(req) {
                crate::codecs::structured::validate(req, &context)?;
            }
            crate::codecs::anthropic_mcp::validate(req, &context)?;
            if crate::codecs::anthropic_web_fetch::has_fetch(req)
                || crate::codecs::anthropic_mcp::has_mcp(req)
            {
                crate::codecs::cache::validate(req, &context)?;
            }
            crate::codecs::anthropic_code_execution::validate(req, &context)?;
            crate::codecs::structured::validate_qwen_output_contract(
                req,
                head.profile,
                &head.model.request_model,
            )?;
        }
        let has_openrouter_server_tools = if crate::codecs::openrouter_server_tools::has_tools(req)
        {
            let head = connections
                .first()
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("no connection served {:?}", req.model),
                })?;
            crate::codecs::openrouter_server_tools::validate(
                req,
                head.profile,
                opts.account_scope.as_deref(),
                true,
            )?
        } else {
            false
        };
        if opts
            .account_scope
            .as_deref()
            .is_some_and(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "account_scope must be a non-empty, non-secret account identifier".into(),
            });
        }
        if let Some(reference) = &req.continuation {
            let head = connections
                .first()
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("no connection served {:?}", req.model),
                })?;
            if head.profile.protocol != ProtocolFamily::OpenAiResponses {
                return Err(LlmError::UnsupportedCapability {
                    message: format!(
                        "profile {:?} cannot continue a Responses reference",
                        head.profile.profile_name
                    ),
                });
            }
            if !supports_continuation(head.profile) {
                return Err(LlmError::UnsupportedCapability {
                    message: format!(
                        "profile {:?} does not declare response continuation support",
                        head.profile.profile_name
                    ),
                });
            }
            reference.validate(
                head.profile,
                &head.model.request_model,
                opts.account_scope.as_deref(),
                req.hosted_file_search()
                    .as_ref()
                    .map(|search| search.workspace_id.as_str()),
            )?;
        }
        if let Some(head) = connections.first() {
            super::options::validate_openrouter_response_cache(
                opts.openrouter_response_cache,
                head.profile,
            )?;
            super::options::validate_mcp_authorizations(
                &opts.mcp_authorizations,
                req,
                head.profile,
            )?;
            if head.profile.provider_id.as_str() == "qwen"
                && head.profile.protocol == ProtocolFamily::OpenAiChat
            {
                let context = CodecContext::for_model(head.profile, head.model, mode)
                    .with_file_scope(opts.file_account_scope.as_deref());
                crate::codecs::qwen_cache::validate(req, &context)?;
            }
        }
        let has_chat_audio = req
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| matches!(block, ContentBlock::Audio { .. }))
            || req.metadata.get("openrouter_chat_audio").is_some();
        if has_chat_audio {
            let head = connections
                .first()
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("no connection served {:?}", req.model),
                })?;
            let openrouter = head.profile.provider_id.as_str() == "openrouter"
                && head.profile.protocol == ProtocolFamily::OpenAiChat;
            let gemini = matches!(
                head.profile.protocol,
                ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
            );
            if !openrouter && (!gemini || req.metadata.get("openrouter_chat_audio").is_some()) {
                return Err(LlmError::UnsupportedCapability {
                    message: "Chat audio requires OpenRouter Chat or Gemini GenerateContent; OpenRouter output settings require OpenRouter".into(),
                });
            }
            let codec = self.codecs.get(&head.profile.protocol).ok_or_else(|| {
                LlmError::UnsupportedCapability {
                    message: "Chat audio profile has no registered codec".into(),
                }
            })?;
            let context = CodecContext::for_model(head.profile, head.model, mode)
                .with_file_scope(opts.file_account_scope.as_deref());
            codec.validate_request(req, &context)?;
        }
        let has_remote_mcp = req.hosted_tools.iter().any(|tool| {
            matches!(
                tool,
                crate::protocol::HostedTool::RemoteMcp(_)
                    | crate::protocol::HostedTool::XaiRemoteMcp(_)
            )
        });
        let has_anthropic_mcp = crate::codecs::anthropic_mcp::has_mcp(req);
        let has_anthropic_fetch = crate::codecs::anthropic_web_fetch::has_fetch(req);
        let has_openai_tool_search = req.hosted_openai_tool_search().is_some();
        // Hosted execution can mutate its container before a transport error.
        // Neither route fallback nor attachment repair may replay the request.
        let has_code_interpreter = req.hosted_code_interpreter().is_some();
        let has_gemini_hosted_tool = crate::codecs::gemini::encode::has_gemini_hosted_tools(req);
        let has_anthropic_execution = crate::codecs::anthropic_code_execution::has_execution(req);
        let prepared_request = self.resolve_before_deadline(req, &opts, started).await?;
        let mut last = None;
        for (attempt_index, attempt) in attempts(
            &connections,
            req.continuation.is_some()
                || opts.openrouter_response_cache.is_some()
                || has_remote_mcp
                || has_anthropic_mcp
                || has_anthropic_fetch
                || has_gemini_hosted_tool
                || has_openrouter_server_tools
                || has_openai_tool_search
                || has_code_interpreter
                || has_anthropic_execution,
        )
        .into_iter()
        .enumerate()
        {
            let outcome = self
                .execute_attempt(&route, &attempt, &prepared_request, &opts, started, mode)
                .await;
            let missing = outcome
                .as_ref()
                .err()
                .and_then(|(_, response, uses)| {
                    response
                        .as_ref()
                        .map(|response| missing_provider_file_uses(response, uses))
                })
                .unwrap_or_default();
            let outcome = if req.continuation.is_none()
                && !has_remote_mcp
                && !has_anthropic_mcp
                && !has_anthropic_fetch
                && !has_gemini_hosted_tool
                && !has_openrouter_server_tools
                && !has_openai_tool_search
                && !has_code_interpreter
                && !has_anthropic_execution
                && !missing.is_empty()
            {
                self.attachments
                    .invalidate_provider_file_cache(&missing)
                    .await;
                self.execute_attempt(&route, &attempt, &prepared_request, &opts, started, mode)
                    .await
                    .map_err(|(e, _, _)| e)
            } else {
                outcome.map_err(|(e, _, _)| e)
            };
            match outcome {
                Ok(output) => return Ok(output),
                Err(error)
                    if !has_openrouter_server_tools
                        && !has_openai_tool_search
                        && !has_code_interpreter
                        && !has_anthropic_execution
                        && !has_anthropic_mcp
                        && !has_anthropic_fetch
                        && (is_attachment_capability_fallback(
                            attempt_index,
                            !prepared_request.attachments.is_empty(),
                            &error,
                        ) || route.failover.matches(&error)) =>
                {
                    last = Some(error)
                }
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| LlmError::ModelUnavailable {
            message: format!("no connection served {:?}", req.model),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_continuation_is_pinned_to_the_head_connection() {
        let profiles = ["primary", "secondary"].map(|name| {
            serde_json::from_value(serde_json::json!({
                "profile_name": name, "provider_id": "acme", "base_url": "https://example.test", "auth": "none", "protocol": "open_ai_chat",
                "connection": {"group": "acme", "connection_id": name},
                "models": [{"request_model": "wire-m", "display_model": "m", "billing_model": "wire-m"}]
            })).unwrap()
        });
        let snapshot = super::super::snapshot::RuntimeSnapshot::new(
            crate::protocol::Region::International,
            profiles.into(),
            None,
        );
        let resolved = snapshot.resolve_request("m", Some("primary")).unwrap();
        assert_eq!(attempts(&resolved.connections, false).len(), 2);
        let continued = attempts(&resolved.connections, true);
        assert_eq!(continued.len(), 1);
        assert_eq!(continued[0].profile.profile_name, "primary");
        assert_eq!(continued[0].model.request_model, "wire-m");
    }
}
