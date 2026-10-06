//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::options::DEFAULT_REQUEST_TIMEOUT;
use super::resolve::{RequestRoute, ResolvedConnection as Attempt};
use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{
    files, ClientSnapshot, PreparedProviderFileUse, ProviderFilePreparation, RequestOptions,
    ResolvedRequest,
};
use crate::codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec};
use crate::protocol::{
    AuthStrategy, ChatRequest, ChatResponse, ContentBlock, ContinuationRef, LlmError,
    ProtocolFamily, ProviderProfile, ResponseId,
};
use crate::transport::{collect_error_body, HttpExecutor, HttpRequest, HttpResponse, Transport};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{future::Future, time::Duration};

use crate::providers::dispatch::{self, ChatBackend};

/// Check caller-supplied file lifetimes independently of route preparation so
/// an already-expired reference cannot trigger attachment reads or uploads.
pub(super) fn request_file_expirations(req: &ChatRequest) -> Vec<String> {
    use crate::protocol::{DocumentSource, ImageSource, VideoSource};
    req.messages
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
        .filter_map(|file| file.expires_at.clone())
        .chain(dispatch::extra_file_expirations(req).map(str::to_owned))
        .collect()
}

fn validate_request_file_expirations(req: &ChatRequest, now: SystemTime) -> Result<(), LlmError> {
    for expiry in request_file_expirations(req) {
        files::validate_file_expiration_at(Some(&expiry), now)?;
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

/// One connection to try: the head of the route, then each sibling in order.
/// The chain holds only the siblings, so the head is prepended here rather than
/// special-cased inside the walk.
pub(super) struct PreparedAttempt {
    pub(super) inference: crate::protocol::InferenceReport,
    pub(super) http: HttpRequest,
    pub(super) context: CodecContext,
    pub(super) codec: Arc<dyn WireCodec>,
    pub(super) backend: &'static dyn ChatBackend,
    pub(super) files: Vec<PreparedProviderFileUse>,
    pub(super) cleanup: Option<Arc<files::AutomaticFileCleanup>>,
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

pub(super) fn missing_provider_file_uses(
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

pub(super) fn continuation_template(
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
    clock: &'a Arc<dyn crate::transport::Clock>,
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
            clock: &client.runtime.clock,
            http: &client.runtime.http,
            codecs: &client.runtime.codecs,
            authenticators: &client.runtime.authenticators,
            attachments: &client.runtime.attachments,
            state: &client.state,
        }
    }

    pub(super) async fn prepare_before_deadline(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt<'_>,
        req: &ResolvedRequest<'_>,
        opts: &RequestOptions,
        started: Instant,
        preparation: (RequestMode, bool),
    ) -> Result<PreparedAttempt, LlmError> {
        let (mode, authenticate) = preparation;
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
        let preparation = self.prepare(
            route,
            attempt,
            req,
            &attempt_opts,
            request_deadline,
            (mode, authenticate),
        );
        let mut prepared = within_deadline(preparation, remaining).await??;
        // Authentication may have awaited a token refresh. The transport
        // receives only the time still available after that work.
        prepared.http.timeout = remaining_timeout(started, opts.total_timeout)?;
        Ok(prepared)
    }

    pub(super) async fn resolve_before_deadline<'a>(
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
        preparation: (RequestMode, bool),
    ) -> Result<PreparedAttempt, LlmError> {
        let (mode, authenticate) = preparation;
        let profile = attempt.profile;
        validate_request_file_expirations(req.request, self.clock.now())?;
        let backend = dispatch::chat(profile);
        let codec = backend.codec(profile, self.codecs)?;
        let validation_context = CodecContext::for_model(profile, attempt.model, mode)
            .with_account_scope(opts.account_scope.as_deref())
            .with_file_scope(opts.file_account_scope.as_deref())
            .with_file_validation_time(self.clock.now());
        backend.validate_preparation(req.request, &validation_context, opts)?;

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
            opts.authenticator
                .as_ref()
                .filter(|_| attempt.profile.profile_name == route.profile_name)
                .map(|auth| auth.0.as_ref())
                .or_else(|| self.authenticators.get(&profile.auth).map(Arc::as_ref))
        };
        let automatic_cleanup = self.attachments.cleanup_lease(
            profile,
            opts.authenticator
                .as_ref()
                .filter(|_| attempt.profile.profile_name == route.profile_name)
                .map(|auth| auth.0.clone())
                .or_else(|| self.authenticators.get(&profile.auth).cloned()),
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
        backend.preflight_media(req.request, &media, &context, codec.as_ref())?;
        let endpoint = self
            .state
            .config
            .first_party_endpoint(&profile.profile_name);
        let inline_image_data_budget_bytes =
            backend.attachment_budget(req, &context, &attempt_opts, codec.as_ref(), endpoint)?;
        let prepared = self
            .attachments
            .prepare_provider_file_inputs(
                profile,
                &attempt.model.request_model,
                req.request,
                &req.attachments,
                ProviderFilePreparation {
                    file_validation_time: self.clock.now(),
                    clock: self.clock.as_ref(),
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
        backend.apply_request_options(opts, profile, &mut http)?;
        if let Some(finalizer) = &opts.finalizer {
            finalizer.finalize(&mut http, profile)?;
        }
        backend.validate_prepared_body(http.body.len(), endpoint)?;

        validate_prepared_file_expirations(req.request, &prepared.uses, self.clock.now())?;
        http.timeout = opts.total_timeout;

        if authenticate {
            if let Some(auth) = authenticator {
                auth.apply(&mut http, profile, credential).await?;
            }
            if profile.auth == AuthStrategy::ChatGptPlan {
                crate::auth::chatgpt_plan::validate_request(
                    &http,
                    profile,
                    Some(&attempt.model.request_model),
                )?;
            }
        }
        Ok(PreparedAttempt {
            inference,
            http,
            context,
            codec,
            backend,
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
        let prepared = self
            .prepare_before_deadline(route, attempt, req, opts, started, (mode, true))
            .await
            .map_err(|e| (e, None, Vec::new()))?;
        let PreparedAttempt {
            inference: mut requested,
            http,
            context,
            codec,
            backend,
            files,
            cleanup,
        } = prepared;
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
            let response_cache =
                backend.response_cache(attempt.profile, &request_url, &response.headers);
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
                .with_pricing(super::FrozenPricing {
                    profile: attempt.profile.clone(),
                    model: attempt.model.clone(),
                })
                .with_provider_observation(backend.stream_observation(&context)),
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
        decoded.response_cache =
            backend.response_cache(attempt.profile, &request_url, &response.headers);
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
    pub(super) fn validate_host_request(
        &self,
        connections: &[Attempt<'_>],
        req: &ChatRequest,
        opts: &RequestOptions,
        mode: RequestMode,
    ) -> Result<(), LlmError> {
        req.validate_hosted_tools()?;
        validate_request_file_expirations(req, self.clock.now())?;
        if let Some(head) = connections.first() {
            let context = CodecContext::for_model(head.profile, head.model, mode)
                .with_account_scope(opts.account_scope.as_deref())
                .with_file_scope(opts.file_account_scope.as_deref())
                .with_file_validation_time(self.clock.now());
            let backend = dispatch::chat(head.profile);
            let codec = backend.codec(head.profile, self.codecs)?;
            backend.validate_host(req, &context, opts, codec.as_ref())?;
        }
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
        Ok(())
    }

    pub(super) async fn run(
        &self,
        resolved: RequestRoute<'_>,
        req: &ChatRequest,
        opts: &RequestOptions,
        mode: RequestMode,
    ) -> Result<RequestOutput, LlmError> {
        let RequestRoute { route, connections } = resolved;
        let opts = execution_options(req, opts, mode);
        let started = Instant::now();
        self.validate_host_request(&connections, req, &opts, mode)?;
        let head = connections
            .first()
            .ok_or_else(|| LlmError::ModelUnavailable {
                message: format!("no connection served {:?}", req.model),
            })?;
        let mut replay = dispatch::chat(head.profile).replay_policy(req, &opts);
        if connections
            .iter()
            .any(|attempt| attempt.profile.auth == AuthStrategy::ChatGptPlan)
        {
            // A plan grant may never become either the source or the target of
            // automatic failover. Account and billing selection stay explicit.
            replay.pin_to_connection = true;
            replay.allow_failover = false;
            replay.repair_missing_files = false;
        }
        let prepared_request = self.resolve_before_deadline(req, &opts, started).await?;
        let mut last = None;
        for (attempt_index, attempt) in attempts(&connections, replay.pin_to_connection)
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
            let outcome = if replay.repair_missing_files && !missing.is_empty() {
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
                    if replay.allow_failover
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

pub(super) fn execution_options(
    req: &ChatRequest,
    opts: &RequestOptions,
    mode: RequestMode,
) -> RequestOptions {
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
    RequestOptions {
        total_timeout: if mode != RequestMode::Stream {
            Some(opts.total_timeout.unwrap_or(default_timeout))
        } else {
            opts.total_timeout
        },
        ..opts.clone()
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
