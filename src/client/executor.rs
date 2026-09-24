//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::options::DEFAULT_REQUEST_TIMEOUT;
use super::resolve::{RequestRoute, ResolvedConnection as Attempt};
use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{
    files, uses_first_party_anthropic_messages, AttachmentKind, LlmClient, PreparedProviderFileUse,
    ProviderFilePreparation, RequestOptions, ResolvedRequest, INLINE_IMAGE_PREFERENCE_LIMIT,
};
use crate::codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec};
use crate::protocol::{
    AuthStrategy, CompletionRequest, CompletionResponse, ContentBlock, LlmError, ProtocolFamily,
    ProviderFileSource, ProviderProfile,
};
use crate::transport::{collect_error_body, HttpExecutor, HttpRequest, HttpResponse, Transport};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{future::Future, time::Duration};

const ANTHROPIC_MAX_REQUEST_BODY_BYTES: usize = 32_000_000;

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

pub(super) struct RequestExecutor<'a> {
    clock: &'a dyn crate::transport::Clock,
    http: &'a Arc<dyn Transport>,
    codecs: &'a std::collections::BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    authenticators:
        &'a std::collections::BTreeMap<AuthStrategy, Arc<dyn crate::auth::Authenticator>>,
    attachments: &'a super::AttachmentManager,
}
pub(super) enum RequestOutput {
    Complete(Box<CompletionResponse>),
    Stream(Box<ModelStream>),
}
impl<'client> RequestExecutor<'client> {
    pub(super) fn new(client: &'client LlmClient) -> Self {
        Self {
            clock: client.clock.as_ref(),
            http: &client.http,
            codecs: &client.codecs,
            authenticators: &client.authenticators,
            attachments: &client.attachments,
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
        req: &'a CompletionRequest,
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
        if req.request.previous_response_id.is_some()
            && profile.protocol != ProtocolFamily::OpenAiResponses
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
        let context = CodecContext::for_model(profile, attempt.model, mode)
            .with_file_scope(attempt_opts.file_account_scope.as_deref());
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
        let anthropic_request_limit = uses_first_party_anthropic_messages(profile);
        let inline_image_data_budget_bytes = if anthropic_request_limit
            && !req.attachments.is_empty()
        {
            // Simulate the exact attachment choices before uploading. If the
            // inline body fits, only large images and supported documents use
            // files. Otherwise, every eligible app image is promoted.
            let inline = codec
                .encoded_body_len(EncodeRequest::new(req.request).with_media(&media), &context)?;
            let promote_small_images = inline > ANTHROPIC_MAX_REQUEST_BODY_BYTES;
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

        let mut http = codec.encode_request(
            EncodeRequest::new(req.request)
                .with_media(&media)
                .with_bindings(&prepared.bindings),
            &context,
        )?;
        if anthropic_request_limit && http.body.len() > ANTHROPIC_MAX_REQUEST_BODY_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: format!(
                    "Anthropic request body is {} bytes; the limit is {} bytes",
                    http.body.len(),
                    ANTHROPIC_MAX_REQUEST_BODY_BYTES
                ),
            });
        }

        http.timeout = opts.total_timeout;

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
            return Ok(RequestOutput::Stream(Box::new(ModelStream::new(
                response,
                codec.stream_decoder(&context),
                attempt.profile.profile_name.clone(),
                cleanup,
                deadline,
                requested,
            ))));
        }
        let response = HttpExecutor::collect_response(response, None)
            .await
            .map_err(|e| (e, None, files.clone()))?;
        let mut decoded = codec
            .decode_response(&response, &context)
            .map_err(|e| (e, Some(response), files))?;
        decoded.executed_profile = Some(attempt.profile.profile_name.clone());
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
        req: &CompletionRequest,
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
        let prepared_request = self.resolve_before_deadline(req, &opts, started).await?;
        let mut last = None;
        for (attempt_index, attempt) in attempts(&connections, req.previous_response_id.is_some())
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
            let outcome = if !missing.is_empty() {
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
                    if is_attachment_capability_fallback(
                        attempt_index,
                        !prepared_request.attachments.is_empty(),
                        &error,
                    ) || route.failover.matches(&error) =>
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
