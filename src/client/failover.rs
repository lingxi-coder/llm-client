//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::options::DEFAULT_REQUEST_TIMEOUT;
use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{
    files, uses_first_party_anthropic_messages, AttachmentKind, LlmClient, PreparedProviderFileUse,
    ProviderFilePreparation, RequestOptions, ResolvedRequest, INLINE_IMAGE_PREFERENCE_LIMIT,
};
use crate::codecs::WireCodec;
use crate::transport::{collect_error_body, HttpRequest, HttpResponse};
use lingxi_agent_api::protocol::{
    AuthStrategy, CompletionRequest, CompletionResponse, ContentBlock, DocumentSource, ImageSource,
    LlmError, ProtocolFamily, ProviderFileSource, ProviderProfile, WebSearchConfig,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ANTHROPIC_MAX_REQUEST_BODY_BYTES: usize = 32_000_000;

fn projected_anthropic_file_request(
    req: &ResolvedRequest,
    profile: &ProviderProfile,
    model: &str,
    opts: &RequestOptions,
    promote_small_images: bool,
) -> Result<CompletionRequest, LlmError> {
    let mut projected = req.request.clone();
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
        let block = projected
            .messages
            .get_mut(payload.message_index)
            .and_then(|message| message.content.get_mut(payload.block_index))
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "resolved attachment no longer matches the request".into(),
            })?;
        match (payload.kind, block) {
            (AttachmentKind::Image, ContentBlock::Image { source }) => {
                *source = ImageSource::ProviderFile { file };
            }
            (AttachmentKind::Document, ContentBlock::Document { source, .. }) => {
                *source = DocumentSource::ProviderFile { file };
            }
            _ => {
                return Err(LlmError::InvalidRequest {
                    message: "resolved attachment kind does not match its request block".into(),
                });
            }
        }
    }
    Ok(projected)
}

/// One connection to try: the head of the route, then each sibling in order.
/// The chain holds only the siblings, so the head is prepended here rather than
/// special-cased inside the walk.
#[derive(Debug, Clone)]
struct Attempt {
    profile_name: String,
    request_model: String,
}

fn deadline_elapsed() -> LlmError {
    LlmError::TransportTimeout {
        message: "request deadline elapsed".into(),
    }
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

fn attempts(route: &ResolvedRoute, continuation: bool) -> Vec<Attempt> {
    let head = std::iter::once(Attempt {
        profile_name: route.profile_name.clone(),
        request_model: route.request_model.clone(),
    });
    if continuation {
        return head.collect();
    }
    head.chain(route.connection_chain.iter().map(|h| Attempt {
        profile_name: h.profile_name.clone(),
        request_model: h.request_model.clone(),
    }))
    .collect()
}

impl LlmClient {
    async fn prepare_before_deadline(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt,
        req: &ResolvedRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<
        (
            HttpRequest,
            Arc<dyn WireCodec>,
            Vec<PreparedProviderFileUse>,
        ),
        LlmError,
    > {
        let remaining = remaining_timeout(started, opts.total_timeout)?;
        let mut attempt_opts = opts.clone();
        attempt_opts.total_timeout = remaining;
        let preparation = self.prepare(route, attempt, req, &attempt_opts);
        let (mut http, codec, prepared_file_uses) = match remaining {
            Some(limit) if tokio::runtime::Handle::try_current().is_ok() => {
                tokio::time::timeout(limit, preparation)
                    .await
                    .map_err(|_| deadline_elapsed())??
            }
            _ => preparation.await?,
        };
        // Authentication may have awaited a token refresh. The transport
        // receives only the time still available after that work.
        http.timeout = remaining_timeout(started, opts.total_timeout)?;
        Ok((http, codec, prepared_file_uses))
    }

    async fn resolve_before_deadline(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<ResolvedRequest, LlmError> {
        let remaining = remaining_timeout(started, opts.total_timeout)?;
        let resolution = self.resolve_attachments(req);
        match remaining {
            Some(limit) if tokio::runtime::Handle::try_current().is_ok() => {
                tokio::time::timeout(limit, resolution)
                    .await
                    .map_err(|_| deadline_elapsed())?
            }
            _ => resolution.await,
        }
    }

    async fn execute_completion_attempt(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt,
        req: &ResolvedRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<CompletionResponse, (LlmError, Option<HttpResponse>, Vec<PreparedProviderFileUse>)>
    {
        let (http, codec, prepared_file_uses) = self
            .prepare_before_deadline(route, attempt, req, opts, started)
            .await
            .map_err(|error| (error, None, Vec::new()))?;
        let response = self
            .http
            .execute(http)
            .await
            .map_err(|error| (error, None, prepared_file_uses.clone()))?;
        codec
            .decode_response(&response)
            .map_err(|error| (error, Some(response), prepared_file_uses))
    }

    async fn open_stream_attempt(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt,
        req: &ResolvedRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<ModelStream, (LlmError, Option<HttpResponse>, Vec<PreparedProviderFileUse>)> {
        let (http, codec, prepared_file_uses) = self
            .prepare_before_deadline(route, attempt, req, opts, started)
            .await
            .map_err(|error| (error, None, Vec::new()))?;
        let response = self
            .http
            .open_stream(http)
            .await
            .map_err(|error| (error, None, prepared_file_uses.clone()))?;
        if !(200..300).contains(&response.status) {
            // Keep a bounded body for classification and stale-file detection.
            let body = collect_error_body(response.body).await;
            let response = HttpResponse {
                status: response.status,
                headers: response.headers,
                body,
            };
            let error = codec.decode_response(&response).err().unwrap_or_else(|| {
                LlmError::ProviderInternal {
                    message: format!("stream request failed with HTTP {}", response.status),
                }
            });
            return Err((error, Some(response), prepared_file_uses));
        }
        let profile_name = attempt.profile_name.clone();
        Ok(ModelStream::new(
            response,
            codec.stream_decoder(),
            profile_name,
        ))
    }

    /// Run a completion with provider-hosted web search enabled. The supplied
    /// configuration replaces `req.web_search` for this call; `req` is unchanged.
    /// The selected profile must declare a compatible `extra.web_search` adapter.
    pub async fn web_search(
        &self,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.complete(&req, opts).await
    }

    /// Run a provider-hosted web search on the selected profile or group.
    /// Scoping binds model resolution, endpoint, and primary credential to the
    /// same route; failover credentials still come from `opts` by profile name.
    pub async fn web_search_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.complete_in(profile, &req, opts).await
    }

    /// Stream a completion with provider-hosted web search enabled. Search
    /// metadata arrives as `StreamEvent::WebSearch` alongside text events.
    pub async fn web_search_stream(
        &self,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.stream(&req, opts).await
    }

    /// Stream a provider-hosted web search on the selected profile or group.
    /// Scoping binds model resolution, endpoint, and primary credential to the
    /// same route; failover credentials still come from `opts` by profile name.
    pub async fn web_search_stream_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.stream_in(profile, &req, opts).await
    }

    /// Encode, authenticate, and hand the request to the transport. One hop.
    async fn prepare(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt,
        req: &ResolvedRequest,
        opts: &RequestOptions,
    ) -> Result<
        (
            HttpRequest,
            Arc<dyn WireCodec>,
            Vec<PreparedProviderFileUse>,
        ),
        LlmError,
    > {
        let profile =
            self.profile(&attempt.profile_name)
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("profile {:?} vanished", attempt.profile_name),
                })?;
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
        let mut hop_route = route.clone();
        hop_route.provider_id = profile.provider_id.clone();
        hop_route.profile_name.clone_from(&attempt.profile_name);
        hop_route.request_model.clone_from(&attempt.request_model);

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

        let credential = if attempt.profile_name == route.profile_name {
            opts.credential.as_ref()
        } else {
            opts.fallback_credentials.get(&attempt.profile_name)
        };
        if attempt.profile_name != route.profile_name
            && profile.auth != AuthStrategy::None
            && credential.is_none()
        {
            return Err(LlmError::Authentication {
                message: format!(
                    "profile {:?} needs its own credential",
                    attempt.profile_name
                ),
            });
        }
        let authenticator = if profile.auth == AuthStrategy::None {
            None
        } else {
            self.authenticators.get(&profile.auth).map(Arc::as_ref)
        };
        let anthropic_request_limit = uses_first_party_anthropic_messages(profile);
        let inline_image_data_budget_bytes = if anthropic_request_limit
            && !req.attachments.is_empty()
        {
            // Simulate the exact attachment choices before uploading. If the
            // inline body fits, only large images and supported documents use
            // files. Otherwise, every eligible app image is promoted.
            let inline = codec.encode_request(&req.request, profile, &hop_route, &attempt_opts)?;
            let promote_small_images = inline.body.len() > ANTHROPIC_MAX_REQUEST_BODY_BYTES;
            let projected = projected_anthropic_file_request(
                req,
                profile,
                &attempt.request_model,
                &attempt_opts,
                promote_small_images,
            )?;
            let projected = codec.encode_request(&projected, profile, &hop_route, &attempt_opts)?;
            if projected.body.len() > ANTHROPIC_MAX_REQUEST_BODY_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "Anthropic request body projects to {} bytes with bounded file IDs; the limit is {} bytes",
                        projected.body.len(),
                        ANTHROPIC_MAX_REQUEST_BODY_BYTES
                    ),
                });
            }
            Some(if promote_small_images { 0 } else { usize::MAX })
        } else {
            None
        };
        let mut attempt_request = req.request.clone();
        let prepared_file_uses = self
            .prepare_provider_file_inputs(
                profile,
                &attempt.request_model,
                &mut attempt_request,
                &req.attachments,
                ProviderFilePreparation {
                    opts: &attempt_opts,
                    stable_account_scope: opts.file_account_scope.as_deref(),
                    inline_image_data_budget_bytes,
                    authenticator,
                    credential,
                },
            )
            .await?;

        let mut http =
            codec.encode_request(&attempt_request, profile, &hop_route, &attempt_opts)?;
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
        Ok((http, codec, prepared_file_uses))
    }

    /// Walk the group's connections until one answers or an error is not a
    /// failover trigger. Returns the last error when the group is spent
    /// (gate 32).
    pub async fn complete(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let route = self.resolve(&req.model)?;
        self.complete_route(route, req, opts).await
    }

    /// Complete a request using the selected profile or group as the starting
    /// route. The selected profile's credential must be in `opts.credential`;
    /// failover credentials are looked up by profile name.
    pub async fn complete_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let route = self.resolve_in(&req.model, Some(profile))?;
        self.complete_route(route, req, opts).await
    }

    async fn complete_route(
        &self,
        route: ResolvedRoute,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let opts = RequestOptions {
            stream: false,
            total_timeout: Some(opts.total_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT)),
            ..opts.clone()
        };
        let started = Instant::now();
        let prepared_request = self.resolve_before_deadline(req, &opts, started).await?;
        let mut last = None;
        for (attempt_index, attempt) in attempts(&route, req.previous_response_id.is_some())
            .into_iter()
            .enumerate()
        {
            let outcome = self
                .execute_completion_attempt(&route, &attempt, &prepared_request, &opts, started)
                .await;
            let missing_file_uses = outcome
                .as_ref()
                .err()
                .and_then(|(_, response, uses)| {
                    response
                        .as_ref()
                        .map(|response| missing_provider_file_uses(response, uses))
                })
                .unwrap_or_default();
            let outcome = if !missing_file_uses.is_empty() {
                self.invalidate_provider_file_cache(&missing_file_uses)
                    .await;
                self.execute_completion_attempt(&route, &attempt, &prepared_request, &opts, started)
                    .await
                    .map_err(|(error, _, _)| error)
            } else {
                outcome.map_err(|(error, _, _)| error)
            };
            match outcome {
                Ok(mut resp) => {
                    resp.executed_profile = Some(attempt.profile_name);
                    return Ok(resp);
                }
                Err(e)
                    if is_attachment_capability_fallback(
                        attempt_index,
                        !prepared_request.attachments.is_empty(),
                        &e,
                    ) =>
                {
                    last = Some(e)
                }
                Err(e) if route.failover.matches(&e) => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or(LlmError::ModelUnavailable {
            message: format!("no connection served {:?}", req.model),
        }))
    }

    /// Same walk, but the connection that answers hands back a stream.
    pub async fn stream(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self.resolve(&req.model)?;
        self.stream_route(route, req, opts).await
    }

    /// Stream a request using the selected profile or group as the starting
    /// route. The selected profile's credential must be in `opts.credential`;
    /// failover credentials are looked up by profile name.
    pub async fn stream_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self.resolve_in(&req.model, Some(profile))?;
        self.stream_route(route, req, opts).await
    }

    async fn stream_route(
        &self,
        route: ResolvedRoute,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let opts = RequestOptions {
            stream: true,
            ..opts.clone()
        };
        let started = Instant::now();
        let prepared_request = self.resolve_before_deadline(req, &opts, started).await?;
        let mut last = None;
        for (attempt_index, attempt) in attempts(&route, req.previous_response_id.is_some())
            .into_iter()
            .enumerate()
        {
            let outcome = self
                .open_stream_attempt(&route, &attempt, &prepared_request, &opts, started)
                .await;
            let missing_file_uses = outcome
                .as_ref()
                .err()
                .and_then(|(_, response, uses)| {
                    response
                        .as_ref()
                        .map(|response| missing_provider_file_uses(response, uses))
                })
                .unwrap_or_default();
            let outcome = if !missing_file_uses.is_empty() {
                self.invalidate_provider_file_cache(&missing_file_uses)
                    .await;
                self.open_stream_attempt(&route, &attempt, &prepared_request, &opts, started)
                    .await
                    .map_err(|(error, _, _)| error)
            } else {
                outcome.map_err(|(error, _, _)| error)
            };
            match outcome {
                Ok(s) => return Ok(s),
                Err(e)
                    if is_attachment_capability_fallback(
                        attempt_index,
                        !prepared_request.attachments.is_empty(),
                        &e,
                    ) =>
                {
                    last = Some(e)
                }
                Err(e) if route.failover.matches(&e) => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or(LlmError::ModelUnavailable {
            message: format!("no connection served {:?}", req.model),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::route::{ConnectionHop, PricingModelRef};
    use super::*;

    #[test]
    fn a_continuation_is_pinned_to_the_head_connection() {
        use lingxi_agent_api::protocol::{FailoverTriggers, ModelCapabilities, ProviderId};

        let route = ResolvedRoute {
            provider_id: ProviderId::new("acme"),
            profile_name: "primary".to_owned(),
            request_model: "wire-m".to_owned(),
            display_model: "m".to_owned(),
            pricing_model: PricingModelRef {
                pricing_provider_id: ProviderId::new("acme"),
                billing_model: "wire-m".to_owned(),
                request_model: "wire-m".to_owned(),
                display_model: "m".to_owned(),
            },
            capabilities: ModelCapabilities::default(),
            connection_chain: vec![ConnectionHop {
                profile_name: "secondary".to_owned(),
                request_model: "wire-m".to_owned(),
            }],
            failover: FailoverTriggers::default(),
        };

        assert_eq!(attempts(&route, false).len(), 2);
        let continued = attempts(&route, true);
        assert_eq!(continued.len(), 1);
        assert_eq!(continued[0].profile_name, "primary");
    }
}
