//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::options::DEFAULT_REQUEST_TIMEOUT;
use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{LlmClient, RequestOptions};
use crate::codecs::WireCodec;
use crate::transport::{HttpRequest, HttpResponse};
use futures::StreamExt;
use lingxi_agent_api::protocol::{
    AuthStrategy, CompletionRequest, CompletionResponse, LlmError, ProtocolFamily, WebSearchConfig,
};
use std::sync::Arc;
use std::time::Instant;

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
        req: &CompletionRequest,
        opts: &RequestOptions,
        started: Instant,
    ) -> Result<(HttpRequest, Arc<dyn WireCodec>), LlmError> {
        let remaining = remaining_timeout(started, opts.total_timeout)?;
        let mut attempt_opts = opts.clone();
        attempt_opts.total_timeout = remaining;
        let preparation = self.prepare(route, attempt, req, &attempt_opts);
        let (mut http, codec) = match remaining {
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
        Ok((http, codec))
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

    /// Encode, authenticate, and hand the request to the transport. One hop.
    async fn prepare(
        &self,
        route: &ResolvedRoute,
        attempt: &Attempt,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<(HttpRequest, Arc<dyn WireCodec>), LlmError> {
        let profile =
            self.profile(&attempt.profile_name)
                .ok_or_else(|| LlmError::ModelUnavailable {
                    message: format!("profile {:?} vanished", attempt.profile_name),
                })?;
        if req.previous_response_id.is_some() && profile.protocol != ProtocolFamily::OpenAiResponses
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
        let mut http = codec.encode_request(req, profile, &hop_route, opts)?;

        http.timeout = opts.total_timeout;

        if profile.auth != AuthStrategy::None {
            let credential = if attempt.profile_name == route.profile_name {
                opts.credential.as_ref()
            } else {
                opts.fallback_credentials.get(&attempt.profile_name)
            };
            if attempt.profile_name != route.profile_name && credential.is_none() {
                return Err(LlmError::Authentication {
                    message: format!(
                        "profile {:?} needs its own credential",
                        attempt.profile_name
                    ),
                });
            }
            if let Some(auth) = self.authenticators.get(&profile.auth) {
                auth.apply(&mut http, profile, credential).await?;
            }
        }
        Ok((http, codec))
    }

    /// Walk the group's connections until one answers or an error is not a
    /// failover trigger. Returns the last error when the group is spent
    /// (gate 32).
    pub async fn complete(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let opts = RequestOptions {
            stream: false,
            total_timeout: Some(opts.total_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT)),
            ..opts.clone()
        };
        let route = self.resolve(&req.model)?;
        let started = Instant::now();
        let mut last = None;
        for attempt in attempts(&route, req.previous_response_id.is_some()) {
            let outcome = async {
                let (http, codec) = self
                    .prepare_before_deadline(&route, &attempt, req, &opts, started)
                    .await?;
                let resp = self.http.execute(http).await?;
                codec.decode_response(&resp)
            }
            .await;
            match outcome {
                Ok(mut resp) => {
                    resp.executed_profile = Some(attempt.profile_name);
                    return Ok(resp);
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
        let opts = RequestOptions {
            stream: true,
            ..opts.clone()
        };
        let route = self.resolve(&req.model)?;
        let started = Instant::now();
        let mut last = None;
        for attempt in attempts(&route, req.previous_response_id.is_some()) {
            let outcome = async {
                let (http, codec) = self
                    .prepare_before_deadline(&route, &attempt, req, &opts, started)
                    .await?;
                let mut resp = self.http.open_stream(http).await?;
                if !(200..300).contains(&resp.status) {
                    // Reuse each codec's status/body classification, including
                    // Retry-After, before deciding whether to try another hop.
                    const MAX_ERROR_BODY: usize = 64 * 1024;
                    let mut body = Vec::new();
                    while body.len() < MAX_ERROR_BODY {
                        let Some(chunk) = resp.body.next().await else {
                            break;
                        };
                        let chunk = chunk?;
                        let take = chunk.len().min(MAX_ERROR_BODY - body.len());
                        body.extend_from_slice(&chunk[..take]);
                    }
                    let response = HttpResponse {
                        status: resp.status,
                        headers: resp.headers,
                        body: body.into(),
                    };
                    return Err(codec.decode_response(&response).err().unwrap_or_else(|| {
                        LlmError::ProviderInternal {
                            message: format!("stream request failed with HTTP {}", response.status),
                        }
                    }));
                }
                Ok::<_, LlmError>(ModelStream::new(
                    resp,
                    codec.stream_decoder(),
                    attempt.profile_name.clone(),
                ))
            }
            .await;
            match outcome {
                Ok(s) => return Ok(s),
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
