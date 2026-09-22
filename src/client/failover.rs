//! `complete` and `stream` walk the route's connection chain until one
//! answers or an error is not a failover trigger (gate 32).

use super::route::ResolvedRoute;
use super::stream::ModelStream;
use super::{LlmClient, RequestOptions};
use crate::codecs::WireCodec;
use crate::transport::{HttpRequest, HttpResponse};
use futures::StreamExt;
use lingxi_agent_api::protocol::{
    AuthStrategy, CompletionRequest, CompletionResponse, LlmError, ProtocolFamily,
};
use std::sync::Arc;

/// One connection to try: the head of the route, then each sibling in order.
/// The chain holds only the siblings, so the head is prepended here rather than
/// special-cased inside the walk.
#[derive(Debug, Clone)]
struct Attempt {
    profile_name: String,
    request_model: String,
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

        if profile.auth != AuthStrategy::None {
            if let Some(auth) = self.authenticators.get(&profile.auth) {
                auth.apply(&mut http, profile, opts.credential.as_ref())
                    .await?;
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
        let route = self.resolve(&req.model)?;
        let mut last = None;
        for attempt in attempts(&route, req.previous_response_id.is_some()) {
            let outcome = async {
                let (http, codec) = self.prepare(&route, &attempt, req, opts).await?;
                let resp = self.services.http.execute(http).await?;
                codec.decode_response(&resp)
            }
            .await;
            match outcome {
                Ok(resp) => return Ok(resp),
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
        let mut last = None;
        for attempt in attempts(&route, req.previous_response_id.is_some()) {
            let outcome = async {
                let (http, codec) = self.prepare(&route, &attempt, req, opts).await?;
                let mut resp = self.services.http.open_stream(http).await?;
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
                Ok::<_, LlmError>(ModelStream::new(resp, codec.stream_decoder()))
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
