//! xAI's single-use deferred Chat Completions lifecycle.
use crate::{
    client::RequestOptions,
    codecs::{CodecContext, EncodeRequest, RequestMode},
    files::provider_file_endpoint_fingerprint,
    protocol::{
        ChatRequest, ChatResponse, LlmError, ProtocolFamily, ProviderId, ProviderProfile, Secret,
        ServiceAuth, ServiceSetting,
    },
    runtime::Deadline,
    runtime::{ClientSnapshot, ClientSource},
    transport::{HttpExecutor, HttpRequest, HttpResponse},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferredApi {
    XaiChat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeferredRoute {
    pub api: DeferredApi,
    /// Full `/v1/chat/completions` URL.
    pub endpoint: String,
    /// Full `/v1/chat/deferred-completion` collection URL.
    pub results_endpoint: String,
    pub auth: ServiceAuth,
}

/// One account-bound result ticket. Pass by value to `fetch_once`.
/// It is intentionally not `Clone`: a completed xAI result is consumed by GET.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeferredJobRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub results_endpoint_fingerprint: String,
    pub account_scope: String,
    pub model: String,
    pub request_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeferredJob {
    pub reference: DeferredJobRef,
    pub native: Value,
}

#[derive(Debug)]
pub struct DeferredCompletion {
    pub response: ChatResponse,
    pub native: Value,
}

#[derive(Debug)]
pub enum DeferredPoll {
    Pending(DeferredJobRef),
    Completed(Box<DeferredCompletion>),
}

#[derive(Debug, thiserror::Error)]
pub enum DeferredError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid deferred response: {0}")]
    InvalidResponse(String),
    #[error("deferred provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of deferred submission is unknown: {source}")]
    SubmitOutcomeUnknown {
        #[source]
        source: LlmError,
    },
    #[error("deferred submission was accepted but its request id is unavailable")]
    SubmitAcceptedUnknownId { native: Value },
    #[error("outcome of single-use deferred fetch is unknown: {source}")]
    FetchOutcomeUnknown {
        reference: Box<DeferredJobRef>,
        #[source]
        source: LlmError,
    },
    #[error("a deferred result was consumed but could not be decoded: {message}")]
    ConsumedInvalidResult { message: String, native: Value },
}

#[derive(Clone, Copy)]
pub struct DeferredService<'a> {
    source: ClientSource<'a>,
}

impl<'a> DeferredService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn submit(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<DeferredJob, DeferredError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        PinnedDeferredService { client: &snapshot }
            .submit(profile_name, request, options)
            .await
    }

    /// Perform exactly one GET. A 202 returns the ticket for later polling;
    /// a 200 consumes it and returns the completion without a reusable ticket.
    pub async fn fetch_once(
        self,
        reference: DeferredJobRef,
        options: &RequestOptions,
    ) -> Result<DeferredPoll, DeferredError> {
        let snapshot = self.source.pin()?;
        PinnedDeferredService { client: &snapshot }
            .fetch_once(reference, options)
            .await
    }
}

struct PinnedDeferredService<'a> {
    client: &'a ClientSnapshot,
}
impl PinnedDeferredService<'_> {
    fn route<'s>(
        &'s self,
        profile_name: &str,
        options: &'s RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s DeferredRoute, &'s str), DeferredError> {
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown deferred profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("deferred profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.deferred else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no deferred route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| invalid("deferred request requires a non-secret account_scope"))?;
        Ok((profile, route, scope))
    }

    async fn submit(
        &self,
        profile_name: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<DeferredJob, DeferredError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        if request.has_response_continuation() || !request.hosted_tools.is_empty() {
            return Err(LlmError::UnsupportedCapability {
                message: "deferred Chat cannot encode Responses continuation or hosted tools"
                    .into(),
            }
            .into());
        }
        let matches = profile
            .models
            .iter()
            .filter(|model| {
                model.display_model == request.model
                    || model.request_model == request.model
                    || model.aliases.iter().any(|alias| alias == &request.model)
            })
            .collect::<Vec<_>>();
        let [model] = matches.as_slice() else {
            return Err(invalid("deferred model is missing or ambiguous on this profile").into());
        };
        let codec = self
            .client
            .runtime
            .codecs
            .get(&ProtocolFamily::OpenAiChat)
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: "OpenAI Chat codec is unavailable".into(),
            })?;
        let context = CodecContext::for_model(profile, model, RequestMode::Complete)
            .with_file_scope(options.file_account_scope.as_deref());
        crate::codecs::structured::validate(request, &context)?;
        codec.validate_request(request, &context)?;
        let mut http = codec.encode_request(EncodeRequest::new(request), &context)?;
        if http.url != route.endpoint {
            return Err(invalid("deferred route differs from the profile's Chat endpoint").into());
        }
        let mut body: Value = serde_json::from_slice(&http.body)
            .map_err(|_| invalid("deferred Chat body is not JSON"))?;
        if body
            .get("deferred")
            .is_some_and(|value| value != &Value::Bool(true))
        {
            return Err(invalid("deferred Chat body conflicts with deferred=true").into());
        }
        body["deferred"] = Value::Bool(true);
        let bytes = serde_json::to_vec(&body)
            .map_err(|_| invalid("deferred Chat body cannot be serialized"))?;
        if bytes.len() > MAX_BODY {
            return Err(invalid("deferred Chat body exceeds 64 MiB").into());
        }
        http.body = bytes.into();
        apply_auth(route, options.credential.as_ref(), &mut http)?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        http.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(http, MAX_BODY)
            .await
            .map_err(|source| match source {
                LlmError::Transport { .. } | LlmError::TransportTimeout { .. } => {
                    DeferredError::SubmitOutcomeUnknown { source }
                }
                other => DeferredError::Llm(other),
            })?;
        let native = parse_json(&response).map_err(|error| {
            if (200..300).contains(&response.status) {
                DeferredError::SubmitAcceptedUnknownId {
                    native: Value::String(String::from_utf8_lossy(&response.body).into_owned()),
                }
            } else {
                error
            }
        })?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error(&response, native));
        }
        let request_id = native["request_id"]
            .as_str()
            .filter(|id| valid_id(id))
            .ok_or_else(|| DeferredError::SubmitAcceptedUnknownId {
                native: native.clone(),
            })?;
        Ok(DeferredJob {
            reference: DeferredJobRef {
                provider_id: profile.provider_id.clone(),
                profile_name: profile.profile_name.clone(),
                endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
                results_endpoint_fingerprint: provider_file_endpoint_fingerprint(
                    &route.results_endpoint,
                ),
                account_scope: scope.into(),
                model: model.request_model.clone(),
                request_id: request_id.into(),
            },
            native,
        })
    }

    async fn fetch_once(
        &self,
        reference: DeferredJobRef,
        options: &RequestOptions,
    ) -> Result<DeferredPoll, DeferredError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.results_endpoint_fingerprint
                != provider_file_endpoint_fingerprint(&route.results_endpoint)
            || reference.account_scope != scope
            || !valid_id(&reference.request_id)
            || reference.model.trim().is_empty()
        {
            return Err(LlmError::PermissionDenied { message: "deferred result belongs to another provider, profile, endpoint, model or account".into() }.into());
        }
        let mut url = url::Url::parse(&route.results_endpoint)
            .map_err(|_| invalid("invalid deferred result URL"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid deferred result URL"))?
            .push(&reference.request_id);
        let mut http = HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url: url.into(),
            headers: vec![],
            body: Vec::new().into(),
            timeout: None,
        };
        apply_auth(route, options.credential.as_ref(), &mut http)?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        http.timeout = deadline.remaining()?;
        let response = match HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(http, MAX_BODY)
            .await
        {
            Ok(response) => response,
            Err(source)
                if matches!(
                    source,
                    LlmError::Transport { .. }
                        | LlmError::TransportTimeout { .. }
                        | LlmError::StreamInterrupted { .. }
                ) =>
            {
                return Err(DeferredError::FetchOutcomeUnknown {
                    reference: Box::new(reference),
                    source,
                })
            }
            Err(other) => return Err(DeferredError::Llm(other)),
        };
        if response.status == 202 {
            return Ok(DeferredPoll::Pending(reference));
        }
        let native =
            parse_json(&response).map_err(|error| DeferredError::ConsumedInvalidResult {
                message: error.to_string(),
                native: Value::String(String::from_utf8_lossy(&response.body).into_owned()),
            })?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error(&response, native));
        }
        // The result can outlive a catalog row update. The original wire model
        // is pinned in the ticket; GET does not need a currently listed row.
        let context = CodecContext::new(profile, &reference.model, RequestMode::Complete);
        let codec = self
            .client
            .runtime
            .codecs
            .get(&ProtocolFamily::OpenAiChat)
            .ok_or_else(|| invalid("OpenAI Chat codec is unavailable"))?;
        let mut decoded = codec
            .decode_response(&response, &context)
            .map_err(|error| DeferredError::ConsumedInvalidResult {
                message: error.to_string(),
                native: native.clone(),
            })?;
        decoded.executed_profile = Some(profile.profile_name.clone());
        Ok(DeferredPoll::Completed(Box::new(DeferredCompletion {
            response: decoded,
            native,
        })))
    }
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
fn apply_auth(
    route: &DeferredRoute,
    credential: Option<&Secret<String>>,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    if route.auth != ServiceAuth::Bearer {
        return Err(invalid("xAI deferred route requires bearer authentication"));
    }
    let secret = credential.ok_or_else(|| LlmError::Authentication {
        message: "deferred route requires a credential".into(),
    })?;
    request.headers.push((
        "authorization".into(),
        format!("Bearer {}", secret.expose_secret()),
    ));
    Ok(())
}
fn parse_json(response: &HttpResponse) -> Result<Value, DeferredError> {
    serde_json::from_slice(&response.body).or_else(|_| {
        if (200..300).contains(&response.status) {
            Err(DeferredError::InvalidResponse(
                "successful response is not JSON".into(),
            ))
        } else {
            Ok(Value::String(
                String::from_utf8_lossy(&response.body).into_owned(),
            ))
        }
    })
}
fn provider_error(response: &HttpResponse, body: Value) -> DeferredError {
    DeferredError::Provider {
        status: response.status,
        request_id: response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned),
        body,
    }
}
pub fn validate_route(profile: &ProviderProfile, route: &DeferredRoute) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "xai"
        || profile.protocol != ProtocolFamily::OpenAiChat
        || route.api != DeferredApi::XaiChat
        || route.auth != ServiceAuth::Bearer
    {
        return Err(invalid(
            "deferred Chat route requires an xAI OpenAI Chat profile and bearer authentication",
        ));
    }
    let endpoint =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid deferred endpoint"))?;
    let results = url::Url::parse(&route.results_endpoint)
        .map_err(|_| invalid("invalid deferred results endpoint"))?;
    let base = url::Url::parse(&profile.base_url)
        .map_err(|_| invalid("invalid deferred profile base URL"))?;
    if endpoint.scheme() != base.scheme()
        || endpoint.host_str() != base.host_str()
        || endpoint.port_or_known_default() != base.port_or_known_default()
        || results.scheme() != base.scheme()
        || results.host_str() != base.host_str()
        || results.port_or_known_default() != base.port_or_known_default()
        || !endpoint.username().is_empty()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || !results.username().is_empty()
        || results.password().is_some()
        || results.query().is_some()
        || results.fragment().is_some()
        || endpoint.path() != format!("{}/chat/completions", base.path().trim_end_matches('/'))
        || results.path()
            != format!(
                "{}/chat/deferred-completion",
                base.path().trim_end_matches('/')
            )
    {
        return Err(invalid(
            "deferred endpoints must match the profile origin and documented Chat paths",
        ));
    }
    Ok(())
}
