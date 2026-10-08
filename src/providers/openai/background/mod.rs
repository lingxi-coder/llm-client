//! OpenAI Responses background jobs. The caller owns polling and persistence.
mod stream;
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
pub use stream::{
    BackgroundChatEvent, BackgroundChatEventStream, BackgroundChatStreamError, BackgroundEvent,
    BackgroundEventCursor, BackgroundEventStream, BackgroundStreamError,
};

const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackgroundRoute {
    /// Full `/v1/responses` collection URL.
    pub endpoint: String,
    pub auth: ServiceAuth,
}

/// A reusable, account-bound job reference. Serialize it for durable polling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackgroundJobRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub model: String,
    pub response_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
}
impl BackgroundStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Queued | Self::InProgress)
    }
}

#[derive(Debug)]
pub struct BackgroundJob {
    pub reference: BackgroundJobRef,
    pub status: BackgroundStatus,
    /// Present for completed and incomplete results when the wire body decodes.
    pub response: Option<ChatResponse>,
    pub native: Value,
}

/// Confirmed deletion of one stored Response. This does not cancel inference.
#[derive(Debug)]
pub struct BackgroundDeletion {
    pub reference: BackgroundJobRef,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum BackgroundError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("background provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid background response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("outcome of background submission is unknown: {source}")]
    SubmitOutcomeUnknown {
        #[source]
        source: LlmError,
    },
    #[error("outcome of background cancellation is unknown: {source}")]
    CancelOutcomeUnknown {
        reference: Box<BackgroundJobRef>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of background deletion is unknown: {source}")]
    DeleteOutcomeUnknown {
        reference: Box<BackgroundJobRef>,
        #[source]
        source: Box<BackgroundError>,
    },
}

#[derive(Clone, Copy)]
pub struct BackgroundService<'a> {
    source: ClientSource<'a>,
}
impl<'a> BackgroundService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn submit(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundJob, BackgroundError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .submit(profile_name, request, options)
            .await
    }

    /// Fetch one snapshot. The caller chooses its polling interval.
    pub async fn get(
        self,
        reference: &BackgroundJobRef,
        options: &RequestOptions,
    ) -> Result<BackgroundJob, BackgroundError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .operate(reference, options, false)
            .await
    }

    /// Request cancellation once. Transport failure leaves the outcome unknown;
    /// use `get` to reconcile rather than blindly resubmitting the operation.
    pub async fn cancel(
        self,
        reference: &BackgroundJobRef,
        options: &RequestOptions,
    ) -> Result<BackgroundJob, BackgroundError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .operate(reference, options, true)
            .await
    }

    /// Delete a stored Response once, separately from cancellation. An
    /// uncertain deletion is never retried automatically.
    pub async fn delete(
        self,
        reference: &BackgroundJobRef,
        options: &RequestOptions,
    ) -> Result<BackgroundDeletion, BackgroundError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .delete(reference, options)
            .await
    }
}

struct Pinned<'a> {
    client: &'a ClientSnapshot,
}
impl Pinned<'_> {
    fn route<'a>(
        &'a self,
        profile_name: &str,
        options: &'a RequestOptions,
    ) -> Result<(&'a ProviderProfile, &'a BackgroundRoute, &'a str), BackgroundError> {
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown background profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("background profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.background else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no background route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| invalid("background request requires a non-secret account_scope"))?;
        Ok((profile, route, scope))
    }

    async fn submit(
        &self,
        profile_name: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BackgroundJob, BackgroundError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        let matches: Vec<_> = profile
            .models
            .iter()
            .filter(|m| {
                m.display_model == request.model
                    || m.request_model == request.model
                    || m.aliases.iter().any(|a| a == &request.model)
            })
            .collect();
        let [model] = matches.as_slice() else {
            return Err(invalid("background model is missing or ambiguous on this profile").into());
        };
        if let Some(reference) = &request.continuation {
            reference.validate(
                profile,
                &model.request_model,
                Some(scope),
                request
                    .hosted_file_search()
                    .as_ref()
                    .map(|search| search.workspace_id.as_str()),
            )?;
        }
        let codec = self.codec()?;
        let context = CodecContext::for_model(profile, model, RequestMode::Complete)
            .with_file_scope(options.file_account_scope.as_deref());
        crate::codecs::structured::validate(request, &context)?;
        codec.validate_request(request, &context)?;
        let mut http = codec.encode_request(EncodeRequest::new(request), &context)?;
        if http.url != route.endpoint {
            return Err(
                invalid("background route differs from the profile's Responses endpoint").into(),
            );
        }
        let mut body: Value = serde_json::from_slice(&http.body)
            .map_err(|_| invalid("background request body is not JSON"))?;
        if body.get("stream") == Some(&Value::Bool(true)) {
            return Err(invalid("use a separate background streaming API for stream=true").into());
        }
        if body
            .get("background")
            .is_some_and(|v| v != &Value::Bool(true))
        {
            return Err(invalid("background request conflicts with background=true").into());
        }
        body["background"] = Value::Bool(true);
        http.body = serde_json::to_vec(&body)
            .map_err(|_| invalid("background body cannot be serialized"))?
            .into();
        if http.body.len() > MAX_BODY {
            return Err(invalid("background body exceeds 64 MiB").into());
        }
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
                LlmError::Transport { .. }
                | LlmError::TransportTimeout { .. }
                | LlmError::StreamInterrupted { .. } => {
                    BackgroundError::SubmitOutcomeUnknown { source }
                }
                other => BackgroundError::Llm(other),
            })?;
        let native = parse_json(&response)?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error(&response, native));
        }
        let reference = make_ref(profile, route, scope, &model.request_model, &native)?;
        self.job(reference, response, native)
    }

    async fn operate(
        &self,
        reference: &BackgroundJobRef,
        options: &RequestOptions,
        cancel: bool,
    ) -> Result<BackgroundJob, BackgroundError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        validate_reference(profile, route, scope, reference)?;
        let mut url =
            url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid background endpoint"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid background endpoint"))?
            .push(&reference.response_id);
        if cancel {
            url.path_segments_mut()
                .map_err(|_| invalid("invalid background endpoint"))?
                .push("cancel");
        }
        let mut http = HttpRequest {
            http1_header_layout: None,
            method: if cancel { "POST" } else { "GET" }.into(),
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
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(http, MAX_BODY)
            .await
            .map_err(|source| {
                if cancel
                    && matches!(
                        source,
                        LlmError::Transport { .. }
                            | LlmError::TransportTimeout { .. }
                            | LlmError::StreamInterrupted { .. }
                    )
                {
                    BackgroundError::CancelOutcomeUnknown {
                        reference: Box::new(reference.clone()),
                        source,
                    }
                } else {
                    BackgroundError::Llm(source)
                }
            })?;
        let native = parse_json(&response)?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error(&response, native));
        }
        if native.get("id").and_then(Value::as_str) != Some(reference.response_id.as_str()) {
            return Err(BackgroundError::InvalidResponse {
                message: "response id differs from job reference".into(),
                native,
            });
        }
        self.job(reference.clone(), response, native)
    }

    async fn delete(
        &self,
        reference: &BackgroundJobRef,
        options: &RequestOptions,
    ) -> Result<BackgroundDeletion, BackgroundError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        validate_reference(profile, route, scope, reference)?;
        let mut url =
            url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid background endpoint"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid background endpoint"))?
            .push(&reference.response_id);
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let mut http = HttpRequest {
            http1_header_layout: None,
            method: "DELETE".into(),
            url: url.into(),
            headers: vec![],
            body: Default::default(),
            timeout: deadline.remaining()?,
        };
        apply_auth(route, options.credential.as_ref(), &mut http)?;
        let uncertain = |source| BackgroundError::DeleteOutcomeUnknown {
            reference: Box::new(reference.clone()),
            source: Box::new(source),
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(http, 1024 * 1024)
            .await
            .map_err(|source| uncertain(BackgroundError::Llm(source)))?;
        let native = parse_json(&response).map_err(uncertain)?;
        if !(200..300).contains(&response.status) {
            let unknown = response.status >= 500 || response.status == 408;
            let error = provider_error(&response, native);
            return Err(if unknown { uncertain(error) } else { error });
        }
        if native.get("id").and_then(Value::as_str) != Some(&reference.response_id)
            || native.get("object").and_then(Value::as_str) != Some("response")
            || native.get("deleted").and_then(Value::as_bool) != Some(true)
        {
            return Err(uncertain(BackgroundError::InvalidResponse {
                message: "missing or mismatched Response deletion receipt".into(),
                native,
            }));
        }
        Ok(BackgroundDeletion {
            reference: reference.clone(),
            native,
        })
    }

    fn job(
        &self,
        reference: BackgroundJobRef,
        response: HttpResponse,
        native: Value,
    ) -> Result<BackgroundJob, BackgroundError> {
        let status = match native.get("status").and_then(Value::as_str) {
            Some("queued") => BackgroundStatus::Queued,
            Some("in_progress") => BackgroundStatus::InProgress,
            Some("completed") => BackgroundStatus::Completed,
            Some("failed") => BackgroundStatus::Failed,
            Some("cancelled") => BackgroundStatus::Cancelled,
            Some("incomplete") => BackgroundStatus::Incomplete,
            _ => {
                return Err(BackgroundError::InvalidResponse {
                    message: "missing or unknown response status".into(),
                    native,
                })
            }
        };
        let decoded = if matches!(
            status,
            BackgroundStatus::Completed | BackgroundStatus::Incomplete
        ) {
            let profile = self
                .client
                .native_profile(&reference.profile_name)
                .ok_or_else(|| invalid("background profile disappeared"))?;
            let context = CodecContext::new(profile, &reference.model, RequestMode::Complete);
            let mut decoded = self
                .codec()?
                .decode_response(&response, &context)
                .map_err(|e| BackgroundError::InvalidResponse {
                    message: e.to_string(),
                    native: native.clone(),
                })?;
            decoded.executed_profile = Some(profile.profile_name.clone());
            if status == BackgroundStatus::Completed
                && profile
                    .extra
                    .get("supports_previous_response_id")
                    .and_then(Value::as_bool)
                    == Some(true)
            {
                if let Some(id) = decoded.response_id.clone() {
                    decoded.continuation = Some(crate::protocol::ContinuationRef::scoped(
                        id,
                        profile,
                        &reference.model,
                        &reference.account_scope,
                        None,
                    ));
                }
            }
            Some(decoded)
        } else {
            None
        };
        Ok(BackgroundJob {
            reference,
            status,
            response: decoded,
            native,
        })
    }

    fn codec(&self) -> Result<&dyn crate::codecs::WireCodec, LlmError> {
        self.client
            .runtime
            .codecs
            .get(&ProtocolFamily::OpenAiResponses)
            .map(|c| c.as_ref())
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: "OpenAI Responses codec is unavailable".into(),
            })
    }
}

fn validate_reference(
    profile: &ProviderProfile,
    route: &BackgroundRoute,
    scope: &str,
    reference: &BackgroundJobRef,
) -> Result<(), BackgroundError> {
    if reference.provider_id != profile.provider_id
        || reference.profile_name != profile.profile_name
        || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
        || reference.account_scope != scope
        || reference.model.trim().is_empty()
        || !valid_id(&reference.response_id)
    {
        return Err(LlmError::PermissionDenied {
            message:
                "background job belongs to another provider, profile, endpoint, model or account"
                    .into(),
        }
        .into());
    }
    Ok(())
}

fn make_ref(
    profile: &ProviderProfile,
    route: &BackgroundRoute,
    scope: &str,
    model: &str,
    native: &Value,
) -> Result<BackgroundJobRef, BackgroundError> {
    let id = native
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| valid_id(id))
        .ok_or_else(|| BackgroundError::InvalidResponse {
            message: "accepted response has no valid id; submission may have succeeded".into(),
            native: native.clone(),
        })?;
    Ok(BackgroundJobRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
        account_scope: scope.into(),
        model: model.into(),
        response_id: id.into(),
    })
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn apply_auth(
    route: &BackgroundRoute,
    credential: Option<&Secret<String>>,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    if route.auth != ServiceAuth::Bearer {
        return Err(invalid(
            "OpenAI background route requires bearer authentication",
        ));
    }
    let secret = credential.ok_or_else(|| LlmError::Authentication {
        message: "background route requires a credential".into(),
    })?;
    request.headers.push((
        "authorization".into(),
        format!("Bearer {}", secret.expose_secret()),
    ));
    Ok(())
}
fn parse_json(response: &HttpResponse) -> Result<Value, BackgroundError> {
    serde_json::from_slice(&response.body).or_else(|_| {
        if (200..300).contains(&response.status) {
            Err(BackgroundError::InvalidResponse {
                message: "successful response is not JSON".into(),
                native: Value::String(String::from_utf8_lossy(&response.body).into_owned()),
            })
        } else {
            Ok(Value::String(
                String::from_utf8_lossy(&response.body).into_owned(),
            ))
        }
    })
}
fn provider_error(response: &HttpResponse, body: Value) -> BackgroundError {
    BackgroundError::Provider {
        status: response.status,
        request_id: response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned),
        body,
    }
}

pub fn validate_route(profile: &ProviderProfile, route: &BackgroundRoute) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "openai"
        || profile.protocol != ProtocolFamily::OpenAiResponses
        || route.auth != ServiceAuth::Bearer
    {
        return Err(invalid("background route requires a first-party OpenAI Responses profile and bearer authentication"));
    }
    let endpoint =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid background endpoint"))?;
    let base = url::Url::parse(&profile.base_url)
        .map_err(|_| invalid("invalid background profile base URL"))?;
    if endpoint.scheme() != base.scheme()
        || endpoint.host_str() != base.host_str()
        || endpoint.port_or_known_default() != base.port_or_known_default()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || endpoint.path() != format!("{}/responses", base.path().trim_end_matches('/'))
    {
        return Err(invalid(
            "background endpoint must match the profile origin and Responses path",
        ));
    }
    Ok(())
}
