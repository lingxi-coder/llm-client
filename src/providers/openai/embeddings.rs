use crate::client::{ClientSnapshot, ClientSource, RequestOptions};
use crate::embeddings::{self, backend::*, *};
use crate::protocol::{LlmError, ProviderId, ProviderProfile};
use crate::runtime::Deadline;
use crate::transport::{HttpExecutor, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, time::Duration};

pub(crate) struct OpenAiEmbeddings;
impl EmbeddingBackend for OpenAiEmbeddings {
    fn encode(
        &self,
        profile: &ProviderProfile,
        route: &EmbeddingRoute,
        req: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError> {
        validate_input(route, req)?;
        if is_first_party_openai_embedding_route(&profile.provider_id, route)
            && req.input.len() > 2048
        {
            return Err(invalid(
                "OpenAI Embeddings accepts at most 2048 inputs per request",
            ));
        }
        if is_first_party_openai_embedding_route(&profile.provider_id, route)
            && req.model == "text-embedding-ada-002"
            && req.dimensions.is_some()
        {
            return Err(invalid(
                "text-embedding-ada-002 does not support the dimensions parameter",
            ));
        }
        crate::codecs::openai::embeddings::reject_task(req)?;
        post(
            route.endpoint.clone(),
            crate::codecs::openai::embeddings::body(req),
        )
    }
    fn vectors(
        &self,
        body: &Value,
        req: &EmbeddingRequest,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
        decode_vectors(&body["data"], Some("index"), "embedding", req)
    }
    fn usage(&self, body: &Value) -> EmbeddingUsage {
        usage(body["usage"].clone(), "prompt_tokens", "total_tokens")
    }
}
/// True only for OpenAI's documented first-party API hosts and embeddings
/// operation. OpenAI-compatible gateways deliberately do not inherit these
/// limits or model-directory behavior.
pub(crate) fn is_first_party_openai_embedding_route(
    provider_id: &ProviderId,
    route: &EmbeddingRoute,
) -> bool {
    provider_id.as_str() == "openai"
        && route.api == EmbeddingApi::OpenAi
        && is_first_party_openai_endpoint(&route.endpoint, "/v1/embeddings")
}

pub(crate) fn is_first_party_openai_model_directory(
    provider_id: &ProviderId,
    route: &EmbeddingRoute,
) -> bool {
    is_first_party_openai_embedding_route(provider_id, route)
        && route
            .models_endpoint
            .as_deref()
            .is_some_and(|endpoint| is_first_party_openai_endpoint(endpoint, "/v1/models"))
}

fn is_first_party_openai_endpoint(endpoint: &str, expected_path: &str) -> bool {
    let Ok(url) = url::Url::parse(endpoint) else {
        return false;
    };
    matches!(url.scheme(), "https")
        && matches!(
            url.host_str(),
            Some(
                "api.openai.com"
                    | "us.api.openai.com"
                    | "eu.api.openai.com"
                    | "au.api.openai.com"
                    | "ca.api.openai.com"
                    | "jp.api.openai.com"
                    | "in.api.openai.com"
                    | "sg.api.openai.com"
                    | "kr.api.openai.com"
                    | "gb.api.openai.com"
                    | "ae.api.openai.com"
            )
        )
        && url.path() == expected_path
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

/// One currently documented OpenAI embedding model visible to the configured
/// API key. OpenAI's `/v1/models` response is a general model directory and
/// does not label model capabilities, so this list is intentionally limited
/// to the model IDs named by the Embeddings API contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiEmbeddingModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    /// Present when OpenAI has announced a model shutdown date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutdown_date: Option<String>,
    /// Full provider row, including fields unknown to this client version.
    pub native: Value,
}

/// One unpaginated page from OpenAI's general model directory, filtered to
/// its documented embedding model IDs. `native` retains all model rows, so
/// an unrecognized ID is not treated as evidence that it lacks embeddings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiEmbeddingModelPage {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub models: Vec<OpenAiEmbeddingModel>,
    pub native: Value,
}

pub(crate) async fn list_openai_models(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    options: &RequestOptions,
) -> Result<OpenAiEmbeddingModelPage, EmbeddingError> {
    let profile = snapshot
        .profile(profile_name)
        .ok_or_else(|| invalid("unknown embedding profile"))?;
    if !profile.supports_region(snapshot.region()) {
        return Err(invalid("embedding profile is unavailable in this region").into());
    }
    let ServiceSetting::Enabled(route) = &profile.embeddings else {
        return Err(LlmError::UnsupportedCapability {
            message: "profile has no enabled embedding route".into(),
        }
        .into());
    };
    if route.api != EmbeddingApi::OpenAi
        || !is_first_party_openai_model_directory(&profile.provider_id, route)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI embedding model discovery requires the first-party OpenAI Embeddings and Models routes".into(),
        }
        .into());
    }
    validate_route(route)?;
    let endpoint = route
        .models_endpoint
        .as_deref()
        .ok_or_else(|| invalid("OpenAI model-directory route is not configured"))?;
    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ));
    let mut request = HttpRequest {
        http1_header_layout: None,
        method: "GET".into(),
        url: endpoint.into(),
        headers: vec![("accept".into(), "application/json".into())],
        body: Vec::new().into(),
        timeout: deadline.remaining()?,
    };
    apply_auth(&route.auth, options, &mut request)?;
    let response = HttpExecutor::new(snapshot.runtime.http.as_ref())
        .with_deadline(deadline)
        .execute_bounded(request, MAX_RESPONSE_BYTES)
        .await?;
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let body = decode_response_body(&response.body, response.status, "OpenAI model directory")?;
    if !(200..300).contains(&response.status) {
        return Err(EmbeddingError::Provider {
            status: response.status,
            request_id,
            body,
        });
    }
    let object = body.as_object().ok_or_else(|| {
        EmbeddingError::InvalidResponse("OpenAI model-directory body is not an object".into())
    })?;
    if object.get("object").and_then(Value::as_str) != Some("list") {
        return Err(EmbeddingError::InvalidResponse(
            "OpenAI model-directory object is not `list`".into(),
        ));
    }
    let rows = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            EmbeddingError::InvalidResponse("OpenAI model-directory data is missing".into())
        })?;
    let mut seen = BTreeSet::new();
    let mut models = Vec::new();
    for row in rows {
        let row_object = row.as_object().ok_or_else(|| {
            EmbeddingError::InvalidResponse("OpenAI model-directory row is not an object".into())
        })?;
        let id = required_string(row_object, "id")?;
        if !seen.insert(id.clone()) {
            return Err(EmbeddingError::InvalidResponse(
                "OpenAI model-directory contains duplicate IDs".into(),
            ));
        }
        if is_documented_openai_embedding_model(&id) {
            models.push(parse_openai_embedding_model(row, row_object)?);
        }
    }

    Ok(OpenAiEmbeddingModelPage {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        models,
        native: body,
    })
}

fn is_documented_openai_embedding_model(id: &str) -> bool {
    matches!(
        id,
        "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002"
    )
}

fn parse_openai_embedding_model(
    value: &Value,
    object: &serde_json::Map<String, Value>,
) -> Result<OpenAiEmbeddingModel, EmbeddingError> {
    let model_object = required_string(object, "object")?;
    if model_object != "model" {
        return Err(EmbeddingError::InvalidResponse(
            "OpenAI model-directory row object is not `model`".into(),
        ));
    }
    let created = object
        .get("created")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            EmbeddingError::InvalidResponse(
                "OpenAI model-directory `created` field is missing or invalid".into(),
            )
        })?;
    let owned_by = required_string(object, "owned_by")?;
    let shutdown_date = optional_string(object, "shutdown_date")?;
    Ok(OpenAiEmbeddingModel {
        id: required_string(object, "id")?,
        object: model_object,
        created,
        owned_by,
        shutdown_date,
        native: value.clone(),
    })
}

fn required_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<String, EmbeddingError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            EmbeddingError::InvalidResponse(format!(
                "OpenAI model `{field}` field is missing or invalid"
            ))
        })
}
fn optional_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, EmbeddingError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(EmbeddingError::InvalidResponse(format!(
            "OpenAI model `{field}` field is not a string"
        ))),
    }
}
fn decode_response_body(
    bytes: &[u8],
    status: u16,
    operation: &str,
) -> Result<Value, EmbeddingError> {
    match serde_json::from_slice(bytes) {
        Ok(value) => Ok(value),
        Err(_) if !(200..300).contains(&status) => {
            Ok(Value::String(String::from_utf8_lossy(bytes).into_owned()))
        }
        Err(_) => Err(EmbeddingError::InvalidResponse(format!(
            "{operation} response body is not JSON"
        ))),
    }
}
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
#[derive(Clone, Copy)]
pub struct Embeddings<'a> {
    source: ClientSource<'a>,
    profile: &'a str,
}
impl<'a> Embeddings<'a> {
    pub(crate) fn new(source: ClientSource<'a>, profile: &'a str) -> Self {
        Self { source, profile }
    }
    pub async fn embed(
        &self,
        input: &EmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let snapshot = self.source.pin()?;
        embeddings::execute(&snapshot, self.profile, input, options).await
    }
    pub async fn list_models(
        &self,
        options: &RequestOptions,
    ) -> Result<OpenAiEmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.pin()?;
        list_openai_models(&snapshot, self.profile, options).await
    }
}
