//! Provider-specific embedding model directories.
//!
//! OpenAI, OpenRouter, and Gemini each expose different model-directory
//! shapes; their typed pages and pagination behavior remain separate.

use super::{
    apply_auth, invalid, is_first_party_openai_model_directory, valid_gemini_embedding_model_id,
    validate_route, EmbeddingApi, EmbeddingError, EmbeddingRoute, ServiceSetting,
};
use crate::{
    client::{ClientSnapshot, RequestOptions},
    protocol::{LlmError, ProviderId, ProviderProfile, Region},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fmt, time::Duration};
use url::Url;

const DEFAULT_PAGE_SIZE: i32 = 50;
const MAX_RETURNED_MODELS: i32 = 1000;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

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
        .provider(profile_name)
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

/// Query one page from Gemini's native `models.list` endpoint.
///
/// `page_size` is sent only when explicitly set on the first request. When a
/// continuation is supplied, omitting it reuses the exact original parameter
/// shape; an explicit value must match that original value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiEmbeddingModelListQuery {
    /// Optional positive native `pageSize` value. The API defaults to 50 and
    /// caps its returned page at 1000 models even when this is larger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_size: Option<i32>,
    /// Opaque, scoped continuation returned by a previous Gemini page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_token: Option<GeminiEmbeddingPageToken>,
}

/// A Gemini `models.list` continuation bound to its original query identity.
///
/// The Google page token remains opaque. This wrapper prevents accidentally
/// reusing it with a different profile, route, region, declared account scope,
/// or `pageSize` parameter shape.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiEmbeddingPageToken {
    token: String,
    page_size: Option<i32>,
    provider_id: ProviderId,
    profile_name: String,
    embedding_endpoint: String,
    models_endpoint: String,
    region: Region,
    account_scope: Option<String>,
}

impl fmt::Debug for GeminiEmbeddingPageToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiEmbeddingPageToken")
            .field("token", &"<redacted>")
            .field("page_size", &self.page_size)
            .field("provider_id", &self.provider_id)
            .field("profile_name", &self.profile_name)
            .field("embedding_endpoint", &self.embedding_endpoint)
            .field("models_endpoint", &self.models_endpoint)
            .field("region", &self.region)
            .field("account_scope", &self.account_scope)
            .finish()
    }
}

impl GeminiEmbeddingPageToken {
    /// The exact `pageSize` parameter shape captured by this continuation.
    #[must_use]
    pub fn page_size(&self) -> Option<i32> {
        self.page_size
    }
}

/// One Gemini model that explicitly advertises the `embedContent` method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingModel {
    /// Gemini API resource name, such as `models/gemini-embedding-001`.
    pub resource_name: String,
    /// Bare resource-name suffix, suitable for `EmbeddingRequest.model`.
    pub id: String,
    /// Provider-reported base model ID, retained separately from the exact
    /// resource ID because it can identify a different alias or version.
    pub base_model_id: String,
    /// Provider-reported model version.
    pub version: String,
    /// Provider-reported display name, when present.
    pub display_name: Option<String>,
    /// Provider-reported description, when present.
    pub description: Option<String>,
    /// Provider-reported input token limit, when present.
    pub input_token_limit: Option<u64>,
    /// Provider-reported output token limit, when present.
    pub output_token_limit: Option<u64>,
    /// Exact provider-reported method list.
    pub supported_generation_methods: Vec<String>,
    /// Full model object, including fields unknown to this client version.
    pub native: Value,
}

/// A page from Google's Gemini embedding model directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingModelPage {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub models: Vec<GeminiEmbeddingModel>,
    /// Present whenever Google returns `nextPageToken`, even if filtering
    /// leaves this page with no embedding models.
    pub next_page_token: Option<GeminiEmbeddingPageToken>,
    /// Full list response, including unfiltered models and unknown fields.
    pub native: Value,
}

pub(crate) async fn list_gemini_models(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    query: &GeminiEmbeddingModelListQuery,
    options: &RequestOptions,
) -> Result<GeminiEmbeddingModelPage, EmbeddingError> {
    let PreparedList {
        profile,
        route,
        endpoint,
        page_size,
        raw_page_token,
    } = prepare_list(snapshot, profile_name, query, options)?;

    let mut url = Url::parse(endpoint)
        .map_err(|_| invalid("invalid Gemini embedding model-directory URL"))?;
    if page_size.is_some() || raw_page_token.is_some() {
        let mut pairs = url.query_pairs_mut();
        if let Some(page_size) = page_size {
            pairs.append_pair("pageSize", &page_size.to_string());
        }
        if let Some(page_token) = raw_page_token {
            pairs.append_pair("pageToken", page_token);
        }
    }

    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ));
    let mut request = HttpRequest {
        method: "GET".into(),
        url: url.into(),
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
    let body = decode_response_body(&response.body, response.status, "Gemini model directory")?;
    if !(200..300).contains(&response.status) {
        return Err(EmbeddingError::Provider {
            status: response.status,
            request_id,
            body,
        });
    }

    let object = body.as_object().ok_or_else(|| {
        EmbeddingError::InvalidResponse("Gemini model-directory body is not an object".into())
    })?;
    let rows = match object.get("models") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(rows)) => rows.as_slice(),
        Some(_) => {
            return Err(EmbeddingError::InvalidResponse(
                "Gemini model-directory `models` field is not an array".into(),
            ));
        }
    };
    let wire_page_size = page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    let maximum_rows = usize::try_from(wire_page_size.min(MAX_RETURNED_MODELS)).map_err(|_| {
        EmbeddingError::InvalidResponse("Gemini pageSize is not a positive integer".into())
    })?;
    if rows.len() > maximum_rows {
        return Err(EmbeddingError::InvalidResponse(
            "Gemini model-directory page exceeds requested pageSize".into(),
        ));
    }

    let mut seen_names = BTreeSet::new();
    let mut models = Vec::new();
    for row in rows {
        let model = parse_model(row)?;
        if !seen_names.insert(model.resource_name.clone()) {
            return Err(EmbeddingError::InvalidResponse(
                "Gemini model-directory contains duplicate resource names".into(),
            ));
        }
        if model.supports_embedding() {
            models.push(model);
        }
    }

    let returned_token = optional_string(object, "nextPageToken")?;
    let next_page_token = if let Some(token) = returned_token.filter(|token| !token.is_empty()) {
        if token.trim().is_empty() || raw_page_token.is_some_and(|current| current == token) {
            return Err(EmbeddingError::InvalidResponse(
                "Gemini model directory returned an empty or repeated page token".into(),
            ));
        }
        Some(GeminiEmbeddingPageToken {
            token,
            page_size,
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            embedding_endpoint: route.endpoint.clone(),
            models_endpoint: endpoint.to_owned(),
            region: snapshot.region(),
            account_scope: options.account_scope.clone(),
        })
    } else {
        None
    };

    Ok(GeminiEmbeddingModelPage {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        models,
        next_page_token,
        native: body,
    })
}

pub(crate) async fn get_gemini_model(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    resource_name: &str,
    options: &RequestOptions,
) -> Result<GeminiEmbeddingModel, EmbeddingError> {
    let (_, route, endpoint) = prepare_route(snapshot, profile_name, options)?;
    let model_id = resource_model_id(resource_name)
        .ok_or_else(|| invalid("Gemini model resource name must be `models/{model}`"))?;
    let mut url = Url::parse(endpoint)
        .map_err(|_| invalid("invalid Gemini embedding model-directory URL"))?;
    url.path_segments_mut()
        .map_err(|_| invalid("Gemini model-directory URL cannot contain path segments"))?
        .push(model_id);

    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ));
    let mut request = HttpRequest {
        method: "GET".into(),
        url: url.into(),
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
    let body = decode_response_body(&response.body, response.status, "Gemini model")?;
    if !(200..300).contains(&response.status) {
        return Err(EmbeddingError::Provider {
            status: response.status,
            request_id,
            body,
        });
    }
    let model = parse_model(&body)?;
    if model.resource_name != resource_name {
        return Err(EmbeddingError::InvalidResponse(
            "Gemini models.get returned a different resource name".into(),
        ));
    }
    if !model.supports_embedding() {
        return Err(LlmError::UnsupportedCapability {
            message: "Gemini model does not advertise the embedContent method".into(),
        }
        .into());
    }
    Ok(model)
}

struct PreparedList<'a> {
    profile: &'a ProviderProfile,
    route: &'a EmbeddingRoute,
    endpoint: &'a str,
    page_size: Option<i32>,
    raw_page_token: Option<&'a str>,
}

fn prepare_list<'a>(
    snapshot: &'a ClientSnapshot,
    profile_name: &str,
    query: &'a GeminiEmbeddingModelListQuery,
    options: &RequestOptions,
) -> Result<PreparedList<'a>, EmbeddingError> {
    let (profile, route, endpoint) = prepare_route(snapshot, profile_name, options)?;
    if query.page_size.is_some_and(|size| size <= 0) {
        return Err(invalid("Gemini pageSize must be a positive integer").into());
    }

    let (page_size, raw_page_token) = match query.page_token.as_ref() {
        None => (query.page_size, None),
        Some(cursor) => {
            validate_page_token(cursor, profile, route, endpoint, snapshot.region(), options)?;
            if query
                .page_size
                .is_some_and(|page_size| Some(page_size) != cursor.page_size)
            {
                return Err(invalid(
                    "Gemini pageSize must match the parameter shape captured by the page token",
                )
                .into());
            }
            (cursor.page_size, Some(cursor.token.as_str()))
        }
    };
    Ok(PreparedList {
        profile,
        route,
        endpoint,
        page_size,
        raw_page_token,
    })
}

fn prepare_route<'a>(
    snapshot: &'a ClientSnapshot,
    profile_name: &str,
    options: &RequestOptions,
) -> Result<(&'a ProviderProfile, &'a EmbeddingRoute, &'a str), EmbeddingError> {
    let profile = snapshot
        .provider(profile_name)
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
    if route.api != EmbeddingApi::Gemini {
        return Err(LlmError::UnsupportedCapability {
            message: "Gemini model discovery requires a Gemini embedding route".into(),
        }
        .into());
    }
    validate_route(route)?;
    let endpoint =
        route
            .models_endpoint
            .as_deref()
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: "Gemini embedding model-directory route is not configured".into(),
            })?;
    if options
        .account_scope
        .as_deref()
        .is_some_and(|scope| scope.trim().is_empty())
    {
        return Err(invalid("account_scope must not be empty or whitespace").into());
    }
    Ok((profile, route, endpoint))
}

fn validate_page_token(
    cursor: &GeminiEmbeddingPageToken,
    profile: &ProviderProfile,
    route: &EmbeddingRoute,
    endpoint: &str,
    region: Region,
    options: &RequestOptions,
) -> Result<(), EmbeddingError> {
    if cursor.token.trim().is_empty() || cursor.page_size.is_some_and(|size| size <= 0) {
        return Err(invalid("invalid Gemini embedding page token").into());
    }
    if cursor.provider_id != profile.provider_id
        || cursor.profile_name != profile.profile_name
        || cursor.embedding_endpoint != route.endpoint
        || cursor.models_endpoint != endpoint
        || cursor.region != region
        || cursor.account_scope != options.account_scope
    {
        return Err(invalid(
            "Gemini embedding page token does not match the selected profile, route, region, or declared account scope",
        )
        .into());
    }
    Ok(())
}

impl GeminiEmbeddingModel {
    fn supports_embedding(&self) -> bool {
        self.supported_generation_methods
            .iter()
            .any(|method| method == "embedContent")
    }
}

fn parse_model(value: &Value) -> Result<GeminiEmbeddingModel, EmbeddingError> {
    let object = value.as_object().ok_or_else(|| {
        EmbeddingError::InvalidResponse("Gemini model entry is not an object".into())
    })?;
    let resource_name = required_string(object, "name")?;
    let id = resource_model_id(&resource_name)
        .ok_or_else(|| {
            EmbeddingError::InvalidResponse(
                "Gemini model `name` is not a `models/{model}` resource name".into(),
            )
        })?
        .to_owned();
    let base_model_id = required_string(object, "baseModelId")?;
    let version = required_string(object, "version")?;
    if version.trim().is_empty() {
        return Err(EmbeddingError::InvalidResponse(
            "Gemini model version is empty".into(),
        ));
    }
    let display_name = optional_string(object, "displayName")?;
    let description = optional_string(object, "description")?;
    let input_token_limit = optional_u64(object, "inputTokenLimit")?;
    let output_token_limit = optional_u64(object, "outputTokenLimit")?;
    let supported_generation_methods = optional_string_array(object, "supportedGenerationMethods")?;
    Ok(GeminiEmbeddingModel {
        resource_name,
        id,
        base_model_id,
        version,
        display_name,
        description,
        input_token_limit,
        output_token_limit,
        supported_generation_methods,
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
                "Gemini model `{field}` field is missing or invalid"
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
            "Gemini model `{field}` field is not a string"
        ))),
    }
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, EmbeddingError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            EmbeddingError::InvalidResponse(format!(
                "Gemini model `{field}` field is not a nonnegative integer"
            ))
        }),
    }
}

fn optional_string_array(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, EmbeddingError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    EmbeddingError::InvalidResponse(format!(
                        "Gemini model `{field}` contains a non-string value"
                    ))
                })
            })
            .collect(),
        Some(_) => Err(EmbeddingError::InvalidResponse(format!(
            "Gemini model `{field}` field is not an array"
        ))),
    }
}

fn resource_model_id(resource_name: &str) -> Option<&str> {
    resource_name
        .strip_prefix("models/")
        .filter(|id| valid_gemini_embedding_model_id(id))
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
