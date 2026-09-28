use crate::client::{ClientSnapshot, ClientSource, RequestOptions};
use crate::embeddings::{self, backend::*, *};
use crate::protocol::{LlmError, ProviderId, ProviderProfile, Region, UsageState};
use crate::runtime::Deadline;
use crate::transport::{HttpExecutor, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, time::Duration};

use base64::Engine;
use std::fmt;
use url::Url;
/// One part of a `gemini-embedding-2` content item.
///
/// Gemini combines every part in this value into one embedding. For separate
/// vectors, make separate service calls or use the Gemini Batch API.
#[derive(Debug, Clone, PartialEq)]
pub enum GeminiEmbeddingPart {
    Text(String),
    Media(GeminiEmbeddingMedia),
}

/// Media supported by the Gemini Embedding 2 endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiEmbeddingMedia {
    /// Supported MIME types are image/png, image/jpeg, audio/mpeg, audio/wav,
    /// video/mp4, video/quicktime, and application/pdf.
    pub mime_type: String,
    pub source: GeminiEmbeddingSource,
    /// Required for audio and video, and rounded up to whole seconds by the
    /// caller. It is local validation metadata and is not sent on the wire.
    pub duration_seconds: Option<u32>,
    /// Required for PDF. It is local validation metadata and is not sent on
    /// the wire.
    pub page_count: Option<u8>,
}

/// Inline bytes or a URI already uploaded to the caller's Gemini Files API
/// account. This client does not upload, download, or poll media.
#[derive(Debug, Clone, PartialEq)]
pub enum GeminiEmbeddingSource {
    Inline(Vec<u8>),
    FileUri(String),
}

/// A single Gemini Embedding 2 input. All parts produce one aggregated vector.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiMultimodalEmbeddingRequest {
    pub model: String,
    pub parts: Vec<GeminiEmbeddingPart>,
    pub dimensions: Option<usize>,
}
pub(crate) struct GoogleEmbeddings;
impl EmbeddingBackend for GoogleEmbeddings {
    fn encode(
        &self,
        _profile: &ProviderProfile,
        route: &EmbeddingRoute,
        req: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError> {
        validate_input(route, req)?;
        let mut endpoint = route.endpoint.clone();
        let body = {
            if !valid_gemini_embedding_model_id(&req.model) || !endpoint.contains("{model}") {
                return Err(invalid(
                    "Gemini embedding route requires a model path placeholder and a bare model ID",
                ));
            }
            if matches!(
                req.model.as_str(),
                "gemini-embedding-001" | "gemini-embedding-2"
            ) && req
                .dimensions
                .is_some_and(|dimensions| !(128..=3072).contains(&dimensions))
            {
                return Err(invalid(
                    "Gemini embedding output dimensions must be between 128 and 3072",
                ));
            }
            if req.model == "gemini-embedding-2" && req.task.is_some() {
                return Err(invalid(
                    "Gemini Embedding 2 does not support taskType; include a task instruction in text parts instead",
                ));
            }
            endpoint = endpoint.replace("{model}", &req.model);
            let mut config = json!({});
            if matches!(
                req.model.as_str(),
                "gemini-embedding-001" | "gemini-embedding-2"
            ) {
                // Keep the provider from silently truncating either model's
                // documented input-token limit.
                config["autoTruncate"] = json!(false);
            }
            if let Some(dim) = req.dimensions {
                config["outputDimensionality"] = json!(dim);
            }
            if let Some(task) = req.task {
                config["taskType"] = json!(match task {
                    EmbeddingTask::RetrievalQuery => "RETRIEVAL_QUERY",
                    EmbeddingTask::RetrievalDocument => "RETRIEVAL_DOCUMENT",
                    EmbeddingTask::SemanticSimilarity => "SEMANTIC_SIMILARITY",
                    EmbeddingTask::Classification => "CLASSIFICATION",
                    EmbeddingTask::Clustering => "CLUSTERING",
                });
            }
            json!({"requests":req.input.iter().map(|text|json!({"model":format!("models/{}",req.model),"content":{"parts":[{"text":text}]},"embedContentConfig":config})).collect::<Vec<_>>()})
        };
        post(endpoint, body)
    }
    fn vectors(
        &self,
        body: &Value,
        req: &EmbeddingRequest,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
        decode_vectors(&body["embeddings"], None, "values", req)
    }
    fn usage(&self, body: &Value) -> EmbeddingUsage {
        usage(
            body["usageMetadata"].clone(),
            "promptTokenCount",
            "totalTokenCount",
        )
    }
}
pub(crate) fn valid_gemini_embedding_model_id(model_id: &str) -> bool {
    !model_id.is_empty()
        && model_id != "."
        && model_id != ".."
        && model_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}
fn encode_gemini_multimodal(
    route: &EmbeddingRoute,
    req: &GeminiMultimodalEmbeddingRequest,
) -> Result<HttpRequest, LlmError> {
    if req.model != "gemini-embedding-2" {
        return Err(invalid(
            "Gemini multimodal embeddings require the gemini-embedding-2 model",
        ));
    }
    if req.parts.is_empty() {
        return Err(invalid(
            "Gemini embedding content must contain at least one part",
        ));
    }
    if req
        .dimensions
        .is_some_and(|dimensions| !(128..=3072).contains(&dimensions))
    {
        return Err(invalid(
            "Gemini embedding output dimensions must be between 128 and 3072",
        ));
    }
    if !valid_gemini_embedding_model_id(&req.model) || !route.endpoint.contains("{model}") {
        return Err(invalid(
            "Gemini embedding route requires a model path placeholder and a bare model ID",
        ));
    }
    validate_route(route)?;

    let mut image_count = 0usize;
    let mut pdf_count = 0usize;
    let mut encoded_parts = Vec::with_capacity(req.parts.len());
    let mut inline_size = 0usize;
    for part in &req.parts {
        match part {
            GeminiEmbeddingPart::Text(text) => {
                if text.trim().is_empty() {
                    return Err(invalid("Gemini embedding text parts cannot be empty"));
                }
                inline_size = inline_size
                    .checked_add(text.len())
                    .ok_or_else(|| invalid("Gemini embedding content is too large"))?;
                encoded_parts.push(json!({"text":text}));
            }
            GeminiEmbeddingPart::Media(media) => {
                let mime = media.mime_type.to_ascii_lowercase();
                let is_image = matches!(mime.as_str(), "image/png" | "image/jpeg");
                let is_audio = matches!(mime.as_str(), "audio/mpeg" | "audio/wav");
                let is_video = matches!(mime.as_str(), "video/mp4" | "video/quicktime");
                let is_pdf = mime == "application/pdf";
                if !is_image && !is_audio && !is_video && !is_pdf {
                    return Err(invalid(
                        "Gemini Embedding 2 supports PNG/JPEG images, MP3/WAV audio, MP4/MOV video, and PDF",
                    ));
                }
                if is_image {
                    image_count += 1;
                    if image_count > 6 {
                        return Err(invalid(
                            "Gemini Embedding 2 accepts at most 6 images per request",
                        ));
                    }
                    if media.duration_seconds.is_some() || media.page_count.is_some() {
                        return Err(invalid(
                            "duration and page-count metadata are not valid for images",
                        ));
                    }
                } else if is_audio {
                    validate_duration(media.duration_seconds, 180, "audio")?;
                    if media.page_count.is_some() {
                        return Err(invalid("page-count metadata is not valid for audio"));
                    }
                } else if is_video {
                    validate_duration(media.duration_seconds, 120, "video")?;
                    if media.page_count.is_some() {
                        return Err(invalid("page-count metadata is not valid for video"));
                    }
                } else {
                    pdf_count += 1;
                    if pdf_count > 1 {
                        return Err(invalid(
                            "Gemini Embedding 2 accepts at most one PDF per request",
                        ));
                    }
                    if media.duration_seconds.is_some()
                        || !media
                            .page_count
                            .is_some_and(|pages| (1..=6).contains(&pages))
                    {
                        return Err(invalid(
                            "PDF input requires a page count between 1 and 6 and no duration",
                        ));
                    }
                }
                let data = match &media.source {
                    GeminiEmbeddingSource::Inline(bytes) => {
                        if bytes.is_empty() {
                            return Err(invalid("inline Gemini embedding media cannot be empty"));
                        }
                        inline_size = inline_size
                            .checked_add(bytes.len())
                            .ok_or_else(|| invalid("Gemini embedding content is too large"))?;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                        json!({"inline_data":{"mime_type":mime,"data":encoded}})
                    }
                    GeminiEmbeddingSource::FileUri(uri) => {
                        validate_gemini_file_uri(uri)?;
                        json!({"file_data":{"mime_type":mime,"file_uri":uri}})
                    }
                };
                encoded_parts.push(data);
            }
        }
    }
    if inline_size > 64 * 1024 * 1024 {
        return Err(invalid("Gemini embedding content exceeds 64 MiB"));
    }

    let endpoint = route.endpoint.replace("{model}", &req.model);
    if !endpoint.ends_with(":batchEmbedContents") {
        return Err(invalid(
            "Gemini multimodal embeddings require a batchEmbedContents route",
        ));
    }
    let mut embed_config = json!({"autoTruncate":false});
    if let Some(dimensions) = req.dimensions {
        embed_config["outputDimensionality"] = json!(dimensions);
    }
    let mut embed_request = json!({
        "model": format!("models/{}", req.model),
        "content": {"parts": encoded_parts},
    });
    embed_request["embedContentConfig"] = embed_config;
    let body = json!({"requests":[embed_request]});
    let parsed =
        url::Url::parse(&endpoint).map_err(|_| invalid("invalid Gemini embedding endpoint"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid(
            "Gemini embedding endpoint must be an HTTP URL without credentials, query or fragment",
        ));
    }
    let body = serde_json::to_vec(&body)
        .map_err(|_| invalid("Gemini embedding request cannot be serialized"))?;
    if body.len() > 64 * 1024 * 1024 {
        return Err(invalid("Gemini embedding request exceeds 64 MiB"));
    }
    Ok(HttpRequest {
        method: "POST".into(),
        url: endpoint,
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.into(),
        timeout: None,
    })
}

fn validate_duration(
    duration_seconds: Option<u32>,
    maximum: u32,
    kind: &str,
) -> Result<(), LlmError> {
    if !duration_seconds.is_some_and(|duration| (1..=maximum).contains(&duration)) {
        return Err(invalid(format!(
            "Gemini Embedding 2 {kind} requires a declared duration from 1 to {maximum} seconds"
        )));
    }
    Ok(())
}

fn validate_gemini_file_uri(uri: &str) -> Result<(), LlmError> {
    let parsed = url::Url::parse(uri).map_err(|_| invalid("invalid Gemini Files API URI"))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || parsed.path().is_empty()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid(
            "Gemini Files API URI must be HTTPS without credentials, query or fragment",
        ));
    }
    Ok(())
}

const DEFAULT_PAGE_SIZE: i32 = 50;
const MAX_RETURNED_MODELS: i32 = 1000;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

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
pub(crate) async fn embed_multimodal(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    input: &GeminiMultimodalEmbeddingRequest,
    options: &RequestOptions,
) -> Result<EmbeddingResponse, EmbeddingError> {
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
    if route.api != EmbeddingApi::Gemini {
        return Err(LlmError::UnsupportedCapability {
            message: "multimodal embedding adapter is only available for Gemini".into(),
        }
        .into());
    }
    let mut request = encode_gemini_multimodal(route, input)?;
    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ));
    apply_auth(&route.auth, options, &mut request)?;
    request.timeout = deadline.remaining()?;
    let response = HttpExecutor::new(snapshot.runtime.http.as_ref())
        .with_deadline(deadline)
        .execute_bounded(request, 64 * 1024 * 1024)
        .await?;
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let body: Value = match serde_json::from_slice(&response.body) {
        Ok(body) => body,
        Err(_) if !(200..300).contains(&response.status) => {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }
        Err(_) => return Err(EmbeddingError::InvalidResponse("body is not JSON".into())),
    };
    if !(200..300).contains(&response.status) {
        return Err(EmbeddingError::Provider {
            status: response.status,
            request_id,
            body,
        });
    }
    let response_request = EmbeddingRequest {
        model: input.model.clone(),
        input: vec![String::new()],
        dimensions: input.dimensions,
        task: None,
    };
    let vectors = GoogleEmbeddings.vectors(&body, &response_request)?;
    let native = body["usageMetadata"].clone();
    let input_tokens = native["promptTokenCount"].as_u64();
    let total_tokens = native["totalTokenCount"].as_u64();
    let invalid_usage = (!native.is_null() && !native.is_object())
        || ["promptTokenCount", "totalTokenCount"].iter().any(|key| {
            native
                .get(*key)
                .is_some_and(|value| value.as_u64().is_none())
        })
        || input_tokens
            .zip(total_tokens)
            .is_some_and(|(input, total)| input > total);
    let state = if invalid_usage {
        UsageState::Invalid
    } else if native.is_null() {
        UsageState::Missing
    } else if input_tokens.is_some() && total_tokens.is_some() {
        UsageState::Complete
    } else {
        UsageState::Partial
    };
    Ok(EmbeddingResponse {
        provider_id: profile.provider_id.clone(),
        executed_profile: profile.profile_name.clone(),
        requested_model: input.model.clone(),
        model: body["model"].as_str().map(str::to_owned),
        request_id: request_id.or_else(|| body["request_id"].as_str().map(str::to_owned)),
        vectors,
        usage: EmbeddingUsage {
            state,
            input_tokens,
            total_tokens,
            native,
        },
        response_cache: None,
    })
}
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
    pub async fn embed_multimodal(
        &self,
        input: &GeminiMultimodalEmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let snapshot = self.source.pin()?;
        embed_multimodal(&snapshot, self.profile, input, options).await
    }
    pub async fn list_models(
        &self,
        query: &GeminiEmbeddingModelListQuery,
        options: &RequestOptions,
    ) -> Result<GeminiEmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.pin()?;
        list_gemini_models(&snapshot, self.profile, query, options).await
    }
    pub async fn get_model(
        &self,
        resource_name: &str,
        options: &RequestOptions,
    ) -> Result<GeminiEmbeddingModel, EmbeddingError> {
        let snapshot = self.source.pin()?;
        get_gemini_model(&snapshot, self.profile, resource_name, options).await
    }
}
