//! Independent embedding routes. No automatic splitting or vector normalization.
mod catalog;
pub use crate::protocol::{ServiceAuth, ServiceSetting};
pub use catalog::{
    GeminiEmbeddingModel, GeminiEmbeddingModelListQuery, GeminiEmbeddingModelPage,
    GeminiEmbeddingPageToken, OpenAiEmbeddingModel, OpenAiEmbeddingModelPage,
};

use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    protocol::{LlmError, ProviderId, UsageState},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingApi {
    OpenAi,
    OpenRouter,
    Gemini,
    Qwen,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRoute {
    pub api: EmbeddingApi,
    /// Complete operation URL. Gemini uses a literal `{model}` path placeholder.
    pub endpoint: String,
    pub auth: ServiceAuth,
    /// Optional independent model-directory endpoint. Only documented
    /// provider directories are enabled; it is never inferred from Chat.
    #[serde(default)]
    pub models_endpoint: Option<String>,
    /// Optional explicit limit; exceeding it fails without splitting the request.
    #[serde(default)]
    pub max_inputs: Option<usize>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingTask {
    RetrievalQuery,
    RetrievalDocument,
    SemanticSimilarity,
    Classification,
    Clustering,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub task: Option<EmbeddingTask>,
}

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingVector {
    pub index: usize,
    pub values: Vec<f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub state: UsageState,
    pub input_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub native: Value,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub provider_id: ProviderId,
    pub executed_profile: String,
    pub requested_model: String,
    /// Absent when the provider does not report the executed model.
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub vectors: Vec<EmbeddingVector>,
    pub usage: EmbeddingUsage,
    /// Explicit OpenRouter gateway response-cache observation, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_cache: Option<crate::protocol::ResponseCacheObservation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
    pub input_modalities: Vec<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingModelPage {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub models: Vec<EmbeddingModel>,
    pub total_count: Option<u64>,
    pub next_offset: Option<u64>,
    pub native: Value,
}
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid embedding response: {0}")]
    InvalidResponse(String),
    #[error("embedding provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
}
/// A service handle whose live configuration is captured once per operation.
#[derive(Clone, Copy)]
pub struct EmbeddingService<'a> {
    source: ClientSource<'a>,
}

impl<'a> EmbeddingService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn embed(
        self,
        profile_name: &str,
        input: &EmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let snapshot = self.source.snapshot();
        PinnedEmbeddingService { client: &snapshot }
            .embed(profile_name, input, options)
            .await
    }

    /// Embed one multimodal Gemini Embedding 2 content item into one vector.
    /// Text and media parts are aggregated as documented by Gemini. Uploaded
    /// file URIs remain caller-owned and are passed through as-is.
    pub async fn embed_gemini_multimodal(
        self,
        profile_name: &str,
        input: &GeminiMultimodalEmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let snapshot = self.source.snapshot();
        PinnedEmbeddingService { client: &snapshot }
            .embed_gemini_multimodal(profile_name, input, options)
            .await
    }

    /// List one offset-based page from OpenRouter's embedding model directory.
    /// Gemini uses [`Self::list_gemini_models`] because its native page-token
    /// contract is not interchangeable with OpenRouter offsets.
    pub async fn list_models(
        self,
        profile_name: &str,
        offset: u64,
        limit: u16,
        options: &RequestOptions,
    ) -> Result<EmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.snapshot();
        PinnedEmbeddingService { client: &snapshot }
            .list_models(profile_name, offset, limit, options)
            .await
    }

    /// List the currently available, explicitly documented OpenAI embedding
    /// models from the first-party Models API. The native response retains
    /// every model row, including IDs this client does not classify as an
    /// embedding model.
    pub async fn list_openai_models(
        self,
        profile_name: &str,
        options: &RequestOptions,
    ) -> Result<OpenAiEmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.snapshot();
        catalog::list_openai_models(&snapshot, profile_name, options).await
    }

    /// List one page from Google's independent Gemini model directory,
    /// retaining its native `pageToken` continuation contract.
    pub async fn list_gemini_models(
        self,
        profile_name: &str,
        query: &GeminiEmbeddingModelListQuery,
        options: &RequestOptions,
    ) -> Result<GeminiEmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.snapshot();
        catalog::list_gemini_models(&snapshot, profile_name, query, options).await
    }

    /// Retrieve one Google Gemini model resource. Models without the exact
    /// `embedContent` method are rejected as unsupported by this service.
    pub async fn get_gemini_model(
        self,
        profile_name: &str,
        resource_name: &str,
        options: &RequestOptions,
    ) -> Result<GeminiEmbeddingModel, EmbeddingError> {
        let snapshot = self.source.snapshot();
        catalog::get_gemini_model(&snapshot, profile_name, resource_name, options).await
    }
}

#[derive(Clone, Copy)]
struct PinnedEmbeddingService<'a> {
    client: &'a ClientSnapshot,
}

impl<'a> PinnedEmbeddingService<'a> {
    async fn embed_gemini_multimodal(
        self,
        profile_name: &str,
        input: &GeminiMultimodalEmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown embedding profile"))?;
        if !profile.supports_region(self.client.region()) {
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
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
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
        let vectors = decode_vectors(EmbeddingApi::Gemini, &body, &response_request)?;
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

    async fn list_models(
        self,
        profile_name: &str,
        offset: u64,
        limit: u16,
        options: &RequestOptions,
    ) -> Result<EmbeddingModelPage, EmbeddingError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown embedding profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("embedding profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.embeddings else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled embedding route".into(),
            }
            .into());
        };
        if route.api != EmbeddingApi::OpenRouter {
            return Err(LlmError::UnsupportedCapability {
                message: "this embedding adapter has no model-directory contract".into(),
            }
            .into());
        }
        let endpoint =
            route
                .models_endpoint
                .as_deref()
                .ok_or_else(|| LlmError::UnsupportedCapability {
                    message: "embedding model-directory route is not configured".into(),
                })?;
        if !(1..=1000).contains(&limit) {
            return Err(invalid("embedding model-directory limit must be 1–1000").into());
        }
        let mut url = url::Url::parse(endpoint)
            .map_err(|_| invalid("invalid embedding model-directory URL"))?;
        url.query_pairs_mut()
            .append_pair("offset", &offset.to_string())
            .append_pair("limit", &limit.to_string());
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
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, 64 * 1024 * 1024)
            .await?;
        let body: Value = match serde_json::from_slice(&response.body) {
            Ok(value) => value,
            Err(_) if !(200..300).contains(&response.status) => {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            }
            Err(_) => {
                return Err(EmbeddingError::InvalidResponse(
                    "model-directory body is not JSON".into(),
                ));
            }
        };
        if !(200..300).contains(&response.status) {
            return Err(EmbeddingError::Provider {
                status: response.status,
                request_id: response.header("x-request-id").map(str::to_owned),
                body,
            });
        }
        let rows = body["data"].as_array().ok_or_else(|| {
            EmbeddingError::InvalidResponse("model-directory data is missing".into())
        })?;
        if rows.len() > usize::from(limit) {
            return Err(EmbeddingError::InvalidResponse(
                "model-directory page exceeds requested limit".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        let models = rows
            .iter()
            .map(|row| {
                let id = row["id"]
                    .as_str()
                    .filter(|id| !id.trim().is_empty())
                    .ok_or_else(|| {
                        EmbeddingError::InvalidResponse("model-directory row has no id".into())
                    })?;
                if !seen.insert(id) {
                    return Err(EmbeddingError::InvalidResponse(
                        "model-directory contains duplicate IDs".into(),
                    ));
                }
                let modalities = row["architecture"]["input_modalities"]
                    .as_array()
                    .ok_or_else(|| {
                        EmbeddingError::InvalidResponse(
                            "model-directory row has no input_modalities".into(),
                        )
                    })?
                    .iter()
                    .map(|item| {
                        item.as_str().map(str::to_owned).ok_or_else(|| {
                            EmbeddingError::InvalidResponse(
                                "model-directory modality is invalid".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(EmbeddingModel {
                    id: id.into(),
                    name: row["name"].as_str().map(str::to_owned),
                    context_length: row["context_length"].as_u64(),
                    input_modalities: modalities,
                    native: row.clone(),
                })
            })
            .collect::<Result<Vec<_>, EmbeddingError>>()?;
        let total_count = match body.get("total_count") {
            None | Some(Value::Null) => None,
            Some(value) => Some(value.as_u64().ok_or_else(|| {
                EmbeddingError::InvalidResponse("model-directory total_count is invalid".into())
            })?),
        };
        let next = offset.checked_add(rows.len() as u64);
        if total_count
            .zip(next)
            .is_some_and(|(total, next)| next > total)
        {
            return Err(EmbeddingError::InvalidResponse(
                "model-directory page exceeds total_count".into(),
            ));
        }
        Ok(EmbeddingModelPage {
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            models,
            total_count,
            next_offset: next
                .filter(|next| !rows.is_empty() && total_count.is_some_and(|total| *next < total)),
            native: body,
        })
    }

    async fn embed(
        self,
        profile_name: &str,
        input: &EmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown embedding profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("embedding profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.embeddings else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled embedding route".into(),
            }
            .into());
        };
        let mut request = encode(&profile.provider_id, route, input)?;
        if options.openrouter_response_cache.is_some() {
            if route.api != EmbeddingApi::OpenRouter
                || request.url != "https://openrouter.ai/api/v1/embeddings"
            {
                return Err(LlmError::UnsupportedCapability {
                    message: "embedding response caching requires the official OpenRouter Embeddings route".into(),
                }
                .into());
            }
            crate::client::apply_openrouter_response_cache(
                options.openrouter_response_cache,
                profile,
                &mut request,
            )?;
        }
        let request_url = request.url.clone();
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        apply_auth(&route.auth, options, &mut request)?;
        request.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, 64 * 1024 * 1024)
            .await?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let response_cache =
            crate::client::openrouter_observation(profile, &request_url, &response.headers);
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
        let vectors = decode_vectors(route.api, &body, input)?;
        let native = match route.api {
            EmbeddingApi::Gemini => body["usageMetadata"].clone(),
            _ => body["usage"].clone(),
        };
        let (input_tokens, total_tokens) = match route.api {
            EmbeddingApi::Gemini => (
                native["promptTokenCount"].as_u64(),
                native["totalTokenCount"].as_u64(),
            ),
            EmbeddingApi::Qwen => (
                native["total_tokens"].as_u64(),
                native["total_tokens"].as_u64(),
            ),
            EmbeddingApi::OpenAi | EmbeddingApi::OpenRouter => (
                native["prompt_tokens"].as_u64(),
                native["total_tokens"].as_u64(),
            ),
        };
        let token_keys: &[&str] = match route.api {
            EmbeddingApi::OpenAi | EmbeddingApi::OpenRouter => &["prompt_tokens", "total_tokens"],
            EmbeddingApi::Qwen => &["total_tokens"],
            EmbeddingApi::Gemini => &["promptTokenCount", "totalTokenCount"],
        };
        let invalid_usage = (!native.is_null() && !native.is_object())
            || token_keys
                .iter()
                .any(|key| native.get(*key).is_some_and(|v| v.as_u64().is_none()))
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
            response_cache,
        })
    }
}
fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

fn apply_auth(
    auth: &ServiceAuth,
    options: &RequestOptions,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    match auth {
        ServiceAuth::None => Ok(()),
        ServiceAuth::Bearer | ServiceAuth::ApiKey { .. } => {
            let secret = options
                .credential
                .as_ref()
                .ok_or_else(|| LlmError::Authentication {
                    message: "embedding route requires a credential".into(),
                })?;
            let (header, value) = match auth {
                ServiceAuth::Bearer => (
                    "authorization",
                    format!("Bearer {}", secret.expose_secret()),
                ),
                ServiceAuth::ApiKey { header } => (header.as_str(), secret.expose_secret().clone()),
                ServiceAuth::None => unreachable!(),
            };
            request.headers.push((header.into(), value));
            Ok(())
        }
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

fn encode(
    provider_id: &ProviderId,
    route: &EmbeddingRoute,
    req: &EmbeddingRequest,
) -> Result<HttpRequest, LlmError> {
    if req.model.trim().is_empty()
        || req.input.is_empty()
        || req.input.iter().any(String::is_empty)
        || req.dimensions == Some(0)
    {
        return Err(invalid(
            "embedding model, nonempty inputs and positive dimensions are required",
        ));
    }
    if route.max_inputs.is_some_and(|max| req.input.len() > max) {
        return Err(invalid(
            "embedding request exceeds route input limit; the caller must split it",
        ));
    }
    if is_first_party_openai_embedding_route(provider_id, route) && req.input.len() > 2048 {
        return Err(invalid(
            "OpenAI Embeddings accepts at most 2048 inputs per request",
        ));
    }
    if is_first_party_openai_embedding_route(provider_id, route)
        && req.model == "text-embedding-ada-002"
        && req.dimensions.is_some()
    {
        return Err(invalid(
            "text-embedding-ada-002 does not support the dimensions parameter",
        ));
    }
    if let Some(limits) = documented_model_limits(provider_id, route, &req.model) {
        if req.input.len() > limits.max_inputs {
            return Err(invalid(format!(
                "embedding model {} accepts at most {} inputs; the caller must split it",
                req.model, limits.max_inputs
            )));
        }
        if let Some(dimensions) = req.dimensions {
            match limits.dimensions {
                Some(allowed) if allowed.contains(&dimensions) => {}
                Some(_) => {
                    return Err(invalid(format!(
                        "embedding model {} does not support output dimension {dimensions}",
                        req.model
                    )));
                }
                None => {
                    return Err(invalid(format!(
                        "embedding model {} does not support a custom output dimension",
                        req.model
                    )));
                }
            }
        }
    }
    validate_route(route)?;
    let mut endpoint = route.endpoint.clone();
    if req
        .input
        .iter()
        .try_fold(0usize, |total, text| total.checked_add(text.len()))
        .is_none_or(|size| size > 64 * 1024 * 1024)
    {
        return Err(invalid("embedding input exceeds 64 MiB"));
    }
    let body = match route.api {
        EmbeddingApi::OpenAi => {
            if req.task.is_some() {
                return Err(LlmError::UnsupportedCapability {
                    message: "OpenAI embedding wire cannot encode task type".into(),
                });
            }
            let mut body = json!({"model":req.model,"input":req.input,"encoding_format":"float"});
            if let Some(dim) = req.dimensions {
                body["dimensions"] = json!(dim);
            }
            body
        }
        EmbeddingApi::OpenRouter => {
            let mut body = json!({"model":req.model,"input":req.input,"encoding_format":"float"});
            if let Some(dim) = req.dimensions {
                body["dimensions"] = json!(dim);
            }
            if let Some(task) = req.task {
                body["input_type"] = json!(match task {
                    EmbeddingTask::RetrievalQuery => "search_query",
                    EmbeddingTask::RetrievalDocument => "search_document",
                    _ => return Err(LlmError::UnsupportedCapability {
                        message: "OpenRouter embedding input_type supports retrieval query or document in this client".into(),
                    }),
                });
            }
            body
        }
        EmbeddingApi::Gemini => {
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
        }
        EmbeddingApi::Qwen => {
            let mut parameters = json!({});
            if let Some(dim) = req.dimensions {
                parameters["dimension"] = json!(dim);
            }
            if let Some(task) = req.task {
                parameters["text_type"] = json!(match task {
                    EmbeddingTask::RetrievalQuery => "query",
                    EmbeddingTask::RetrievalDocument => "document",
                    _ =>
                        return Err(LlmError::UnsupportedCapability {
                            message: "Qwen text embedding task must be retrieval query or document"
                                .into()
                        }),
                });
            }
            json!({"model":req.model,"input":{"texts":req.input},"parameters":parameters})
        }
    };
    let parsed = url::Url::parse(&endpoint).map_err(|_| invalid("invalid embedding endpoint"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid(
            "embedding endpoint must be an HTTP URL without credentials, query or fragment",
        ));
    }
    let body =
        serde_json::to_vec(&body).map_err(|_| invalid("embedding request cannot be serialized"))?;
    if body.len() > 64 * 1024 * 1024 {
        return Err(invalid("embedding request exceeds 64 MiB"));
    }
    Ok(HttpRequest {
        method: "POST".into(),
        url: endpoint,
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.into(),
        timeout: None,
    })
}

struct DocumentedModelLimits {
    max_inputs: usize,
    /// `None` means the provider explicitly excludes this model from the
    /// custom-dimension parameter. Unknown models have no
    /// `DocumentedModelLimits` entry at all.
    dimensions: Option<&'static [usize]>,
}

fn documented_model_limits(
    provider_id: &ProviderId,
    route: &EmbeddingRoute,
    model: &str,
) -> Option<DocumentedModelLimits> {
    const QWEN_37: &[usize] = &[2560, 2048, 1536, 1024, 768, 512, 256];
    const QWEN_37_FLASH: &[usize] = &[1024, 768, 512, 256];
    const QWEN_V4: &[usize] = &[2048, 1536, 1024, 768, 512, 256, 128, 64];
    const QWEN_V3: &[usize] = &[1024, 768, 512, 256, 128, 64];
    const GLM_EMBEDDING_3: &[usize] = &[2048, 1024, 512, 256];

    if provider_id.as_str() == "qwen" && route.api == EmbeddingApi::Qwen {
        return match model {
            "qwen3.7-text-embedding" => Some(DocumentedModelLimits {
                max_inputs: 20,
                dimensions: Some(QWEN_37),
            }),
            "qwen3.7-text-embedding-flash" => Some(DocumentedModelLimits {
                max_inputs: 20,
                dimensions: Some(QWEN_37_FLASH),
            }),
            "text-embedding-v4" => Some(DocumentedModelLimits {
                max_inputs: 10,
                dimensions: Some(QWEN_V4),
            }),
            "text-embedding-v3" => Some(DocumentedModelLimits {
                max_inputs: 10,
                dimensions: Some(QWEN_V3),
            }),
            "text-embedding-v1" | "text-embedding-v2" => Some(DocumentedModelLimits {
                max_inputs: 25,
                dimensions: None,
            }),
            _ => None,
        };
    }

    // BigModel's GLM route uses the OpenAI wire format. Match the provider's
    // documented operation URL so these model-specific limits do not leak to
    // unrelated OpenAI-compatible services.
    if provider_id.as_str() == "zhipu"
        && route.api == EmbeddingApi::OpenAi
        && route.endpoint == "https://open.bigmodel.cn/api/paas/v4/embeddings"
        && model == "embedding-3"
    {
        return Some(DocumentedModelLimits {
            max_inputs: 64,
            dimensions: Some(GLM_EMBEDDING_3),
        });
    }

    None
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

fn decode_vectors(
    api: EmbeddingApi,
    body: &Value,
    req: &EmbeddingRequest,
) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
    let error = |msg: &str| EmbeddingError::InvalidResponse(msg.into());
    let array = match api {
        EmbeddingApi::OpenAi | EmbeddingApi::OpenRouter => &body["data"],
        EmbeddingApi::Gemini => &body["embeddings"],
        EmbeddingApi::Qwen => &body["output"]["embeddings"],
    }
    .as_array()
    .ok_or_else(|| error("missing embeddings"))?;
    if array.len() != req.input.len() {
        return Err(error("embedding count differs from input count"));
    }
    let mut slots = vec![None; req.input.len()];
    let mut dimensions = req.dimensions;
    for (position, item) in array.iter().enumerate() {
        let index = match api {
            EmbeddingApi::Gemini => position,
            _ => {
                let key = if api == EmbeddingApi::Qwen {
                    "text_index"
                } else {
                    "index"
                };
                item[key]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| error("invalid embedding index"))?
            }
        };
        let values = item[if api == EmbeddingApi::Gemini {
            "values"
        } else {
            "embedding"
        }]
        .as_array()
        .ok_or_else(|| error("missing float vector"))?;
        let vector = values
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| error("vector contains an invalid number"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if vector.is_empty() || dimensions.is_some_and(|n| n != vector.len()) {
            return Err(error(
                "vector dimensions differ from the requested or preceding vector",
            ));
        }
        dimensions = Some(vector.len());
        let slot = slots
            .get_mut(index)
            .ok_or_else(|| error("embedding index is out of range"))?;
        if slot.is_some() {
            return Err(error("duplicate embedding index"));
        }
        *slot = Some(EmbeddingVector {
            index,
            values: vector,
        });
    }
    slots
        .into_iter()
        .map(|slot| slot.ok_or_else(|| error("missing embedding index")))
        .collect()
}

/// Validate route identity before persisting it or acquiring a credential.
pub(crate) fn validate_route(route: &EmbeddingRoute) -> Result<(), LlmError> {
    if route.max_inputs == Some(0) {
        return Err(invalid("embedding max_inputs must be positive"));
    }
    if route.api == EmbeddingApi::Gemini && !route.endpoint.contains("{model}") {
        return Err(invalid("Gemini embedding endpoint requires {model}"));
    }
    let sample = route.endpoint.replace("{model}", "embedding-model");
    let parsed = url::Url::parse(&sample).map_err(|_| invalid("invalid embedding endpoint"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid(
            "embedding endpoint must be HTTP without credentials, query or fragment",
        ));
    }
    if let Some(endpoint) = &route.models_endpoint {
        let models = url::Url::parse(endpoint)
            .map_err(|_| invalid("invalid embedding model-directory endpoint"))?;
        let expected_models_path =
            match route.api {
                EmbeddingApi::OpenRouter => None,
                EmbeddingApi::Gemini => {
                    let operation_suffix = "/models/embedding-model:batchEmbedContents";
                    let prefix = parsed.path().strip_suffix(operation_suffix).ok_or_else(|| {
                    invalid("Gemini embedding endpoint must use /models/{model}:batchEmbedContents")
                })?;
                    Some(format!("{prefix}/models"))
                }
                EmbeddingApi::OpenAi => {
                    let prefix = parsed.path().strip_suffix("/embeddings").ok_or_else(|| {
                        invalid("OpenAI embedding endpoint must end with /embeddings")
                    })?;
                    Some(format!("{prefix}/models"))
                }
                EmbeddingApi::Qwen => {
                    return Err(invalid(
                        "this embedding adapter has no model-directory contract",
                    ));
                }
            };
        if models.scheme() != parsed.scheme()
            || models.host_str() != parsed.host_str()
            || models.port_or_known_default() != parsed.port_or_known_default()
            || !models.username().is_empty()
            || models.password().is_some()
            || models.query().is_some()
            || models.fragment().is_some()
            || match route.api {
                EmbeddingApi::OpenRouter => !models.path().ends_with("/embeddings/models"),
                EmbeddingApi::Gemini | EmbeddingApi::OpenAi => {
                    expected_models_path.as_deref() != Some(models.path())
                }
                EmbeddingApi::Qwen => true,
            }
        {
            return Err(invalid(
                "embedding model-directory endpoint must share the embedding origin and use the documented route path",
            ));
        }
    }
    if let ServiceAuth::ApiKey { header } = &route.auth {
        if header.is_empty()
            || !header
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || [
                "host",
                "content-type",
                "content-length",
                "connection",
                "transfer-encoding",
            ]
            .iter()
            .any(|name| header.eq_ignore_ascii_case(name))
        {
            return Err(invalid("invalid embedding authentication header"));
        }
    }
    Ok(())
}
