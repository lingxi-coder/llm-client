use crate::client::{ClientSnapshot, ClientSource, RequestOptions};
use crate::embeddings::{self, backend::*, *};
use crate::protocol::{LlmError, ProviderId, ProviderProfile};
use crate::runtime::Deadline;
use crate::transport::{HttpExecutor, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, time::Duration};

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
pub(crate) struct OpenRouterEmbeddings;
impl EmbeddingBackend for OpenRouterEmbeddings {
    fn encode(
        &self,
        _profile: &ProviderProfile,
        route: &EmbeddingRoute,
        req: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError> {
        validate_input(route, req)?;
        let mut body = crate::codecs::openai::embeddings::body(req);
        if let Some(task) = req.task {
            body["input_type"] = json!(match task {
                    EmbeddingTask::RetrievalQuery => "search_query",
                    EmbeddingTask::RetrievalDocument => "search_document",
                    _ => return Err(LlmError::UnsupportedCapability {
                        message: "OpenRouter embedding input_type supports retrieval query or document in this client".into(),
                    }),
                });
        }
        post(route.endpoint.clone(), body)
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
    fn request_options(
        &self,
        profile: &ProviderProfile,
        route: &EmbeddingRoute,
        request: &mut HttpRequest,
        options: &RequestOptions,
    ) -> Result<(), LlmError> {
        if options.openrouter_response_cache.is_some() {
            if route.api != EmbeddingApi::OpenRouter
                || request.url != "https://openrouter.ai/api/v1/embeddings"
            {
                return Err(LlmError::UnsupportedCapability{message:"embedding response caching requires the official OpenRouter Embeddings route".into()});
            }
            super::response_cache::apply_openrouter_response_cache(
                options.openrouter_response_cache,
                profile,
                request,
            )?;
        }
        Ok(())
    }
    fn response_cache(
        &self,
        profile: &ProviderProfile,
        url: &str,
        response: &crate::transport::HttpResponse,
    ) -> Option<crate::protocol::ResponseCacheObservation> {
        super::response_cache::openrouter_observation(profile, url, &response.headers)
    }
}
pub(crate) async fn list_models(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    offset: u64,
    limit: u16,
    options: &RequestOptions,
) -> Result<EmbeddingModelPage, EmbeddingError> {
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
    let mut url =
        url::Url::parse(endpoint).map_err(|_| invalid("invalid embedding model-directory URL"))?;
    url.query_pairs_mut()
        .append_pair("offset", &offset.to_string())
        .append_pair("limit", &limit.to_string());
    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(120)),
    ));
    let mut request = HttpRequest {
        http1_header_layout: None,
        method: "GET".into(),
        url: url.into(),
        headers: vec![("accept".into(), "application/json".into())],
        body: Vec::new().into(),
        timeout: deadline.remaining()?,
    };
    apply_auth(&route.auth, options, &mut request)?;
    let response = HttpExecutor::new(snapshot.runtime.http.as_ref())
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
    let rows = body["data"]
        .as_array()
        .ok_or_else(|| EmbeddingError::InvalidResponse("model-directory data is missing".into()))?;
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
        offset: u64,
        limit: u16,
        options: &RequestOptions,
    ) -> Result<EmbeddingModelPage, EmbeddingError> {
        let snapshot = self.source.pin()?;
        list_models(&snapshot, self.profile, offset, limit, options).await
    }
}
