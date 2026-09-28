//! Independent embedding routes. No automatic splitting or vector normalization.
pub(crate) mod backend;
pub use crate::protocol::{ServiceAuth, ServiceSetting};

use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    protocol::{LlmError, ProviderId, UsageState},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

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
        execute(&snapshot, profile_name, input, options).await
    }
}
pub(crate) async fn execute(
    snapshot: &ClientSnapshot,
    profile_name: &str,
    input: &EmbeddingRequest,
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
    let backend = backend::resolve(profile, route.api)?;
    let mut request = backend.encode(profile, route, input)?;
    backend.request_options(profile, route, &mut request, options)?;
    let request_url = request.url.clone();
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
    let response_cache = backend.response_cache(profile, &request_url, &response);
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
    let vectors = backend.vectors(&body, input)?;
    let usage = backend.usage(&body);
    Ok(EmbeddingResponse {
        provider_id: profile.provider_id.clone(),
        executed_profile: profile.profile_name.clone(),
        requested_model: input.model.clone(),
        model: body["model"].as_str().map(str::to_owned),
        request_id: request_id.or_else(|| body["request_id"].as_str().map(str::to_owned)),
        vectors,
        usage,
        response_cache,
    })
}
pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
pub(crate) fn apply_auth(
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
