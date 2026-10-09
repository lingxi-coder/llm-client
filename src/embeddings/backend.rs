//! Embedding capability dispatch and shared protocol mechanics.
use super::*;
use crate::protocol::{ProviderProfile, ResponseCacheObservation};
use crate::transport::HttpResponse;

pub(crate) trait EmbeddingBackend: Sync {
    fn encode(
        &self,
        profile: &ProviderProfile,
        route: &EmbeddingRoute,
        request: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError>;
    fn vectors(
        &self,
        body: &Value,
        request: &EmbeddingRequest,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingError>;
    fn usage(&self, body: &Value) -> EmbeddingUsage;
    fn request_options(
        &self,
        _profile: &ProviderProfile,
        _route: &EmbeddingRoute,
        _request: &mut HttpRequest,
        options: &RequestOptions,
    ) -> Result<(), LlmError> {
        if options.openrouter_response_cache.is_some() {
            return Err(LlmError::UnsupportedCapability {
                message:
                    "embedding response caching requires the official OpenRouter Embeddings route"
                        .into(),
            });
        }
        Ok(())
    }
    fn response_cache(
        &self,
        _profile: &ProviderProfile,
        _url: &str,
        _response: &HttpResponse,
    ) -> Option<ResponseCacheObservation> {
        None
    }
}
pub(crate) fn resolve(
    profile: &ProviderProfile,
    api: EmbeddingApi,
) -> Result<&'static dyn EmbeddingBackend, LlmError> {
    use crate::providers::{google, openai, openrouter, qwen, zhipu};
    let backend: &dyn EmbeddingBackend = match (profile.provider_id.as_str(), api) {
        ("openai", EmbeddingApi::OpenAi) => &openai::embeddings::OpenAiEmbeddings,
        ("google", EmbeddingApi::Gemini) => &google::embeddings::GoogleEmbeddings,
        ("openrouter", EmbeddingApi::OpenRouter) => &openrouter::embeddings::OpenRouterEmbeddings,
        ("qwen", EmbeddingApi::Qwen) => &qwen::embeddings::QwenEmbeddings,
        ("zhipu", EmbeddingApi::OpenAi) => &zhipu::embeddings::ZhipuEmbeddings,
        (
            "openai" | "google" | "openrouter" | "qwen" | "zhipu" | "anthropic" | "minimax"
            | "kimi" | "xai" | "deepseek" | "github-copilot",
            _,
        ) => {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "provider {} does not support the configured {api:?} embedding API",
                    profile.provider_id
                ),
            })
        }
        (_, EmbeddingApi::OpenAi) => &openai::embeddings::OpenAiEmbeddings,
        (_, EmbeddingApi::OpenRouter) => &openrouter::embeddings::OpenRouterEmbeddings,
        (_, EmbeddingApi::Gemini) => &google::embeddings::GoogleEmbeddings,
        (_, EmbeddingApi::Qwen) => &qwen::embeddings::QwenEmbeddings,
    };
    Ok(backend)
}
pub(crate) fn validate_input(
    route: &EmbeddingRoute,
    req: &EmbeddingRequest,
) -> Result<(), LlmError> {
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
    validate_route(route)?;
    if req
        .input
        .iter()
        .try_fold(0usize, |total, text| total.checked_add(text.len()))
        .is_none_or(|size| size > 64 * 1024 * 1024)
    {
        return Err(invalid("embedding input exceeds 64 MiB"));
    }
    Ok(())
}
pub(crate) fn post(endpoint: String, body: Value) -> Result<HttpRequest, LlmError> {
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
        http1_header_layout: None,
        method: "POST".into(),
        url: endpoint,
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.into(),
        timeout: None,
    })
}
pub(crate) fn decode_vectors(
    array: &Value,
    index_key: Option<&str>,
    value_key: &str,
    req: &EmbeddingRequest,
) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
    let error = |msg: &str| EmbeddingError::InvalidResponse(msg.into());
    let array = array
        .as_array()
        .ok_or_else(|| error("missing embeddings"))?;
    if array.len() != req.input.len() {
        return Err(error("embedding count differs from input count"));
    }
    let mut slots = vec![None; req.input.len()];
    let mut dimensions = req.dimensions;
    for (position, item) in array.iter().enumerate() {
        let index = if let Some(key) = index_key {
            item[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| error("invalid embedding index"))?
        } else {
            position
        };
        let values = item[value_key]
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
pub(crate) fn usage(native: Value, input_key: &str, total_key: &str) -> EmbeddingUsage {
    let input_tokens = native[input_key].as_u64();
    let total_tokens = native[total_key].as_u64();
    let invalid_usage = (!native.is_null() && !native.is_object())
        || [input_key, total_key]
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
    EmbeddingUsage {
        state,
        input_tokens,
        total_tokens,
        native,
    }
}
pub(crate) struct DocumentedModelLimits {
    pub max_inputs: usize,
    /// `None` means the provider explicitly excludes this model from the
    /// custom-dimension parameter. Unknown models have no
    /// `DocumentedModelLimits` entry at all.
    pub dimensions: Option<&'static [usize]>,
}

pub(crate) fn validate_limits(
    req: &EmbeddingRequest,
    limits: Option<DocumentedModelLimits>,
) -> Result<(), LlmError> {
    if let Some(limits) = limits {
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
    Ok(())
}
