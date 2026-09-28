//! Synchronous text reranking through OpenRouter's native Rerank API.
//!
//! Each call submits one query and a list of text documents, and returns the
//! provider-ranked original indices, scores, and unmodified response payload.
//! The service does not retry ambiguous requests.

use crate::{
    client::RequestOptions,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::Bytes;
use serde_json::{json, Value};
use std::fmt;
use thiserror::Error;
use url::Url;

/// OpenRouter's documented native rerank endpoint.
pub const OPENROUTER_RERANK_ENDPOINT: &str = "https://openrouter.ai/api/v1/rerank";

const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Explicit identity and endpoint for one OpenRouter account connection.
///
/// The endpoint is an argument so endpoint scope is visible to the caller.
/// This service accepts only OpenRouter's documented HTTPS origin and exact
/// rerank path; it will not forward an API key to arbitrary hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenRouterRerankScope {
    profile_name: String,
    account_scope: String,
    endpoint: String,
}

impl OpenRouterRerankScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        endpoint: impl AsRef<str>,
    ) -> Result<Self, OpenRouterRerankError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if profile_name.trim().is_empty() || account_scope.trim().is_empty() {
            return Err(OpenRouterRerankError::InvalidInput(
                "profile name and account scope must be non-empty".into(),
            ));
        }
        let endpoint = validate_endpoint(endpoint.as_ref())?;
        Ok(Self {
            profile_name,
            account_scope,
            endpoint,
        })
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn validate(&self) -> Result<(), OpenRouterRerankError> {
        if self.profile_name.trim().is_empty() || self.account_scope.trim().is_empty() {
            return Err(OpenRouterRerankError::InvalidInput(
                "profile name and account scope must be non-empty".into(),
            ));
        }
        let validated = validate_endpoint(&self.endpoint)?;
        if validated != self.endpoint {
            return Err(OpenRouterRerankError::InvalidInput(
                "OpenRouter rerank endpoint is not in canonical form".into(),
            ));
        }
        Ok(())
    }
}

fn validate_endpoint(endpoint: &str) -> Result<String, OpenRouterRerankError> {
    let parsed = Url::parse(endpoint).map_err(|_| {
        OpenRouterRerankError::InvalidInput("endpoint must be a valid HTTPS URL".into())
    })?;
    if parsed.scheme() != "https"
        || parsed.host_str() != Some("openrouter.ai")
        || parsed.port_or_known_default() != Some(443)
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/api/v1/rerank"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(OpenRouterRerankError::InvalidInput(
            "endpoint must be https://openrouter.ai/api/v1/rerank".into(),
        ));
    }
    Ok(parsed.to_string())
}

/// One text query and an ordered list of text documents to rerank.
#[derive(Clone, PartialEq, Eq)]
pub struct OpenRouterRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
}

impl OpenRouterRerankRequest {
    pub fn new<I, S>(model: impl Into<String>, query: impl Into<String>, documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            model: model.into(),
            query: query.into(),
            documents: documents.into_iter().map(Into::into).collect(),
            top_n: None,
        }
    }

    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = Some(top_n);
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn documents(&self) -> &[String] {
        &self.documents
    }

    pub fn top_n(&self) -> Option<usize> {
        self.top_n
    }

    fn validate(&self) -> Result<(), OpenRouterRerankError> {
        if self.model.trim().is_empty() {
            return Err(OpenRouterRerankError::InvalidInput(
                "model slug must be non-empty".into(),
            ));
        }
        if self.query.trim().is_empty() {
            return Err(OpenRouterRerankError::InvalidInput(
                "query must be non-empty".into(),
            ));
        }
        if self.documents.is_empty() {
            return Err(OpenRouterRerankError::InvalidInput(
                "at least one document is required".into(),
            ));
        }
        if self
            .documents
            .iter()
            .any(|document| document.trim().is_empty())
        {
            return Err(OpenRouterRerankError::InvalidInput(
                "documents must be non-empty".into(),
            ));
        }
        if self.top_n == Some(0) {
            return Err(OpenRouterRerankError::InvalidInput(
                "top_n must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for OpenRouterRerankRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenRouterRerankRequest")
            .field("model", &self.model)
            .field("query", &"<redacted query>")
            .field(
                "documents",
                &format_args!("<{} redacted documents>", self.documents.len()),
            )
            .field("top_n", &self.top_n)
            .finish()
    }
}

/// One ranked hit. `index` maps to the original request's `documents` array.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouterRerankHit {
    pub index: usize,
    pub relevance_score: f64,
    /// Provider-returned document object, when included in the response.
    pub document: Option<Value>,
}

/// Common usage values with the complete provider object retained.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouterRerankUsage {
    pub search_units: Option<u64>,
    pub total_tokens: Option<u64>,
    pub native: Value,
}

/// One successful OpenRouter rerank response.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouterRerankResult {
    scope: OpenRouterRerankScope,
    pub id: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// Provider-ranked hits, retaining each original input index and score.
    pub results: Vec<OpenRouterRerankHit>,
    pub usage: Option<OpenRouterRerankUsage>,
    /// Complete response JSON for fields not yet modeled by this crate.
    pub native: Value,
}

impl OpenRouterRerankResult {
    pub fn scope(&self) -> &OpenRouterRerankScope {
        &self.scope
    }
}

/// Dispatch knowledge after a rerank request fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRouterRerankDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum OpenRouterRerankError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid OpenRouter rerank input: {0}")]
    InvalidInput(String),
    #[error("OpenRouter rerank account scope does not match the configured service")]
    ScopeMismatch,
    #[error("OpenRouter rerank outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        id: Option<String>,
    },
    #[error("OpenRouter rerank request was rejected with HTTP {status}: {message}")]
    Rejected {
        status: u16,
        code: Option<String>,
        message: String,
        id: Option<String>,
    },
    #[error("OpenRouter accepted the rerank request but returned an invalid response: {message}")]
    AcceptedInvalidResponse { message: String, id: Option<String> },
}

impl OpenRouterRerankError {
    pub fn dispatch(&self) -> OpenRouterRerankDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::ScopeMismatch => {
                OpenRouterRerankDispatch::NotSent
            }
            Self::OutcomeUnknown { .. } => OpenRouterRerankDispatch::Unknown,
            Self::Rejected { .. } => OpenRouterRerankDispatch::Rejected,
            Self::AcceptedInvalidResponse { .. } => OpenRouterRerankDispatch::Accepted,
        }
    }
}

/// OpenRouter native rerank client using a caller-selected transport.
#[derive(Clone)]
pub struct OpenRouterRerankService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    scope: OpenRouterRerankScope,
}

impl<'a> OpenRouterRerankService<'a> {
    fn pin(&self) -> Result<Self, crate::protocol::LlmError> {
        let mut pinned = self.clone();
        pinned.binding = self
            .binding
            .as_ref()
            .map(crate::providers::binding::ProviderBinding::pinned)
            .transpose()?;
        Ok(pinned)
    }

    pub(crate) fn with_binding(
        mut self,
        binding: &crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding.clone());
        self
    }

    pub fn new(
        transport: &'a dyn Transport,
        scope: OpenRouterRerankScope,
    ) -> Result<Self, OpenRouterRerankError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            transport,
            scope,
        })
    }

    pub fn scope(&self) -> &OpenRouterRerankScope {
        &self.scope
    }

    /// Rerank one text query and candidate list in a single request.
    pub async fn rerank(
        &self,
        request: &OpenRouterRerankRequest,
        options: &RequestOptions,
    ) -> Result<OpenRouterRerankResult, OpenRouterRerankError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        request.validate()?;
        if options.account_scope.as_deref() != Some(pinned_service.scope.account_scope()) {
            return Err(OpenRouterRerankError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "OpenRouter rerank requires an API key for the selected account".into(),
            })?;

        let mut payload = json!({
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
        });
        if let Some(top_n) = request.top_n {
            payload["top_n"] = json!(top_n);
        }
        let body = serde_json::to_vec(&payload).map_err(|error| {
            OpenRouterRerankError::InvalidInput(format!("could not encode request: {error}"))
        })?;
        if body.len() > MAX_REQUEST_BYTES {
            return Err(LlmError::RequestTooLarge {
                message: format!("OpenRouter rerank request exceeds {MAX_REQUEST_BYTES} bytes"),
            }
            .into());
        }

        let response = HttpExecutor::new(pinned_service.transport)
            .execute_bounded(
                HttpRequest {
                    method: "POST".into(),
                    url: pinned_service.scope.endpoint.clone(),
                    headers: vec![
                        (
                            "authorization".into(),
                            format!("Bearer {}", credential.expose_secret()),
                        ),
                        ("content-type".into(), "application/json".into()),
                        ("accept".into(), "application/json".into()),
                    ],
                    body: Bytes::from(body),
                    timeout: options.total_timeout,
                },
                MAX_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| OpenRouterRerankError::OutcomeUnknown { source, id: None })?;

        if !(200..300).contains(&response.status) {
            let (code, message, id) = error_details(&response.body);
            if (400..500).contains(&response.status) {
                return Err(OpenRouterRerankError::Rejected {
                    status: response.status,
                    code,
                    message,
                    id,
                });
            }
            return Err(OpenRouterRerankError::OutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "OpenRouter returned HTTP {} after rerank submission: {}",
                        response.status, message
                    ),
                },
                id,
            });
        }

        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            OpenRouterRerankError::AcceptedInvalidResponse {
                message: format!("response JSON could not be decoded: {error}"),
                id: None,
            }
        })?;
        decode_result(value, &pinned_service.scope, request.documents.len())
    }
}

fn decode_result(
    value: Value,
    scope: &OpenRouterRerankScope,
    document_count: usize,
) -> Result<OpenRouterRerankResult, OpenRouterRerankError> {
    let id = value.get("id").and_then(Value::as_str).map(str::to_owned);
    if let Some(error) = value.get("error") {
        let (code, message, _) = error_value_details(error);
        return Err(OpenRouterRerankError::AcceptedInvalidResponse {
            message: code.map_or(message.clone(), |code| format!("{code}: {message}")),
            id,
        });
    }
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| OpenRouterRerankError::AcceptedInvalidResponse {
            message: "missing results array".into(),
            id: id.clone(),
        })?;
    let mut hits = Vec::with_capacity(results.len());
    let mut seen = std::collections::BTreeSet::new();
    for item in results {
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < document_count)
            .ok_or_else(|| OpenRouterRerankError::AcceptedInvalidResponse {
                message: "result contains a missing or out-of-range document index".into(),
                id: id.clone(),
            })?;
        if !seen.insert(index) {
            return Err(OpenRouterRerankError::AcceptedInvalidResponse {
                message: "result contains a duplicate document index".into(),
                id,
            });
        }
        let relevance_score = item
            .get("relevance_score")
            .and_then(Value::as_f64)
            .filter(|score| score.is_finite())
            .ok_or_else(|| OpenRouterRerankError::AcceptedInvalidResponse {
                message: "result contains a missing or non-finite relevance score".into(),
                id: id.clone(),
            })?;
        hits.push(OpenRouterRerankHit {
            index,
            relevance_score,
            document: item.get("document").cloned(),
        });
    }

    let usage = value
        .get("usage")
        .filter(|usage| !usage.is_null())
        .map(|usage| OpenRouterRerankUsage {
            search_units: usage.get("search_units").and_then(Value::as_u64),
            total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
            native: usage.clone(),
        });
    Ok(OpenRouterRerankResult {
        scope: scope.clone(),
        id,
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        provider: value
            .get("provider")
            .and_then(Value::as_str)
            .map(str::to_owned),
        results: hits,
        usage,
        native: value,
    })
}

fn error_details(body: &[u8]) -> (Option<String>, String, Option<String>) {
    match serde_json::from_slice::<Value>(body) {
        Ok(value) => error_value_details(&value),
        Err(_) => (None, String::from_utf8_lossy(body).into_owned(), None),
    }
}

fn error_value_details(value: &Value) -> (Option<String>, String, Option<String>) {
    let error = value.get("error").unwrap_or(value);
    let code = error.get("code").and_then(|code| match code {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    });
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string());
    let id = value.get("id").and_then(Value::as_str).map(str::to_owned);
    (code, message, id)
}
