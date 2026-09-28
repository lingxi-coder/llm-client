//! Synchronous text reranking through Alibaba Cloud Model Studio's
//! provider-native Qwen3.7-Text-Rerank HTTP API.
//!
//! The caller supplies one query and its candidate documents. This module
//! sends exactly one request, returns the provider's original document
//! indices and relevance scores, and never retries or changes account scope.

use crate::{
    client::RequestOptions,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::Bytes;
use serde_json::{json, Value};
use std::fmt;
use thiserror::Error;

/// Model Studio's documented Qwen3.7 text rerank route in the Beijing region.
pub const QWEN_RERANK_BEIJING_ENDPOINT_SUFFIX: &str =
    ".cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank";
/// Model used by this provider-native route.
pub const QWEN_RERANK_MODEL: &str = "qwen3.7-text-rerank";

const MAX_DOCUMENTS: usize = 500;
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Region used by the documented Qwen3.7 text-rerank HTTP route.
///
/// The current HTTP reference gives the exact workspace URL for Beijing. It
/// does not give an exact regional URL for this model/API elsewhere, so this
/// type intentionally does not expose an inferred Singapore route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QwenRerankRegion {
    Beijing,
}

/// Stable caller-owned identity for a Model Studio account and workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenRerankScope {
    profile_name: String,
    account_scope: String,
    region: QwenRerankRegion,
    workspace_id: String,
}

impl QwenRerankScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenRerankRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, QwenRerankError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn region(&self) -> QwenRerankRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    fn validate(&self) -> Result<(), QwenRerankError> {
        if self.profile_name.trim().is_empty() || self.account_scope.trim().is_empty() {
            return Err(QwenRerankError::InvalidInput(
                "profile name and account scope must be non-empty".into(),
            ));
        }
        let id = self.workspace_id.as_bytes();
        if id.is_empty()
            || id.len() > 63
            || !id[0].is_ascii_alphanumeric()
            || !id[id.len() - 1].is_ascii_alphanumeric()
            || !id
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        {
            return Err(QwenRerankError::InvalidInput(
                "workspace ID must be one DNS label containing only letters, digits, and internal hyphens".into(),
            ));
        }
        Ok(())
    }

    fn endpoint(&self) -> String {
        match self.region {
            QwenRerankRegion::Beijing => {
                format!(
                    "https://{}{}",
                    self.workspace_id, QWEN_RERANK_BEIJING_ENDPOINT_SUFFIX
                )
            }
        }
    }
}

/// One synchronous query and its ordered candidate documents.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenRerankRequest {
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    instruction: Option<String>,
}

impl QwenRerankRequest {
    pub fn new<I, S>(query: impl Into<String>, documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            query: query.into(),
            documents: documents.into_iter().map(Into::into).collect(),
            top_n: None,
            instruction: None,
        }
    }

    /// Request at most `top_n` results. The provider documents values larger
    /// than the document count as equivalent to returning all documents.
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = Some(top_n);
        self
    }

    /// Add Model Studio's optional English sorting instruction.
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
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

    pub fn instruction(&self) -> Option<&str> {
        self.instruction.as_deref()
    }

    fn validate(&self) -> Result<(), QwenRerankError> {
        if self.query.trim().is_empty() {
            return Err(QwenRerankError::InvalidInput(
                "query must be non-empty".into(),
            ));
        }
        if self.documents.is_empty() {
            return Err(QwenRerankError::InvalidInput(
                "at least one candidate document is required".into(),
            ));
        }
        if self.documents.len() > MAX_DOCUMENTS {
            return Err(QwenRerankError::InvalidInput(format!(
                "qwen3.7-text-rerank accepts at most {MAX_DOCUMENTS} documents"
            )));
        }
        if self
            .documents
            .iter()
            .any(|document| document.trim().is_empty())
        {
            return Err(QwenRerankError::InvalidInput(
                "candidate documents must be non-empty".into(),
            ));
        }
        if self.top_n == Some(0) {
            return Err(QwenRerankError::InvalidInput(
                "top_n must be greater than zero".into(),
            ));
        }
        if self
            .instruction
            .as_deref()
            .is_some_and(|instruction| instruction.trim().is_empty())
        {
            return Err(QwenRerankError::InvalidInput(
                "instruction must be non-empty when supplied".into(),
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for QwenRerankRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenRerankRequest")
            .field("query", &"<redacted query>")
            .field(
                "documents",
                &format_args!("<{} redacted documents>", self.documents.len()),
            )
            .field("top_n", &self.top_n)
            .field(
                "instruction",
                &self.instruction.as_ref().map(|_| "<redacted instruction>"),
            )
            .finish()
    }
}

/// One result with its original index into [`QwenRerankRequest::documents`].
#[derive(Debug, Clone, PartialEq)]
pub struct QwenRerankHit {
    pub index: usize,
    pub relevance_score: f64,
}

/// Token accounting returned by Model Studio when present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenRerankUsage {
    pub prompt_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Typed results from one provider-native Qwen3.7 text-rerank call.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenRerankResult {
    scope: QwenRerankScope,
    pub request_id: Option<String>,
    pub model: Option<String>,
    /// Provider-sorted results. Each index addresses the input request's
    /// original `documents` array.
    pub results: Vec<QwenRerankHit>,
    pub usage: Option<QwenRerankUsage>,
    /// Original provider response, retained for fields added by Model Studio.
    pub native: Value,
}

impl QwenRerankResult {
    pub fn scope(&self) -> &QwenRerankScope {
        &self.scope
    }
}

/// What can be concluded after a rerank call fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenRerankDispatchOutcome {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Error)]
pub enum QwenRerankError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Qwen rerank input: {0}")]
    InvalidInput(String),
    #[error("Qwen rerank account scope does not match the configured service")]
    ScopeMismatch,
    #[error("Qwen rerank request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen rerank request was rejected with HTTP {status}: {message}")]
    Rejected {
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen rerank accepted the request but returned an invalid response: {message}")]
    AcceptedInvalidResponse {
        message: String,
        request_id: Option<String>,
    },
}

impl QwenRerankError {
    /// Classify whether a failed request was sent, rejected, or may have been
    /// processed. The service never retries a request automatically.
    pub fn dispatch_outcome(&self) -> QwenRerankDispatchOutcome {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) | Self::ScopeMismatch => {
                QwenRerankDispatchOutcome::NotSent
            }
            Self::OutcomeUnknown { .. } => QwenRerankDispatchOutcome::Unknown,
            Self::Rejected { .. } => QwenRerankDispatchOutcome::Rejected,
            Self::AcceptedInvalidResponse { .. } => QwenRerankDispatchOutcome::Accepted,
        }
    }
}

/// Synchronous provider-native Model Studio Qwen3.7 rerank service.
#[derive(Clone)]
pub struct QwenRerankService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    scope: QwenRerankScope,
}

impl<'a> QwenRerankService<'a> {
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
        scope: QwenRerankScope,
    ) -> Result<Self, QwenRerankError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            transport,
            scope,
        })
    }

    pub fn scope(&self) -> &QwenRerankScope {
        &self.scope
    }

    /// Rerank the supplied candidate documents in one synchronous API call.
    pub async fn rerank(
        &self,
        request: &QwenRerankRequest,
        options: &RequestOptions,
    ) -> Result<QwenRerankResult, QwenRerankError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        request.validate()?;
        if options.account_scope.as_deref() != Some(pinned_service.scope.account_scope()) {
            return Err(QwenRerankError::ScopeMismatch);
        }
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen rerank requires the API key for the selected Model Studio region"
                    .into(),
            })?;

        let mut parameters = serde_json::Map::new();
        if let Some(top_n) = request.top_n {
            parameters.insert("top_n".into(), json!(top_n));
        }
        if let Some(instruction) = &request.instruction {
            parameters.insert("instruct".into(), json!(instruction));
        }
        let mut payload = json!({
            "model": QWEN_RERANK_MODEL,
            "input": {
                "query": request.query,
                "documents": request.documents,
            },
        });
        if !parameters.is_empty() {
            payload["parameters"] = Value::Object(parameters);
        }
        let body = serde_json::to_vec(&payload).map_err(|error| {
            QwenRerankError::InvalidInput(format!("could not encode rerank request: {error}"))
        })?;

        let response = HttpExecutor::new(pinned_service.transport)
            .execute_bounded(
                HttpRequest {
                    method: "POST".into(),
                    url: pinned_service.scope.endpoint(),
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
            .map_err(|source| QwenRerankError::OutcomeUnknown {
                source,
                request_id: None,
            })?;

        if !(200..300).contains(&response.status) {
            let (code, message, request_id) = error_details(&response.body);
            if (400..500).contains(&response.status) {
                return Err(QwenRerankError::Rejected {
                    status: response.status,
                    code,
                    message,
                    request_id,
                });
            }
            return Err(QwenRerankError::OutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "Model Studio returned HTTP {} after rerank submission: {}",
                        response.status, message
                    ),
                },
                request_id,
            });
        }

        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            QwenRerankError::AcceptedInvalidResponse {
                message: format!("response JSON could not be decoded: {error}"),
                request_id: None,
            }
        })?;
        decode_result(value, &pinned_service.scope, request.documents.len())
    }
}

fn decode_result(
    value: Value,
    scope: &QwenRerankScope,
    document_count: usize,
) -> Result<QwenRerankResult, QwenRerankError> {
    let request_id = value
        .get("request_id")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(code) = value
        .get("code")
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty())
    {
        let message = value
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Model Studio reported a rerank error")
            .to_owned();
        return Err(QwenRerankError::AcceptedInvalidResponse {
            message: format!("{code}: {message}"),
            request_id,
        });
    }

    let results = value
        .pointer("/output/results")
        .and_then(Value::as_array)
        .ok_or_else(|| QwenRerankError::AcceptedInvalidResponse {
            message: "missing output.results array".into(),
            request_id: request_id.clone(),
        })?;
    let mut hits = Vec::with_capacity(results.len());
    let mut seen = std::collections::BTreeSet::new();
    for item in results {
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < document_count)
            .ok_or_else(|| QwenRerankError::AcceptedInvalidResponse {
                message: "result contains a missing or out-of-range document index".into(),
                request_id: request_id.clone(),
            })?;
        if !seen.insert(index) {
            return Err(QwenRerankError::AcceptedInvalidResponse {
                message: "result contains a duplicate document index".into(),
                request_id,
            });
        }
        let relevance_score = item
            .get("relevance_score")
            .and_then(Value::as_f64)
            .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
            .ok_or_else(|| QwenRerankError::AcceptedInvalidResponse {
                message: "result contains a missing or out-of-range relevance score".into(),
                request_id: request_id.clone(),
            })?;
        hits.push(QwenRerankHit {
            index,
            relevance_score,
        });
    }

    let usage = value
        .get("usage")
        .and_then(Value::as_object)
        .map(|usage| QwenRerankUsage {
            prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
            total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        });
    Ok(QwenRerankResult {
        scope: scope.clone(),
        request_id,
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        results: hits,
        usage,
        native: value,
    })
}

fn error_details(body: &[u8]) -> (Option<String>, String, Option<String>) {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        let code = value
            .get("code")
            .and_then(Value::as_str)
            .filter(|code| !code.is_empty())
            .map(str::to_owned);
        let message = value
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());
        let request_id = value
            .get("request_id")
            .or_else(|| value.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        (code, message, request_id)
    } else {
        (None, String::from_utf8_lossy(body).into_owned(), None)
    }
}
