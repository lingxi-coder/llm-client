//! Native Knowledge Search calls for published Model Studio retrieval
//! services. This is separate from host-side RAG orchestration and from the
//! lower-level direct-index `retrieve` operation.

use super::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use url::Url;

const KNOWLEDGE_SEARCH_PATH: &str = "/api/v1/indices/knowledge/search";
const MAX_FILTER_BYTES: usize = 80_000;
const MAX_UNSTRUCTURED_TAG_VALUES: usize = 1_000;

/// Opaque reference to a published Knowledge Search service in one workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeSearchRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    agent_id: String,
}

impl QwenKnowledgeSearchRef {
    /// Bind the `agent_id` of a service that the caller has already published.
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        agent_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let agent_id = agent_id.into();
        if !valid_resource_id(&agent_id) {
            return Err(invalid("Knowledge Search agent_id is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            agent_id,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

impl QwenKnowledgeScope {
    /// Bind a published service ID from this exact Model Studio workspace.
    pub fn knowledge_search_ref(
        &self,
        agent_id: impl Into<String>,
    ) -> Result<QwenKnowledgeSearchRef, QwenKnowledgeError> {
        QwenKnowledgeSearchRef::from_scope(self, agent_id)
    }
}

/// Per-Knowledge-Base runtime filters for a published search service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeSearchKbConfig {
    knowledge: QwenKnowledgeRef,
    search_filters: Vec<Value>,
    known_unstructured: bool,
}

impl QwenKnowledgeSearchKbConfig {
    pub fn new(knowledge: QwenKnowledgeRef) -> Self {
        Self {
            knowledge,
            search_filters: Vec::new(),
            known_unstructured: false,
        }
    }

    /// Mark a config whose Knowledge Base is known to be unstructured so the
    /// documented unstructured `tags` array limit can be checked locally.
    pub fn for_unstructured(knowledge: QwenKnowledgeRef) -> Self {
        Self {
            knowledge,
            search_filters: Vec::new(),
            known_unstructured: true,
        }
    }

    pub fn with_search_filters(mut self, filters: impl IntoIterator<Item = Value>) -> Self {
        self.search_filters = filters.into_iter().collect();
        self
    }

    pub fn knowledge(&self) -> &QwenKnowledgeRef {
        &self.knowledge
    }

    pub fn search_filters(&self) -> &[Value] {
        &self.search_filters
    }
}

/// One request to an already-published Model Studio Knowledge Search service.
///
/// The request surface intentionally exposes only the fields documented by
/// the endpoint. Search strategy, routing, weights, and reranking are held by
/// the published service and are not caller-side overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeSearchRequest {
    search: QwenKnowledgeSearchRef,
    agent_version: Option<String>,
    query: String,
    images: Option<Vec<String>>,
    kb_search_configs: Vec<QwenKnowledgeSearchKbConfig>,
}

impl QwenKnowledgeSearchRequest {
    /// Build a text search request for a published service.
    pub fn new(search: QwenKnowledgeSearchRef, query: impl Into<String>) -> Self {
        Self {
            search,
            agent_version: None,
            query: query.into(),
            images: None,
            kb_search_configs: Vec::new(),
        }
    }

    /// Build an image-only search. The documented API requires an empty
    /// `query` string alongside `images` for this mode.
    pub fn images_only(
        search: QwenKnowledgeSearchRef,
        images: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            search,
            agent_version: None,
            query: String::new(),
            images: Some(images.into_iter().map(Into::into).collect()),
            kb_search_configs: Vec::new(),
        }
    }

    pub fn with_agent_version(mut self, agent_version: impl Into<String>) -> Self {
        self.agent_version = Some(agent_version.into());
        self
    }

    /// Add image URLs to a text request for multimodal search.
    pub fn with_images(mut self, images: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.images = Some(images.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_kb_search_config(mut self, config: QwenKnowledgeSearchKbConfig) -> Self {
        self.kb_search_configs.push(config);
        self
    }

    pub fn with_kb_search_configs(
        mut self,
        configs: impl IntoIterator<Item = QwenKnowledgeSearchKbConfig>,
    ) -> Self {
        self.kb_search_configs = configs.into_iter().collect();
        self
    }

    pub fn search_ref(&self) -> &QwenKnowledgeSearchRef {
        &self.search
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn images(&self) -> Option<&[String]> {
        self.images.as_deref()
    }

    pub fn kb_search_configs(&self) -> &[QwenKnowledgeSearchKbConfig] {
        &self.kb_search_configs
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.query.is_empty() && self.images.as_ref().is_none_or(|images| images.is_empty()) {
            return Err(invalid(
                "Knowledge Search requires a nonempty query or at least one image URL",
            ));
        }
        if let Some(images) = &self.images {
            for image in images {
                validate_image_url(image)?;
            }
        }

        let mut seen_knowledge_ids = HashSet::with_capacity(self.kb_search_configs.len());
        let mut filter_bytes = 0usize;
        for config in &self.kb_search_configs {
            if !seen_knowledge_ids.insert(config.knowledge.index_id.as_str()) {
                return Err(invalid(
                    "kb_search_configs cannot contain duplicate Knowledge Base IDs",
                ));
            }
            for filter in &config.search_filters {
                if !filter.is_object() {
                    return Err(invalid(
                        "each Knowledge Search search_filters item must be a JSON object",
                    ));
                }
                if config.known_unstructured {
                    validate_tag_array_lengths(filter)?;
                }
            }
            if !config.search_filters.is_empty() {
                let bytes = serde_json::to_vec(&config.search_filters)
                    .map_err(|_| invalid("Knowledge Search filters cannot be encoded as JSON"))?
                    .len();
                filter_bytes = filter_bytes.checked_add(bytes).ok_or_else(|| {
                    invalid("Knowledge Search total search_filters size is too large")
                })?;
                if filter_bytes > MAX_FILTER_BYTES {
                    return Err(invalid(
                        "Knowledge Search total search_filters size must not exceed 80000 bytes",
                    ));
                }
            }
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = Map::new();
        body.insert("agent_id".into(), json!(self.search.agent_id));
        if let Some(agent_version) = &self.agent_version {
            body.insert("agent_version".into(), json!(agent_version));
        }
        body.insert("query".into(), json!(self.query));
        if let Some(images) = &self.images {
            body.insert("images".into(), json!(images));
        }
        if !self.kb_search_configs.is_empty() {
            body.insert(
                "kb_search_configs".into(),
                json!(self
                    .kb_search_configs
                    .iter()
                    .map(QwenKnowledgeSearchKbConfig::to_value)
                    .collect::<Vec<_>>()),
            );
        }
        Value::Object(body)
    }
}

impl QwenKnowledgeSearchKbConfig {
    fn to_value(&self) -> Value {
        let mut value = Map::new();
        value.insert("id".into(), json!(self.knowledge.index_id));
        if !self.search_filters.is_empty() {
            value.insert("search_filters".into(), json!(self.search_filters));
        }
        Value::Object(value)
    }
}

/// One native result node. Unknown node and metadata fields remain available
/// in `native` and `metadata` respectively.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenKnowledgeSearchNode {
    pub score: Option<f64>,
    pub text: Option<String>,
    /// Scoped owner when Model Studio returns `metadata.pipeline_id`.
    pub knowledge: Option<QwenKnowledgeRef>,
    pub metadata: Value,
    pub native: Value,
}

/// One response from the published Knowledge Search service.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenKnowledgeSearchResult {
    pub search: QwenKnowledgeSearchRef,
    pub total: u64,
    pub cost_time_ms: u64,
    pub nodes: Vec<QwenKnowledgeSearchNode>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// Query one published Knowledge Search service once.
    ///
    /// This calls Model Studio's native retrieval service. It does not create
    /// or publish an Agent, run host-side RAG orchestration, or retry queries.
    pub async fn knowledge_search(
        &self,
        request: &QwenKnowledgeSearchRequest,
    ) -> Result<QwenKnowledgeSearchResult, QwenKnowledgeError> {
        self.validate_knowledge_search_ref(&request.search)?;
        request.validate()?;
        for config in &request.kb_search_configs {
            self.validate_knowledge_ref(&config.knowledge)?;
        }
        let response = self
            .request_json(
                "POST",
                KNOWLEDGE_SEARCH_PATH,
                &[],
                Some(request.to_body()),
                "knowledge_search",
                false,
            )
            .await?;
        decode_search_result(&self.scope, request, response)
    }

    fn validate_knowledge_search_ref(
        &self,
        reference: &QwenKnowledgeSearchRef,
    ) -> Result<(), QwenKnowledgeError> {
        let belongs_to_scope =
            QwenKnowledgeSearchRef::from_scope(reference.scope(), reference.agent_id())
                .is_ok_and(|canonical| canonical == *reference)
                && reference.scope() == self.scope();
        if !belongs_to_scope {
            return Err(LlmError::PermissionDenied {
                message: "Qwen Knowledge Search reference belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        Ok(())
    }
}

fn decode_search_result(
    scope: &QwenKnowledgeScope,
    request: &QwenKnowledgeSearchRequest,
    response: DecodedEnvelope,
) -> Result<QwenKnowledgeSearchResult, QwenKnowledgeError> {
    let data = response
        .native
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            response_invalid(
                "knowledge_search",
                "successful response omitted data object",
                &response,
                QwenKnowledgeDispatch::NotSent,
            )
        })?;
    let total = required_u64(data, "total", &response)?;
    let cost_time_ms = required_u64(data, "cost_time", &response)?;
    let nodes = data.get("nodes").and_then(Value::as_array).ok_or_else(|| {
        response_invalid(
            "knowledge_search",
            "successful response omitted data.nodes array",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let nodes = nodes
        .iter()
        .map(|node| decode_search_node(scope, node, &response))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(QwenKnowledgeSearchResult {
        search: request.search.clone(),
        total,
        cost_time_ms,
        nodes,
        request_id: response.request_id,
        native: response.native,
    })
}

fn decode_search_node(
    scope: &QwenKnowledgeScope,
    node: &Value,
    response: &DecodedEnvelope,
) -> Result<QwenKnowledgeSearchNode, QwenKnowledgeError> {
    let Some(node_object) = node.as_object() else {
        return Err(response_invalid(
            "knowledge_search",
            "data.nodes contained a non-object node",
            response,
            QwenKnowledgeDispatch::NotSent,
        ));
    };
    let Some(metadata) = node_object
        .get("metadata")
        .filter(|value| value.is_object())
    else {
        return Err(response_invalid(
            "knowledge_search",
            "result node omitted metadata object",
            response,
            QwenKnowledgeDispatch::NotSent,
        ));
    };
    if let Some(workspace_id) = metadata.get("workspace_id") {
        let Some(workspace_id) = workspace_id.as_str() else {
            return Err(response_invalid(
                "knowledge_search",
                "result metadata.workspace_id was not a string",
                response,
                QwenKnowledgeDispatch::NotSent,
            ));
        };
        if workspace_id != scope.workspace_id() {
            return Err(response_invalid(
                "knowledge_search",
                "result metadata.workspace_id did not match the current workspace",
                response,
                QwenKnowledgeDispatch::NotSent,
            ));
        }
    }
    let knowledge = match metadata.get("pipeline_id") {
        None => None,
        Some(pipeline_id) => {
            let Some(pipeline_id) = pipeline_id.as_str() else {
                return Err(response_invalid(
                    "knowledge_search",
                    "result metadata.pipeline_id was not a string",
                    response,
                    QwenKnowledgeDispatch::NotSent,
                ));
            };
            Some(scope.knowledge_ref(pipeline_id).map_err(|_| {
                response_invalid(
                    "knowledge_search",
                    "result metadata.pipeline_id was invalid",
                    response,
                    QwenKnowledgeDispatch::NotSent,
                )
            })?)
        }
    };
    let score = match node_object.get("score") {
        None => None,
        Some(score) => match score.as_f64() {
            Some(score) => Some(score),
            None => {
                return Err(response_invalid(
                    "knowledge_search",
                    "result node score was not numeric",
                    response,
                    QwenKnowledgeDispatch::NotSent,
                ));
            }
        },
    };
    let text = match node_object.get("text") {
        None => None,
        Some(text) => match text.as_str() {
            Some(text) => Some(text.to_owned()),
            None => {
                return Err(response_invalid(
                    "knowledge_search",
                    "result node text was not a string",
                    response,
                    QwenKnowledgeDispatch::NotSent,
                ));
            }
        },
    };
    Ok(QwenKnowledgeSearchNode {
        score,
        text,
        knowledge,
        metadata: metadata.clone(),
        native: node.clone(),
    })
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    response: &DecodedEnvelope,
) -> Result<u64, QwenKnowledgeError> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        response_invalid(
            "knowledge_search",
            &format!("data omitted a valid {field}"),
            response,
            QwenKnowledgeDispatch::NotSent,
        )
    })
}

fn validate_tag_array_lengths(value: &Value) -> Result<(), QwenKnowledgeError> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == "tags"
                    && child
                        .as_array()
                        .is_some_and(|tags| tags.len() > MAX_UNSTRUCTURED_TAG_VALUES)
                {
                    return Err(invalid(
                        "Knowledge Search unstructured filter tags array must not exceed 1000 items",
                    ));
                }
                validate_tag_array_lengths(child)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                validate_tag_array_lengths(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_image_url(value: &str) -> Result<(), QwenKnowledgeError> {
    let url = Url::parse(value).map_err(|_| invalid("Knowledge Search image URL is malformed"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(invalid(
            "Knowledge Search image URL must use HTTP or HTTPS and include a host",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidInput(message.into())
}
