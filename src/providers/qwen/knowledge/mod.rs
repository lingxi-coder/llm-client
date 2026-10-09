//! Scoped access to Alibaba Cloud Model Studio knowledge bases.
//!
//! This is a native Model Studio service, separate from OpenAI vector stores
//! and Qwen's Responses `file_search` tool. It implements the documented
//! Beijing RAG REST knowledge-base, document, file, and import-job routes.
//! Resource references retain the account, profile, region, and workspace that
//! supplied each ID.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use url::Url;

pub mod chat;
pub use chat::{
    QwenKnowledgeChatContent, QwenKnowledgeChatContentPart, QwenKnowledgeChatError,
    QwenKnowledgeChatEvent, QwenKnowledgeChatMessage, QwenKnowledgeChatProviderError,
    QwenKnowledgeChatRef, QwenKnowledgeChatRequest, QwenKnowledgeChatRole, QwenKnowledgeChatStream,
    QwenKnowledgeChatToolCall,
};
pub mod search;
pub use search::{
    QwenKnowledgeSearchKbConfig, QwenKnowledgeSearchNode, QwenKnowledgeSearchRef,
    QwenKnowledgeSearchRequest, QwenKnowledgeSearchResult,
};
pub mod categories;
pub use categories::{
    QwenKnowledgeCategory, QwenKnowledgeCategoryCreateRequest, QwenKnowledgeCategoryCreateResult,
    QwenKnowledgeCategoryDeleteResult, QwenKnowledgeCategoryListRequest, QwenKnowledgeCategoryPage,
    QwenKnowledgeCategoryRef,
};
pub mod connectors;
pub use connectors::{
    QwenKnowledgeConnectorCreateRequest, QwenKnowledgeConnectorCreateResult,
    QwenKnowledgeConnectorDetails, QwenKnowledgeConnectorLookup, QwenKnowledgeConnectorRef,
    QwenKnowledgeConnectorStoreType, QwenKnowledgeOssImportFile, QwenKnowledgeOssImportRequest,
    QwenKnowledgeOssImportResult,
};
pub mod chunks;
pub use chunks::{
    QwenKnowledgeAddChunkRequest, QwenKnowledgeChunk, QwenKnowledgeChunkFields,
    QwenKnowledgeChunkMutationResult, QwenKnowledgeChunkPage, QwenKnowledgeChunkRef,
    QwenKnowledgeListChunksRequest, QwenKnowledgeUpdateChunkRequest,
};
pub mod monitoring;
pub use monitoring::{QwenKnowledgeMonitoringRequest, QwenKnowledgeMonitoringResult};
pub mod file_management;
pub use file_management::{
    QwenKnowledgeDataFile, QwenKnowledgeFileDeleteResult, QwenKnowledgeFileListRequest,
    QwenKnowledgeFileListResult, QwenKnowledgeFileTagUpdate, QwenKnowledgeFileTagUpdateItemResult,
    QwenKnowledgeFileTagUpdateMode, QwenKnowledgeFileTagUpdateRequest,
    QwenKnowledgeFileTagUpdateResult,
};
pub mod files;
pub use files::{
    QwenKnowledgeCategoryType, QwenKnowledgeFileDetails, QwenKnowledgeFileRef,
    QwenKnowledgeFileRegistration, QwenKnowledgeFileUploadLease, QwenKnowledgeFileUploadRequest,
    QwenKnowledgeParser, QwenKnowledgeParserConfig, QwenKnowledgeRegisterFileRequest,
};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_QUERY_CHARS: usize = 8 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const RETRIEVE_PATH: &str = "/api/v1/indices/rag/index/retrieve";
const CREATE_INDEX_PATH: &str = "/api/v1/indices/rag/index/create_v2";
const LIST_INDEX_PATH: &str = "/api/v1/indices/rag/index/list";
const UPDATE_INDEX_PATH: &str = "/api/v1/indices/rag/index/update";
const DELETE_INDEX_PATH: &str = "/api/v1/indices/rag/index/delete";
const LIST_DOCUMENTS_PATH: &str = "/api/v1/indices/rag/index/files";
const DOCUMENT_DETAILS_PATH: &str = "/api/v1/indices/rag/list/index/file/details";
const DELETE_DOCUMENTS_PATH: &str = "/api/v1/indices/rag/index/delete_file";
const SUBMIT_IMPORT_PATH: &str = "/api/v1/indices/rag/index/job/create";
const IMPORT_STATUS_PATH: &str = "/api/v1/indices/rag/index_job/status";

/// Model Studio regions relevant to the managed knowledge-base API.
///
/// The current RAG REST reference documents the workspace URL and low-level
/// retrieve route for Beijing. Singapore remains represented in resource
/// scopes because Model Studio knowledge resources are region-bound, but this
/// service refuses to infer that the Beijing REST route is available there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenKnowledgeRegion {
    Beijing,
    Singapore,
}

impl QwenKnowledgeRegion {
    fn host_suffix(self) -> &'static str {
        match self {
            Self::Beijing => "cn-beijing.maas.aliyuncs.com",
            Self::Singapore => "ap-southeast-1.maas.aliyuncs.com",
        }
    }

    fn supports_rag_rest(self) -> bool {
        matches!(self, Self::Beijing)
    }
}

/// Explicit identity for one Qwen Model Studio workspace and account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    region: QwenKnowledgeRegion,
    workspace_id: String,
}

impl QwenKnowledgeScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenKnowledgeRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        let scope = Self {
            provider_id: ProviderId::from("qwen"),
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn region(&self) -> QwenKnowledgeRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    /// Bind an ID obtained from Model Studio to this exact connection scope.
    pub fn knowledge_ref(
        &self,
        index_id: impl Into<String>,
    ) -> Result<QwenKnowledgeRef, QwenKnowledgeError> {
        QwenKnowledgeRef::from_scope(self, index_id)
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.provider_id.as_str() != "qwen"
            || self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || !valid_workspace_id(&self.workspace_id)
        {
            return Err(invalid(
                "Qwen knowledge scope requires a profile, non-secret account scope, and valid workspace ID",
            ));
        }
        Ok(())
    }

    fn endpoint_fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(&format!(
            "https://{}.{}",
            self.workspace_id,
            self.region.host_suffix()
        ))
    }
}

/// Opaque identity for a knowledge base in one Qwen workspace/account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    index_id: String,
}

impl QwenKnowledgeRef {
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        index_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let index_id = index_id.into();
        if !valid_resource_id(&index_id) {
            return Err(invalid("Qwen knowledge index ID is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            index_id,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn index_id(&self) -> &str {
        &self.index_id
    }

    pub fn document_ref(
        &self,
        document_id: impl Into<String>,
    ) -> Result<QwenKnowledgeDocumentRef, QwenKnowledgeError> {
        QwenKnowledgeDocumentRef::new(self, document_id)
    }

    pub fn job_ref(
        &self,
        job_id: impl Into<String>,
    ) -> Result<QwenKnowledgeJobRef, QwenKnowledgeError> {
        QwenKnowledgeJobRef::new(self, job_id)
    }
}

/// A document identity scoped to one knowledge base and Model Studio workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeDocumentRef {
    knowledge: QwenKnowledgeRef,
    document_id: String,
}

impl QwenKnowledgeDocumentRef {
    pub fn new(
        knowledge: &QwenKnowledgeRef,
        document_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        let document_id = document_id.into();
        if !valid_resource_id(&document_id) {
            return Err(invalid("Qwen knowledge document ID is invalid"));
        }
        Ok(Self {
            knowledge: knowledge.clone(),
            document_id,
        })
    }

    pub fn knowledge(&self) -> &QwenKnowledgeRef {
        &self.knowledge
    }

    pub fn document_id(&self) -> &str {
        &self.document_id
    }
}

/// An import task identity scoped to one knowledge base and workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeJobRef {
    knowledge: QwenKnowledgeRef,
    job_id: String,
}

impl QwenKnowledgeJobRef {
    pub fn new(
        knowledge: &QwenKnowledgeRef,
        job_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        let job_id = job_id.into();
        if !valid_resource_id(&job_id) {
            return Err(invalid("Qwen knowledge import job ID is invalid"));
        }
        Ok(Self {
            knowledge: knowledge.clone(),
            job_id,
        })
    }

    pub fn knowledge(&self) -> &QwenKnowledgeRef {
        &self.knowledge
    }

    pub fn job_id(&self) -> &str {
        &self.job_id
    }
}

/// One low-level retrieval request. `top_k` is omitted to use Model Studio's
/// default; the RAG REST reference does not publish a numeric range for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeRetrieveRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

impl QwenKnowledgeRetrieveRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: None,
        }
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.query.trim().is_empty() || self.query.chars().count() > MAX_QUERY_CHARS {
            return Err(invalid(format!(
                "Qwen knowledge query must contain 1 to {MAX_QUERY_CHARS} characters"
            )));
        }
        Ok(())
    }
}

/// Paginated knowledge-base list request. The REST API uses `page_number` in
/// the query string; it does not use `page_num` for this endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeListRequest {
    pub page_number: u32,
    pub page_size: u32,
    pub pipeline_name: Option<String>,
}

impl Default for QwenKnowledgeListRequest {
    fn default() -> Self {
        Self {
            page_number: 1,
            page_size: 10,
            pipeline_name: None,
        }
    }
}

impl QwenKnowledgeListRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_page(mut self, page_number: u32, page_size: u32) -> Self {
        self.page_number = page_number;
        self.page_size = page_size;
        self
    }

    pub fn with_name_filter(mut self, pipeline_name: impl Into<String>) -> Self {
        self.pipeline_name = Some(pipeline_name.into());
        self
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        validate_page(self.page_number, self.page_size, 100)?;
        if self.pipeline_name.as_ref().is_some_and(|name| {
            name.trim().is_empty() || name.chars().count() > 20 || has_control(name)
        }) {
            return Err(invalid(
                "pipeline_name filter must contain 1 to 20 characters",
            ));
        }
        Ok(())
    }
}

/// The structure mode for a newly created knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenKnowledgeStructureType {
    Unstructured,
    Structured,
}

/// A source descriptor included in the `create_v2` request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct QwenKnowledgeDataSource {
    pub source_type: String,
}

impl QwenKnowledgeDataSource {
    pub fn new(source_type: impl Into<String>) -> Self {
        Self {
            source_type: source_type.into(),
        }
    }
}

/// Typed request for the documented create-and-import endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenKnowledgeCreateRequest {
    pub name: String,
    pub description: String,
    pub structure_type: QwenKnowledgeStructureType,
    pub sink_type: String,
    pub source_type: String,
    pub document_ids: Vec<String>,
    pub category_ids: Vec<String>,
    pub data_sources: Vec<QwenKnowledgeDataSource>,
    pub knowledge_type: Option<String>,
    pub knowledge_scene: Option<String>,
    pub embedding_model_name: Option<String>,
    pub multimodal_embedding_model_name: Option<String>,
    pub chunk_size: Option<u32>,
}

impl QwenKnowledgeCreateRequest {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        document_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            structure_type: QwenKnowledgeStructureType::Unstructured,
            sink_type: "DEFAULT".into(),
            source_type: "DATA_CENTER_FILE".into(),
            document_ids: document_ids.into_iter().map(Into::into).collect(),
            category_ids: Vec::new(),
            data_sources: vec![QwenKnowledgeDataSource::new("DATA_CENTER_FILE")],
            knowledge_type: None,
            knowledge_scene: None,
            embedding_model_name: None,
            multimodal_embedding_model_name: None,
            chunk_size: None,
        }
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.name.trim().is_empty() || self.name.chars().count() > 20 || has_control(&self.name)
        {
            return Err(invalid(
                "knowledge-base name must contain 1 to 20 characters",
            ));
        }
        if self.description.trim().is_empty()
            || self.description.chars().count() > 200
            || has_control(&self.description)
        {
            return Err(invalid(
                "knowledge-base description must contain 1 to 200 characters",
            ));
        }
        if self.document_ids.is_empty() || self.document_ids.iter().any(|id| !valid_resource_id(id))
        {
            return Err(invalid("create_v2 requires at least one valid document ID"));
        }
        if self.category_ids.iter().any(|id| !valid_resource_id(id)) {
            return Err(invalid("category IDs must be valid resource IDs"));
        }
        if self.source_type.trim().is_empty()
            || has_control(&self.source_type)
            || self.sink_type.trim().is_empty()
        {
            return Err(invalid("source_type and sink_type must be nonempty"));
        }
        if self.data_sources.is_empty()
            || self.data_sources.iter().any(|source| {
                source.source_type.trim().is_empty() || has_control(&source.source_type)
            })
        {
            return Err(invalid("create_v2 requires a nonempty data_sources list"));
        }
        if self.knowledge_type.is_some() != self.knowledge_scene.is_some() {
            return Err(invalid(
                "knowledge_type and knowledge_scene must be supplied together",
            ));
        }
        if let (Some(knowledge_type), Some(scene)) = (
            self.knowledge_type.as_deref(),
            self.knowledge_scene.as_deref(),
        ) {
            let compatible = match knowledge_type {
                "document" => {
                    self.structure_type == QwenKnowledgeStructureType::Unstructured
                        && matches!(
                            scene,
                            "basic_document_qa"
                                | "visual_perception_qa"
                                | "lite_document_qa"
                                | "visual_document_qa"
                        )
                }
                "table" => {
                    self.structure_type == QwenKnowledgeStructureType::Structured
                        && scene == "basic_table_qa"
                }
                "image" => {
                    self.structure_type == QwenKnowledgeStructureType::Unstructured
                        && scene == "image_qa"
                }
                "multimedia" => {
                    self.structure_type == QwenKnowledgeStructureType::Unstructured
                        && scene == "basic_multimedia_qa"
                }
                _ => false,
            };
            if !compatible {
                return Err(invalid(
                    "knowledge_type, knowledge_scene, and structure_type do not match the documented combinations",
                ));
            }
            if scene == "lite_document_qa" && self.sink_type != "BUILT_IN" {
                return Err(invalid("lite_document_qa requires sink_type BUILT_IN"));
            }
            if matches!(scene, "visual_perception_qa" | "image_qa")
                && self
                    .multimodal_embedding_model_name
                    .as_ref()
                    .is_none_or(|model| model.trim().is_empty() || has_control(model))
            {
                return Err(invalid(
                    "visual_perception_qa and image_qa require a multimodal embedding model",
                ));
            }
        }
        if self
            .embedding_model_name
            .as_ref()
            .is_some_and(|model| model.trim().is_empty() || has_control(model))
            || self
                .multimodal_embedding_model_name
                .as_ref()
                .is_some_and(|model| model.trim().is_empty() || has_control(model))
        {
            return Err(invalid(
                "embedding model names must be nonempty and contain no control characters",
            ));
        }
        if self.chunk_size == Some(0) {
            return Err(invalid("chunk_size must be positive when specified"));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({
            "name": self.name,
            "description": self.description,
            "structureType": self.structure_type,
            "sinkType": self.sink_type,
            "sourceType": self.source_type,
            "docIds": self.document_ids,
            "dataSources": self.data_sources,
        });
        if !self.category_ids.is_empty() {
            body["categoryIds"] = json!(self.category_ids);
        }
        if let Some(value) = &self.knowledge_type {
            body["knowledgeType"] = json!(value);
        }
        if let Some(value) = &self.knowledge_scene {
            body["knowledgeScene"] = json!(value);
        }
        if let Some(value) = &self.embedding_model_name {
            body["embeddingModelName"] = json!(value);
        }
        if let Some(value) = &self.multimodal_embedding_model_name {
            body["multimodalEmbeddingModelName"] = json!(value);
        }
        if let Some(value) = self.chunk_size {
            body["chunkSize"] = json!(value);
        }
        body
    }
}

/// The optional fields accepted by the knowledge-base update endpoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QwenKnowledgeUpdateRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub rerank_min_score: Option<f64>,
}

impl QwenKnowledgeUpdateRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_rerank_min_score(mut self, score: f64) -> Self {
        self.rerank_min_score = Some(score);
        self
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.name.is_none() && self.description.is_none() && self.rerank_min_score.is_none() {
            return Err(invalid("at least one knowledge-base field must be updated"));
        }
        if self.name.as_ref().is_some_and(|name| {
            name.trim().is_empty() || name.chars().count() > 20 || has_control(name)
        }) {
            return Err(invalid(
                "knowledge-base name must contain 1 to 20 characters",
            ));
        }
        if self
            .description
            .as_ref()
            .is_some_and(|description| description.trim().is_empty() || has_control(description))
        {
            return Err(invalid(
                "knowledge-base description must be nonempty and contain no control characters",
            ));
        }
        if self
            .rerank_min_score
            .is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
        {
            return Err(invalid("rerank_min_score must be between 0 and 1"));
        }
        Ok(())
    }

    fn to_body(&self, knowledge: &QwenKnowledgeRef) -> Value {
        let mut body = json!({"id": knowledge.index_id});
        if let Some(value) = &self.name {
            body["name"] = json!(value);
        }
        if let Some(value) = &self.description {
            body["description"] = json!(value);
        }
        if let Some(value) = self.rerank_min_score {
            body["rerankMinScore"] = json!(value);
        }
        body
    }
}

/// A specific file or category source for an import job. This avoids the
/// endpoint's hazardous implicit default of importing every workspace file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenKnowledgeImportSource {
    Files(Vec<String>),
    Categories(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenKnowledgeChunkMode {
    H1,
    H2,
    H3,
    H4,
    H5,
    Length,
    Page,
    Regex,
}

impl QwenKnowledgeChunkMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::H1 => "h1",
            Self::H2 => "h2",
            Self::H3 => "h3",
            Self::H4 => "h4",
            Self::H5 => "h5",
            Self::Length => "length",
            Self::Page => "page",
            Self::Regex => "regex",
        }
    }
}

/// Options for appending existing data-center files to a knowledge base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeImportRequest {
    pub source: QwenKnowledgeImportSource,
    pub chunk_mode: Option<QwenKnowledgeChunkMode>,
    pub chunk_size: Option<u32>,
    pub overlap_size: Option<u32>,
    pub separator: Option<String>,
    pub enable_headers: Option<bool>,
}

impl QwenKnowledgeImportRequest {
    pub fn from_files(document_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            source: QwenKnowledgeImportSource::Files(
                document_ids.into_iter().map(Into::into).collect(),
            ),
            chunk_mode: None,
            chunk_size: None,
            overlap_size: None,
            separator: None,
            enable_headers: None,
        }
    }

    pub fn from_categories(category_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            source: QwenKnowledgeImportSource::Categories(
                category_ids.into_iter().map(Into::into).collect(),
            ),
            chunk_mode: None,
            chunk_size: None,
            overlap_size: None,
            separator: None,
            enable_headers: None,
        }
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let ids = match &self.source {
            QwenKnowledgeImportSource::Files(ids) | QwenKnowledgeImportSource::Categories(ids) => {
                ids
            }
        };
        if ids.is_empty() || ids.iter().any(|id| !valid_resource_id(id)) {
            return Err(invalid(
                "import requires a nonempty list of valid file or category IDs",
            ));
        }
        if self
            .chunk_size
            .is_some_and(|size| !(1..=6000).contains(&size))
        {
            return Err(invalid("chunk_size must be between 1 and 6000"));
        }
        if self.overlap_size.is_some_and(|size| size > 1024) {
            return Err(invalid("overlap_size must be at most 1024"));
        }
        if self.chunk_mode == Some(QwenKnowledgeChunkMode::Length) && self.chunk_size.is_none() {
            return Err(invalid("chunk_size is required for length chunking"));
        }
        if self.chunk_mode == Some(QwenKnowledgeChunkMode::Regex)
            && self
                .separator
                .as_ref()
                .is_none_or(|value| value.is_empty() || has_control(value))
        {
            return Err(invalid("separator is required for regex chunking"));
        }
        Ok(())
    }

    fn to_body(&self, knowledge: &QwenKnowledgeRef) -> Value {
        let mut body = json!({"indexId": knowledge.index_id});
        match &self.source {
            QwenKnowledgeImportSource::Files(ids) => {
                body["sourceType"] = json!("DATA_CENTER_FILE");
                body["docIds"] = json!(ids);
            }
            QwenKnowledgeImportSource::Categories(ids) => {
                body["sourceType"] = json!("DATA_CENTER_CATEGORY");
                body["categoryIds"] = json!(ids);
            }
        }
        if let Some(mode) = self.chunk_mode {
            body["chunkMode"] = json!(mode.as_str());
        }
        if let Some(size) = self.chunk_size {
            body["chunkSize"] = json!(size);
        }
        if let Some(size) = self.overlap_size {
            body["overlapSize"] = json!(size);
        }
        if let Some(separator) = &self.separator {
            body["separator"] = json!(separator);
        }
        if let Some(enable_headers) = self.enable_headers {
            body["enableHeaders"] = json!(enable_headers);
        }
        body
    }
}

/// Pagination used by the separate document list and details routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenKnowledgeDocumentPageRequest {
    pub page_number: u32,
    pub page_size: u32,
}

impl Default for QwenKnowledgeDocumentPageRequest {
    fn default() -> Self {
        Self {
            page_number: 1,
            page_size: 10,
        }
    }
}

impl QwenKnowledgeDocumentPageRequest {
    pub fn new(page_number: u32, page_size: u32) -> Self {
        Self {
            page_number,
            page_size,
        }
    }

    fn validate_list(&self) -> Result<(), QwenKnowledgeError> {
        validate_page(self.page_number, self.page_size, 100)
    }

    fn validate_details(&self) -> Result<(), QwenKnowledgeError> {
        validate_page(self.page_number, self.page_size, 10)
    }
}

/// One retrieved document chunk. Native metadata and the complete node remain
/// available because the provider can add fields independently of this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeMatch {
    pub text: String,
    pub score: Option<f64>,
    pub metadata: Value,
    pub native: Value,
}

/// Low-level RAG retrieval output from Model Studio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeRetrieveResult {
    pub matches: Vec<QwenKnowledgeMatch>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeBase {
    pub knowledge: QwenKnowledgeRef,
    pub name: Option<String>,
    pub description: Option<String>,
    pub structure_type: Option<String>,
    pub status: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeListResult {
    pub bases: Vec<QwenKnowledgeBase>,
    pub page_number: u32,
    pub page_size: u32,
    pub total_count: u64,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeCreateResult {
    pub knowledge: QwenKnowledgeRef,
    pub job: QwenKnowledgeJobRef,
    pub status: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeUpdateResult {
    pub knowledge: QwenKnowledgeRef,
    pub created_at: Option<u64>,
    pub updated_at: Option<u64>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeDeleteResult {
    pub request_id: Option<String>,
    pub deleted_document_ids: Option<Vec<String>>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeDocument {
    pub reference: QwenKnowledgeDocumentRef,
    pub name: Option<String>,
    pub document_type: Option<String>,
    pub status: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
    pub size_bytes: Option<u64>,
    pub ingestion_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeDocumentPage {
    pub documents: Vec<QwenKnowledgeDocument>,
    pub page_number: u32,
    pub page_size: u32,
    pub total_count: u64,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeImportJobResult {
    pub reference: QwenKnowledgeJobRef,
    pub status: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeImportJobStatus {
    pub reference: QwenKnowledgeJobRef,
    pub ingestion_status: Option<String>,
    pub ingestion_message: Option<String>,
    pub total_count: u64,
    pub page_number: u32,
    pub page_size: u32,
    pub documents: Vec<QwenKnowledgeDocument>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenKnowledgeDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, thiserror::Error)]
pub enum QwenKnowledgeError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Qwen knowledge input: {0}")]
    InvalidInput(String),
    #[error("Qwen knowledge API returned HTTP {status}")]
    Provider {
        status: u16,
        code: Option<String>,
        message: Option<String>,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: QwenKnowledgeDispatch,
    },
    #[error("invalid Qwen knowledge response for {operation}: {message}")]
    InvalidResponse {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
        dispatch: QwenKnowledgeDispatch,
    },
    #[error("Qwen knowledge write outcome is unknown for {operation}: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("Qwen knowledge write outcome is unknown for {operation}: {message}")]
    ResponseOutcomeUnknown {
        operation: &'static str,
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
}

impl QwenKnowledgeError {
    /// Reports write dispatch certainty. Errors from read-only operations
    /// return `NotSent` because they do not create or modify resources.
    pub fn dispatch(&self) -> QwenKnowledgeDispatch {
        match self {
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { dispatch, .. } => *dispatch,
            Self::OutcomeUnknown { .. } | Self::ResponseOutcomeUnknown { .. } => {
                QwenKnowledgeDispatch::Unknown
            }
            Self::Llm(_) | Self::InvalidInput(_) => QwenKnowledgeDispatch::NotSent,
        }
    }
}

/// Access to caller-scoped Qwen knowledge resources in one Model Studio workspace.
///
/// The API key and account identity are supplied by the caller. This service
/// receives the API key through each operation's request options. Import documents by IDs already
/// registered in the workspace's data center; direct OSS file upload is a
/// separate workflow and is not performed by this service.
#[derive(Clone)]
pub struct QwenKnowledgeService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    http: &'a dyn Transport,
    scope: QwenKnowledgeScope,
}

impl<'a> QwenKnowledgeService<'a> {
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

    fn request_credential<'o>(
        &self,
        request_options: &'o crate::RequestOptions,
    ) -> Result<&'o str, QwenKnowledgeError> {
        if request_options
            .account_scope
            .as_deref()
            .is_some_and(|account| account != self.scope.account_scope())
        {
            return Err(crate::protocol::LlmError::PermissionDenied {
                message: "request account scope does not match the native resource scope".into(),
            }
            .into());
        }

        let credential = request_options
            .credential
            .as_ref()
            .map(|value| value.expose_secret().as_str())
            .unwrap_or_default();
        if credential.trim().is_empty() {
            return Err(invalid("Qwen Model Studio API key must be non-empty"));
        }
        Ok(credential)
    }

    pub fn new(
        http: &'a dyn Transport,
        scope: QwenKnowledgeScope,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;

        Ok(Self {
            binding: None,
            http,
            scope,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn knowledge_ref(
        &self,
        index_id: impl Into<String>,
    ) -> Result<QwenKnowledgeRef, QwenKnowledgeError> {
        self.scope.knowledge_ref(index_id)
    }

    /// Retrieve chunks from one already-created knowledge base.
    pub async fn retrieve(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeRetrieveRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeRetrieveResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        request.validate()?;
        let mut body = json!({"index_id": knowledge.index_id, "query": request.query});
        if let Some(top_k) = request.top_k {
            body["top_k"] = json!(top_k);
        }
        let response = pinned_service
            .request_json(
                "POST",
                RETRIEVE_PATH,
                &[],
                Some(body),
                "retrieve",
                false,
                request_options,
            )
            .await?;
        decode_retrieve(response)
    }

    /// List a page of knowledge bases in this workspace. The current RAG REST
    /// catalog has no dedicated knowledge-base detail route.
    pub async fn list_knowledge_bases(
        &self,
        request: &QwenKnowledgeListRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeListResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        let mut query = vec![
            ("page_number", request.page_number.to_string()),
            ("page_size", request.page_size.to_string()),
        ];
        if let Some(name) = &request.pipeline_name {
            query.push(("pipeline_name", name.clone()));
        }
        let response = pinned_service
            .request_json(
                "GET",
                LIST_INDEX_PATH,
                &query,
                None,
                "list_knowledge_bases",
                false,
                request_options,
            )
            .await?;
        decode_knowledge_list(&pinned_service.scope, response)
    }

    /// Find one knowledge-base record by exact ID using only the documented
    /// paginated list endpoint; this performs one GET per page until found.
    pub async fn find_knowledge_base(
        &self,
        knowledge: &QwenKnowledgeRef,
        request_options: &crate::RequestOptions,
    ) -> Result<Option<QwenKnowledgeBase>, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        let deadline = crate::runtime::Deadline::after(request_options.total_timeout);
        let mut page_options = request_options.clone();
        let mut page_number = 1_u32;
        loop {
            page_options.total_timeout = deadline.remaining()?;
            let page = deadline
                .run(pinned_service.list_knowledge_bases(
                    &QwenKnowledgeListRequest::new().with_page(page_number, 100),
                    &page_options,
                ))
                .await??;
            if let Some(base) = page
                .bases
                .into_iter()
                .find(|base| base.knowledge.index_id == knowledge.index_id)
            {
                return Ok(Some(base));
            }
            let pages = page.total_count.div_ceil(u64::from(page.page_size.max(1)));
            if u64::from(page_number) >= pages || page.page_size == 0 {
                return Ok(None);
            }
            page_number = page_number
                .checked_add(1)
                .ok_or_else(|| invalid("knowledge-base page number overflowed"))?;
        }
    }

    /// Create a knowledge base and import its initial files in one request.
    pub async fn create_knowledge_base(
        &self,
        request: &QwenKnowledgeCreateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeCreateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        let response = pinned_service
            .request_json(
                "POST",
                CREATE_INDEX_PATH,
                &[],
                Some(request.to_body()),
                "create_knowledge_base",
                true,
                request_options,
            )
            .await?;
        let data = response.native.get("data").unwrap_or(&Value::Null);
        let index_id = data
            .get("pipelineId")
            .and_then(Value::as_str)
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| {
                response_invalid(
                    "create_knowledge_base",
                    "response omitted data.pipelineId",
                    &response,
                    QwenKnowledgeDispatch::Accepted,
                )
            })?;
        let job_id = data
            .get("ingestionId")
            .and_then(Value::as_str)
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| {
                response_invalid(
                    "create_knowledge_base",
                    "response omitted data.ingestionId",
                    &response,
                    QwenKnowledgeDispatch::Accepted,
                )
            })?;
        let knowledge = pinned_service.knowledge_ref(index_id)?;
        let job = knowledge.job_ref(job_id)?;
        let status = data
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok(QwenKnowledgeCreateResult {
            knowledge,
            job,
            status,
            request_id: response.request_id,
            native: response.native,
        })
    }

    /// Update name, description, or rerank threshold. Only the fields
    /// documented by Model Studio are accepted.
    pub async fn update_knowledge_base(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeUpdateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeUpdateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        request.validate()?;
        let response = pinned_service
            .request_json(
                "POST",
                UPDATE_INDEX_PATH,
                &[],
                Some(request.to_body(knowledge)),
                "update_knowledge_base",
                true,
                request_options,
            )
            .await?;
        let data = response.native.get("data").unwrap_or(&Value::Null);
        let returned_id = data
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| {
                response_invalid(
                    "update_knowledge_base",
                    "response omitted data.id",
                    &response,
                    QwenKnowledgeDispatch::Accepted,
                )
            })?;
        if returned_id != knowledge.index_id {
            return Err(response_invalid(
                "update_knowledge_base",
                "response ID did not match the requested knowledge base",
                &response,
                QwenKnowledgeDispatch::Accepted,
            ));
        }
        Ok(QwenKnowledgeUpdateResult {
            knowledge: knowledge.clone(),
            created_at: data.get("created_at").and_then(Value::as_u64),
            updated_at: data.get("updated_at").and_then(Value::as_u64),
            request_id: response.request_id,
            native: response.native,
        })
    }

    /// Permanently delete a knowledge base and its documents/chunks.
    pub async fn delete_knowledge_base(
        &self,
        knowledge: &QwenKnowledgeRef,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeDeleteResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        let response = pinned_service
            .request_json(
                "POST",
                DELETE_INDEX_PATH,
                &[],
                Some(json!({"index_id": knowledge.index_id})),
                "delete_knowledge_base",
                true,
                request_options,
            )
            .await?;
        Ok(QwenKnowledgeDeleteResult {
            request_id: response.request_id,
            deleted_document_ids: None,
            native: response.native,
        })
    }

    /// List documents. The endpoint-specific reference uses `page_number` in
    /// the query string, despite a different spelling in the overview table.
    pub async fn list_documents(
        &self,
        knowledge: &QwenKnowledgeRef,
        page: QwenKnowledgeDocumentPageRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeDocumentPage, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        page.validate_list()?;
        let query = vec![
            ("index_id", knowledge.index_id.clone()),
            ("page_number", page.page_number.to_string()),
            ("page_size", page.page_size.to_string()),
        ];
        let response = pinned_service
            .request_json(
                "GET",
                LIST_DOCUMENTS_PATH,
                &query,
                None,
                "list_documents",
                false,
                request_options,
            )
            .await?;
        decode_document_page(knowledge, response, "list_documents")
    }

    /// Get document parsing/indexing details using the separately documented
    /// camelCase request schema and its 10-item page limit.
    pub async fn list_document_details(
        &self,
        knowledge: &QwenKnowledgeRef,
        page: QwenKnowledgeDocumentPageRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeDocumentPage, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        page.validate_details()?;
        let body = json!({
            "indexId": knowledge.index_id,
            "pageNumber": page.page_number,
            "pageSize": page.page_size,
        });
        let response = pinned_service
            .request_json(
                "POST",
                DOCUMENT_DETAILS_PATH,
                &[],
                Some(body),
                "list_document_details",
                false,
                request_options,
            )
            .await?;
        decode_document_page(knowledge, response, "list_document_details")
    }

    /// Delete selected document IDs and their chunks from a knowledge base.
    pub async fn delete_documents(
        &self,
        knowledge: &QwenKnowledgeRef,
        documents: &[QwenKnowledgeDocumentRef],
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeDeleteResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        if documents.is_empty() {
            return Err(invalid(
                "at least one scoped document reference is required",
            ));
        }
        for document in documents {
            pinned_service.validate_document_ref(knowledge, document)?;
        }
        let ids = documents
            .iter()
            .map(|document| document.document_id.clone())
            .collect::<Vec<_>>();
        let response = pinned_service
            .request_json(
                "POST",
                DELETE_DOCUMENTS_PATH,
                &[],
                Some(json!({"index_id": knowledge.index_id, "doc_ids": ids})),
                "delete_documents",
                true,
                request_options,
            )
            .await?;
        let deleted_document_ids = response
            .native
            .get("data")
            .and_then(|data| data.get("deleted"))
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .ok_or_else(|| {
                response_invalid(
                    "delete_documents",
                    "response omitted data.deleted",
                    &response,
                    QwenKnowledgeDispatch::Accepted,
                )
            })?;
        Ok(QwenKnowledgeDeleteResult {
            request_id: response.request_id,
            deleted_document_ids: Some(deleted_document_ids),
            native: response.native,
        })
    }

    /// Append existing data-center documents/categories and return the task
    /// reference needed by `get_import_job_status`.
    pub async fn submit_import_job(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeImportRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeImportJobResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_knowledge_ref(knowledge)?;
        request.validate()?;
        let response = pinned_service
            .request_json(
                "POST",
                SUBMIT_IMPORT_PATH,
                &[],
                Some(request.to_body(knowledge)),
                "submit_import_job",
                true,
                request_options,
            )
            .await?;
        let data = response.native.get("data").unwrap_or(&Value::Null);
        let job_id = data
            .get("ingestionId")
            .and_then(Value::as_str)
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| {
                response_invalid(
                    "submit_import_job",
                    "response omitted data.ingestionId",
                    &response,
                    QwenKnowledgeDispatch::Accepted,
                )
            })?;
        let reference = knowledge.job_ref(job_id)?;
        Ok(QwenKnowledgeImportJobResult {
            reference,
            status: data
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id: response.request_id,
            native: response.native,
        })
    }

    /// Query one import job once. Pacing remains caller-managed.
    pub async fn get_import_job_status(
        &self,
        job: &QwenKnowledgeJobRef,
        page: QwenKnowledgeDocumentPageRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeImportJobStatus, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_job_ref(job)?;
        page.validate_list()?;
        let query = vec![
            ("index_id", job.knowledge.index_id.clone()),
            ("job_id", job.job_id.clone()),
            ("page_number", page.page_number.to_string()),
            ("page_size", page.page_size.to_string()),
        ];
        let response = pinned_service
            .request_json(
                "GET",
                IMPORT_STATUS_PATH,
                &query,
                None,
                "get_import_job_status",
                false,
                request_options,
            )
            .await?;
        decode_import_job_status(job, response)
    }

    #[allow(clippy::too_many_arguments)]
    async fn request_json(
        &self,
        method: &str,
        path: &str,
        query: &[(&str, String)],
        body: Option<Value>,
        operation: &'static str,
        write: bool,
        request_options: &crate::RequestOptions,
    ) -> Result<DecodedEnvelope, QwenKnowledgeError> {
        self.ensure_supported_region()?;
        let url = self.operation_url(path, query)?;
        let body_bytes = match body {
            Some(value) => serde_json::to_vec(&value)
                .map_err(|_| invalid("Qwen knowledge request body cannot be encoded"))?,
            None => Vec::new(),
        };
        let mut headers = vec![
            (
                "authorization".into(),
                format!("Bearer {}", self.request_credential(request_options)?),
            ),
            ("accept".into(), "application/json".into()),
        ];
        if method == "POST" {
            headers.push(("content-type".into(), "application/json".into()));
        }
        let request = HttpRequest {
            http1_header_layout: None,
            method: method.into(),
            url: url.to_string(),
            headers,
            body: Bytes::from(body_bytes),
            timeout: request_options.total_timeout.or(Some(DEFAULT_TIMEOUT)),
        };
        let response = HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if write {
                    QwenKnowledgeError::OutcomeUnknown { operation, source }
                } else {
                    QwenKnowledgeError::Llm(source)
                }
            })?;
        decode_envelope(response, operation, write)
    }

    fn ensure_supported_region(&self) -> Result<(), QwenKnowledgeError> {
        if !self.scope.region.supports_rag_rest() {
            return Err(LlmError::UnsupportedCapability {
                message: "the current Model Studio RAG REST reference documents this API family for Beijing; Singapore routes are not specified".into(),
            }
            .into());
        }
        Ok(())
    }

    fn operation_url(
        &self,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<Url, QwenKnowledgeError> {
        let mut url = Url::parse(&format!(
            "https://{}.{}",
            self.scope.workspace_id,
            self.scope.region.host_suffix()
        ))
        .map_err(|_| invalid("Qwen knowledge workspace URL is invalid"))?;
        url.set_path(path);
        if !query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (name, value) in query {
                pairs.append_pair(name, value);
            }
        }
        Ok(url)
    }

    fn validate_knowledge_ref(
        &self,
        knowledge: &QwenKnowledgeRef,
    ) -> Result<(), QwenKnowledgeError> {
        if knowledge.scope != self.scope
            || knowledge.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || !valid_resource_id(&knowledge.index_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "Qwen knowledge reference belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        Ok(())
    }

    fn validate_document_ref(
        &self,
        knowledge: &QwenKnowledgeRef,
        document: &QwenKnowledgeDocumentRef,
    ) -> Result<(), QwenKnowledgeError> {
        if document.knowledge != *knowledge || !valid_resource_id(&document.document_id) {
            return Err(LlmError::PermissionDenied {
                message: "Qwen document reference belongs to another knowledge base or workspace"
                    .into(),
            }
            .into());
        }
        self.validate_knowledge_ref(knowledge)
    }

    fn validate_job_ref(&self, job: &QwenKnowledgeJobRef) -> Result<(), QwenKnowledgeError> {
        self.validate_knowledge_ref(&job.knowledge)?;
        if !valid_resource_id(&job.job_id) {
            return Err(invalid("Qwen knowledge import job ID is invalid"));
        }
        Ok(())
    }
}

struct DecodedEnvelope {
    native: Value,
    request_id: Option<String>,
}

fn decode_envelope(
    response: HttpResponse,
    operation: &'static str,
    write: bool,
) -> Result<DecodedEnvelope, QwenKnowledgeError> {
    let success_policy = match operation {
        // These endpoint pages show successful examples with `code: Success`,
        // `status_code: 200`, and `request_id`, while their response-field
        // tables list `success` separately. Keep this exception local to the
        // documented chunk mutations and monitoring query.
        "update_chunk" | "delete_chunks" | "get_knowledge_base_monitoring" => {
            EnvelopeSuccessPolicy::AllowDocumentedStatusOnlySample
        }
        _ => EnvelopeSuccessPolicy::RequireSuccessBoolean,
    };
    decode_envelope_with_policy(response, operation, write, success_policy)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvelopeSuccessPolicy {
    RequireSuccessBoolean,
    AllowDocumentedStatusOnlySample,
}

fn decode_envelope_with_policy(
    response: HttpResponse,
    operation: &'static str,
    write: bool,
    success_policy: EnvelopeSuccessPolicy,
) -> Result<DecodedEnvelope, QwenKnowledgeError> {
    let header_request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let parsed = serde_json::from_slice::<Value>(&response.body);
    let native = match parsed {
        Ok(value) => value,
        Err(_) if write && (200..300).contains(&response.status) => {
            return Err(QwenKnowledgeError::ResponseOutcomeUnknown {
                operation,
                message: "successful HTTP response was not valid JSON".into(),
                request_id: header_request_id,
                native: Box::new(Value::Null),
            });
        }
        Err(_) if !(200..300).contains(&response.status) => {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }
        Err(_) => {
            return Err(QwenKnowledgeError::InvalidResponse {
                operation,
                message: "successful HTTP response was not valid JSON".into(),
                request_id: header_request_id,
                native: Box::new(Value::Null),
                dispatch: QwenKnowledgeDispatch::NotSent,
            });
        }
    };
    let request_id = request_id(&native, header_request_id);
    if !(200..300).contains(&response.status) {
        let dispatch = if write && (response.status == 408 || response.status >= 500) {
            QwenKnowledgeDispatch::Unknown
        } else if write {
            QwenKnowledgeDispatch::Rejected
        } else {
            QwenKnowledgeDispatch::NotSent
        };
        return Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: native
                .get("code")
                .and_then(Value::as_str)
                .map(str::to_owned),
            message: native
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id,
            native: Box::new(native),
            dispatch,
        });
    }
    let code = native.get("code").and_then(Value::as_str);
    let success = native.get("success").and_then(Value::as_bool);
    let business_status_code = native
        .get("status_code")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok());
    let has_malformed_business_status =
        native.get("status_code").is_some() && business_status_code.is_none();
    if has_malformed_business_status {
        if write {
            return Err(QwenKnowledgeError::ResponseOutcomeUnknown {
                operation,
                message: "provider response status_code was not an integer".into(),
                request_id,
                native: Box::new(native),
            });
        }
        return Err(QwenKnowledgeError::InvalidResponse {
            operation,
            message: "provider response status_code was not an integer".into(),
            request_id,
            native: Box::new(native),
            dispatch: QwenKnowledgeDispatch::NotSent,
        });
    }

    if let Some(status) = business_status_code.filter(|status| !(200..300).contains(status)) {
        return Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: code.map(str::to_owned),
            message: native
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id,
            native: Box::new(native),
            dispatch: if write && (status == 408 || status >= 500) {
                QwenKnowledgeDispatch::Unknown
            } else if write {
                QwenKnowledgeDispatch::Rejected
            } else {
                QwenKnowledgeDispatch::NotSent
            },
        });
    }

    let allow_status_only_sample = success_policy
        == EnvelopeSuccessPolicy::AllowDocumentedStatusOnlySample
        && native.get("success").is_none()
        && business_status_code == Some(200);
    if code == Some("Success") && (success == Some(true) || allow_status_only_sample) {
        return Ok(DecodedEnvelope { native, request_id });
    }
    if success == Some(false) || code.is_some_and(|code| code != "Success") {
        return Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: code.map(str::to_owned),
            message: native
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id,
            native: Box::new(native),
            dispatch: if write {
                QwenKnowledgeDispatch::Rejected
            } else {
                QwenKnowledgeDispatch::NotSent
            },
        });
    }
    if write {
        Err(QwenKnowledgeError::ResponseOutcomeUnknown {
            operation,
            message: "successful HTTP response omitted the documented success envelope".into(),
            request_id,
            native: Box::new(native),
        })
    } else {
        Err(QwenKnowledgeError::InvalidResponse {
            operation,
            message: "successful response omitted the documented success envelope".into(),
            request_id,
            native: Box::new(native),
            dispatch: QwenKnowledgeDispatch::NotSent,
        })
    }
}

fn decode_retrieve(
    response: DecodedEnvelope,
) -> Result<QwenKnowledgeRetrieveResult, QwenKnowledgeError> {
    let native = response.native;
    let rows = native
        .get("data")
        .and_then(|data| data.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            response_invalid(
                "retrieve",
                "response omitted data.nodes",
                &DecodedEnvelope {
                    native: native.clone(),
                    request_id: response.request_id.clone(),
                },
                QwenKnowledgeDispatch::NotSent,
            )
        })?;
    let matches = rows
        .iter()
        .cloned()
        .map(|node| {
            let text = node
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("retrieve", "node omitted text", node.clone()))?
                .to_owned();
            let score = node.get("score").and_then(Value::as_f64);
            let metadata = node.get("metadata").cloned().unwrap_or(Value::Null);
            Ok(QwenKnowledgeMatch {
                text,
                score,
                metadata,
                native: node,
            })
        })
        .collect::<Result<Vec<_>, QwenKnowledgeError>>()?;
    Ok(QwenKnowledgeRetrieveResult {
        matches,
        request_id: response.request_id,
        native,
    })
}

fn decode_knowledge_list(
    scope: &QwenKnowledgeScope,
    response: DecodedEnvelope,
) -> Result<QwenKnowledgeListResult, QwenKnowledgeError> {
    let data = response.native.get("data").ok_or_else(|| {
        response_invalid(
            "list_knowledge_bases",
            "response omitted data",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let page_number = required_u32(data, "page_number", "list_knowledge_bases", &response)?;
    let page_size = required_u32(data, "page_size", "list_knowledge_bases", &response)?;
    let total_count = data
        .get("total_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            response_invalid(
                "list_knowledge_bases",
                "data omitted total_count",
                &response,
                QwenKnowledgeDispatch::NotSent,
            )
        })?;
    let rows = data.get("rows").and_then(Value::as_array).ok_or_else(|| {
        response_invalid(
            "list_knowledge_bases",
            "data omitted rows",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let bases = rows
        .iter()
        .cloned()
        .map(|row| {
            let id = row
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| valid_resource_id(id))
                .ok_or_else(|| {
                    bad(
                        "list_knowledge_bases",
                        "row omitted a valid id",
                        row.clone(),
                    )
                })?;
            Ok(QwenKnowledgeBase {
                knowledge: QwenKnowledgeRef::from_scope(scope, id)?,
                name: row.get("name").and_then(Value::as_str).map(str::to_owned),
                description: row
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                structure_type: row
                    .get("structureType")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                status: row.get("status").and_then(Value::as_str).map(str::to_owned),
                native: row,
            })
        })
        .collect::<Result<Vec<_>, QwenKnowledgeError>>()?;
    Ok(QwenKnowledgeListResult {
        bases,
        page_number,
        page_size,
        total_count,
        request_id: response.request_id,
        native: response.native,
    })
}

fn decode_document_page(
    knowledge: &QwenKnowledgeRef,
    response: DecodedEnvelope,
    operation: &'static str,
) -> Result<QwenKnowledgeDocumentPage, QwenKnowledgeError> {
    let data = response.native.get("data").ok_or_else(|| {
        response_invalid(
            operation,
            "response omitted data",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let page_number = required_u32(data, "page_number", operation, &response)?;
    let page_size = required_u32(data, "page_size", operation, &response)?;
    let total_count = data
        .get("total_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            response_invalid(
                operation,
                "data omitted total_count",
                &response,
                QwenKnowledgeDispatch::NotSent,
            )
        })?;
    let rows = data.get("rows").and_then(Value::as_array).ok_or_else(|| {
        response_invalid(
            operation,
            "data omitted rows",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let documents = decode_documents(knowledge, rows, operation)?;
    Ok(QwenKnowledgeDocumentPage {
        documents,
        page_number,
        page_size,
        total_count,
        request_id: response.request_id,
        native: response.native,
    })
}

fn decode_import_job_status(
    job: &QwenKnowledgeJobRef,
    response: DecodedEnvelope,
) -> Result<QwenKnowledgeImportJobStatus, QwenKnowledgeError> {
    let data = response.native.get("data").ok_or_else(|| {
        response_invalid(
            "get_import_job_status",
            "response omitted data",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let id = data.get("id").and_then(Value::as_str).ok_or_else(|| {
        response_invalid(
            "get_import_job_status",
            "data omitted id",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    if id != job.job_id {
        return Err(response_invalid(
            "get_import_job_status",
            "response job ID did not match the scoped reference",
            &response,
            QwenKnowledgeDispatch::NotSent,
        ));
    }
    let page_number = required_u32(data, "page_number", "get_import_job_status", &response)?;
    let page_size = required_u32(data, "page_size", "get_import_job_status", &response)?;
    let total_count = data.get("total_count").and_then(Value::as_u64).unwrap_or(0);
    let rows = data
        .get("rows")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let documents = decode_documents(&job.knowledge, rows, "get_import_job_status")?;
    Ok(QwenKnowledgeImportJobStatus {
        reference: job.clone(),
        ingestion_status: data
            .get("ingestion_status")
            .and_then(Value::as_str)
            .map(str::to_owned),
        ingestion_message: data
            .get("ingestion_message")
            .and_then(Value::as_str)
            .map(str::to_owned),
        total_count,
        page_number,
        page_size,
        documents,
        request_id: response.request_id,
        native: response.native,
    })
}

fn decode_documents(
    knowledge: &QwenKnowledgeRef,
    rows: &[Value],
    operation: &'static str,
) -> Result<Vec<QwenKnowledgeDocument>, QwenKnowledgeError> {
    rows.iter()
        .cloned()
        .map(|row| {
            let id = row
                .get("doc_id")
                .and_then(Value::as_str)
                .filter(|id| valid_resource_id(id))
                .ok_or_else(|| {
                    bad(
                        operation,
                        "document row omitted a valid doc_id",
                        row.clone(),
                    )
                })?;
            Ok(QwenKnowledgeDocument {
                reference: knowledge.document_ref(id)?,
                name: row
                    .get("doc_name")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                document_type: row
                    .get("doc_type")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                status: row.get("status").and_then(Value::as_str).map(str::to_owned),
                code: row.get("code").and_then(Value::as_str).map(str::to_owned),
                message: row
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                size_bytes: row.get("size").and_then(Value::as_u64),
                ingestion_id: row
                    .get("ingestion_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: row,
            })
        })
        .collect()
}

fn valid_workspace_id(value: &str) -> bool {
    value.starts_with("llm-")
        && value.len() > 4
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn valid_resource_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn invalid(message: impl Into<String>) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidInput(message.into())
}

fn bad(operation: &'static str, message: &str, native: Value) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidResponse {
        operation,
        message: message.into(),
        request_id: None,
        native: Box::new(native),
        dispatch: QwenKnowledgeDispatch::NotSent,
    }
}

fn response_invalid(
    operation: &'static str,
    message: &str,
    response: &DecodedEnvelope,
    dispatch: QwenKnowledgeDispatch,
) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidResponse {
        operation,
        message: message.into(),
        request_id: response.request_id.clone(),
        native: Box::new(response.native.clone()),
        dispatch,
    }
}

fn request_id(native: &Value, header: Option<String>) -> Option<String> {
    header
        .or_else(|| {
            native
                .get("request_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            native
                .get("requestId")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn required_u32(
    object: &Value,
    field: &str,
    operation: &'static str,
    response: &DecodedEnvelope,
) -> Result<u32, QwenKnowledgeError> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            response_invalid(
                operation,
                &format!("data omitted a valid {field}"),
                response,
                QwenKnowledgeDispatch::NotSent,
            )
        })
}

fn validate_page(
    page_number: u32,
    page_size: u32,
    maximum_page_size: u32,
) -> Result<(), QwenKnowledgeError> {
    if page_number == 0 || page_size == 0 || page_size > maximum_page_size {
        return Err(invalid(format!(
            "page_number must be positive and page_size must be between 1 and {maximum_page_size}"
        )));
    }
    Ok(())
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}
