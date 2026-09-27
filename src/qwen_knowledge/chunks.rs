//! Manual chunk creation, listing, update, and deletion for Qwen Model Studio.
//!
//! Chunk IDs are bound to the Qwen knowledge reference and, when the provider
//! returns a document ID, to that document as well. The update endpoint
//! requires both IDs, so an unassociated chunk reference cannot be updated.

use super::*;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

const ADD_CHUNK_PATH: &str = "/api/v1/indices/rag/index/chunk/create";
const LIST_CHUNKS_PATH: &str = "/api/v1/indices/rag/index/chunklist";
const UPDATE_CHUNK_PATH: &str = "/api/v1/indices/rag/index/chunk/update";
const DELETE_CHUNKS_PATH: &str = "/api/v1/indices/rag/index/chunk/delete";
const MAX_CHUNK_PAGE_SIZE: u32 = 100;
const MAX_DELETE_CHUNKS: usize = 10;
const MAX_CHUNK_TEXT_CHARS: usize = 6_000;
const MAX_CHUNK_TITLE_CHARS: usize = 50;

/// One chunk under a knowledge base and, when known, its source document.
///
/// The chunk ID is opaque; Model Studio does not document a length limit for
/// it. A document association is required for updates because the provider's
/// update request also requires `dataId`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeChunkRef {
    knowledge: QwenKnowledgeRef,
    chunk_id: String,
    document: Option<QwenKnowledgeDocumentRef>,
}

impl QwenKnowledgeChunkRef {
    /// Construct a chunk ref when its source document is known.
    pub fn new(
        knowledge: &QwenKnowledgeRef,
        document: &QwenKnowledgeDocumentRef,
        chunk_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        let chunk_id = chunk_id.into();
        validate_chunk_id(&chunk_id)?;
        if document.knowledge() != knowledge {
            return Err(LlmError::PermissionDenied {
                message: "Qwen chunk document reference belongs to another knowledge base".into(),
            }
            .into());
        }
        Ok(Self {
            knowledge: knowledge.clone(),
            chunk_id,
            document: Some(document.clone()),
        })
    }

    /// Construct a chunk ref without a document association.
    ///
    /// Such a ref can be deleted, but cannot be updated because the update API
    /// requires the originating `dataId`.
    pub fn without_document(
        knowledge: &QwenKnowledgeRef,
        chunk_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        let chunk_id = chunk_id.into();
        validate_chunk_id(&chunk_id)?;
        Ok(Self {
            knowledge: knowledge.clone(),
            chunk_id,
            document: None,
        })
    }

    pub fn knowledge(&self) -> &QwenKnowledgeRef {
        &self.knowledge
    }

    pub fn chunk_id(&self) -> &str {
        &self.chunk_id
    }

    pub fn document(&self) -> Option<&QwenKnowledgeDocumentRef> {
        self.document.as_ref()
    }
}

/// Provider-native fields for one manually added chunk.
///
/// Document knowledge bases use the fixed `content`, `title`, and
/// `image_urls` fields. Table and image knowledge bases use the custom field
/// map from their spreadsheet schema.
#[derive(Debug, Clone, PartialEq)]
pub enum QwenKnowledgeChunkFields {
    Document {
        content: String,
        title: Option<String>,
        image_urls: Vec<String>,
    },
    Custom(BTreeMap<String, Value>),
}

impl QwenKnowledgeChunkFields {
    pub fn document(content: impl Into<String>) -> Self {
        Self::Document {
            content: content.into(),
            title: None,
            image_urls: Vec::new(),
        }
    }

    pub fn custom(fields: BTreeMap<String, Value>) -> Self {
        Self::Custom(fields)
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Result<Self, QwenKnowledgeError> {
        if let Self::Document { title: current, .. } = &mut self {
            *current = Some(title.into());
            Ok(self)
        } else {
            Err(invalid("title is only supported for document chunk fields"))
        }
    }

    pub fn with_image_urls(
        mut self,
        urls: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, QwenKnowledgeError> {
        if let Self::Document { image_urls, .. } = &mut self {
            *image_urls = urls.into_iter().map(Into::into).collect();
            Ok(self)
        } else {
            Err(invalid(
                "image_urls is only supported for document chunk fields",
            ))
        }
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        match self {
            Self::Document {
                content,
                title,
                image_urls,
            } => {
                if content.chars().count() > MAX_CHUNK_TEXT_CHARS {
                    return Err(invalid(
                        "Qwen document chunk content must not exceed 6000 characters",
                    ));
                }
                if title
                    .as_ref()
                    .is_some_and(|value| value.chars().count() > MAX_CHUNK_TITLE_CHARS)
                {
                    return Err(invalid(
                        "Qwen document chunk title must not exceed 50 characters",
                    ));
                }
                if image_urls.len() > 10 {
                    return Err(invalid(
                        "Qwen document chunk supports at most 10 image_urls",
                    ));
                }
                Ok(())
            }
            Self::Custom(fields) => {
                for (key, value) in fields {
                    if value
                        .as_str()
                        .is_some_and(|text| text.chars().count() > MAX_CHUNK_TEXT_CHARS)
                    {
                        return Err(invalid(
                            "Qwen custom chunk string values must not exceed 6000 characters",
                        ));
                    }
                    if key == "image_url"
                        && value
                            .as_str()
                            .is_some_and(|images| images.split(',').count() > 5)
                    {
                        return Err(invalid(
                            "Qwen image_url custom field supports at most 5 images",
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Self::Document {
                content,
                title,
                image_urls,
            } => {
                let mut field = Map::new();
                field.insert("content".into(), json!(content));
                if let Some(title) = title {
                    field.insert("title".into(), json!(title));
                }
                if !image_urls.is_empty() {
                    field.insert("image_urls".into(), json!(image_urls));
                }
                Value::Object(field)
            }
            Self::Custom(fields) => Value::Object(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
        }
    }
}

/// Request to add one chunk to a Qwen knowledge base.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenKnowledgeAddChunkRequest {
    document: Option<QwenKnowledgeDocumentRef>,
    fields: QwenKnowledgeChunkFields,
}

impl QwenKnowledgeAddChunkRequest {
    pub fn new(fields: QwenKnowledgeChunkFields) -> Self {
        Self {
            document: None,
            fields,
        }
    }

    pub fn for_document(
        document: &QwenKnowledgeDocumentRef,
        fields: QwenKnowledgeChunkFields,
    ) -> Self {
        Self {
            document: Some(document.clone()),
            fields,
        }
    }

    pub fn document(&self) -> Option<&QwenKnowledgeDocumentRef> {
        self.document.as_ref()
    }

    pub fn fields(&self) -> &QwenKnowledgeChunkFields {
        &self.fields
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        self.fields.validate()
    }

    fn to_body(&self, knowledge: &QwenKnowledgeRef) -> Value {
        let mut body = json!({
            "pipelineId": knowledge.index_id(),
            "field": self.fields.to_value(),
        });
        if let Some(document) = &self.document {
            body["dataId"] = json!(document.document_id());
        }
        body
    }
}

/// Pagination and optional document filter for the provider's chunk list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeListChunksRequest {
    pub page_num: u32,
    pub page_size: u32,
    pub document: Option<QwenKnowledgeDocumentRef>,
}

impl Default for QwenKnowledgeListChunksRequest {
    fn default() -> Self {
        Self {
            page_num: 1,
            page_size: 20,
            document: None,
        }
    }
}

impl QwenKnowledgeListChunksRequest {
    pub fn new(page_num: u32, page_size: u32) -> Self {
        Self {
            page_num,
            page_size,
            document: None,
        }
    }

    pub fn for_document(mut self, document: &QwenKnowledgeDocumentRef) -> Self {
        self.document = Some(document.clone());
        self
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.page_num == 0 || !(1..=MAX_CHUNK_PAGE_SIZE).contains(&self.page_size) {
            return Err(invalid(
                "Qwen chunk pagination requires page_num >= 1 and page_size from 1 to 100",
            ));
        }
        Ok(())
    }

    fn to_body(&self, knowledge: &QwenKnowledgeRef) -> Value {
        let mut body = json!({
            "indexId": knowledge.index_id(),
            "pageNum": self.page_num,
            "pageSize": self.page_size,
        });
        if let Some(document) = &self.document {
            body["docId"] = json!(document.document_id());
        }
        body
    }
}

/// Typed chunk content returned by the Qwen list-chunks endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeChunk {
    pub reference: QwenKnowledgeChunkRef,
    pub text: Option<String>,
    pub score: Option<f64>,
    pub metadata: Value,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeChunkPage {
    pub chunks: Vec<QwenKnowledgeChunk>,
    pub total_count: u64,
    pub page_num: u32,
    pub page_size: u32,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Fields accepted by the provider's single-chunk update endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeUpdateChunkRequest {
    content: String,
    title: Option<String>,
    is_displayed_chunk_content: bool,
}

impl QwenKnowledgeUpdateChunkRequest {
    pub fn new(content: impl Into<String>, is_displayed_chunk_content: bool) -> Self {
        Self {
            content: content.into(),
            title: None,
            is_displayed_chunk_content,
        }
    }

    /// Set a replacement title. `Some("")` clears the existing title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub fn is_displayed_chunk_content(&self) -> bool {
        self.is_displayed_chunk_content
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let content_chars = self.content.chars().count();
        if !(10..=MAX_CHUNK_TEXT_CHARS).contains(&content_chars) {
            return Err(invalid(
                "Qwen updated chunk content must contain 10 to 6000 characters",
            ));
        }
        if self
            .title
            .as_ref()
            .is_some_and(|title| title.chars().count() > MAX_CHUNK_TITLE_CHARS)
        {
            return Err(invalid(
                "Qwen updated chunk title must contain at most 50 characters",
            ));
        }
        Ok(())
    }

    fn to_body(&self, chunk: &QwenKnowledgeChunkRef) -> Result<Value, QwenKnowledgeError> {
        let document = chunk.document.as_ref().ok_or_else(|| {
            invalid("Qwen chunk update requires a chunk reference bound to a document")
        })?;
        let mut body = json!({
            "pipelineId": chunk.knowledge.index_id(),
            "chunkId": chunk.chunk_id,
            "dataId": document.document_id(),
            "content": self.content,
            "isDisplayedChunkContent": self.is_displayed_chunk_content,
        });
        if let Some(title) = &self.title {
            body["title"] = json!(title);
        }
        Ok(body)
    }
}

/// Acknowledgment returned by a chunk mutation. It retains the raw response
/// because the documented add/update/delete responses do not return chunk IDs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeChunkMutationResult {
    pub request_id: Option<String>,
    pub native: Value,
}

impl QwenKnowledgeService<'_> {
    /// Add one chunk. The create API is described as idempotent, but this
    /// client still makes only one transport attempt and never retries.
    pub async fn add_chunk(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeAddChunkRequest,
    ) -> Result<QwenKnowledgeChunkMutationResult, QwenKnowledgeError> {
        self.validate_knowledge_ref(knowledge)?;
        request.validate()?;
        if let Some(document) = &request.document {
            self.validate_document_ref(knowledge, document)?;
        }
        let response = self
            .request_json(
                "POST",
                ADD_CHUNK_PATH,
                &[],
                Some(request.to_body(knowledge)),
                "add_chunk",
                true,
            )
            .await?;
        Ok(mutation_result(response))
    }

    /// List one page of chunks. The provider requires `docId` for document and
    /// multimedia knowledge bases; pass a scoped document reference when
    /// querying those knowledge-base types.
    pub async fn list_chunks(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeListChunksRequest,
    ) -> Result<QwenKnowledgeChunkPage, QwenKnowledgeError> {
        self.validate_knowledge_ref(knowledge)?;
        request.validate()?;
        if let Some(document) = &request.document {
            self.validate_document_ref(knowledge, document)?;
        }
        let response = self
            .request_json(
                "POST",
                LIST_CHUNKS_PATH,
                &[],
                Some(request.to_body(knowledge)),
                "list_chunks",
                false,
            )
            .await?;
        decode_chunks_page(knowledge, request, response)
    }

    /// Update one chunk's content, title, or retrieval visibility. The
    /// provider requires its source document ID, which is carried by the chunk
    /// reference returned from `list_chunks` when metadata contains `doc_id`.
    pub async fn update_chunk(
        &self,
        chunk: &QwenKnowledgeChunkRef,
        request: &QwenKnowledgeUpdateChunkRequest,
    ) -> Result<QwenKnowledgeChunkMutationResult, QwenKnowledgeError> {
        self.validate_chunk_ref(&chunk.knowledge, chunk)?;
        request.validate()?;
        let body = request.to_body(chunk)?;
        let response = self
            .request_json(
                "POST",
                UPDATE_CHUNK_PATH,
                &[],
                Some(body),
                "update_chunk",
                true,
            )
            .await?;
        Ok(mutation_result(response))
    }

    /// Delete 1–10 chunk references from the same knowledge base.
    /// Deletion is irreversible.
    pub async fn delete_chunks(
        &self,
        knowledge: &QwenKnowledgeRef,
        chunks: &[QwenKnowledgeChunkRef],
    ) -> Result<QwenKnowledgeChunkMutationResult, QwenKnowledgeError> {
        self.validate_knowledge_ref(knowledge)?;
        if chunks.is_empty() || chunks.len() > MAX_DELETE_CHUNKS {
            return Err(invalid("Qwen chunk deletion requires 1 to 10 chunks"));
        }
        for chunk in chunks {
            self.validate_chunk_ref(knowledge, chunk)?;
        }
        let ids = chunks
            .iter()
            .map(|chunk| chunk.chunk_id.as_str())
            .collect::<Vec<_>>();
        let response = self
            .request_json(
                "POST",
                DELETE_CHUNKS_PATH,
                &[],
                Some(json!({"pipelineId": knowledge.index_id(), "chunkIds": ids})),
                "delete_chunks",
                true,
            )
            .await?;
        Ok(mutation_result(response))
    }

    fn validate_chunk_ref(
        &self,
        knowledge: &QwenKnowledgeRef,
        chunk: &QwenKnowledgeChunkRef,
    ) -> Result<(), QwenKnowledgeError> {
        self.validate_knowledge_ref(knowledge)?;
        if &chunk.knowledge != knowledge {
            return Err(LlmError::PermissionDenied {
                message: "Qwen chunk reference belongs to another knowledge base or workspace"
                    .into(),
            }
            .into());
        }
        validate_chunk_id(&chunk.chunk_id)?;
        if let Some(document) = &chunk.document {
            self.validate_document_ref(knowledge, document)?;
        }
        Ok(())
    }
}

fn mutation_result(response: DecodedEnvelope) -> QwenKnowledgeChunkMutationResult {
    QwenKnowledgeChunkMutationResult {
        request_id: response.request_id,
        native: response.native,
    }
}

fn decode_chunks_page(
    knowledge: &QwenKnowledgeRef,
    request: &QwenKnowledgeListChunksRequest,
    response: DecodedEnvelope,
) -> Result<QwenKnowledgeChunkPage, QwenKnowledgeError> {
    let data = response.native.get("data").ok_or_else(|| {
        response_invalid(
            "list_chunks",
            "response omitted data",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let total_count = data.get("total").and_then(Value::as_u64).ok_or_else(|| {
        response_invalid(
            "list_chunks",
            "data omitted total",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let rows = data.get("nodes").and_then(Value::as_array).ok_or_else(|| {
        response_invalid(
            "list_chunks",
            "data omitted nodes",
            &response,
            QwenKnowledgeDispatch::NotSent,
        )
    })?;
    let expected_document_id = request
        .document
        .as_ref()
        .map(QwenKnowledgeDocumentRef::document_id);
    let chunks = rows
        .iter()
        .cloned()
        .map(|row| {
            decode_chunk(
                knowledge,
                expected_document_id,
                response.request_id.as_ref(),
                row,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(QwenKnowledgeChunkPage {
        chunks,
        total_count,
        page_num: request.page_num,
        page_size: request.page_size,
        request_id: response.request_id,
        native: response.native,
    })
}

fn decode_chunk(
    knowledge: &QwenKnowledgeRef,
    expected_document_id: Option<&str>,
    request_id: Option<&String>,
    native: Value,
) -> Result<QwenKnowledgeChunk, QwenKnowledgeError> {
    let metadata = native
        .get("metadata")
        .cloned()
        .ok_or_else(|| chunk_bad("chunk omitted metadata", request_id, native.clone()))?;
    let chunk_id = metadata
        .get("_id")
        .and_then(Value::as_str)
        .filter(|id| validate_chunk_id(id).is_ok())
        .ok_or_else(|| chunk_bad("metadata omitted a valid _id", request_id, native.clone()))?;
    if metadata
        .get("pipeline_id")
        .and_then(Value::as_str)
        .is_some_and(|pipeline_id| pipeline_id != knowledge.index_id())
    {
        return Err(chunk_bad(
            "chunk metadata belongs to another knowledge base",
            request_id,
            native,
        ));
    }
    if metadata
        .get("workspace_id")
        .and_then(Value::as_str)
        .is_some_and(|workspace_id| workspace_id != knowledge.scope.workspace_id())
    {
        return Err(chunk_bad(
            "chunk metadata belongs to another workspace",
            request_id,
            native,
        ));
    }
    let document = match metadata.get("doc_id") {
        Some(Value::String(document_id)) => {
            Some(knowledge.document_ref(document_id.clone()).map_err(|_| {
                chunk_bad(
                    "metadata contains an invalid doc_id",
                    request_id,
                    native.clone(),
                )
            })?)
        }
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err(chunk_bad(
                "metadata doc_id is not a string",
                request_id,
                native,
            ));
        }
    };
    if let (Some(expected), Some(actual)) = (expected_document_id, document.as_ref()) {
        if actual.document_id() != expected {
            return Err(chunk_bad(
                "chunk metadata doc_id did not match the requested document filter",
                request_id,
                native,
            ));
        }
    }
    let reference = match document.as_ref() {
        Some(document) => QwenKnowledgeChunkRef::new(knowledge, document, chunk_id.to_owned())?,
        None => QwenKnowledgeChunkRef::without_document(knowledge, chunk_id.to_owned())?,
    };
    Ok(QwenKnowledgeChunk {
        reference,
        text: native
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned),
        score: native.get("score").and_then(Value::as_f64),
        metadata,
        native,
    })
}

fn chunk_bad(message: &str, request_id: Option<&String>, native: Value) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidResponse {
        operation: "list_chunks",
        message: message.into(),
        request_id: request_id.cloned(),
        native: Box::new(native),
        dispatch: QwenKnowledgeDispatch::NotSent,
    }
}

fn validate_chunk_id(chunk_id: &str) -> Result<(), QwenKnowledgeError> {
    if chunk_id.is_empty() {
        return Err(invalid("Qwen chunk ID must be nonempty"));
    }
    Ok(())
}
