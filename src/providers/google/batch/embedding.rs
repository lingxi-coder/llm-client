//! Typed lifecycle for Gemini's asynchronous `EmbedContent` Batch API.

use super::{
    decode_json, embedding_mutation_transport_error, encode_path_segment, ensure_success,
    invalid_input, invalid_response, is_embedding_batch_type, is_generation_batch_type,
    normalize_model_name, scope_mismatch, valid_gemini_file_id, valid_identity, valid_resource_id,
    validate_credential, GeminiBatchError, GeminiBatchFileRef, GeminiBatchListOptions,
    GeminiBatchScope, GeminiBatchService, GeminiBatchState, MAX_CONTROL_RESPONSE_BYTES,
    MAX_FILE_INPUT_BYTES, MAX_INLINE_REQUEST_BYTES,
};
use crate::{
    protocol::{LlmError, Secret},
    providers::google::embeddings::{GeminiEmbeddingMedia, GeminiEmbeddingSource},
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{stream, stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    pin::Pin,
    task::{Context, Poll},
};

/// An input part for one Gemini embedding request.
#[derive(Debug, Clone, PartialEq)]
pub enum GeminiBatchEmbeddingPart {
    Text(String),
    Media(GeminiEmbeddingMedia),
}

/// Typed `Content` value for an embedding request.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchEmbeddingContent {
    parts: Vec<GeminiBatchEmbeddingPart>,
}

impl GeminiBatchEmbeddingContent {
    pub fn new(parts: Vec<GeminiBatchEmbeddingPart>) -> Result<Self, GeminiBatchError> {
        if parts.is_empty() {
            return Err(invalid_input(
                "embedding content must contain at least one part",
            ));
        }
        if parts.iter().any(
            |part| matches!(part, GeminiBatchEmbeddingPart::Text(text) if text.trim().is_empty()),
        ) {
            return Err(invalid_input("embedding text parts cannot be empty"));
        }
        Ok(Self { parts })
    }

    pub fn text(text: impl Into<String>) -> Result<Self, GeminiBatchError> {
        Self::new(vec![GeminiBatchEmbeddingPart::Text(text.into())])
    }

    pub fn parts(&self) -> &[GeminiBatchEmbeddingPart] {
        &self.parts
    }
}

/// Current, non-deprecated `EmbedContentConfig.taskType` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeminiBatchEmbeddingTaskType {
    RetrievalQuery,
    RetrievalDocument,
    SemanticSimilarity,
    Classification,
    Clustering,
    QuestionAnswering,
    FactVerification,
    CodeRetrievalQuery,
}

impl GeminiBatchEmbeddingTaskType {
    fn wire_name(self) -> &'static str {
        match self {
            Self::RetrievalQuery => "RETRIEVAL_QUERY",
            Self::RetrievalDocument => "RETRIEVAL_DOCUMENT",
            Self::SemanticSimilarity => "SEMANTIC_SIMILARITY",
            Self::Classification => "CLASSIFICATION",
            Self::Clustering => "CLUSTERING",
            Self::QuestionAnswering => "QUESTION_ANSWERING",
            Self::FactVerification => "FACT_VERIFICATION",
            Self::CodeRetrievalQuery => "CODE_RETRIEVAL_QUERY",
        }
    }
}

/// Current `embedContentConfig` fields used for one embedding request.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiBatchEmbeddingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<GeminiBatchEmbeddingTaskType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,
    #[serde(default)]
    pub auto_truncate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_ocr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_track_extraction: Option<bool>,
}

impl GeminiBatchEmbeddingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_task_type(mut self, task_type: GeminiBatchEmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Result<Self, GeminiBatchError> {
        let title = title.into();
        if !valid_identity(&title) {
            return Err(invalid_input("embedding title must be non-empty"));
        }
        self.title = Some(title);
        Ok(self)
    }

    pub fn with_output_dimensionality(mut self, dimensions: u32) -> Result<Self, GeminiBatchError> {
        if dimensions == 0 {
            return Err(invalid_input(
                "embedding output dimensionality must be greater than zero",
            ));
        }
        self.output_dimensionality = Some(dimensions);
        Ok(self)
    }

    pub fn with_auto_truncate(mut self, auto_truncate: bool) -> Self {
        self.auto_truncate = auto_truncate;
        self
    }

    pub fn with_document_ocr(mut self, enabled: bool) -> Self {
        self.document_ocr = Some(enabled);
        self
    }

    pub fn with_audio_track_extraction(mut self, enabled: bool) -> Self {
        self.audio_track_extraction = Some(enabled);
        self
    }

    fn encode(&self) -> Value {
        let mut config = Map::new();
        config.insert("autoTruncate".into(), json!(self.auto_truncate));
        if let Some(task_type) = self.task_type {
            config.insert("taskType".into(), json!(task_type.wire_name()));
        }
        if let Some(title) = &self.title {
            config.insert("title".into(), json!(title));
        }
        if let Some(dimensions) = self.output_dimensionality {
            config.insert("outputDimensionality".into(), json!(dimensions));
        }
        if let Some(enabled) = self.document_ocr {
            config.insert("documentOcr".into(), json!(enabled));
        }
        if let Some(enabled) = self.audio_track_extraction {
            config.insert("audioTrackExtraction".into(), json!(enabled));
        }
        Value::Object(config)
    }
}

/// One `EmbedContentRequest` body. The batch model is applied to the native
/// request when the create body is encoded, preventing cross-model rows.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchEmbedContentRequest {
    content: GeminiBatchEmbeddingContent,
    config: GeminiBatchEmbeddingConfig,
}

impl GeminiBatchEmbedContentRequest {
    pub fn new(content: GeminiBatchEmbeddingContent) -> Self {
        Self {
            content,
            config: GeminiBatchEmbeddingConfig::default(),
        }
    }

    pub fn text(text: impl Into<String>) -> Result<Self, GeminiBatchError> {
        Ok(Self::new(GeminiBatchEmbeddingContent::text(text)?))
    }

    pub fn with_config(mut self, config: GeminiBatchEmbeddingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn content(&self) -> &GeminiBatchEmbeddingContent {
        &self.content
    }

    pub fn config(&self) -> &GeminiBatchEmbeddingConfig {
        &self.config
    }

    fn encode(&self, model: &str) -> Result<Value, GeminiBatchError> {
        if self.config.output_dimensionality == Some(0)
            || self
                .config
                .title
                .as_deref()
                .is_some_and(|title| !valid_identity(title))
        {
            return Err(invalid_input("embedding config contains an invalid value"));
        }
        if matches!(model, "gemini-embedding-001" | "gemini-embedding-2")
            && self
                .config
                .output_dimensionality
                .is_some_and(|dimensions| !(128..=3072).contains(&dimensions))
        {
            return Err(invalid_input(
                "Gemini embedding output dimensionality must be between 128 and 3072",
            ));
        }
        if model == "gemini-embedding-2" && self.config.task_type.is_some() {
            return Err(invalid_input(
                "Gemini Embedding 2 does not support taskType; include task instructions in text parts instead",
            ));
        }
        if self.config.title.is_some()
            && self.config.task_type != Some(GeminiBatchEmbeddingTaskType::RetrievalDocument)
        {
            return Err(invalid_input(
                "embedding title is only valid for RETRIEVAL_DOCUMENT",
            ));
        }
        if model != "gemini-embedding-2"
            && (self.config.document_ocr.is_some() || self.config.audio_track_extraction.is_some())
        {
            return Err(invalid_input(
                "document OCR and audio-track extraction require gemini-embedding-2",
            ));
        }
        let parts = encode_embedding_parts(model, &self.content.parts)?;
        Ok(json!({
            "model": format!("models/{model}"),
            "content": {"parts": parts},
            "embedContentConfig": self.config.encode(),
        }))
    }
}

/// One inline embedding request with optional correlation metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiBatchEmbedContentItem {
    request: GeminiBatchEmbedContentRequest,
    metadata: Option<Value>,
}

impl GeminiBatchEmbedContentItem {
    pub fn new(request: GeminiBatchEmbedContentRequest) -> Self {
        Self {
            request,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Result<Self, GeminiBatchError> {
        if !metadata.is_object() {
            return Err(invalid_input(
                "embedding request metadata must be a JSON object",
            ));
        }
        self.metadata = Some(metadata);
        Ok(self)
    }

    pub fn with_key(self, key: impl Into<String>) -> Result<Self, GeminiBatchError> {
        let key = key.into();
        if !valid_identity(&key) {
            return Err(invalid_input("embedding request key must be non-empty"));
        }
        self.with_metadata(json!({"key": key}))
    }

    pub fn request(&self) -> &GeminiBatchEmbedContentRequest {
        &self.request
    }

    pub fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
}

/// One keyed request row in a Gemini embedding Batch JSONL input file.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiEmbeddingBatchJsonlRequest {
    key: String,
    request: GeminiBatchEmbedContentRequest,
}

impl GeminiEmbeddingBatchJsonlRequest {
    pub fn new(
        key: impl Into<String>,
        request: GeminiBatchEmbedContentRequest,
    ) -> Result<Self, GeminiBatchError> {
        let key = key.into();
        if !valid_identity(&key) {
            return Err(invalid_input(
                "embedding JSONL request key must be non-empty",
            ));
        }
        Ok(Self { key, request })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn request(&self) -> &GeminiBatchEmbedContentRequest {
        &self.request
    }
}

/// Typed keyed rows ready for the Gemini Files API embedding workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiEmbeddingBatchJsonlInput {
    model: String,
    requests: Vec<GeminiEmbeddingBatchJsonlRequest>,
}

impl GeminiEmbeddingBatchJsonlInput {
    pub fn new(
        model: impl Into<String>,
        requests: Vec<GeminiEmbeddingBatchJsonlRequest>,
    ) -> Result<Self, GeminiBatchError> {
        let model = normalize_model_name(&model.into())?;
        if requests.is_empty() {
            return Err(invalid_input(
                "embedding JSONL input must contain at least one request",
            ));
        }
        let mut keys = BTreeSet::new();
        if requests
            .iter()
            .any(|request| !keys.insert(request.key.as_str()))
        {
            return Err(invalid_input("embedding JSONL request keys must be unique"));
        }
        Ok(Self { model, requests })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn requests(&self) -> &[GeminiEmbeddingBatchJsonlRequest] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Encode keyed REST request rows with a trailing newline after each row.
    pub fn to_bytes(&self) -> Result<Bytes, GeminiBatchError> {
        let size_bytes = self.encoded_size_bytes()?;
        let capacity = usize::try_from(size_bytes)
            .map_err(|_| invalid_input("embedding JSONL input cannot fit in memory"))?;
        let mut bytes = Vec::with_capacity(capacity);
        for request in &self.requests {
            let encoded = encode_embedding_jsonl_request(&self.model, request)?;
            bytes.extend_from_slice(&encoded);
            bytes.push(b'\n');
        }
        Ok(Bytes::from(bytes))
    }

    /// Calculate exact UTF-8 size and validate every row before uploading.
    pub fn encoded_size_bytes(&self) -> Result<u64, GeminiBatchError> {
        let mut size_bytes = 0_u64;
        for request in &self.requests {
            let row_len =
                u64::try_from(encode_embedding_jsonl_request(&self.model, request)?.len())
                    .map_err(|_| invalid_input("embedding JSONL line size overflowed"))?;
            size_bytes = size_bytes
                .checked_add(row_len)
                .and_then(|size| size.checked_add(1))
                .ok_or_else(|| invalid_input("embedding JSONL file size overflowed"))?;
            if size_bytes > MAX_FILE_INPUT_BYTES {
                return Err(invalid_input(
                    "embedding JSONL input file must not exceed 2 GB",
                ));
            }
        }
        Ok(size_bytes)
    }

    /// Produce one-shot JSONL rows after prevalidating exact length and schema.
    pub fn into_stream(
        self,
    ) -> Result<(u64, BoxStream<'static, Result<Bytes, LlmError>>), GeminiBatchError> {
        let size_bytes = self.encoded_size_bytes()?;
        let model = self.model;
        let lines = self.requests.into_iter().map(move |request| {
            encode_embedding_jsonl_request(&model, &request)
                .map(|mut line| {
                    line.push(b'\n');
                    Bytes::from(line)
                })
                .map_err(|error| LlmError::InvalidRequest {
                    message: error.to_string(),
                })
        });
        Ok((size_bytes, stream::iter(lines).boxed()))
    }
}

fn encode_embedding_jsonl_request(
    model: &str,
    row: &GeminiEmbeddingBatchJsonlRequest,
) -> Result<Vec<u8>, GeminiBatchError> {
    let value = json!({
        "key": row.key,
        "request": row.request.encode(model)?,
    });
    serde_json::to_vec(&value)
        .map_err(|_| invalid_input("embedding JSONL request could not be encoded"))
}

/// A Gemini Files API resource used as an embedding batch input or output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiEmbeddingBatchFileRef {
    scope: GeminiBatchScope,
    file_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_keys_in_order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimensions: Option<Vec<Option<u32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimension: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimensions_by_key: Option<BTreeMap<String, Option<u32>>>,
}

impl GeminiEmbeddingBatchFileRef {
    pub fn from_resource_name(
        scope: &GeminiBatchScope,
        resource_name: impl AsRef<str>,
    ) -> Result<Self, GeminiBatchError> {
        scope.validate()?;
        let file_id = resource_name
            .as_ref()
            .strip_prefix("files/")
            .filter(|id| valid_gemini_file_id(id))
            .ok_or_else(|| invalid_input("Gemini file name must be a valid files/{id} resource"))?;
        Ok(Self {
            scope: scope.clone(),
            file_id: file_id.to_owned(),
            model: None,
            expected_output_keys_in_order: None,
            expected_output_dimensions: None,
            expected_output_dimension: None,
            expected_output_dimensions_by_key: None,
        })
    }

    /// Record one uniform expected vector width for this file's requests.
    pub fn with_expected_output_dimension(
        mut self,
        dimension: u32,
    ) -> Result<Self, GeminiBatchError> {
        if dimension == 0 {
            return Err(invalid_input("expected output dimension must be positive"));
        }
        if self
            .expected_output_dimensions
            .as_ref()
            .is_some_and(|dimensions| {
                dimensions
                    .iter()
                    .flatten()
                    .any(|expected| *expected != dimension)
            })
        {
            return Err(invalid_input(
                "uniform output dimension conflicts with typed request dimensions",
            ));
        }
        self.expected_output_dimension = Some(dimension);
        Ok(self)
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn resource_name(&self) -> String {
        format!("files/{}", self.file_id)
    }

    pub fn scope(&self) -> &GeminiBatchScope {
        &self.scope
    }

    pub fn expected_output_dimension(&self) -> Option<u32> {
        self.expected_output_dimension
    }

    pub fn expected_output_keys_in_order(&self) -> Option<&[String]> {
        self.expected_output_keys_in_order.as_deref()
    }

    pub fn expected_output_dimensions(&self) -> Option<&[Option<u32>]> {
        self.expected_output_dimensions.as_deref()
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum EmbeddingInputSource {
    Inline(Vec<GeminiBatchEmbedContentItem>),
    File(Box<GeminiEmbeddingBatchFileRef>),
}

/// Inline embedding requests or a previously uploaded embedding JSONL file.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiEmbeddingBatchInput {
    source: EmbeddingInputSource,
    expected_output_dimensions: Option<Vec<Option<u32>>>,
    expected_output_keys_in_order: Option<Vec<String>>,
    expected_output_dimension: Option<u32>,
    expected_output_dimensions_by_key: Option<BTreeMap<String, Option<u32>>>,
}

impl GeminiEmbeddingBatchInput {
    pub fn new(requests: Vec<GeminiBatchEmbedContentItem>) -> Result<Self, GeminiBatchError> {
        if requests.is_empty() {
            return Err(invalid_input(
                "inline embedding batch must contain at least one request",
            ));
        }
        let expected_output_dimensions = Some(
            requests
                .iter()
                .map(|item| item.request.config.output_dimensionality)
                .collect(),
        );
        Ok(Self {
            source: EmbeddingInputSource::Inline(requests),
            expected_output_dimensions,
            expected_output_keys_in_order: None,
            expected_output_dimension: None,
            expected_output_dimensions_by_key: None,
        })
    }

    pub fn from_file(file: GeminiEmbeddingBatchFileRef) -> Self {
        let expected_output_dimensions = file.expected_output_dimensions.clone();
        let expected_output_keys_in_order = file.expected_output_keys_in_order.clone();
        let expected_output_dimension = file.expected_output_dimension;
        let expected_output_dimensions_by_key = file.expected_output_dimensions_by_key.clone();
        Self {
            source: EmbeddingInputSource::File(Box::new(file)),
            expected_output_dimensions,
            expected_output_keys_in_order,
            expected_output_dimension,
            expected_output_dimensions_by_key,
        }
    }

    /// Record one uniform output dimension for a caller-prepared JSONL file.
    pub fn with_expected_output_dimension(
        mut self,
        dimension: u32,
    ) -> Result<Self, GeminiBatchError> {
        if dimension == 0 {
            return Err(invalid_input("expected output dimension must be positive"));
        }
        if matches!(&self.source, EmbeddingInputSource::Inline(_)) {
            return Err(invalid_input(
                "inline output dimensions come from each typed request config",
            ));
        }
        if self
            .expected_output_dimensions
            .as_ref()
            .is_some_and(|dimensions| {
                dimensions
                    .iter()
                    .flatten()
                    .any(|expected| *expected != dimension)
            })
        {
            return Err(invalid_input(
                "uniform output dimension conflicts with typed request dimensions",
            ));
        }
        self.expected_output_dimension = Some(dimension);
        if let EmbeddingInputSource::File(file) = &mut self.source {
            file.expected_output_dimension = Some(dimension);
        }
        Ok(self)
    }

    pub fn requests(&self) -> &[GeminiBatchEmbedContentItem] {
        match &self.source {
            EmbeddingInputSource::Inline(requests) => requests,
            EmbeddingInputSource::File(_) => &[],
        }
    }

    pub fn file(&self) -> Option<&GeminiEmbeddingBatchFileRef> {
        match &self.source {
            EmbeddingInputSource::File(file) => Some(file.as_ref()),
            EmbeddingInputSource::Inline(_) => None,
        }
    }

    pub fn expected_output_dimensions(&self) -> Option<&[Option<u32>]> {
        self.expected_output_dimensions.as_deref()
    }

    pub fn expected_output_keys_in_order(&self) -> Option<&[String]> {
        self.expected_output_keys_in_order.as_deref()
    }

    pub fn expected_output_dimension(&self) -> Option<u32> {
        self.expected_output_dimension
    }

    pub fn expected_output_dimensions_by_key(&self) -> Option<&BTreeMap<String, Option<u32>>> {
        self.expected_output_dimensions_by_key.as_ref()
    }
}

/// Typed create body for `models.asyncBatchEmbedContent`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiEmbeddingBatchCreateRequest {
    model: String,
    display_name: String,
    input: GeminiEmbeddingBatchInput,
}

impl GeminiEmbeddingBatchCreateRequest {
    pub fn new(
        model: impl Into<String>,
        display_name: impl Into<String>,
        input: GeminiEmbeddingBatchInput,
    ) -> Result<Self, GeminiBatchError> {
        let model = normalize_model_name(&model.into())?;
        let display_name = display_name.into();
        if !valid_identity(&display_name) {
            return Err(invalid_input("batch display name must be non-empty"));
        }
        if input
            .file()
            .and_then(|file| file.model.as_deref())
            .is_some_and(|file_model| file_model != model)
        {
            return Err(invalid_input(
                "embedding input file was encoded for a different model",
            ));
        }
        Ok(Self {
            model,
            display_name,
            input,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn input(&self) -> &GeminiEmbeddingBatchInput {
        &self.input
    }

    fn encode(&self) -> Result<Bytes, GeminiBatchError> {
        let input_config = match &self.input.source {
            EmbeddingInputSource::Inline(requests) => {
                let mut encoded_requests = Vec::with_capacity(requests.len());
                for item in requests {
                    let mut row = Map::new();
                    row.insert("request".into(), item.request.encode(&self.model)?);
                    if let Some(metadata) = &item.metadata {
                        row.insert("metadata".into(), metadata.clone());
                    }
                    encoded_requests.push(Value::Object(row));
                }
                json!({"requests": {"requests": encoded_requests}})
            }
            EmbeddingInputSource::File(file) => {
                json!({"file_name": file.resource_name()})
            }
        };
        let body = json!({
            "batch": {
                "display_name": self.display_name,
                "input_config": input_config,
            }
        });
        let bytes = serde_json::to_vec(&body)
            .map_err(|_| invalid_input("embedding batch request could not be encoded"))?;
        if matches!(&self.input.source, EmbeddingInputSource::Inline(_))
            && bytes.len() >= MAX_INLINE_REQUEST_BYTES
        {
            return Err(invalid_input(
                "inline embedding batch request must be smaller than 20 MB",
            ));
        }
        Ok(Bytes::from(bytes))
    }
}

/// Identity for an embedding Batch operation, distinct from a generateContent
/// `GeminiBatchRef`. Create-time model/dimension expectations are retained;
/// list/import references expose only what Google returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum GeminiEmbeddingBatchOperation {
    EmbedContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiEmbeddingBatchRef {
    operation: GeminiEmbeddingBatchOperation,
    scope: GeminiBatchScope,
    batch_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimensions: Option<Vec<Option<u32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_keys_in_order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimensions_by_key: Option<BTreeMap<String, Option<u32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_output_dimension: Option<u32>,
}

impl GeminiEmbeddingBatchRef {
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn operation(&self) -> &'static str {
        match self.operation {
            GeminiEmbeddingBatchOperation::EmbedContent => "embed_content",
        }
    }

    pub fn resource_name(&self) -> String {
        format!("batches/{}", self.batch_id)
    }

    pub fn scope(&self) -> &GeminiBatchScope {
        &self.scope
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn expected_output_dimensions(&self) -> Option<&[Option<u32>]> {
        self.expected_output_dimensions.as_deref()
    }

    pub fn expected_output_keys_in_order(&self) -> Option<&[String]> {
        self.expected_output_keys_in_order.as_deref()
    }

    pub fn expected_output_dimension(&self) -> Option<u32> {
        self.expected_output_dimension
    }

    pub fn expected_output_dimensions_by_key(&self) -> Option<&BTreeMap<String, Option<u32>>> {
        self.expected_output_dimensions_by_key.as_ref()
    }
}

/// A typed snapshot of an embedding Batch operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingBatchSnapshot {
    pub reference: GeminiEmbeddingBatchRef,
    pub done: Option<bool>,
    pub state: Option<GeminiBatchState>,
    pub metadata: Option<Value>,
    pub error: Option<Value>,
    pub response: Option<Value>,
    pub output_file: Option<GeminiEmbeddingBatchFileRef>,
    pub native: Value,
}

/// One provider-paginated embedding Batch page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingBatchListPage {
    pub batches: Vec<GeminiEmbeddingBatchSnapshot>,
    pub next_page_token: Option<String>,
    pub native: Value,
}

/// A validated embedding returned by `EmbedContentResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiContentEmbedding {
    /// Google omits values for first-party calls; the tensor shape remains available.
    pub values: Option<Vec<f64>>,
    pub shape: Option<Vec<u32>>,
    pub native: Value,
}

/// Typed successful item response from `models.embedContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbedContentResponse {
    pub embedding: GeminiContentEmbedding,
    pub usage_metadata: Option<Value>,
    pub native: Value,
}

/// Typed provider `google.rpc.Status` returned for one failed item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingItemError {
    pub code: Option<i32>,
    pub message: Option<String>,
    pub details: Vec<Value>,
    pub native: Value,
}

/// One inline item result. The provider keeps request-level errors separate
/// from operation-level failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingBatchItemResult {
    pub metadata: Option<Value>,
    pub response: Option<GeminiEmbedContentResponse>,
    pub error: Option<GeminiEmbeddingItemError>,
    pub native: Value,
}

/// Results from one explicit status read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiEmbeddingBatchResults {
    pub snapshot: GeminiEmbeddingBatchSnapshot,
    /// `None` means Google has not returned inline output or the batch used a file.
    pub items: Option<Vec<GeminiEmbeddingBatchItemResult>>,
}

/// Incremental typed reader for a Gemini embedding result JSONL file.
pub struct GeminiEmbeddingBatchResultStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    buffer: Vec<u8>,
    pending_chunk: Option<Bytes>,
    chunk_offset: usize,
    expected_output_dimensions: Option<Vec<Option<u32>>>,
    expected_output_keys_in_order: Option<Vec<String>>,
    expected_output_dimension: Option<u32>,
    expected_output_dimensions_by_key: Option<BTreeMap<String, Option<u32>>>,
    seen_keys: BTreeSet<String>,
    row_count: usize,
    finished: bool,
}

impl GeminiEmbeddingBatchResultStream {
    fn new(
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        expected_output_dimensions: Option<Vec<Option<u32>>>,
        expected_output_keys_in_order: Option<Vec<String>>,
        expected_output_dimension: Option<u32>,
        expected_output_dimensions_by_key: Option<BTreeMap<String, Option<u32>>>,
    ) -> Self {
        Self {
            body,
            buffer: Vec::new(),
            pending_chunk: None,
            chunk_offset: 0,
            expected_output_dimensions,
            expected_output_keys_in_order,
            expected_output_dimension,
            expected_output_dimensions_by_key,
            seen_keys: BTreeSet::new(),
            row_count: 0,
            finished: false,
        }
    }

    fn decode_next_line(
        &mut self,
        line: &[u8],
    ) -> Result<GeminiEmbeddingBatchItemResult, GeminiBatchError> {
        if self
            .expected_output_dimensions
            .as_ref()
            .is_some_and(|dimensions| self.row_count >= dimensions.len())
        {
            return Err(invalid_response(
                "embedding JSONL result count exceeds the known input count",
            ));
        }
        let ordered_key = self
            .expected_output_keys_in_order
            .as_ref()
            .and_then(|keys| keys.get(self.row_count))
            .map(String::as_str);
        let expected_dimension = self
            .expected_output_dimensions
            .as_ref()
            .and_then(|dimensions| dimensions.get(self.row_count))
            .copied()
            .flatten();
        let item = parse_embedding_result_line(
            line,
            expected_dimension,
            self.expected_output_dimension,
            self.expected_output_dimensions_by_key.as_ref(),
            ordered_key,
        )?;
        self.row_count += 1;
        if let (Some(dimensions), Some(key)) = (
            self.expected_output_dimensions_by_key.as_ref(),
            item.metadata
                .as_ref()
                .and_then(|metadata| metadata.get("key"))
                .and_then(Value::as_str),
        ) {
            if !dimensions.contains_key(key) {
                return Err(invalid_response(
                    "embedding JSONL result contains an unknown request key",
                ));
            }
            if !self.seen_keys.insert(key.to_owned()) {
                return Err(invalid_response(
                    "embedding JSONL results contain a duplicate request key",
                ));
            }
        }
        Ok(item)
    }
}

impl Stream for GeminiEmbeddingBatchResultStream {
    type Item = Result<GeminiEmbeddingBatchItemResult, GeminiBatchError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        loop {
            if let Some(newline) = this.buffer.iter().position(|byte| *byte == b'\n') {
                if newline > MAX_CONTROL_RESPONSE_BYTES {
                    this.finished = true;
                    return Poll::Ready(Some(Err(invalid_response(
                        "embedding JSONL result line exceeds the 64 MiB limit",
                    ))));
                }
                let mut line = this.buffer.drain(..=newline).collect::<Vec<_>>();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if line.is_empty() {
                    continue;
                }
                let item = this.decode_next_line(&line);
                if item.is_err() {
                    this.finished = true;
                }
                return Poll::Ready(Some(item));
            }
            if let Some(chunk) = this.pending_chunk.as_ref() {
                let start = this.chunk_offset;
                if let Some(relative_newline) =
                    chunk[start..].iter().position(|byte| *byte == b'\n')
                {
                    let end = start + relative_newline + 1;
                    let line_bytes = relative_newline + this.buffer.len();
                    if line_bytes > MAX_CONTROL_RESPONSE_BYTES {
                        this.finished = true;
                        return Poll::Ready(Some(Err(invalid_response(
                            "embedding JSONL result line exceeds the 64 MiB limit",
                        ))));
                    }
                    this.buffer.extend_from_slice(&chunk[start..end]);
                    this.chunk_offset = end;
                    if end == chunk.len() {
                        this.pending_chunk = None;
                        this.chunk_offset = 0;
                    }
                    continue;
                }
                let remaining = &chunk[start..];
                if this.buffer.len().saturating_add(remaining.len()) > MAX_CONTROL_RESPONSE_BYTES {
                    this.finished = true;
                    return Poll::Ready(Some(Err(invalid_response(
                        "embedding JSONL result line exceeds the 64 MiB limit",
                    ))));
                }
                this.buffer.extend_from_slice(remaining);
                this.pending_chunk = None;
                this.chunk_offset = 0;
                continue;
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Err(error))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(error.into())));
                }
                Poll::Ready(Some(Ok(chunk))) => {
                    this.pending_chunk = Some(chunk);
                }
                Poll::Ready(None) => {
                    this.finished = true;
                    if !this.buffer.is_empty() {
                        if this.buffer.len() > MAX_CONTROL_RESPONSE_BYTES {
                            return Poll::Ready(Some(Err(invalid_response(
                                "embedding JSONL result line exceeds the 64 MiB limit",
                            ))));
                        }
                        let line = std::mem::take(&mut this.buffer);
                        let item = match this.decode_next_line(&line) {
                            Ok(item) => item,
                            Err(error) => {
                                return Poll::Ready(Some(Err(error)));
                            }
                        };
                        if this
                            .expected_output_dimensions
                            .as_ref()
                            .is_some_and(|dimensions| this.row_count != dimensions.len())
                        {
                            this.finished = true;
                            return Poll::Ready(Some(Err(invalid_response(
                                "embedding JSONL result count differs from the known input count",
                            ))));
                        }
                        return Poll::Ready(Some(Ok(item)));
                    }
                    if this
                        .expected_output_dimensions
                        .as_ref()
                        .is_some_and(|dimensions| this.row_count != dimensions.len())
                    {
                        this.finished = true;
                        return Poll::Ready(Some(Err(invalid_response(
                            "embedding JSONL result count differs from the known input count",
                        ))));
                    }
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl<'a> GeminiBatchService<'a> {
    /// Enqueue an inline or file-backed `EmbedContent` Batch operation.
    pub async fn create_embedding_batch(
        &self,
        request: &GeminiEmbeddingBatchCreateRequest,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchSnapshot, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        if let Some(file) = request.input.file() {
            pinned_service.validate_embedding_file_ref(file)?;
        }
        let body = request.encode()?;
        let path = format!(
            "/models/{}:asyncBatchEmbedContent",
            encode_path_segment(&request.model)
        );
        let response = pinned_service
            .send("POST", &path, Some(body), credential)
            .await
            .map_err(|error| embedding_mutation_transport_error("create", None, error))?;
        ensure_success(&response)?;
        let mut snapshot = decode_json(&response.body)
            .and_then(|native| pinned_service.decode_embedding_snapshot_value(native, None))
            .map_err(|error| GeminiBatchError::EmbeddingOutcomeUnknownResponse {
                operation: "create",
                reference: None,
                reason: error.to_string(),
            })?;
        if snapshot
            .reference
            .model
            .as_deref()
            .is_some_and(|model| model != request.model)
        {
            return Err(GeminiBatchError::EmbeddingOutcomeUnknownResponse {
                operation: "create",
                reference: None,
                reason: "Google returned a different embedding model".into(),
            });
        }
        snapshot.reference.model = Some(request.model.clone());
        snapshot.reference.expected_output_dimensions =
            request.input.expected_output_dimensions.clone();
        snapshot.reference.expected_output_keys_in_order =
            request.input.expected_output_keys_in_order.clone();
        snapshot.reference.expected_output_dimension = request.input.expected_output_dimension;
        snapshot.reference.expected_output_dimensions_by_key =
            request.input.expected_output_dimensions_by_key.clone();
        if let Some(file) = snapshot.output_file.as_mut() {
            file.model = Some(request.model.clone());
            file.expected_output_keys_in_order =
                request.input.expected_output_keys_in_order.clone();
            file.expected_output_dimensions = request.input.expected_output_dimensions.clone();
            file.expected_output_dimension = request.input.expected_output_dimension;
            file.expected_output_dimensions_by_key =
                request.input.expected_output_dimensions_by_key.clone();
        }
        Ok(snapshot)
    }

    /// Fetch and type-check one embedding Batch operation.
    pub async fn get_embedding_batch(
        &self,
        reference: &GeminiEmbeddingBatchRef,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchSnapshot, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_embedding_reference(reference)?;
        let response = pinned_service
            .send(
                "GET",
                &format!("/batches/{}", encode_path_segment(&reference.batch_id)),
                None,
                credential,
            )
            .await?;
        ensure_success(&response)?;
        let snapshot = pinned_service
            .decode_embedding_snapshot_value(decode_json(&response.body)?, Some(reference))?;
        pinned_service.ensure_same_embedding_batch(reference, &snapshot.reference)?;
        Ok(snapshot)
    }

    /// Fetch one page and retain only operations explicitly typed by Google as
    /// `EmbedContentBatch`; generation and unknown operation kinds are not
    /// silently exposed as embeddings.
    pub async fn list_embedding_batches(
        &self,
        options: &GeminiBatchListOptions,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchListPage, GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        let mut url = pinned_service.url("/batches")?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(page_size) = options.page_size {
                if page_size == 0 {
                    return Err(invalid_input("page size must be greater than zero"));
                }
                query.append_pair("pageSize", &page_size.to_string());
            }
            if let Some(page_token) = options.page_token.as_deref() {
                if page_token.trim().is_empty() {
                    return Err(invalid_input("page token must be non-empty"));
                }
                query.append_pair("pageToken", page_token);
            }
        }
        let response = pinned_service
            .send_url("GET", url, None, credential)
            .await?;
        ensure_success(&response)?;
        let native = decode_json(&response.body)?;
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("list response must be a JSON object"))?;
        let token = decode_next_page_token(object, options.page_token.as_deref())?;
        let operations = match object.get("operations") {
            Some(Value::Array(operations)) => operations,
            Some(_) => return Err(invalid_response("operations must be an array")),
            None => {
                return Ok(GeminiEmbeddingBatchListPage {
                    batches: Vec::new(),
                    next_page_token: token,
                    native,
                });
            }
        };
        let mut seen = BTreeSet::new();
        let mut batches = Vec::new();
        for operation in operations {
            let name = operation
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_response("operation is missing its resource name"))?;
            let id = name
                .strip_prefix("batches/")
                .filter(|id| valid_resource_id(id))
                .ok_or_else(|| invalid_response("operation name is not a batches resource"))?;
            if !seen.insert(id.to_owned()) {
                return Err(invalid_response("list page contains duplicate batch IDs"));
            }
            let type_name = operation
                .get("metadata")
                .and_then(|metadata| metadata.get("@type"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_response("operation metadata is missing its @type"))?;
            if is_embedding_batch_type(type_name) {
                batches
                    .push(pinned_service.decode_embedding_snapshot_value(operation.clone(), None)?);
            } else if !is_generation_batch_type(type_name) {
                return Err(invalid_response("list operation has an unknown batch type"));
            }
        }
        Ok(GeminiEmbeddingBatchListPage {
            batches,
            next_page_token: token,
            native,
        })
    }

    /// Request best-effort cancellation of an embedding operation.
    pub async fn cancel_embedding_batch(
        &self,
        reference: &GeminiEmbeddingBatchRef,
        credential: &Secret<String>,
    ) -> Result<(), GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_embedding_reference(reference)?;
        let path = format!(
            "/batches/{}:cancel",
            encode_path_segment(&reference.batch_id)
        );
        let response = pinned_service
            .send("POST", &path, None, credential)
            .await
            .map_err(|error| {
                embedding_mutation_transport_error("cancel", Some(reference.clone()), error)
            })?;
        ensure_success(&response)?;
        ensure_embedding_empty_success("cancel", reference, &response.body)
    }

    /// Delete an embedding operation record without cancelling its execution.
    pub async fn delete_embedding_batch(
        &self,
        reference: &GeminiEmbeddingBatchRef,
        credential: &Secret<String>,
    ) -> Result<(), GeminiBatchError> {
        let pinned_service = self.pin()?;
        validate_credential(credential)?;
        pinned_service.validate_embedding_reference(reference)?;
        let response = pinned_service
            .send(
                "DELETE",
                &format!("/batches/{}", encode_path_segment(&reference.batch_id)),
                None,
                credential,
            )
            .await
            .map_err(|error| {
                embedding_mutation_transport_error("delete", Some(reference.clone()), error)
            })?;
        ensure_success(&response)?;
        ensure_embedding_empty_success("delete", reference, &response.body)
    }

    /// Read one status snapshot and decode inline per-item embeddings/errors.
    pub async fn embedding_batch_results(
        &self,
        reference: &GeminiEmbeddingBatchRef,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchResults, GeminiBatchError> {
        let pinned_service = self.pin()?;
        let snapshot = pinned_service
            .get_embedding_batch(reference, credential)
            .await?;
        let values = snapshot
            .response
            .as_ref()
            .and_then(|response| response.get("output"))
            .and_then(|output| output.get("inlinedResponses"))
            .and_then(|inlined| inlined.get("inlinedResponses"));
        let items = values
            .map(|values| {
                parse_embedding_results(
                    values,
                    snapshot.reference.expected_output_dimensions.as_deref(),
                )
            })
            .transpose()?;
        Ok(GeminiEmbeddingBatchResults { snapshot, items })
    }

    /// Stream a Gemini embedding input JSONL file using the Files API resumable
    /// upload, returning a reference that is accepted only by embedding batches.
    pub async fn upload_embedding_input_stream(
        &self,
        filename: &str,
        size_bytes: u64,
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchFileRef, GeminiBatchError> {
        let pinned_service = self.pin()?;
        let file = pinned_service
            .upload_input_stream(filename, size_bytes, body, credential)
            .await?;
        GeminiEmbeddingBatchFileRef::from_resource_name(&pinned_service.scope, file.resource_name())
    }

    /// Encode and upload typed keyed embedding JSONL rows through the Gemini
    /// Files API. Result rows are checked in documented input order; keys and
    /// dimensions are preserved for keyed and keyless responses.
    pub async fn upload_embedding_input_jsonl(
        &self,
        filename: &str,
        input: GeminiEmbeddingBatchJsonlInput,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchFileRef, GeminiBatchError> {
        let pinned_service = self.pin()?;
        let model = input.model.clone();
        let expected_output_keys_in_order =
            Some(input.requests.iter().map(|row| row.key.clone()).collect());
        let expected_output_dimensions = Some(
            input
                .requests
                .iter()
                .map(|row| row.request.config.output_dimensionality)
                .collect(),
        );
        let expected_output_dimensions_by_key = Some(
            input
                .requests
                .iter()
                .map(|row| (row.key.clone(), row.request.config.output_dimensionality))
                .collect(),
        );
        let (size_bytes, body) = input.into_stream()?;
        let mut file = pinned_service
            .upload_embedding_input_stream(filename, size_bytes, body, credential)
            .await?;
        file.model = Some(model);
        file.expected_output_keys_in_order = expected_output_keys_in_order;
        file.expected_output_dimensions = expected_output_dimensions;
        file.expected_output_dimensions_by_key = expected_output_dimensions_by_key;
        Ok(file)
    }

    /// Download a completed file-backed embedding Batch output as raw JSONL.
    pub async fn download_embedding_results(
        &self,
        file: &GeminiEmbeddingBatchFileRef,
        credential: &Secret<String>,
    ) -> Result<GeminiEmbeddingBatchResultStream, GeminiBatchError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_embedding_file_ref(file)?;
        let generic = GeminiBatchFileRef {
            scope: file.scope.clone(),
            file_id: file.file_id.clone(),
        };
        let raw = pinned_service
            .download_results(&generic, credential)
            .await?;
        Ok(GeminiEmbeddingBatchResultStream::new(
            raw.body,
            file.expected_output_dimensions.clone(),
            file.expected_output_keys_in_order.clone(),
            file.expected_output_dimension,
            file.expected_output_dimensions_by_key.clone(),
        ))
    }

    fn decode_embedding_snapshot_value(
        &self,
        native: Value,
        expected: Option<&GeminiEmbeddingBatchRef>,
    ) -> Result<GeminiEmbeddingBatchSnapshot, GeminiBatchError> {
        let object = native
            .as_object()
            .ok_or_else(|| invalid_response("operation must be a JSON object"))?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_response("operation is missing its resource name"))?;
        let batch_id = name
            .strip_prefix("batches/")
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| invalid_response("operation name is not a batches resource"))?
            .to_owned();
        let metadata = object.get("metadata").cloned();
        let type_name = metadata
            .as_ref()
            .and_then(|metadata| metadata.get("@type"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_response("embedding operation metadata is missing its @type"))?;
        if !is_embedding_batch_type(type_name) {
            return Err(invalid_response(
                "operation is not an EmbedContentBatch resource",
            ));
        }
        let provider_model = metadata
            .as_ref()
            .and_then(|metadata| metadata.get("model"))
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| invalid_response("embedding batch model must be a string"))
                    .and_then(|value| {
                        normalize_model_name(value)
                            .map_err(|_| invalid_response("embedding batch model is invalid"))
                    })
            })
            .transpose()?;
        if expected
            .and_then(|reference| reference.model.as_deref())
            .zip(provider_model.as_deref())
            .is_some_and(|(expected, actual)| expected != actual)
        {
            return Err(invalid_response(
                "Google returned a different embedding batch model",
            ));
        }
        let model = expected
            .and_then(|reference| reference.model.clone())
            .or(provider_model);
        let response = object.get("response").cloned();
        let output_file = response
            .as_ref()
            .and_then(|value| value.get("output"))
            .and_then(|value| value.get("responsesFile"))
            .map(|value| {
                let name = value.as_str().ok_or_else(|| {
                    invalid_response("response.output.responsesFile must be a file resource name")
                })?;
                GeminiEmbeddingBatchFileRef::from_resource_name(&self.scope, name).map(
                    |mut file| {
                        file.model = expected.and_then(|reference| reference.model.clone());
                        file.expected_output_keys_in_order = expected
                            .and_then(|reference| reference.expected_output_keys_in_order.clone());
                        file.expected_output_dimensions = expected
                            .and_then(|reference| reference.expected_output_dimensions.clone());
                        file.expected_output_dimension =
                            expected.and_then(|reference| reference.expected_output_dimension);
                        file.expected_output_dimensions_by_key = expected.and_then(|reference| {
                            reference.expected_output_dimensions_by_key.clone()
                        });
                        file
                    },
                )
            })
            .transpose()?;
        let state_value = metadata
            .as_ref()
            .and_then(|value| value.get("state"))
            .or_else(|| response.as_ref().and_then(|value| value.get("state")));
        let state = state_value
            .and_then(Value::as_str)
            .map(GeminiBatchState::parse);
        Ok(GeminiEmbeddingBatchSnapshot {
            reference: GeminiEmbeddingBatchRef {
                operation: GeminiEmbeddingBatchOperation::EmbedContent,
                scope: self.scope.clone(),
                batch_id,
                model,
                expected_output_dimensions: expected
                    .and_then(|reference| reference.expected_output_dimensions.clone()),
                expected_output_keys_in_order: expected
                    .and_then(|reference| reference.expected_output_keys_in_order.clone()),
                expected_output_dimensions_by_key: expected
                    .and_then(|reference| reference.expected_output_dimensions_by_key.clone()),
                expected_output_dimension: expected
                    .and_then(|reference| reference.expected_output_dimension),
            },
            done: object.get("done").and_then(Value::as_bool),
            state,
            metadata,
            error: object.get("error").cloned(),
            response,
            output_file,
            native,
        })
    }

    fn validate_embedding_file_ref(
        &self,
        reference: &GeminiEmbeddingBatchFileRef,
    ) -> Result<(), GeminiBatchError> {
        if reference.scope != self.scope
            || !valid_gemini_file_id(&reference.file_id)
            || reference
                .model
                .as_deref()
                .is_some_and(|model| normalize_model_name(model).ok().as_deref() != Some(model))
            || reference.expected_output_dimension == Some(0)
            || !valid_expected_output_metadata(
                reference.expected_output_keys_in_order.as_deref(),
                reference.expected_output_dimensions.as_deref(),
                reference.expected_output_dimensions_by_key.as_ref(),
            )
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_embedding_reference(
        &self,
        reference: &GeminiEmbeddingBatchRef,
    ) -> Result<(), GeminiBatchError> {
        if reference.scope != self.scope
            || reference.operation != GeminiEmbeddingBatchOperation::EmbedContent
            || !valid_resource_id(&reference.batch_id)
            || reference
                .model
                .as_deref()
                .is_some_and(|model| normalize_model_name(model).ok().as_deref() != Some(model))
            || reference
                .expected_output_dimensions
                .as_ref()
                .is_some_and(|dimensions| {
                    dimensions.is_empty() || dimensions.iter().flatten().any(|n| *n == 0)
                })
            || reference.expected_output_dimension == Some(0)
            || !valid_expected_output_metadata(
                reference.expected_output_keys_in_order.as_deref(),
                reference.expected_output_dimensions.as_deref(),
                reference.expected_output_dimensions_by_key.as_ref(),
            )
        {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn ensure_same_embedding_batch(
        &self,
        expected: &GeminiEmbeddingBatchRef,
        actual: &GeminiEmbeddingBatchRef,
    ) -> Result<(), GeminiBatchError> {
        if expected.scope != actual.scope || expected.batch_id != actual.batch_id {
            return Err(invalid_response(
                "Google returned a different embedding batch resource name",
            ));
        }
        Ok(())
    }
}

fn encode_embedding_parts(
    model: &str,
    parts: &[GeminiBatchEmbeddingPart],
) -> Result<Vec<Value>, GeminiBatchError> {
    let mut encoded = Vec::with_capacity(parts.len());
    let mut bytes = 0usize;
    let mut image_count = 0usize;
    let mut pdf_count = 0usize;
    for part in parts {
        match part {
            GeminiBatchEmbeddingPart::Text(text) => {
                bytes = bytes
                    .checked_add(text.len())
                    .ok_or_else(|| invalid_input("embedding content is too large"))?;
                encoded.push(json!({"text": text}));
            }
            GeminiBatchEmbeddingPart::Media(media) => {
                if model != "gemini-embedding-2" {
                    return Err(invalid_input(
                        "Gemini embedding media requires the gemini-embedding-2 model",
                    ));
                }
                let mime = media.mime_type.to_ascii_lowercase();
                let image = matches!(mime.as_str(), "image/png" | "image/jpeg");
                let audio = matches!(mime.as_str(), "audio/mpeg" | "audio/wav");
                let video = matches!(mime.as_str(), "video/mp4" | "video/quicktime");
                let pdf = mime == "application/pdf";
                if !image && !audio && !video && !pdf {
                    return Err(invalid_input(
                        "Gemini Embedding 2 supports PNG/JPEG images, MP3/WAV audio, MP4/MOV video, and PDF",
                    ));
                }
                if image {
                    image_count += 1;
                    if image_count > 6
                        || media.duration_seconds.is_some()
                        || media.page_count.is_some()
                    {
                        return Err(invalid_input(
                            "Gemini Embedding 2 accepts at most 6 images without duration or page-count metadata",
                        ));
                    }
                } else if audio {
                    if !media
                        .duration_seconds
                        .is_some_and(|duration| (1..=180).contains(&duration))
                        || media.page_count.is_some()
                    {
                        return Err(invalid_input(
                            "Gemini Embedding 2 audio requires a duration from 1 to 180 seconds and no page count",
                        ));
                    }
                } else if video {
                    if !media
                        .duration_seconds
                        .is_some_and(|duration| (1..=120).contains(&duration))
                        || media.page_count.is_some()
                    {
                        return Err(invalid_input(
                            "Gemini Embedding 2 video requires a duration from 1 to 120 seconds and no page count",
                        ));
                    }
                } else {
                    pdf_count += 1;
                    if pdf_count > 1
                        || media.duration_seconds.is_some()
                        || !media
                            .page_count
                            .is_some_and(|pages| (1..=6).contains(&pages))
                    {
                        return Err(invalid_input(
                            "Gemini Embedding 2 PDF input requires one PDF with 1 to 6 pages and no duration",
                        ));
                    }
                }
                let value = match &media.source {
                    GeminiEmbeddingSource::Inline(data) => {
                        if data.is_empty() {
                            return Err(invalid_input("inline embedding media cannot be empty"));
                        }
                        bytes = bytes
                            .checked_add(data.len())
                            .ok_or_else(|| invalid_input("embedding content is too large"))?;
                        json!({"inlineData": {"mimeType": mime, "data": STANDARD.encode(data)}})
                    }
                    GeminiEmbeddingSource::FileUri(uri) => {
                        validate_embedding_file_uri(uri)?;
                        json!({"fileData": {"mimeType": mime, "fileUri": uri}})
                    }
                };
                encoded.push(value);
            }
        }
    }
    if bytes > MAX_INLINE_REQUEST_BYTES {
        return Err(invalid_input("embedding content exceeds 20 MB"));
    }
    Ok(encoded)
}

fn validate_embedding_file_uri(uri: &str) -> Result<(), GeminiBatchError> {
    let parsed =
        url::Url::parse(uri).map_err(|_| invalid_input("Gemini Files API URI is invalid"))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || parsed.path().is_empty()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_input(
            "Gemini Files API URI must be HTTPS without credentials, query, or fragment",
        ));
    }
    Ok(())
}

fn decode_next_page_token(
    object: &Map<String, Value>,
    requested: Option<&str>,
) -> Result<Option<String>, GeminiBatchError> {
    let token = match object.get("nextPageToken") {
        None => return Ok(None),
        Some(Value::String(token)) if token.is_empty() => return Ok(None),
        Some(Value::String(token)) => token,
        Some(_) => return Err(invalid_response("nextPageToken must be a string")),
    };
    if requested == Some(token.as_str()) {
        return Err(invalid_response(
            "nextPageToken did not advance the requested page cursor",
        ));
    }
    Ok(Some(token.clone()))
}

fn valid_expected_output_metadata(
    keys_in_order: Option<&[String]>,
    dimensions_in_order: Option<&[Option<u32>]>,
    dimensions_by_key: Option<&BTreeMap<String, Option<u32>>>,
) -> bool {
    let Some(keys) = keys_in_order else {
        return dimensions_by_key.is_none()
            && dimensions_in_order.is_none_or(|dimensions| {
                !dimensions.is_empty() && !dimensions.iter().flatten().any(|n| *n == 0)
            });
    };
    let Some(dimensions) = dimensions_in_order else {
        return false;
    };
    if keys.is_empty()
        || keys.len() != dimensions.len()
        || dimensions.iter().flatten().any(|dimension| *dimension == 0)
    {
        return false;
    }
    let mut unique_keys = BTreeSet::new();
    if !keys
        .iter()
        .all(|key| valid_identity(key) && unique_keys.insert(key.as_str()))
    {
        return false;
    }
    match dimensions_by_key {
        Some(by_key) => {
            by_key.len() == keys.len()
                && keys
                    .iter()
                    .zip(dimensions)
                    .all(|(key, dimension)| by_key.get(key).copied() == Some(*dimension))
                && by_key
                    .iter()
                    .all(|(key, dimension)| valid_identity(key) && *dimension != Some(0))
        }
        None => true,
    }
}

fn parse_embedding_results(
    value: &Value,
    expected_dimensions: Option<&[Option<u32>]>,
) -> Result<Vec<GeminiEmbeddingBatchItemResult>, GeminiBatchError> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid_response("inlinedResponses must be an array"))?;
    if expected_dimensions.is_some_and(|expected| expected.len() != values.len()) {
        return Err(invalid_response(
            "embedding result count differs from the known input count",
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, native)| {
            let object = native
                .as_object()
                .ok_or_else(|| invalid_response("inline embedding result must be a JSON object"))?;
            let metadata = object.get("metadata").cloned();
            if metadata
                .as_ref()
                .is_some_and(|metadata| !metadata.is_object())
            {
                return Err(invalid_response(
                    "embedding result metadata must be an object",
                ));
            }
            let response_value = object.get("response");
            let error_value = object.get("error");
            if response_value.is_some() == error_value.is_some() {
                return Err(invalid_response(
                    "inline embedding result must contain exactly one of response or error",
                ));
            }
            let expected = expected_dimensions
                .and_then(|dimensions| dimensions.get(index))
                .copied()
                .flatten();
            let response = response_value
                .map(|value| parse_embed_content_response(value, expected))
                .transpose()?;
            let error = error_value.map(parse_embedding_item_error).transpose()?;
            Ok(GeminiEmbeddingBatchItemResult {
                metadata,
                response,
                error,
                native: native.clone(),
            })
        })
        .collect()
}

fn parse_embedding_result_line(
    line: &[u8],
    ordered_dimension: Option<u32>,
    uniform_dimension: Option<u32>,
    dimensions_by_key: Option<&BTreeMap<String, Option<u32>>>,
    ordered_key: Option<&str>,
) -> Result<GeminiEmbeddingBatchItemResult, GeminiBatchError> {
    let native = decode_json(line)?;
    let object = native
        .as_object()
        .ok_or_else(|| invalid_response("embedding JSONL row must be a JSON object"))?;
    let top_level_key = match object.get("key") {
        Some(Value::String(key)) => Some(key.as_str()),
        Some(_) => return Err(invalid_response("embedding JSONL key must be a string")),
        None => None,
    };
    let metadata = match object.get("metadata") {
        Some(Value::Object(metadata)) => Some(metadata),
        Some(_) => {
            return Err(invalid_response(
                "embedding result metadata must be an object",
            ))
        }
        None => None,
    };
    let metadata_key = match metadata.and_then(|metadata| metadata.get("key")) {
        Some(Value::String(key)) => Some(key.as_str()),
        Some(_) => {
            return Err(invalid_response(
                "embedding JSONL metadata key must be a string",
            ))
        }
        None => None,
    };
    if top_level_key
        .zip(metadata_key)
        .is_some_and(|(top_level, metadata)| top_level != metadata)
    {
        return Err(invalid_response(
            "embedding JSONL top-level and metadata keys conflict",
        ));
    }
    let explicit_key = top_level_key.or(metadata_key);
    if explicit_key
        .zip(ordered_key)
        .is_some_and(|(actual, expected)| actual != expected)
    {
        return Err(invalid_response(
            "embedding JSONL result key does not match the documented input order",
        ));
    }
    let key = explicit_key.or(ordered_key);
    let expected_dimension = match (dimensions_by_key, explicit_key) {
        (Some(dimensions), Some(key)) => dimensions
            .get(key)
            .copied()
            .ok_or_else(|| {
                invalid_response("embedding JSONL result contains an unknown request key")
            })?
            .or(ordered_dimension)
            .or(uniform_dimension),
        _ => ordered_dimension.or(uniform_dimension),
    };
    let metadata_with_key = |metadata: Option<Value>| -> Option<Value> {
        match (metadata, key) {
            (Some(Value::Object(mut metadata)), Some(key)) => {
                metadata.entry("key").or_insert_with(|| json!(key));
                Some(Value::Object(metadata))
            }
            (Some(metadata), _) => Some(metadata),
            (None, Some(key)) => Some(json!({"key": key})),
            (None, None) => None,
        }
    };
    if object.contains_key("response") || object.contains_key("error") {
        let expected = [expected_dimension];
        let expected = expected_dimension.map(|_| expected.as_slice());
        let mut item =
            parse_embedding_results(&Value::Array(vec![native.clone()]), expected)?.remove(0);
        item.metadata = metadata_with_key(item.metadata);
        return Ok(item);
    }
    if object.contains_key("embedding") {
        let response = parse_embed_content_response(&native, expected_dimension)?;
        return Ok(GeminiEmbeddingBatchItemResult {
            metadata: metadata_with_key(object.get("metadata").cloned()),
            response: Some(response),
            error: None,
            native,
        });
    }
    if object.contains_key("code") || object.contains_key("message") {
        return Ok(GeminiEmbeddingBatchItemResult {
            metadata: metadata_with_key(object.get("metadata").cloned()),
            response: None,
            error: Some(parse_embedding_item_error(&native)?),
            native,
        });
    }
    Err(invalid_response(
        "embedding JSONL row is neither an embedding response nor a status object",
    ))
}

fn parse_embed_content_response(
    value: &Value,
    expected_dimension: Option<u32>,
) -> Result<GeminiEmbedContentResponse, GeminiBatchError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_response("embedding response must be a JSON object"))?;
    let embedding_value = object
        .get("embedding")
        .ok_or_else(|| invalid_response("embedding response is missing embedding"))?;
    let embedding_object = embedding_value
        .as_object()
        .ok_or_else(|| invalid_response("embedding must be a JSON object"))?;
    let values = match embedding_object.get("values") {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) => {
            let decoded = values
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| invalid_response("embedding values must be finite numbers"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if decoded.is_empty() {
                return Err(invalid_response("embedding values cannot be empty"));
            }
            if expected_dimension.is_some_and(|dimension| dimension as usize != decoded.len()) {
                return Err(invalid_response(
                    "embedding value count differs from outputDimensionality",
                ));
            }
            Some(decoded)
        }
        Some(_) => return Err(invalid_response("embedding values must be an array")),
    };
    let shape = match embedding_object.get("shape") {
        None | Some(Value::Null) => None,
        Some(Value::Array(shape)) => {
            let decoded = shape
                .iter()
                .map(|dimension| {
                    dimension
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| {
                            invalid_response("embedding shape must contain non-negative integers")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_none()
                && expected_dimension.is_some()
                && decoded
                    .last()
                    .is_some_and(|dimension| Some(*dimension) != expected_dimension)
            {
                return Err(invalid_response(
                    "embedding tensor shape differs from outputDimensionality",
                ));
            }
            Some(decoded)
        }
        Some(_) => return Err(invalid_response("embedding shape must be an array")),
    };
    Ok(GeminiEmbedContentResponse {
        embedding: GeminiContentEmbedding {
            values,
            shape,
            native: embedding_value.clone(),
        },
        usage_metadata: object.get("usageMetadata").cloned(),
        native: value.clone(),
    })
}

fn parse_embedding_item_error(value: &Value) -> Result<GeminiEmbeddingItemError, GeminiBatchError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_response("embedding item error must be a JSON object"))?;
    let code = match object.get("code") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| invalid_response("embedding item error code must be an integer"))?,
        ),
    };
    let message = match object.get("message") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => {
            return Err(invalid_response(
                "embedding item error message must be a string",
            ));
        }
    };
    let details = match object.get("details") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(details)) => details.clone(),
        Some(_) => {
            return Err(invalid_response(
                "embedding item error details must be an array",
            ));
        }
    };
    Ok(GeminiEmbeddingItemError {
        code,
        message,
        details,
        native: value.clone(),
    })
}

fn ensure_embedding_empty_success(
    operation: &'static str,
    reference: &GeminiEmbeddingBatchRef,
    body: &[u8],
) -> Result<(), GeminiBatchError> {
    let value =
        decode_json(body).map_err(|error| GeminiBatchError::EmbeddingOutcomeUnknownResponse {
            operation,
            reference: Some(Box::new(reference.clone())),
            reason: error.to_string(),
        })?;
    if value.as_object().is_none_or(|object| !object.is_empty()) {
        return Err(GeminiBatchError::EmbeddingOutcomeUnknownResponse {
            operation,
            reference: Some(Box::new(reference.clone())),
            reason: "expected the documented empty JSON object".into(),
        });
    }
    Ok(())
}
