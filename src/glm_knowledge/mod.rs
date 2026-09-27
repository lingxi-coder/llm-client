//! Native Zhipu GLM Knowledge Base API.
//!
//! This service is deliberately separate from the OpenAI-compatible vector
//! store API in [`crate::retrieval`]. References bind a Zhipu knowledge base to
//! the provider profile, configured endpoint, and account that created it.

use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    files::{
        append_field, exact_upload_stream, multipart_boundary, provider_file_endpoint_fingerprint,
        sanitize_filename, validate_media_type, UploadFileStream,
    },
    protocol::{LlmError, ProviderId, ProviderProfile, ServiceAuth, ServiceSetting},
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest},
};
use bytes::{Bytes, BytesMut};
use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fmt, time::Duration};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_QUERY_CHARS: usize = 1000;

/// Zhipu Knowledge Base API root. The endpoint is the complete
/// `.../api/llm-application/open` service root; each operation appends its
/// documented resource path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeRoute {
    pub endpoint: String,
    pub auth: ServiceAuth,
}

/// Opaque identity for one Zhipu knowledge base.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub knowledge_id: String,
}

/// Opaque identity for one document within a Zhipu knowledge base.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeDocumentRef {
    pub knowledge: GlmKnowledgeRef,
    pub document_id: String,
}

/// Documented embedding identifiers accepted when creating a knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmKnowledgeEmbedding {
    Embedding2,
    Embedding3,
    Embedding3Pro,
}

impl GlmKnowledgeEmbedding {
    const fn wire_id(self) -> u8 {
        match self {
            Self::Embedding2 => 3,
            Self::Embedding3 => 11,
            Self::Embedding3Pro => 12,
        }
    }
}

impl Serialize for GlmKnowledgeEmbedding {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.wire_id())
    }
}

impl<'de> Deserialize<'de> for GlmKnowledgeEmbedding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match u8::deserialize(deserializer)? {
            3 => Ok(Self::Embedding2),
            11 => Ok(Self::Embedding3),
            12 => Ok(Self::Embedding3Pro),
            value => Err(serde::de::Error::custom(format!(
                "unsupported GLM knowledge embedding id {value}"
            ))),
        }
    }
}

/// Whether Zhipu should apply contextual augmentation to an indexed document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmKnowledgeContextual {
    Disabled,
    Enabled,
}

impl Serialize for GlmKnowledgeContextual {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(u8::from(matches!(self, Self::Enabled)))
    }
}

impl<'de> Deserialize<'de> for GlmKnowledgeContextual {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match u8::deserialize(deserializer)? {
            0 => Ok(Self::Disabled),
            1 => Ok(Self::Enabled),
            value => Err(serde::de::Error::custom(format!(
                "invalid GLM contextual setting {value}"
            ))),
        }
    }
}

/// Request accepted by Zhipu's `POST /knowledge` endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmCreateKnowledgeRequest {
    pub embedding_id: GlmKnowledgeEmbedding,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<GlmKnowledgeEmbeddingModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual: Option<GlmKnowledgeContextual>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<GlmKnowledgeBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<GlmKnowledgeIcon>,
}

/// String form of the same embedding family used by the GLM API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmKnowledgeEmbeddingModel {
    #[serde(rename = "Embedding-2")]
    Embedding2,
    #[serde(rename = "Embedding-3")]
    Embedding3,
    #[serde(rename = "Embedding-3-pro")]
    Embedding3Pro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmKnowledgeBackground {
    #[serde(rename = "blue")]
    Blue,
    #[serde(rename = "red")]
    Red,
    #[serde(rename = "orange")]
    Orange,
    #[serde(rename = "purple")]
    Purple,
    #[serde(rename = "sky")]
    Sky,
    #[serde(rename = "green")]
    Green,
    #[serde(rename = "yellow")]
    Yellow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmKnowledgeIcon {
    #[serde(rename = "question")]
    Question,
    #[serde(rename = "book")]
    Book,
    #[serde(rename = "seal")]
    Seal,
    #[serde(rename = "wrench")]
    Wrench,
    #[serde(rename = "tag")]
    Tag,
    #[serde(rename = "horn")]
    Horn,
    #[serde(rename = "house")]
    House,
}

/// Knowledge base created through Zhipu's management endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledge {
    pub reference: GlmKnowledgeRef,
    /// Provider-native metadata, retained because Zhipu returns more fields
    /// than the create response currently documents.
    pub native: Value,
}

/// Optional pagination controls for `GET /knowledge`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeListRequest {
    /// One-based page number. The provider defaults to page 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    /// Page size. The provider defaults to 10 entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u32>,
}

/// One page returned by `GET /knowledge`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeListResult {
    pub knowledge_bases: Vec<GlmKnowledge>,
    pub total: u64,
    pub native: Value,
}

/// Partial update accepted by Zhipu's `PUT /knowledge/{id}` endpoint.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmUpdateKnowledgeRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_id: Option<GlmKnowledgeEmbedding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<GlmKnowledgeEmbeddingModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual: Option<GlmKnowledgeContextual>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<GlmKnowledgeBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<GlmKnowledgeIcon>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_header: Option<Value>,
}

impl fmt::Debug for GlmUpdateKnowledgeRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmUpdateKnowledgeRequest")
            .field("embedding_id", &self.embedding_id)
            .field("embedding_model", &self.embedding_model)
            .field("contextual", &self.contextual)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("background", &self.background)
            .field("icon", &self.icon)
            .field(
                "callback_url",
                &self.callback_url.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "callback_header",
                &self.callback_header.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// A successful knowledge-base update or delete response, preserved verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeMutationResult {
    pub native: Value,
}

/// Optional pagination and name search for `GET /document`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeDocumentListRequest {
    /// One-based page number. The provider defaults to page 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    /// Page size. The provider defaults to 10 entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u32>,
    /// Optional document-name search term.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word: Option<String>,
}

/// A document row returned by the scoped GLM document-list endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeDocument {
    pub reference: GlmKnowledgeDocumentRef,
    pub knowledge_type: Option<u8>,
    pub custom_separator: Option<Vec<String>>,
    pub sentence_size: Option<u32>,
    pub length: Option<u64>,
    pub word_num: Option<u64>,
    pub name: Option<String>,
    pub url: Option<String>,
    pub embedding_stat: Option<i64>,
    pub fail_info: Option<Value>,
    pub native: Value,
}

/// One page returned by `GET /document`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeDocumentListResult {
    pub documents: Vec<GlmKnowledgeDocument>,
    pub total: u64,
    pub native: Value,
}

/// URL based document ingestion supported by `POST /document/upload_url`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmUploadUrlDocumentsRequest {
    pub documents: Vec<GlmUrlDocumentInput>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmUrlDocumentInput {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge_type: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_separator: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_header: Option<Value>,
}

impl fmt::Debug for GlmUrlDocumentInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmUrlDocumentInput")
            .field("url", &self.url)
            .field("knowledge_type", &self.knowledge_type)
            .field("custom_separator", &self.custom_separator)
            .field("sentence_size", &self.sentence_size)
            .field(
                "callback_url",
                &self.callback_url.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "callback_header",
                &self.callback_header.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmUploadUrlDocumentsResult {
    pub succeeded: Vec<GlmUploadedUrlDocument>,
    pub failed: Vec<GlmFailedUrlDocument>,
    pub native: Value,
}

/// Multipart parsing controls accepted by Zhipu's current Knowledge API file
/// upload route. File bytes are passed separately as one-shot streams.
#[derive(Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct GlmUploadFileDocumentsRequest {
    /// Document slicing strategy: 1, 2, 3, 5, 6, or 7. Omit for provider
    /// auto-detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge_type: Option<u8>,
    /// Custom separators for `knowledge_type = 5`; emitted as repeated form
    /// fields as documented for the string-array option.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_separator: Option<Vec<String>>,
    /// Custom chunk size (20–2000) for `knowledge_type = 5`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence_size: Option<u32>,
    /// Whether the provider should parse images in documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_image: Option<bool>,
    /// Provider callback URL. Its value is redacted from Debug output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    /// Headers sent by Zhipu when invoking the callback. Values are redacted
    /// from Debug output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_header: Option<Value>,
    /// Optional decimal document-word limit. The provider documents a string
    /// form and requires digits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_num_limit: Option<String>,
    /// Optional caller request identifier, sent as the documented `req_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl fmt::Debug for GlmUploadFileDocumentsRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlmUploadFileDocumentsRequest")
            .field("knowledge_type", &self.knowledge_type)
            .field("custom_separator", &self.custom_separator)
            .field("sentence_size", &self.sentence_size)
            .field("parse_image", &self.parse_image)
            .field(
                "callback_url",
                &self.callback_url.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "callback_header",
                &self.callback_header.as_ref().map(|_| "<redacted>"),
            )
            .field("word_num_limit", &self.word_num_limit)
            .field("request_id", &self.request_id)
            .finish()
    }
}

/// Result row for one successfully accepted file. Provider-specific fields
/// remain available in `native`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmUploadedFileDocument {
    pub reference: GlmKnowledgeDocumentRef,
    pub file_name: Option<String>,
    pub native: Value,
}

/// Per-file ingestion failure returned by Zhipu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlmFailedFileDocument {
    pub file_name: Option<String>,
    pub fail_reason: Option<String>,
    pub native: Value,
}

/// File upload outcome. Rows that the provider reports as accepted or failed
/// are retained independently; unexpected success rows remain in `unresolved`
/// so a malformed ID cannot hide an accepted upload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmUploadFileDocumentsResult {
    pub succeeded: Vec<GlmUploadedFileDocument>,
    pub failed: Vec<GlmFailedFileDocument>,
    pub unresolved: Vec<Value>,
    /// Submitted file names for which the provider returned no usable result.
    pub missing_files: Vec<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmUploadedUrlDocument {
    pub reference: GlmKnowledgeDocumentRef,
    pub url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlmFailedUrlDocument {
    pub url: Option<String>,
    pub fail_reason: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmKnowledgeRecallMethod {
    Embedding,
    Keyword,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GlmKnowledgeRerankModel {
    Rerank,
    RerankPro,
}

/// Query controls for `POST /knowledge/retrieve`.
///
/// Knowledge IDs are passed as scoped [`GlmKnowledgeRef`] values to
/// [`GlmKnowledgeService::retrieve`], so callers cannot accidentally send an
/// ID from another account or provider profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmKnowledgeRetrieveRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<GlmKnowledgeDocumentRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_method: Option<GlmKnowledgeRecallMethod>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_ratio: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_model: Option<GlmKnowledgeRerankModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fractional_threshold: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeRetrieveResult {
    pub matches: Vec<GlmKnowledgeMatch>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmKnowledgeMatch {
    pub text: String,
    pub score: Option<f64>,
    pub metadata: GlmKnowledgeMatchMetadata,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GlmKnowledgeMatchMetadata {
    pub knowledge_id: Option<String>,
    pub document_id: Option<String>,
    pub document_name: Option<String>,
    pub document_url: Option<String>,
    pub contextual_text: Option<String>,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum GlmKnowledgeError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("Zhipu Knowledge Base API returned HTTP {status} (provider code {code:?})")]
    Provider {
        status: u16,
        code: Option<i64>,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid Zhipu Knowledge Base response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("outcome of Zhipu Knowledge Base {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Zhipu Knowledge Base {operation} is unknown from the provider response: {message}")]
    ResponseOutcomeUnknown {
        operation: &'static str,
        message: String,
        native: Value,
    },
}

/// Service façade. Every operation captures one immutable client snapshot.
#[derive(Clone, Copy)]
pub struct GlmKnowledgeService<'a> {
    source: ClientSource<'a>,
}

impl<'a> GlmKnowledgeService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    /// List personal Zhipu knowledge bases for the selected profile/account.
    pub async fn list_knowledge(
        self,
        profile_name: &str,
        request: &GlmKnowledgeListRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeListResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .list_knowledge(profile_name, request, options)
            .await
    }

    /// Create a personal Zhipu knowledge base.
    pub async fn create_knowledge(
        self,
        profile_name: &str,
        request: &GlmCreateKnowledgeRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledge, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .create_knowledge(profile_name, request, options)
            .await
    }

    /// Read one personal knowledge base through its account-scoped reference.
    pub async fn get_knowledge(
        self,
        knowledge: &GlmKnowledgeRef,
        options: &RequestOptions,
    ) -> Result<GlmKnowledge, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .get_knowledge(knowledge, options)
            .await
    }

    /// Update documented metadata on one account-scoped knowledge base.
    pub async fn update_knowledge(
        self,
        knowledge: &GlmKnowledgeRef,
        request: &GlmUpdateKnowledgeRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeMutationResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .update_knowledge(knowledge, request, options)
            .await
    }

    /// Delete one account-scoped personal knowledge base.
    pub async fn delete_knowledge(
        self,
        knowledge: &GlmKnowledgeRef,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeMutationResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .delete_knowledge(knowledge, options)
            .await
    }

    /// List documents in one account-scoped knowledge base.
    pub async fn list_documents(
        self,
        knowledge: &GlmKnowledgeRef,
        request: &GlmKnowledgeDocumentListRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeDocumentListResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .list_documents(knowledge, request, options)
            .await
    }

    /// Add externally hosted URLs to a scoped Zhipu knowledge base.
    ///
    /// This only submits the documented ingestion request. The provider may
    /// index asynchronously; callers can inspect provider-native response
    /// fields and configure a callback URL when the platform supports it.
    pub async fn upload_url_documents(
        self,
        knowledge: &GlmKnowledgeRef,
        request: &GlmUploadUrlDocumentsRequest,
        options: &RequestOptions,
    ) -> Result<GlmUploadUrlDocumentsResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .upload_url_documents(knowledge, request, options)
            .await
    }

    /// Upload one or more caller-owned file streams to a scoped Zhipu
    /// knowledge base. Each stream is consumed once; an interrupted request
    /// is reported as an unknown outcome and is never retried.
    pub async fn upload_file_documents(
        self,
        knowledge: &GlmKnowledgeRef,
        files: Vec<UploadFileStream>,
        request: &GlmUploadFileDocumentsRequest,
        options: &RequestOptions,
    ) -> Result<GlmUploadFileDocumentsResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .upload_file_documents(knowledge, files, request, options)
            .await
    }

    /// Retrieve ranked text chunks from one or more Zhipu personal knowledge
    /// bases. This is the native GLM retrieval API, separate from
    /// [`crate::retrieval::RetrievalService`].
    pub async fn retrieve(
        self,
        knowledge_bases: &[GlmKnowledgeRef],
        request: &GlmKnowledgeRetrieveRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeRetrieveResult, GlmKnowledgeError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .retrieve(knowledge_bases, request, options)
            .await
    }
}

struct Pinned<'a> {
    client: &'a ClientSnapshot,
}

impl Pinned<'_> {
    fn route<'s, 'o>(
        &'s self,
        profile_name: &str,
        options: &'o RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s GlmKnowledgeRoute, &'o str), GlmKnowledgeError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown GLM knowledge profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("GLM knowledge profile is unavailable in this region").into());
        }
        if !matches!(profile.provider_id.as_str(), "zhipu" | "glm") {
            return Err(LlmError::UnsupportedCapability {
                message: "native GLM Knowledge Base API requires a Zhipu/GLM profile".into(),
            }
            .into());
        }
        let ServiceSetting::Enabled(route) = &profile.glm_knowledge else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled GLM Knowledge Base route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| {
                invalid("GLM knowledge operations require a non-secret account_scope")
            })?;
        Ok((profile, route, scope))
    }

    fn checked_knowledge<'s>(
        &'s self,
        reference: &GlmKnowledgeRef,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s GlmKnowledgeRoute), GlmKnowledgeError> {
        let (profile, route, account_scope) = self.route(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.account_scope != account_scope
            || !valid_id(&reference.knowledge_id)
        {
            return Err(LlmError::PermissionDenied {
                message:
                    "GLM knowledge base belongs to another provider, profile, endpoint or account"
                        .into(),
            }
            .into());
        }
        Ok((profile, route))
    }

    async fn list_knowledge(
        &self,
        profile_name: &str,
        request: &GlmKnowledgeListRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeListResult, GlmKnowledgeError> {
        validate_page(request.page, request.size)?;
        let (profile, route, scope) = self.route(profile_name, options)?;
        let query = page_query(request.page, request.size);
        let url = route_url_with_query(route, &["knowledge"], &query)?;
        let response = self
            .send_json_at(route, options, "GET", url, None, "list_knowledge")
            .await?;
        let native = success_json(response, "list_knowledge")?;
        let data = native
            .get("data")
            .ok_or_else(|| bad("knowledge list response omitted data", native.clone()))?;
        let rows = data
            .get("list")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("knowledge list response omitted data.list", native.clone()))?;
        let total = data
            .get("total")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad("knowledge list response omitted data.total", native.clone()))?;
        let knowledge_bases = rows
            .iter()
            .cloned()
            .map(|row| {
                let id = row
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| valid_id(id))
                    .ok_or_else(|| bad("knowledge list row omitted a valid id", row.clone()))?;
                Ok(GlmKnowledge {
                    reference: knowledge_reference(profile, route, scope, id),
                    native: row,
                })
            })
            .collect::<Result<Vec<_>, GlmKnowledgeError>>()?;
        Ok(GlmKnowledgeListResult {
            knowledge_bases,
            total,
            native,
        })
    }

    async fn create_knowledge(
        &self,
        profile_name: &str,
        request: &GlmCreateKnowledgeRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledge, GlmKnowledgeError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        validate_create_request(request)?;
        let body = serde_json::to_value(request)
            .map_err(|_| invalid("GLM knowledge create request cannot be serialized"))?;
        let response = self
            .send_json(
                route,
                options,
                "POST",
                &["knowledge"],
                Some(body),
                "create_knowledge",
            )
            .await?;
        let native = success_json(response, "create_knowledge")?;
        let data = native
            .get("data")
            .ok_or_else(|| bad("create response omitted data", native.clone()))?;
        let id = data["id"]
            .as_str()
            .filter(|id| valid_id(id))
            .ok_or_else(|| {
                bad(
                    "create response omitted a valid knowledge id",
                    native.clone(),
                )
            })?;
        Ok(GlmKnowledge {
            reference: knowledge_reference(profile, route, scope, id),
            native,
        })
    }

    async fn get_knowledge(
        &self,
        reference: &GlmKnowledgeRef,
        options: &RequestOptions,
    ) -> Result<GlmKnowledge, GlmKnowledgeError> {
        let (profile, route) = self.checked_knowledge(reference, options)?;
        let response = self
            .send_json(
                route,
                options,
                "GET",
                &["knowledge", &reference.knowledge_id],
                None,
                "get_knowledge",
            )
            .await?;
        let native = success_json(response, "get_knowledge")?;
        let data = native
            .get("data")
            .ok_or_else(|| bad("knowledge detail response omitted data", native.clone()))?;
        let id = data
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| valid_id(id))
            .ok_or_else(|| {
                bad(
                    "knowledge detail response omitted a valid id",
                    native.clone(),
                )
            })?;
        if id != reference.knowledge_id {
            return Err(bad(
                "knowledge detail response returned a different id",
                native,
            ));
        }
        Ok(GlmKnowledge {
            reference: knowledge_reference(profile, route, &reference.account_scope, id),
            native,
        })
    }

    async fn update_knowledge(
        &self,
        reference: &GlmKnowledgeRef,
        request: &GlmUpdateKnowledgeRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeMutationResult, GlmKnowledgeError> {
        let (_profile, route) = self.checked_knowledge(reference, options)?;
        validate_update_request(request)?;
        let body = serde_json::to_value(request)
            .map_err(|_| invalid("GLM knowledge update cannot be serialized"))?;
        let response = self
            .send_json(
                route,
                options,
                "PUT",
                &["knowledge", &reference.knowledge_id],
                Some(body),
                "update_knowledge",
            )
            .await?;
        Ok(GlmKnowledgeMutationResult {
            native: success_json(response, "update_knowledge")?,
        })
    }

    async fn delete_knowledge(
        &self,
        reference: &GlmKnowledgeRef,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeMutationResult, GlmKnowledgeError> {
        let (_profile, route) = self.checked_knowledge(reference, options)?;
        let response = self
            .send_json(
                route,
                options,
                "DELETE",
                &["knowledge", &reference.knowledge_id],
                None,
                "delete_knowledge",
            )
            .await?;
        Ok(GlmKnowledgeMutationResult {
            native: success_json(response, "delete_knowledge")?,
        })
    }

    async fn list_documents(
        &self,
        reference: &GlmKnowledgeRef,
        request: &GlmKnowledgeDocumentListRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeDocumentListResult, GlmKnowledgeError> {
        let (_profile, route) = self.checked_knowledge(reference, options)?;
        validate_page(request.page, request.size)?;
        let mut query = vec![("knowledge_id".to_owned(), reference.knowledge_id.clone())];
        query.extend(page_query(request.page, request.size));
        if let Some(word) = request.word.as_deref().filter(|word| !word.is_empty()) {
            query.push(("word".to_owned(), word.to_owned()));
        }
        let url = route_url_with_query(route, &["document"], &query)?;
        let response = self
            .send_json_at(route, options, "GET", url, None, "list_documents")
            .await?;
        let native = success_json(response, "list_documents")?;
        let data = native
            .get("data")
            .ok_or_else(|| bad("document list response omitted data", native.clone()))?;
        let rows = data
            .get("list")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("document list response omitted data.list", native.clone()))?;
        let total = data
            .get("total")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad("document list response omitted data.total", native.clone()))?;
        let documents = rows
            .iter()
            .cloned()
            .map(|row| decode_document(reference, row))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GlmKnowledgeDocumentListResult {
            documents,
            total,
            native,
        })
    }

    async fn upload_url_documents(
        &self,
        reference: &GlmKnowledgeRef,
        request: &GlmUploadUrlDocumentsRequest,
        options: &RequestOptions,
    ) -> Result<GlmUploadUrlDocumentsResult, GlmKnowledgeError> {
        let (_profile, route) = self.checked_knowledge(reference, options)?;
        validate_upload_request(request)?;
        let upload_detail = request
            .documents
            .iter()
            .map(|item| {
                let mut detail = serde_json::Map::new();
                detail.insert("url".into(), Value::String(item.url.clone()));
                if let Some(kind) = item.knowledge_type {
                    detail.insert("knowledge_type".into(), json!(kind));
                }
                if let Some(separators) = &item.custom_separator {
                    detail.insert("custom_separator".into(), json!(separators));
                }
                if let Some(size) = item.sentence_size {
                    detail.insert("sentence_size".into(), json!(size));
                }
                if let Some(callback_url) = &item.callback_url {
                    detail.insert("callback_url".into(), json!(callback_url));
                }
                if let Some(callback_header) = &item.callback_header {
                    detail.insert("callback_header".into(), callback_header.clone());
                }
                Value::Object(detail)
            })
            .collect::<Vec<_>>();
        let body = json!({
            "knowledge_id": reference.knowledge_id,
            "upload_detail": upload_detail,
        });
        let response = self
            .send_json(
                route,
                options,
                "POST",
                &["document", "upload_url"],
                Some(body),
                "upload_url_documents",
            )
            .await?;
        let native = success_json(response, "upload_url_documents")?;
        let data = native
            .get("data")
            .ok_or_else(|| bad("URL upload response omitted data", native.clone()))?;
        let succeeded = data
            .get("successInfos")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|row| {
                let id = row["documentId"]
                    .as_str()
                    .filter(|id| valid_id(id))
                    .ok_or_else(|| {
                        bad("URL upload success omitted a document id", native.clone())
                    })?;
                Ok(GlmUploadedUrlDocument {
                    reference: GlmKnowledgeDocumentRef {
                        knowledge: reference.clone(),
                        document_id: id.into(),
                    },
                    url: row["url"].as_str().map(str::to_owned),
                })
            })
            .collect::<Result<Vec<_>, GlmKnowledgeError>>()?;
        let failed = data
            .get("failedInfos")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|row| GlmFailedUrlDocument {
                url: row["url"].as_str().map(str::to_owned),
                fail_reason: row["failReason"].as_str().map(str::to_owned),
                native: row,
            })
            .collect();
        Ok(GlmUploadUrlDocumentsResult {
            succeeded,
            failed,
            native,
        })
    }

    async fn upload_file_documents(
        &self,
        reference: &GlmKnowledgeRef,
        files: Vec<UploadFileStream>,
        request: &GlmUploadFileDocumentsRequest,
        options: &RequestOptions,
    ) -> Result<GlmUploadFileDocumentsResult, GlmKnowledgeError> {
        let (_profile, route) = self.checked_knowledge(reference, options)?;
        validate_file_upload_request(&files, request)?;
        if options
            .total_timeout
            .is_some_and(|timeout| timeout.is_zero())
        {
            return Err(invalid("GLM file upload timeout must be positive").into());
        }
        let credential = options.credential.as_ref().ok_or_else(|| {
            invalid("Zhipu Knowledge Base API requires a caller-supplied API key")
        })?;
        if route.auth != ServiceAuth::Bearer {
            return Err(invalid("Zhipu Knowledge Base API requires Bearer authentication").into());
        }
        let credential_value = credential.expose_secret();
        if credential_value.trim().is_empty() || credential_value.chars().any(char::is_control) {
            return Err(
                invalid("Zhipu API key must not be empty or contain control characters").into(),
            );
        }
        let upload_url = route_url(
            route,
            &["document", "upload_document", &reference.knowledge_id],
        )?
        .to_string();
        let submitted_file_names = files
            .iter()
            .map(|file| sanitize_filename(file.filename()))
            .collect::<Vec<_>>();

        let boundary = multipart_boundary();
        let (prefix, file_headers, suffix, content_length) =
            file_upload_multipart_parts(&boundary, &files, request)?;
        let mut streams = Vec::with_capacity(files.len());
        for (file, header) in files.into_iter().zip(file_headers) {
            let (_, _, size_bytes, body) = file.into_parts();
            streams.push((header, body, size_bytes));
        }
        let body = file_upload_multipart_stream(prefix, streams, suffix);
        let deadline = Deadline::after(options.total_timeout);
        let http_request = HttpStreamRequest {
            method: "POST".into(),
            url: upload_url,
            headers: vec![
                ("authorization".into(), format!("Bearer {credential_value}")),
                ("accept".into(), "application/json".into()),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body,
            content_length,
            timeout: options.total_timeout.or(Some(DEFAULT_TIMEOUT)),
        };
        let executor = HttpExecutor::new(self.client.runtime.http.as_ref()).with_deadline(deadline);
        let response = executor.send_stream(http_request).await.map_err(|source| {
            if matches!(&source, LlmError::UnsupportedCapability { .. }) {
                GlmKnowledgeError::Llm(source)
            } else {
                GlmKnowledgeError::OutcomeUnknown {
                    operation: "upload_file_documents",
                    source,
                }
            }
        })?;
        let response = HttpExecutor::collect_response(response, Some(MAX_RESPONSE_BYTES))
            .await
            .map_err(|source| GlmKnowledgeError::OutcomeUnknown {
                operation: "upload_file_documents",
                source,
            })?;
        if response.status == 408 || response.status >= 500 {
            let native = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            });
            return Err(GlmKnowledgeError::ResponseOutcomeUnknown {
                operation: "upload_file_documents",
                message: format!(
                    "provider returned HTTP {} after upload dispatch",
                    response.status
                ),
                native,
            });
        }
        let native = match success_json(response, "upload_file_documents") {
            Ok(native) => native,
            Err(GlmKnowledgeError::InvalidResponse { message, native }) => {
                return Err(GlmKnowledgeError::ResponseOutcomeUnknown {
                    operation: "upload_file_documents",
                    message,
                    native,
                });
            }
            Err(error) => return Err(error),
        };
        decode_file_upload_result(reference, &submitted_file_names, native)
    }

    async fn retrieve(
        &self,
        knowledge_bases: &[GlmKnowledgeRef],
        request: &GlmKnowledgeRetrieveRequest,
        options: &RequestOptions,
    ) -> Result<GlmKnowledgeRetrieveResult, GlmKnowledgeError> {
        if knowledge_bases.is_empty() || knowledge_bases.len() > 20 {
            return Err(invalid("retrieve requires 1 to 20 GLM knowledge references").into());
        }
        validate_retrieve_request(request)?;
        let mut resolved = Vec::with_capacity(knowledge_bases.len());
        for reference in knowledge_bases {
            resolved.push((reference, self.checked_knowledge(reference, options)?));
        }
        let (profile, route) = resolved[0].1;
        if resolved.iter().any(|(_, (other_profile, other_route))| {
            other_profile.profile_name != profile.profile_name
                || other_route.endpoint != route.endpoint
        }) {
            return Err(
                invalid("one GLM retrieval request cannot mix profiles or endpoints").into(),
            );
        }
        for document in &request.documents {
            if !knowledge_bases
                .iter()
                .any(|knowledge| knowledge == &document.knowledge)
                || !valid_id(&document.document_id)
            {
                return Err(LlmError::PermissionDenied {
                    message: "GLM retrieval document is not scoped to a requested knowledge base"
                        .into(),
                }
                .into());
            }
        }
        let mut body = serde_json::to_value(request)
            .map_err(|_| invalid("GLM knowledge retrieval request cannot be serialized"))?;
        let object = body
            .as_object_mut()
            .ok_or_else(|| invalid("GLM knowledge retrieval request must be an object"))?;
        object.remove("documents");
        object.insert(
            "knowledge_ids".into(),
            json!(knowledge_bases
                .iter()
                .map(|reference| &reference.knowledge_id)
                .collect::<Vec<_>>()),
        );
        if !request.documents.is_empty() {
            object.insert(
                "document_ids".into(),
                json!(request
                    .documents
                    .iter()
                    .map(|document| &document.document_id)
                    .collect::<Vec<_>>()),
            );
        }
        if let Some(rerank) = request.rerank {
            object.insert("rerank_status".into(), json!(u8::from(rerank)));
            object.remove("rerank");
        }
        let response = self
            .send_json(
                route,
                options,
                "POST",
                &["knowledge", "retrieve"],
                Some(body),
                "retrieve",
            )
            .await?;
        let native = success_json(response, "retrieve")?;
        let rows = native
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("retrieve response omitted data array", native.clone()))?;
        let matches = rows
            .iter()
            .cloned()
            .map(decode_match)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GlmKnowledgeRetrieveResult { matches, native })
    }

    async fn send_json(
        &self,
        route: &GlmKnowledgeRoute,
        options: &RequestOptions,
        method: &str,
        path: &[&str],
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<HttpResponse, GlmKnowledgeError> {
        let url = route_url(route, path)?;
        self.send_json_at(route, options, method, url, body, operation)
            .await
    }

    async fn send_json_at(
        &self,
        route: &GlmKnowledgeRoute,
        options: &RequestOptions,
        method: &str,
        url: url::Url,
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<HttpResponse, GlmKnowledgeError> {
        let credential = options.credential.as_ref().ok_or_else(|| {
            invalid("Zhipu Knowledge Base API requires a caller-supplied API key")
        })?;
        if route.auth != ServiceAuth::Bearer {
            return Err(invalid("Zhipu Knowledge Base API requires Bearer authentication").into());
        }
        let body_bytes = match body {
            Some(value) => serde_json::to_vec(&value)
                .map_err(|_| invalid("GLM knowledge request cannot be serialized"))?,
            None => Vec::new(),
        };
        if body_bytes.len() > MAX_REQUEST_BYTES {
            return Err(invalid("GLM knowledge request exceeds the 1 MiB client limit").into());
        }
        let request = HttpRequest {
            method: method.to_owned(),
            url: url.to_string(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                ),
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(body_bytes),
            timeout: options.total_timeout.or(Some(DEFAULT_TIMEOUT)),
        };
        let executor = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(Deadline::after(options.total_timeout));
        executor
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                if matches!(method, "POST" | "PUT" | "DELETE") {
                    GlmKnowledgeError::OutcomeUnknown { operation, source }
                } else {
                    GlmKnowledgeError::Llm(source)
                }
            })
    }
}

fn decode_match(native: Value) -> Result<GlmKnowledgeMatch, GlmKnowledgeError> {
    let text = native
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("retrieve result omitted text", native.clone()))?
        .to_owned();
    let meta = native.get("metadata").cloned().unwrap_or(Value::Null);
    let metadata = GlmKnowledgeMatchMetadata {
        knowledge_id: meta["knowledge_id"].as_str().map(str::to_owned),
        document_id: meta["doc_id"].as_str().map(str::to_owned),
        document_name: meta["doc_name"].as_str().map(str::to_owned),
        document_url: meta["doc_url"].as_str().map(str::to_owned),
        contextual_text: meta["contextual_text"].as_str().map(str::to_owned),
        native: meta,
    };
    Ok(GlmKnowledgeMatch {
        text,
        score: native.get("score").and_then(Value::as_f64),
        metadata,
        native,
    })
}

fn decode_document(
    knowledge: &GlmKnowledgeRef,
    native: Value,
) -> Result<GlmKnowledgeDocument, GlmKnowledgeError> {
    let id = native
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("document list row omitted a valid id", native.clone()))?;
    let custom_separator = native
        .get("custom_separator")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        });
    Ok(GlmKnowledgeDocument {
        reference: GlmKnowledgeDocumentRef {
            knowledge: knowledge.clone(),
            document_id: id.to_owned(),
        },
        knowledge_type: native
            .get("knowledge_type")
            .and_then(Value::as_u64)
            .and_then(|value| u8::try_from(value).ok()),
        custom_separator,
        sentence_size: native
            .get("sentence_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        length: native.get("length").and_then(Value::as_u64),
        word_num: native.get("word_num").and_then(Value::as_u64),
        name: native
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        url: native.get("url").and_then(Value::as_str).map(str::to_owned),
        embedding_stat: native.get("embedding_stat").and_then(Value::as_i64),
        fail_info: native
            .get("failInfo")
            .or_else(|| native.get("fail_info"))
            .cloned(),
        native,
    })
}

fn knowledge_reference(
    profile: &ProviderProfile,
    route: &GlmKnowledgeRoute,
    account_scope: &str,
    knowledge_id: &str,
) -> GlmKnowledgeRef {
    GlmKnowledgeRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
        account_scope: account_scope.to_owned(),
        knowledge_id: knowledge_id.to_owned(),
    }
}

fn route_url(route: &GlmKnowledgeRoute, path: &[&str]) -> Result<url::Url, GlmKnowledgeError> {
    let mut url =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid GLM knowledge route"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| invalid("invalid GLM knowledge route"))?;
        for segment in path {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn route_url_with_query(
    route: &GlmKnowledgeRoute,
    path: &[&str],
    query: &[(String, String)],
) -> Result<url::Url, GlmKnowledgeError> {
    let mut url = route_url(route, path)?;
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (name, value) in query {
            pairs.append_pair(name, value);
        }
    }
    Ok(url)
}

fn page_query(page: Option<u32>, size: Option<u32>) -> Vec<(String, String)> {
    let mut query = Vec::with_capacity(2);
    if let Some(page) = page {
        query.push(("page".into(), page.to_string()));
    }
    if let Some(size) = size {
        query.push(("size".into(), size.to_string()));
    }
    query
}

fn validate_page(page: Option<u32>, size: Option<u32>) -> Result<(), GlmKnowledgeError> {
    if page == Some(0) || size == Some(0) {
        return Err(invalid("GLM knowledge page and size must be positive").into());
    }
    Ok(())
}

fn validate_update_request(request: &GlmUpdateKnowledgeRequest) -> Result<(), GlmKnowledgeError> {
    if request.embedding_id.is_none()
        && request.embedding_model.is_none()
        && request.contextual.is_none()
        && request.name.is_none()
        && request.description.is_none()
        && request.background.is_none()
        && request.icon.is_none()
        && request.callback_url.is_none()
        && request.callback_header.is_none()
    {
        return Err(invalid("GLM knowledge update must include at least one field").into());
    }
    if request
        .name
        .as_ref()
        .is_some_and(|name| name.trim().is_empty() || name.len() > 256)
    {
        return Err(invalid("GLM knowledge base name must contain 1 to 256 bytes").into());
    }
    if request
        .description
        .as_ref()
        .is_some_and(|value| value.len() > 4096)
    {
        return Err(invalid("GLM knowledge description exceeds 4096 bytes").into());
    }
    if request
        .callback_header
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(invalid("GLM knowledge callback_header must be an object").into());
    }
    Ok(())
}

/// Validate a Zhipu route before it is published with a provider profile.
pub fn validate_route(
    profile: &ProviderProfile,
    route: &GlmKnowledgeRoute,
) -> Result<(), LlmError> {
    if !matches!(profile.provider_id.as_str(), "zhipu" | "glm") || route.auth != ServiceAuth::Bearer
    {
        return Err(invalid(
            "GLM Knowledge Base requires a Zhipu/GLM provider and Bearer authentication",
        ));
    }
    let url =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid GLM knowledge endpoint"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url
            .path()
            .trim_end_matches('/')
            .ends_with("/api/llm-application/open")
    {
        return Err(invalid(
            "GLM Knowledge Base endpoint must end in /api/llm-application/open without credentials, query or fragment",
        ));
    }
    if url.scheme() == "http" && !matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"))
    {
        return Err(invalid(
            "plain HTTP GLM knowledge endpoints are only allowed on loopback",
        ));
    }
    Ok(())
}

fn validate_create_request(request: &GlmCreateKnowledgeRequest) -> Result<(), GlmKnowledgeError> {
    if request.name.trim().is_empty() || request.name.len() > 256 {
        return Err(invalid("GLM knowledge base name must contain 1 to 256 bytes").into());
    }
    if request
        .description
        .as_ref()
        .is_some_and(|value| value.len() > 4096)
    {
        return Err(invalid("GLM knowledge description exceeds 4096 bytes").into());
    }
    Ok(())
}

fn validate_upload_request(
    request: &GlmUploadUrlDocumentsRequest,
) -> Result<(), GlmKnowledgeError> {
    if request.documents.is_empty() || request.documents.len() > 100 {
        return Err(invalid("GLM URL upload requires 1 to 100 documents").into());
    }
    for item in &request.documents {
        let url =
            url::Url::parse(&item.url).map_err(|_| invalid("GLM URL document URL is invalid"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(invalid("GLM URL document must use an absolute HTTP or HTTPS URL").into());
        }
        if item
            .knowledge_type
            .is_some_and(|kind| !matches!(kind, 1 | 2 | 3 | 5 | 6 | 7))
        {
            return Err(invalid("GLM URL document knowledge_type is unsupported").into());
        }
        if item
            .sentence_size
            .is_some_and(|size| !(20..=2000).contains(&size))
        {
            return Err(invalid("GLM URL document sentence_size must be from 20 to 2000").into());
        }
    }
    Ok(())
}

fn validate_file_upload_request(
    files: &[UploadFileStream],
    request: &GlmUploadFileDocumentsRequest,
) -> Result<(), GlmKnowledgeError> {
    if files.is_empty() || files.len() > 100 {
        return Err(invalid("GLM file upload requires 1 to 100 files").into());
    }
    if request
        .knowledge_type
        .is_some_and(|kind| !matches!(kind, 1 | 2 | 3 | 5 | 6 | 7))
    {
        return Err(invalid("GLM file upload knowledge_type is unsupported").into());
    }
    if request.custom_separator.is_some() && request.knowledge_type != Some(5) {
        return Err(invalid("GLM custom_separator requires knowledge_type 5").into());
    }
    if request
        .sentence_size
        .is_some_and(|size| !(20..=2000).contains(&size))
        || (request.sentence_size.is_some() && request.knowledge_type != Some(5))
    {
        return Err(invalid(
            "GLM sentence_size requires knowledge_type 5 and must be from 20 to 2000",
        )
        .into());
    }
    if request.custom_separator.as_ref().is_some_and(|separators| {
        separators
            .iter()
            .any(|value| value.chars().any(char::is_control))
    }) || request
        .callback_url
        .as_ref()
        .is_some_and(|value| value.chars().any(char::is_control))
        || request
            .request_id
            .as_ref()
            .is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control))
    {
        return Err(
            invalid("GLM file upload form values must not contain control characters").into(),
        );
    }
    if request
        .callback_header
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(invalid("GLM callback_header must be an object").into());
    }
    if request
        .word_num_limit
        .as_ref()
        .is_some_and(|value| value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(invalid("GLM word_num_limit must contain decimal digits").into());
    }
    if request
        .request_id
        .as_ref()
        .is_some_and(|value| value.len() > 256)
    {
        return Err(invalid("GLM req_id exceeds 256 bytes").into());
    }
    for file in files {
        if file.size_bytes() == 0 {
            return Err(invalid("GLM file upload does not accept empty files").into());
        }
        if file.filename().trim().is_empty()
            || file.filename().len() > 4096
            || file.filename().chars().any(char::is_control)
        {
            return Err(invalid("GLM file upload filename is invalid").into());
        }
        validate_media_type(file.media_type())?;
    }
    Ok(())
}

fn file_upload_multipart_parts(
    boundary: &str,
    files: &[UploadFileStream],
    request: &GlmUploadFileDocumentsRequest,
) -> Result<(Bytes, Vec<Bytes>, Bytes, u64), GlmKnowledgeError> {
    let mut prefix = BytesMut::new();
    if let Some(value) = request.knowledge_type {
        append_field(&mut prefix, boundary, "knowledge_type", &value.to_string());
    }
    if let Some(values) = &request.custom_separator {
        for value in values {
            append_field(&mut prefix, boundary, "custom_separator", value);
        }
    }
    if let Some(value) = request.sentence_size {
        append_field(&mut prefix, boundary, "sentence_size", &value.to_string());
    }
    if let Some(value) = request.parse_image {
        append_field(
            &mut prefix,
            boundary,
            "parse_image",
            if value { "true" } else { "false" },
        );
    }
    if let Some(value) = &request.callback_url {
        append_field(&mut prefix, boundary, "callback_url", value);
    }
    if let Some(value) = &request.callback_header {
        let encoded = serde_json::to_string(value)
            .map_err(|_| invalid("GLM callback_header cannot be serialized"))?;
        append_field(&mut prefix, boundary, "callback_header", &encoded);
    }
    if let Some(value) = &request.word_num_limit {
        append_field(&mut prefix, boundary, "word_num_limit", value);
    }
    if let Some(value) = &request.request_id {
        append_field(&mut prefix, boundary, "req_id", value);
    }

    let mut total_file_bytes = 0u64;
    let mut fixed_bytes =
        u64::try_from(prefix.len()).map_err(|_| invalid("GLM multipart request size overflows"))?;
    let mut headers = Vec::with_capacity(files.len());
    for file in files {
        let filename = sanitize_filename(file.filename());
        let header = Bytes::from(format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            file.media_type()
        ));
        fixed_bytes = fixed_bytes
            .checked_add(
                u64::try_from(header.len())
                    .map_err(|_| invalid("GLM multipart request size overflows"))?,
            )
            .and_then(|length| length.checked_add(2))
            .ok_or_else(|| invalid("GLM multipart request size overflows"))?;
        total_file_bytes = total_file_bytes
            .checked_add(file.size_bytes())
            .ok_or_else(|| invalid("GLM multipart request size overflows"))?;
        headers.push(header);
    }
    let suffix = Bytes::from(format!("--{boundary}--\r\n"));
    fixed_bytes = fixed_bytes
        .checked_add(
            u64::try_from(suffix.len())
                .map_err(|_| invalid("GLM multipart request size overflows"))?,
        )
        .ok_or_else(|| invalid("GLM multipart request size overflows"))?;
    if fixed_bytes > MAX_REQUEST_BYTES as u64 {
        return Err(invalid("GLM multipart metadata exceeds the 1 MiB client limit").into());
    }
    let content_length = fixed_bytes
        .checked_add(total_file_bytes)
        .ok_or_else(|| invalid("GLM multipart request size overflows"))?;
    Ok((prefix.freeze(), headers, suffix, content_length))
}

type FileUploadPart = (Bytes, BoxStream<'static, Result<Bytes, LlmError>>, u64);

fn file_upload_multipart_stream(
    prefix: Bytes,
    files: Vec<FileUploadPart>,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    let mut parts: Vec<BoxStream<'static, Result<Bytes, LlmError>>> =
        Vec::with_capacity(1 + files.len() * 3 + 1);
    if !prefix.is_empty() {
        parts.push(stream::once(async move { Ok(prefix) }).boxed());
    }
    for (header, body, size_bytes) in files {
        parts.push(stream::once(async move { Ok(header) }).boxed());
        parts.push(exact_upload_stream(body, size_bytes));
        parts.push(stream::once(async { Ok(Bytes::from_static(b"\r\n")) }).boxed());
    }
    parts.push(stream::once(async move { Ok(suffix) }).boxed());
    stream::iter(parts).flatten().boxed()
}

fn decode_file_upload_result(
    knowledge: &GlmKnowledgeRef,
    submitted_file_names: &[String],
    native: Value,
) -> Result<GlmUploadFileDocumentsResult, GlmKnowledgeError> {
    let Some(data) = native.get("data") else {
        return Err(GlmKnowledgeError::ResponseOutcomeUnknown {
            operation: "upload_file_documents",
            message: "success response omitted data".into(),
            native,
        });
    };
    if data.get("successInfos").is_none() && data.get("failedInfos").is_none() {
        return Err(GlmKnowledgeError::ResponseOutcomeUnknown {
            operation: "upload_file_documents",
            message: "success response omitted successInfos and failedInfos".into(),
            native,
        });
    }
    let success_rows =
        match data.get("successInfos") {
            Some(value) => value.as_array().cloned().ok_or_else(|| {
                GlmKnowledgeError::ResponseOutcomeUnknown {
                    operation: "upload_file_documents",
                    message: "successInfos was not an array".into(),
                    native: native.clone(),
                }
            })?,
            None => Vec::new(),
        };
    let failed_rows =
        match data.get("failedInfos") {
            Some(value) => value.as_array().cloned().ok_or_else(|| {
                GlmKnowledgeError::ResponseOutcomeUnknown {
                    operation: "upload_file_documents",
                    message: "failedInfos was not an array".into(),
                    native: native.clone(),
                }
            })?,
            None => Vec::new(),
        };

    let mut succeeded = Vec::new();
    let mut failed = Vec::new();
    let mut unresolved = Vec::new();
    let mut remaining_names = BTreeMap::<String, usize>::new();
    for name in submitted_file_names {
        *remaining_names.entry(name.clone()).or_default() += 1;
    }
    let mut document_ids = std::collections::BTreeSet::new();
    for row in success_rows {
        let file_name = row.get("fileName").and_then(Value::as_str);
        let matched_name = file_name.filter(|name| consume_file_name(&mut remaining_names, name));
        let id = row
            .get("documentId")
            .and_then(Value::as_str)
            .filter(|id| valid_id(id));
        if let (Some(id), Some(file_name)) = (id, matched_name) {
            if !document_ids.insert(id.to_owned()) {
                unresolved.push(row);
                continue;
            }
            succeeded.push(GlmUploadedFileDocument {
                reference: GlmKnowledgeDocumentRef {
                    knowledge: knowledge.clone(),
                    document_id: id.to_owned(),
                },
                file_name: Some(file_name.to_owned()),
                native: row,
            });
        } else {
            unresolved.push(row);
        }
    }
    for row in failed_rows {
        let file_name = row.get("fileName").and_then(Value::as_str);
        if let Some(file_name) =
            file_name.filter(|name| consume_file_name(&mut remaining_names, name))
        {
            failed.push(GlmFailedFileDocument {
                file_name: Some(file_name.to_owned()),
                fail_reason: row
                    .get("failReason")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: row,
            });
        } else {
            unresolved.push(row);
        }
    }
    let missing_files = remaining_names
        .into_iter()
        .flat_map(|(name, count)| std::iter::repeat_n(name, count))
        .collect();
    Ok(GlmUploadFileDocumentsResult {
        succeeded,
        failed,
        unresolved,
        missing_files,
        native,
    })
}

fn consume_file_name(remaining: &mut BTreeMap<String, usize>, file_name: &str) -> bool {
    let Some(count) = remaining.get_mut(file_name) else {
        return false;
    };
    *count -= 1;
    if *count == 0 {
        remaining.remove(file_name);
    }
    true
}

fn validate_retrieve_request(
    request: &GlmKnowledgeRetrieveRequest,
) -> Result<(), GlmKnowledgeError> {
    if request.query.trim().is_empty() || request.query.chars().count() > MAX_QUERY_CHARS {
        return Err(invalid("GLM knowledge query must contain 1 to 1000 characters").into());
    }
    if request
        .top_k
        .is_some_and(|value| !(1..=20).contains(&value))
        || request
            .top_n
            .is_some_and(|value| !(1..=100).contains(&value))
        || request
            .recall_ratio
            .is_some_and(|value| !(1..100).contains(&value))
        || request
            .fractional_threshold
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value >= 1.0)
        || request.request_id.as_ref().is_some_and(|id| {
            id.trim().is_empty() || id.len() > 128 || id.chars().any(char::is_control)
        })
    {
        return Err(
            invalid("GLM knowledge retrieval controls are outside documented ranges").into(),
        );
    }
    Ok(())
}

fn success_json(
    response: HttpResponse,
    operation: &'static str,
) -> Result<Value, GlmKnowledgeError> {
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let native = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    let code = native.get("code").and_then(Value::as_i64);
    if !(200..300).contains(&response.status) || code.is_some_and(|value| value != 200) {
        return Err(GlmKnowledgeError::Provider {
            status: response.status,
            code,
            request_id,
            body: native,
        });
    }
    if !native.is_object() {
        return Err(bad("success response is not a JSON object", native));
    }
    let _ = operation;
    Ok(native)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id != "."
        && id != ".."
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

fn bad(message: &str, native: Value) -> GlmKnowledgeError {
    GlmKnowledgeError::InvalidResponse {
        message: message.into(),
        native,
    }
}
