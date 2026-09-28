//! Explicit file upload, registration, and metadata calls for the RAG data
//! center. The three upload steps remain separate so callers can observe and
//! recover each provider mutation deliberately.

use super::*;
use crate::{
    files::{exact_upload_stream, validate_media_type, UploadFileStream},
    transport::HttpStreamRequest,
};
use serde_json::{json, Map, Value};
use std::{fmt, time::Duration};
use url::Url;

const REQUEST_UPLOAD_LEASE_PATH: &str = "/api/v1/connector/dash/applyFileUploadLease";
const REGISTER_FILE_PATH: &str = "/api/v1/connector/dash/addFile";
const DESCRIBE_FILE_PATH: &str = "/api/v1/connector/dash/describeFile";
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// The data-center category namespace for a file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QwenKnowledgeCategoryType {
    /// A durable file category used by knowledge bases.
    #[default]
    Unstructured,
    /// A session-scoped file category. Model Studio documents that these
    /// files are temporary and cannot be stored in a long-term knowledge base.
    SessionFile,
}

impl QwenKnowledgeCategoryType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unstructured => "UNSTRUCTURED",
            Self::SessionFile => "SESSION_FILE",
        }
    }
}

/// Lease request metadata. `content_md5` is supplied by the caller; the
/// service does not buffer the body to calculate or reinterpret a digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeFileUploadRequest {
    file_name: String,
    size_bytes: u64,
    content_md5: String,
    category: String,
    category_type: QwenKnowledgeCategoryType,
}

impl QwenKnowledgeFileUploadRequest {
    pub fn new(
        category: impl Into<String>,
        file_name: impl Into<String>,
        size_bytes: u64,
        content_md5: impl Into<String>,
    ) -> Self {
        Self {
            file_name: file_name.into(),
            size_bytes,
            content_md5: content_md5.into(),
            category: category.into(),
            category_type: QwenKnowledgeCategoryType::Unstructured,
        }
    }

    pub fn with_category_type(mut self, category_type: QwenKnowledgeCategoryType) -> Self {
        self.category_type = category_type;
        self
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn category(&self) -> &str {
        &self.category
    }

    pub fn category_type(&self) -> QwenKnowledgeCategoryType {
        self.category_type
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.file_name.trim().is_empty() || has_control(&self.file_name) {
            return Err(invalid(
                "file_name must be nonempty and contain no control characters",
            ));
        }
        if self.category.trim().is_empty() || has_control(&self.category) {
            return Err(invalid(
                "category must be nonempty and contain no control characters",
            ));
        }
        // The documentation's table says Base64 while its example is hex. Keep
        // this opaque and enforce only the documented character-count range.
        let md5_characters = self.content_md5.chars().count();
        if !(1..=64).contains(&md5_characters) || has_control(&self.content_md5) {
            return Err(invalid(
                "content_md5 must contain 1 to 64 non-control characters",
            ));
        }
        Ok(())
    }
}

/// One of the parser types documented by the `addFile` endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QwenKnowledgeParser {
    AutoSelect,
    Docmind,
    DocmindDigital,
    DocmindLlmVersion,
    DashQwenVlParser,
    DocmindLlmVersionMedia,
}

impl QwenKnowledgeParser {
    fn as_str(self) -> &'static str {
        match self {
            Self::AutoSelect => "AUTO_SELECT",
            Self::Docmind => "DOCMIND",
            Self::DocmindDigital => "DOCMIND_DIGITAL",
            Self::DocmindLlmVersion => "DOCMIND_LLM_VERSION",
            Self::DashQwenVlParser => "DASH_QWEN_VL_PARSER",
            Self::DocmindLlmVersionMedia => "DOCMIND_LLM_VERSION_MEDIA",
        }
    }
}

/// Configuration for the documented Qwen VL parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeParserConfig {
    model_prompt: String,
}

impl QwenKnowledgeParserConfig {
    pub fn new(model_prompt: impl Into<String>) -> Self {
        Self {
            model_prompt: model_prompt.into(),
        }
    }

    pub fn model_prompt(&self) -> &str {
        &self.model_prompt
    }

    fn to_value(&self) -> Value {
        json!({"modelName":"qwen3-vl-plus", "modelPrompt":self.model_prompt})
    }
}

/// Options for the explicit `addFile` registration call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeRegisterFileRequest {
    parser: QwenKnowledgeParser,
    parser_config: Option<QwenKnowledgeParserConfig>,
    tags: Vec<String>,
    original_file_url: Option<String>,
}

impl QwenKnowledgeRegisterFileRequest {
    pub fn new(parser: QwenKnowledgeParser) -> Self {
        Self {
            parser,
            parser_config: None,
            tags: Vec::new(),
            original_file_url: None,
        }
    }

    pub fn with_parser_config(mut self, config: QwenKnowledgeParserConfig) -> Self {
        self.parser_config = Some(config);
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_original_file_url(mut self, url: impl Into<String>) -> Self {
        self.original_file_url = Some(url.into());
        self
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.parser == QwenKnowledgeParser::DashQwenVlParser {
            let Some(config) = self.parser_config.as_ref() else {
                return Err(invalid(
                    "DASH_QWEN_VL_PARSER requires parser_config with a model prompt",
                ));
            };
            let length = config.model_prompt.chars().count();
            if !(1..=1500).contains(&length) {
                return Err(invalid(
                    "Qwen VL parser model_prompt must contain 1 to 1500 characters",
                ));
            }
        } else if self.parser_config.is_some() {
            return Err(invalid(
                "parser_config is documented only for DASH_QWEN_VL_PARSER",
            ));
        }
        if self.tags.len() > 100
            || self
                .tags
                .iter()
                .any(|tag| tag.chars().count() > 32 || has_control(tag))
            || self
                .tags
                .iter()
                .map(|tag| tag.chars().count())
                .sum::<usize>()
                > 700
        {
            return Err(invalid(
                "tags allow at most 100 values, 32 characters per tag, and 700 characters total",
            ));
        }
        if self
            .original_file_url
            .as_ref()
            .is_some_and(|url| url.trim().is_empty() || has_control(url))
        {
            return Err(invalid(
                "original_file_url must be nonempty and contain no control characters",
            ));
        }
        Ok(())
    }

    fn to_body(&self, lease: &QwenKnowledgeFileUploadLease) -> Value {
        let mut body = json!({
            "leaseId": lease.lease_id,
            "category": lease.category,
            "categoryType": lease.category_type.as_str(),
            "parser": self.parser.as_str(),
        });
        if !self.tags.is_empty() {
            body["tags"] = json!(self.tags);
        }
        if let Some(url) = &self.original_file_url {
            body["originalFileUrl"] = json!(url);
        }
        if let Some(config) = &self.parser_config {
            body["parserConfig"] = config.to_value();
        }
        body
    }
}

/// A pre-signed upload lease bound to its originating workspace and account.
/// The URL, provider headers, and lease ID are intentionally omitted from
/// `Debug` and are not serializable.
pub struct QwenKnowledgeFileUploadLease {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    file_name: String,
    size_bytes: u64,
    category: String,
    category_type: QwenKnowledgeCategoryType,
    lease_id: String,
    upload_url: Url,
    upload_headers: Vec<(String, String)>,
}

impl fmt::Debug for QwenKnowledgeFileUploadLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeFileUploadLease")
            .field("scope", &self.scope)
            .field("file_name", &self.file_name)
            .field("size_bytes", &self.size_bytes)
            .field("category", &self.category)
            .field("category_type", &self.category_type)
            .field("lease_id", &"<redacted>")
            .field("upload_url", &"<redacted>")
            .field("upload_headers", &"<redacted>")
            .finish()
    }
}

impl QwenKnowledgeFileUploadLease {
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn category(&self) -> &str {
        &self.category
    }

    pub fn category_type(&self) -> QwenKnowledgeCategoryType {
        self.category_type
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }
}

/// A file identity bound to one workspace/account/profile connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeFileRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    file_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    category_type: Option<QwenKnowledgeCategoryType>,
}

impl QwenKnowledgeFileRef {
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        file_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let file_id = file_id.into();
        if !valid_resource_id(&file_id) {
            return Err(invalid("Qwen knowledge file ID is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            file_id,
            category_type: None,
        })
    }

    pub fn from_scope_with_category_type(
        scope: &QwenKnowledgeScope,
        file_id: impl Into<String>,
        category_type: QwenKnowledgeCategoryType,
    ) -> Result<Self, QwenKnowledgeError> {
        let mut reference = Self::from_scope(scope, file_id)?;
        reference.category_type = Some(category_type);
        Ok(reference)
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    /// Category type when it was returned by registration or supplied when
    /// binding a previously registered ID.
    pub fn category_type(&self) -> Option<QwenKnowledgeCategoryType> {
        self.category_type
    }
}

impl QwenKnowledgeScope {
    /// Bind a file ID that was obtained from this exact data-center scope.
    pub fn file_ref(
        &self,
        file_id: impl Into<String>,
    ) -> Result<QwenKnowledgeFileRef, QwenKnowledgeError> {
        QwenKnowledgeFileRef::from_scope(self, file_id)
    }

    /// Bind an existing file ID when its category type is known to the caller.
    pub fn file_ref_with_category_type(
        &self,
        file_id: impl Into<String>,
        category_type: QwenKnowledgeCategoryType,
    ) -> Result<QwenKnowledgeFileRef, QwenKnowledgeError> {
        QwenKnowledgeFileRef::from_scope_with_category_type(self, file_id, category_type)
    }
}

/// Result of a successful `addFile` registration call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileRegistration {
    pub reference: QwenKnowledgeFileRef,
    pub parser: Option<String>,
    pub status: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

/// One `describeFile` response. Unknown provider fields remain available in
/// `native`; this service does not poll when the status is still parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileDetails {
    pub reference: QwenKnowledgeFileRef,
    pub file_name: Option<String>,
    pub file_type: Option<String>,
    pub status: Option<String>,
    pub size_bytes: Option<u64>,
    pub parser: Option<String>,
    pub upload_time: Option<String>,
    pub category: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// Request one Model Studio OSS upload lease. This creates only the lease;
    /// the caller separately uploads bytes with [`Self::upload_file_content`]
    /// and registers the lease with [`Self::register_file`].
    pub async fn request_file_upload_lease(
        &self,
        request: &QwenKnowledgeFileUploadRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileUploadLease, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.ensure_supported_region()?;
        request.validate()?;
        let body = json!({
            "category": request.category,
            "fileName": request.file_name,
            "sizeBytes": request.size_bytes.to_string(),
            "contentMd5": request.content_md5,
            "categoryType": request.category_type.as_str(),
        });
        let envelope = pinned_service
            .file_json_request(
                REQUEST_UPLOAD_LEASE_PATH,
                body,
                "request_file_upload_lease",
                true,
                true,
                request_options,
            )
            .await?;
        let data = envelope.native.get("data").unwrap_or(&Value::Null);
        let malformed = |message: &str| {
            response_outcome_unknown(
                "request_file_upload_lease",
                message,
                envelope.request_id.clone(),
                redact_lease_response(&envelope.native),
            )
        };
        if data.get("type").and_then(Value::as_str) != Some("OSS.PreSignedUrl") {
            return Err(malformed(
                "lease response omitted the documented OSS upload type",
            ));
        }
        let lease_id = data
            .get("leaseId")
            .and_then(Value::as_str)
            .filter(|value| valid_opaque_id(value))
            .ok_or_else(|| malformed("lease response omitted a valid data.leaseId"))?
            .to_owned();
        let param = data
            .get("param")
            .ok_or_else(|| malformed("lease response omitted data.param"))?;
        if param.get("method").and_then(Value::as_str) != Some("PUT") {
            return Err(malformed(
                "lease response did not specify the documented PUT method",
            ));
        }
        let url_text = param
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("lease response omitted data.param.url"))?;
        let upload_url = validate_upload_url(url_text).map_err(&malformed)?;
        let upload_headers = parse_upload_headers(param.get("headers")).map_err(malformed)?;
        Ok(QwenKnowledgeFileUploadLease {
            scope: pinned_service.scope.clone(),
            endpoint_fingerprint: pinned_service.scope.endpoint_fingerprint(),
            file_name: request.file_name.clone(),
            size_bytes: request.size_bytes,
            category: request.category.clone(),
            category_type: request.category_type,
            lease_id,
            upload_url,
            upload_headers,
        })
    }

    /// Upload exact-size bytes to the validated OSS URL returned by Model
    /// Studio. No Model Studio API key is sent to OSS. An interrupted PUT is
    /// reported as an unknown mutation; this method never retries.
    pub async fn upload_file_content(
        &self,
        lease: &QwenKnowledgeFileUploadLease,
        file: UploadFileStream,
        request_options: &crate::RequestOptions,
    ) -> Result<(), QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_lease(lease)?;
        let (file_name, media_type, size_bytes, body) = file.into_parts();
        validate_media_type(&media_type).map_err(QwenKnowledgeError::Llm)?;
        if file_name != lease.file_name || size_bytes != lease.size_bytes {
            return Err(invalid(
                "upload stream filename and declared size must match the requested lease",
            ));
        }
        // The lease's returned headers are authoritative. In particular, a
        // provider may sign an octet-stream Content-Type even when the caller
        // describes the file with a more specific media type.
        let request = HttpStreamRequest {
            method: "PUT".into(),
            url: lease.upload_url.as_str().to_owned(),
            headers: lease.upload_headers.clone(),
            body: exact_upload_stream(body, size_bytes),
            content_length: size_bytes,
            timeout: request_options.total_timeout.or(Some(UPLOAD_TIMEOUT)),
        };
        let response = HttpExecutor::new(pinned_service.http)
            .send_stream(request)
            .await
            .map_err(|source| QwenKnowledgeError::OutcomeUnknown {
                operation: "upload_file_content",
                source,
            })?;
        if (200..300).contains(&response.status) {
            // OSS success bodies are not needed to complete this explicit PUT.
            return Ok(());
        }
        let dispatch = if response.status == 408
            || response.status >= 500
            || (300..400).contains(&response.status)
        {
            QwenKnowledgeDispatch::Unknown
        } else {
            QwenKnowledgeDispatch::Rejected
        };
        Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: None,
            message: None,
            request_id: response
                .header("x-request-id")
                .or_else(|| response.header("x-oss-request-id"))
                .map(str::to_owned),
            native: Box::new(Value::Null),
            dispatch,
        })
    }

    /// Register an already uploaded lease with the workspace data center.
    /// The caller must have completed the separate OSS PUT successfully; this
    /// method does not upload or poll parser status.
    pub async fn register_file(
        &self,
        lease: &QwenKnowledgeFileUploadLease,
        request: &QwenKnowledgeRegisterFileRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileRegistration, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_lease(lease)?;
        pinned_service.ensure_supported_region()?;
        request.validate()?;
        let envelope = pinned_service
            .file_json_request(
                REGISTER_FILE_PATH,
                request.to_body(lease),
                "register_file",
                true,
                false,
                request_options,
            )
            .await?;
        let data = envelope.native.get("data").unwrap_or(&Value::Null);
        let file_id = data
            .get("fileId")
            .and_then(Value::as_str)
            .filter(|value| valid_resource_id(value))
            .ok_or_else(|| {
                response_outcome_unknown(
                    "register_file",
                    "successful registration response omitted a valid data.fileId",
                    envelope.request_id.clone(),
                    redact_lease_response(&envelope.native),
                )
            })?;
        Ok(QwenKnowledgeFileRegistration {
            reference: pinned_service
                .scope
                .file_ref_with_category_type(file_id, lease.category_type)?,
            parser: data
                .get("parser")
                .and_then(Value::as_str)
                .map(str::to_owned),
            status: data
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id: envelope.request_id,
            native: envelope.native,
        })
    }

    /// Read one file's metadata. This is a single request even when the file
    /// is still in `INIT` or `PARSING`; polling remains caller-managed.
    pub async fn describe_file(
        &self,
        file: &QwenKnowledgeFileRef,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileDetails, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_file_ref(file)?;
        pinned_service.ensure_supported_region()?;
        let envelope = pinned_service
            .file_json_request(
                DESCRIBE_FILE_PATH,
                json!({"fileId": file.file_id}),
                "describe_file",
                false,
                false,
                request_options,
            )
            .await?;
        let data = envelope.native.get("data").unwrap_or(&Value::Null);
        let returned_id = data.get("fileId").and_then(Value::as_str);
        if returned_id != Some(file.file_id.as_str()) {
            return Err(invalid_file_response(
                "describe_file",
                "response fileId did not match the requested scoped file",
                envelope.request_id,
                envelope.native,
            ));
        }
        Ok(QwenKnowledgeFileDetails {
            reference: file.clone(),
            file_name: string_field(data, "fileName"),
            file_type: string_field(data, "fileType"),
            status: string_field(data, "status"),
            size_bytes: data.get("sizeBytes").and_then(Value::as_u64),
            parser: string_field(data, "parser"),
            upload_time: string_field(data, "uploadTime"),
            category: string_field(data, "category"),
            request_id: envelope.request_id,
            native: envelope.native,
        })
    }

    /// Create a knowledge base using file references from this connection.
    /// The request's file IDs must exactly match the supplied references.
    pub async fn create_knowledge_base_with_files(
        &self,
        request: &QwenKnowledgeCreateRequest,
        files: &[QwenKnowledgeFileRef],
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeCreateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_file_ids(&request.document_ids, files)?;
        pinned_service
            .create_knowledge_base(request, request_options)
            .await
    }

    /// Append scoped file references to an existing knowledge base.
    pub async fn submit_import_job_with_files(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeImportRequest,
        files: &[QwenKnowledgeFileRef],
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeImportJobResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        let QwenKnowledgeImportSource::Files(ids) = &request.source else {
            return Err(invalid(
                "typed file references require a Files import source",
            ));
        };
        pinned_service.validate_file_ids(ids, files)?;
        pinned_service
            .submit_import_job(knowledge, request, request_options)
            .await
    }

    fn validate_lease(
        &self,
        lease: &QwenKnowledgeFileUploadLease,
    ) -> Result<(), QwenKnowledgeError> {
        self.ensure_supported_region()?;
        if lease.scope != self.scope
            || lease.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || !valid_opaque_id(&lease.lease_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "Qwen upload lease belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_file_ref(
        &self,
        file: &QwenKnowledgeFileRef,
    ) -> Result<(), QwenKnowledgeError> {
        if file.scope != self.scope
            || file.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || !valid_resource_id(&file.file_id)
        {
            return Err(LlmError::PermissionDenied {
                message: "Qwen file reference belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        Ok(())
    }

    fn validate_file_ids(
        &self,
        expected_ids: &[String],
        files: &[QwenKnowledgeFileRef],
    ) -> Result<(), QwenKnowledgeError> {
        if expected_ids.len() != files.len() {
            return Err(invalid(
                "typed file reference count must match the request's file ID count",
            ));
        }
        for (expected, file) in expected_ids.iter().zip(files) {
            self.validate_file_ref(file)?;
            if file.category_type == Some(QwenKnowledgeCategoryType::SessionFile) {
                return Err(invalid(
                    "SESSION_FILE references cannot be imported into a long-term knowledge base",
                ));
            }
            if expected != &file.file_id {
                return Err(invalid(
                    "typed file references must match request file IDs in order",
                ));
            }
        }
        Ok(())
    }

    pub(super) async fn file_json_request(
        &self,
        path: &str,
        body: Value,
        operation: &'static str,
        write: bool,
        redact_lease: bool,
        request_options: &crate::RequestOptions,
    ) -> Result<FileEnvelope, QwenKnowledgeError> {
        self.ensure_supported_region()?;
        let url = self.operation_url(path, &[])?;
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|_| invalid("Qwen file request body cannot be encoded"))?;
        let request = HttpRequest {
            method: "POST".into(),
            url: url.to_string(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", self.request_credential(request_options)?),
                ),
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
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
        decode_file_envelope(response, operation, write, redact_lease)
    }
}

pub(super) struct FileEnvelope {
    pub(super) native: Value,
    pub(super) request_id: Option<String>,
}

fn decode_file_envelope(
    response: HttpResponse,
    operation: &'static str,
    write: bool,
    redact_lease: bool,
) -> Result<FileEnvelope, QwenKnowledgeError> {
    let header_request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let parsed = serde_json::from_slice::<Value>(&response.body);
    let native = match parsed {
        Ok(value) => value,
        Err(_) if write && (200..300).contains(&response.status) => {
            return Err(response_outcome_unknown(
                operation,
                "successful HTTP response was not valid JSON",
                header_request_id,
                Value::Null,
            ));
        }
        Err(_) if !(200..300).contains(&response.status) => {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        }
        Err(_) => {
            return Err(invalid_file_response(
                operation,
                "successful HTTP response was not valid JSON",
                header_request_id,
                Value::Null,
            ));
        }
    };
    let request_id = request_id(&native, header_request_id);
    if !(200..300).contains(&response.status) {
        let dispatch = if write
            && (response.status == 408
                || response.status >= 500
                || (300..400).contains(&response.status))
        {
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
            native: Box::new(if redact_lease {
                redact_lease_response(&native)
            } else {
                native
            }),
            dispatch,
        });
    }

    let code = native.get("code").and_then(Value::as_str);
    if code.is_some_and(|code| code != "Success") {
        return Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: code.map(str::to_owned),
            message: native
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id,
            native: Box::new(if redact_lease {
                redact_lease_response(&native)
            } else {
                native
            }),
            dispatch: if write {
                QwenKnowledgeDispatch::Rejected
            } else {
                QwenKnowledgeDispatch::NotSent
            },
        });
    }
    if code != Some("Success") {
        return Err(if write {
            response_outcome_unknown(
                operation,
                "successful HTTP response omitted the documented code field",
                request_id,
                if redact_lease {
                    redact_lease_response(&native)
                } else {
                    native
                },
            )
        } else {
            invalid_file_response(
                operation,
                "successful response omitted the documented code field",
                request_id,
                native,
            )
        });
    }

    let status = numeric_status(&native, "status");
    let status_code = numeric_status(&native, "status_code");
    let Some(business_status) = reconcile_business_status(status, status_code) else {
        return Err(if write {
            response_outcome_unknown(
                operation,
                "successful response omitted a valid, consistent status or status_code",
                request_id,
                if redact_lease {
                    redact_lease_response(&native)
                } else {
                    native
                },
            )
        } else {
            invalid_file_response(
                operation,
                "successful response omitted a valid, consistent status or status_code",
                request_id,
                native,
            )
        });
    };
    if business_status != 200 {
        return Err(QwenKnowledgeError::Provider {
            status: response.status,
            code: code.map(str::to_owned),
            message: native
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
            request_id,
            native: Box::new(if redact_lease {
                redact_lease_response(&native)
            } else {
                native
            }),
            dispatch: if write && (business_status == 408 || business_status >= 500) {
                QwenKnowledgeDispatch::Unknown
            } else if write {
                QwenKnowledgeDispatch::Rejected
            } else {
                QwenKnowledgeDispatch::NotSent
            },
        });
    }
    let success_field_present = native.get("success").is_some();
    let success_value = native.get("success").and_then(Value::as_bool);
    match (success_field_present, success_value) {
        (false, _) | (true, Some(true)) => {}
        (true, Some(false)) => {
            return Err(QwenKnowledgeError::Provider {
                status: response.status,
                code: code.map(str::to_owned),
                message: native
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                request_id,
                native: Box::new(if redact_lease {
                    redact_lease_response(&native)
                } else {
                    native
                }),
                dispatch: if write {
                    QwenKnowledgeDispatch::Rejected
                } else {
                    QwenKnowledgeDispatch::NotSent
                },
            });
        }
        (true, None) => {
            return Err(if write {
                response_outcome_unknown(
                    operation,
                    "successful response contained a malformed success field",
                    request_id,
                    if redact_lease {
                        redact_lease_response(&native)
                    } else {
                        native
                    },
                )
            } else {
                invalid_file_response(
                    operation,
                    "successful response contained a malformed success field",
                    request_id,
                    native,
                )
            });
        }
    }
    Ok(FileEnvelope { native, request_id })
}

#[derive(Clone, Copy)]
enum NumericStatus {
    Missing,
    Invalid,
    Value(u64),
}

fn numeric_status(value: &Value, field: &str) -> NumericStatus {
    match value.get(field) {
        None => NumericStatus::Missing,
        Some(number) => number
            .as_u64()
            .map(NumericStatus::Value)
            .unwrap_or(NumericStatus::Invalid),
    }
}

fn reconcile_business_status(left: NumericStatus, right: NumericStatus) -> Option<u64> {
    match (left, right) {
        (NumericStatus::Value(left), NumericStatus::Value(right)) if left == right => Some(left),
        (NumericStatus::Value(value), NumericStatus::Missing)
        | (NumericStatus::Missing, NumericStatus::Value(value)) => Some(value),
        (NumericStatus::Missing, NumericStatus::Missing)
        | (NumericStatus::Invalid, _)
        | (_, NumericStatus::Invalid)
        | (NumericStatus::Value(_), NumericStatus::Value(_)) => None,
    }
}

fn validate_upload_url(value: &str) -> Result<Url, &'static str> {
    let url = Url::parse(value).map_err(|_| "lease response contained an invalid OSS URL")?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.port().is_some_and(|port| port != 443)
        || !is_documented_oss_host(url.host_str().unwrap_or_default())
    {
        return Err("lease response URL was not an allowed HTTPS OSS endpoint");
    }
    Ok(url)
}

fn is_documented_oss_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("oss-example.aliyuncs.com") {
        return true;
    }
    let labels = host
        .to_ascii_lowercase()
        .split('.')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if labels.len() != 4 || labels[2] != "aliyuncs" || labels[3] != "com" {
        return false;
    }
    let bucket = &labels[0];
    let region = labels[1].strip_prefix("oss-");
    valid_dns_label(bucket)
        && region.is_some_and(|region| {
            !region.is_empty()
                && region
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                && region.as_bytes()[0].is_ascii_alphanumeric()
                && region.as_bytes()[region.len() - 1].is_ascii_alphanumeric()
        })
}

fn valid_dns_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value.as_bytes()[value.len() - 1].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn parse_upload_headers(value: Option<&Value>) -> Result<Vec<(String, String)>, &'static str> {
    let Some(Value::Object(headers)) = value else {
        return Err("lease response omitted the OSS request headers object");
    };
    if headers.is_empty() {
        return Err("lease response contained no OSS request headers");
    }
    let mut result = Vec::with_capacity(headers.len());
    for (name, value) in headers {
        if !valid_header_name(name) || forbidden_upload_header(name) {
            return Err("lease response contained a forbidden OSS request header");
        }
        if result
            .iter()
            .any(|(existing, _): &(String, String)| existing.eq_ignore_ascii_case(name))
        {
            return Err("lease response contained duplicate OSS request headers");
        }
        let Some(value) = value.as_str() else {
            return Err("lease response OSS header values must be strings");
        };
        if value.chars().any(char::is_control) {
            return Err("lease response contained an invalid OSS header value");
        }
        result.push((name.clone(), value.to_owned()));
    }
    Ok(result)
}

fn valid_header_name(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn forbidden_upload_header(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "cookie2"
            | "host"
            | "content-length"
            | "transfer-encoding"
            | "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "upgrade"
            | "expect"
    )
}

fn valid_opaque_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn redact_lease_response(value: &Value) -> Value {
    fn redact(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut result = Map::new();
                for (key, value) in object {
                    let normalized = key.to_ascii_lowercase();
                    if matches!(
                        normalized.as_str(),
                        "leaseid" | "fileuploadleaseid" | "url" | "headers"
                    ) {
                        result.insert(key.clone(), Value::String("<redacted>".into()));
                    } else {
                        result.insert(key.clone(), redact(value));
                    }
                }
                Value::Object(result)
            }
            Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
            _ => value.clone(),
        }
    }
    redact(value)
}

pub(super) fn response_outcome_unknown(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> QwenKnowledgeError {
    QwenKnowledgeError::ResponseOutcomeUnknown {
        operation,
        message: message.into(),
        request_id,
        native: Box::new(native),
    }
}

pub(super) fn invalid_file_response(
    operation: &'static str,
    message: &str,
    request_id: Option<String>,
    native: Value,
) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidResponse {
        operation,
        message: message.into(),
        request_id,
        native: Box::new(native),
        dispatch: QwenKnowledgeDispatch::NotSent,
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}
