//! Native xAI Collections management and semantic search.
//!
//! Collection management and file upload require an xAI Management/API key,
//! while semantic search uses a regular xAI API key. Credentials are supplied
//! per operation and never appear in a serialized reference or configuration.

mod file_refs;

use crate::{
    files::{provider_file_endpoint_fingerprint, UploadFile, UploadFileStream},
    protocol::{LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpStreamRequest, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashSet, fmt, time::Duration};
use url::Url;

const DEFAULT_API_BASE_URL: &str = "https://api.x.ai/v1";
const DEFAULT_MANAGEMENT_BASE_URL: &str = "https://management-api.x.ai/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// The xAI Files API documents a 50 MB upload limit.
pub const MAX_XAI_COLLECTION_FILE_BYTES: u64 = 50_000_000;
/// xAI documents a 100 MB limit for files uploaded directly to Collections.
pub const MAX_XAI_COLLECTION_DOCUMENT_BYTES: u64 = 100_000_000;

/// Secret-free settings that bind references to one xAI profile and account.
///
/// The default endpoints are the documented xAI API and Management API roots.
/// Custom HTTPS roots are accepted for compatible gateways; plain HTTP is
/// accepted only for localhost and loopback test servers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiCollectionsConfig {
    pub profile_name: String,
    /// Stable, non-secret account identity chosen by the host application.
    pub account_scope: String,
    pub api_base_url: String,
    pub management_base_url: String,
    pub request_timeout: Duration,
}

impl XaiCollectionsConfig {
    /// Construct settings for one profile/account using xAI's documented roots.
    pub fn new(profile_name: impl Into<String>, account_scope: impl Into<String>) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            api_base_url: DEFAULT_API_BASE_URL.into(),
            management_base_url: DEFAULT_MANAGEMENT_BASE_URL.into(),
            request_timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn with_api_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.api_base_url = base_url.into();
        self
    }

    pub fn with_management_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.management_base_url = base_url.into();
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Per-call xAI credentials. This type deliberately has no serialization or
/// cloning implementation; methods use only the key required by that route.
pub struct XaiCollectionsCredentials {
    api_key: Secret<String>,
    management_api_key: Secret<String>,
}

impl XaiCollectionsCredentials {
    pub fn new(api_key: Secret<String>, management_api_key: Secret<String>) -> Self {
        Self {
            api_key,
            management_api_key,
        }
    }
}

impl fmt::Debug for XaiCollectionsCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiCollectionsCredentials")
            .field("api_key", &"<redacted>")
            .field("management_api_key", &"<redacted>")
            .finish()
    }
}

/// Safe, serializable identity shared by xAI collection and file references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCollectionsScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub api_endpoint_fingerprint: String,
    pub management_endpoint_fingerprint: String,
    pub account_scope: String,
}

/// A collection ID bound to the profile, account and API roots that created it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCollectionRef {
    pub scope: XaiCollectionsScope,
    pub collection_id: String,
}

/// An uploaded xAI file that may be added to one or more collections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiUploadedFileRef {
    pub scope: XaiCollectionsScope,
    pub file_id: String,
}

/// A document's membership in one particular collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiDocumentRef {
    pub collection: XaiCollectionRef,
    pub file_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiCollection {
    pub reference: XaiCollectionRef,
    pub name: String,
    pub created_at: Option<String>,
    pub documents_count: Option<u64>,
    pub description: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiUploadedFile {
    pub reference: XaiUploadedFileRef,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub created_at_unix_seconds: Option<u64>,
    pub expires_at_unix_seconds: Option<u64>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum XaiDocumentStatus {
    Unknown,
    Processing,
    Processed,
    Failed,
    Other(String),
}

impl From<String> for XaiDocumentStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "DOCUMENT_STATUS_UNKNOWN" => Self::Unknown,
            "DOCUMENT_STATUS_PROCESSING" => Self::Processing,
            "DOCUMENT_STATUS_PROCESSED" => Self::Processed,
            "DOCUMENT_STATUS_FAILED" => Self::Failed,
            _ => Self::Other(value),
        }
    }
}

impl From<XaiDocumentStatus> for String {
    fn from(status: XaiDocumentStatus) -> Self {
        match status {
            XaiDocumentStatus::Unknown => "DOCUMENT_STATUS_UNKNOWN".into(),
            XaiDocumentStatus::Processing => "DOCUMENT_STATUS_PROCESSING".into(),
            XaiDocumentStatus::Processed => "DOCUMENT_STATUS_PROCESSED".into(),
            XaiDocumentStatus::Failed => "DOCUMENT_STATUS_FAILED".into(),
            XaiDocumentStatus::Other(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiDocument {
    pub reference: XaiDocumentRef,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub created_at: Option<String>,
    pub fields: Value,
    pub status: Option<XaiDocumentStatus>,
    pub error_message: Option<String>,
    pub last_indexed_at: Option<String>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiCollectionPage {
    pub collections: Vec<XaiCollection>,
    pub pagination_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiDocumentPage {
    pub documents: Vec<XaiDocument>,
    pub pagination_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCollectionListRequest {
    pub limit: u8,
    pub pagination_token: Option<String>,
    pub filter: Option<String>,
    pub order: Option<String>,
    pub sort_by: Option<String>,
}

impl Default for XaiCollectionListRequest {
    fn default() -> Self {
        Self {
            limit: 100,
            pagination_token: None,
            filter: None,
            order: None,
            sort_by: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiDocumentListRequest {
    pub limit: u8,
    pub pagination_token: Option<String>,
    pub filter: Option<String>,
    pub order: Option<String>,
    pub sort_by: Option<String>,
}

impl Default for XaiDocumentListRequest {
    fn default() -> Self {
        Self {
            limit: 100,
            pagination_token: None,
            filter: None,
            order: None,
            sort_by: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCollectionFieldDefinition {
    pub key: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub unique: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub inject_into_chunk: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiCreateCollectionRequest {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Provider-native index settings, such as `{"model_name":"grok-embedding-small"}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_configuration: Option<Value>,
    /// Provider-native token/chunk configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_configuration: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub field_definitions: Vec<XaiCollectionFieldDefinition>,
}

/// Fields that can be changed on an existing xAI collection.
///
/// At least one field must be supplied. The provider's update endpoint is a
/// `PUT` operation that accepts the documented fields independently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct XaiUpdateCollectionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Provider-native chunk configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_configuration: Option<Value>,
    /// Collection document field definitions to send with the update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_definitions: Option<Vec<XaiCollectionFieldDefinition>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiSearchRequest {
    pub query: String,
    pub collections: Vec<XaiCollectionRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Optional keyword, semantic, or hybrid retrieval strategy. Omitted
    /// requests use xAI's documented hybrid default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_mode: Option<XaiRetrievalMode>,
}

/// Retrieval strategy supported by the xAI Collections search API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XaiRetrievalMode {
    Keyword,
    Semantic,
    Hybrid,
}

impl XaiRetrievalMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiSearchMatch {
    pub documents: Vec<XaiDocumentRef>,
    pub chunk_id: Option<String>,
    pub content: Option<String>,
    pub score: Option<f64>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XaiSearchResponse {
    pub matches: Vec<XaiSearchMatch>,
    pub request_id: Option<String>,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum XaiCollectionsError {
    #[error(transparent)]
    Transport(#[from] LlmError),
    #[error("invalid xAI Collections request: {0}")]
    InvalidRequest(String),
    #[error("invalid xAI Collections response: {0}")]
    InvalidResponse(String),
    #[error("xAI Collections returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
    },
    #[error(
        "xAI Collections {operation} outcome is unknown; inspect the collection before retrying: {source}"
    )]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: Box<XaiCollectionsError>,
    },
}

/// A small native xAI Collections client. It stores no credentials; pass the
/// credential bundle to each operation so keys remain owned by the host.
pub struct XaiCollectionsClient<'a> {
    transport: &'a dyn Transport,
    config: XaiCollectionsConfig,
    scope: XaiCollectionsScope,
}

impl<'a> XaiCollectionsClient<'a> {
    pub fn new(
        transport: &'a dyn Transport,
        mut config: XaiCollectionsConfig,
    ) -> Result<Self, XaiCollectionsError> {
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must be nonempty"));
        }
        if config.request_timeout.is_zero() {
            return Err(invalid("request_timeout must be positive"));
        }
        config.api_base_url = normalize_base_url(&config.api_base_url)?;
        config.management_base_url = normalize_base_url(&config.management_base_url)?;
        let scope = XaiCollectionsScope {
            provider_id: ProviderId::new("xai"),
            profile_name: config.profile_name.clone(),
            api_endpoint_fingerprint: provider_file_endpoint_fingerprint(&config.api_base_url),
            management_endpoint_fingerprint: provider_file_endpoint_fingerprint(
                &config.management_base_url,
            ),
            account_scope: config.account_scope.clone(),
        };
        Ok(Self {
            transport,
            config,
            scope,
        })
    }

    pub fn scope(&self) -> &XaiCollectionsScope {
        &self.scope
    }

    pub async fn create_collection(
        &self,
        request: &XaiCreateCollectionRequest,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiCollection, XaiCollectionsError> {
        if request.name.trim().is_empty() {
            return Err(invalid("collection name must be nonempty"));
        }
        let mut body = json!({"collection_name": request.name});
        if let Some(description) = &request.description {
            body["collection_description"] = Value::String(description.clone());
        }
        if let Some(index) = &request.index_configuration {
            body["index_configuration"] = index.clone();
        }
        if let Some(chunk) = &request.chunk_configuration {
            body["chunk_configuration"] = chunk.clone();
        }
        if !request.field_definitions.is_empty() {
            if request
                .field_definitions
                .iter()
                .any(|field| field.key.trim().is_empty())
            {
                return Err(invalid("collection field definition keys must be nonempty"));
            }
            body["field_definitions"] = serde_json::to_value(&request.field_definitions)
                .map_err(|_| invalid("collection field definitions cannot be serialized"))?;
        }
        let url = self.management_url(&["collections"])?;
        let (value, _) = self
            .send_json(
                url,
                "POST",
                Some(body),
                &credentials.management_api_key,
                "create collection",
            )
            .await?;
        decode_collection(value, &self.scope)
    }

    pub async fn list_collections(
        &self,
        request: &XaiCollectionListRequest,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiCollectionPage, XaiCollectionsError> {
        validate_page(request.limit, request.pagination_token.as_deref())?;
        validate_collection_list_options(request)?;
        let mut url = self.management_url(&["collections"])?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &request.limit.to_string());
            if let Some(token) = request.pagination_token.as_deref() {
                query.append_pair("pagination_token", token);
            }
            if let Some(filter) = request.filter.as_deref() {
                query.append_pair("filter", filter);
            }
            if let Some(order) = request.order.as_deref() {
                query.append_pair("order", order);
            }
            if let Some(sort_by) = request.sort_by.as_deref() {
                query.append_pair("sort_by", sort_by);
            }
        }
        let (value, _) = self
            .send_json(
                url,
                "GET",
                None,
                &credentials.management_api_key,
                "list collections",
            )
            .await?;
        let rows = value
            .get("collections")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("collection list has no collections array"))?;
        let collections = rows
            .iter()
            .cloned()
            .map(|row| decode_collection(row, &self.scope))
            .collect::<Result<Vec<_>, _>>()?;
        let pagination_token = value
            .get("pagination_token")
            .and_then(nonempty_string)
            .map(str::to_owned);
        Ok(XaiCollectionPage {
            collections,
            pagination_token,
        })
    }

    pub async fn get_collection(
        &self,
        collection: &XaiCollectionRef,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiCollection, XaiCollectionsError> {
        self.check_collection(collection)?;
        let url = self.collection_url(collection, &[])?;
        let (value, _) = self
            .send_json(
                url,
                "GET",
                None,
                &credentials.management_api_key,
                "get collection",
            )
            .await?;
        let decoded = decode_collection(value, &self.scope)?;
        if decoded.reference.collection_id != collection.collection_id {
            return Err(bad("collection lookup returned a different collection ID"));
        }
        Ok(decoded)
    }

    /// Update a collection's documented configuration fields.
    ///
    /// This sends one `PUT` request with the Management API key and returns
    /// the provider's updated collection object. If transport fails after
    /// dispatch, the provider may already have applied the change; inspect the
    /// collection with [`Self::get_collection`] before deciding whether to
    /// submit the update again.
    pub async fn update_collection(
        &self,
        collection: &XaiCollectionRef,
        request: &XaiUpdateCollectionRequest,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiCollection, XaiCollectionsError> {
        self.check_collection(collection)?;
        if request
            .name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(invalid("collection name must be nonempty"));
        }
        if request.field_definitions.as_ref().is_some_and(|fields| {
            fields.is_empty() || fields.iter().any(|field| field.key.trim().is_empty())
        }) {
            return Err(invalid(
                "collection field definitions must be nonempty and have nonempty keys",
            ));
        }
        if request
            .chunk_configuration
            .as_ref()
            .is_some_and(|configuration| !configuration.is_object())
        {
            return Err(invalid("chunk configuration must be a JSON object"));
        }

        let mut body = json!({});
        if let Some(name) = &request.name {
            body["collection_name"] = Value::String(name.clone());
        }
        if let Some(description) = &request.description {
            body["collection_description"] = Value::String(description.clone());
        }
        if let Some(chunk_configuration) = &request.chunk_configuration {
            body["chunk_configuration"] = chunk_configuration.clone();
        }
        if let Some(field_definitions) = &request.field_definitions {
            body["field_definitions"] = serde_json::to_value(field_definitions)
                .map_err(|_| invalid("collection field definitions cannot be serialized"))?;
        }
        if body.as_object().is_none_or(|fields| fields.is_empty()) {
            return Err(invalid("collection update must include at least one field"));
        }

        let url = self.collection_url(collection, &[])?;
        let (value, _) = self
            .send_json(
                url,
                "PUT",
                Some(body),
                &credentials.management_api_key,
                "update collection",
            )
            .await?;
        let updated = decode_collection(value, &self.scope)?;
        if updated.reference.collection_id != collection.collection_id {
            return Err(bad("collection update returned a different collection ID"));
        }
        Ok(updated)
    }

    pub async fn delete_collection(
        &self,
        collection: &XaiCollectionRef,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<(), XaiCollectionsError> {
        self.check_collection(collection)?;
        let url = self.collection_url(collection, &[])?;
        self.send_json(
            url,
            "DELETE",
            None,
            &credentials.management_api_key,
            "delete collection",
        )
        .await?;
        Ok(())
    }

    /// Upload bytes to xAI's Files API. Call [`Self::add_document`] separately
    /// to attach the returned file to a collection; the explicit two-step flow
    /// lets callers recover the uploaded file ID if attachment fails.
    pub async fn upload_file(
        &self,
        file: &UploadFile,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiUploadedFile, XaiCollectionsError> {
        validate_upload(file)?;
        let boundary = multipart_boundary();
        let body = multipart_upload(file, &boundary)?;
        let mut url = parse_base(&self.config.api_base_url)?;
        push_segments(&mut url, &["files"])?;
        let (value, _) = self
            .send(
                url,
                "POST",
                body,
                Some(format!("multipart/form-data; boundary={boundary}")),
                &credentials.api_key,
                "upload file",
            )
            .await?;
        let file_id = value
            .get("id")
            .and_then(nonempty_string)
            .ok_or_else(|| bad("file upload response has no id"))?
            .to_owned();
        Ok(XaiUploadedFile {
            reference: XaiUploadedFileRef {
                scope: self.scope.clone(),
                file_id,
            },
            filename: value
                .get("filename")
                .and_then(Value::as_str)
                .map(str::to_owned),
            size_bytes: value.get("bytes").and_then(json_u64),
            created_at_unix_seconds: value.get("created_at").and_then(json_u64),
            expires_at_unix_seconds: value.get("expires_at").and_then(json_u64),
            native: value,
        })
    }

    /// Upload a document directly through the Collections Management API.
    ///
    /// This is a single streamed multipart mutation using the Management API
    /// key. xAI documents a 100 MB limit for this route. If the transport
    /// fails after dispatch, the document may already have been created; the
    /// returned [`XaiCollectionsError::OutcomeUnknown`] is not safe to retry
    /// blindly. Inspect the collection's documents first.
    pub async fn upload_document_stream(
        &self,
        collection: &XaiCollectionRef,
        file: UploadFileStream,
        fields: Option<&Value>,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiDocument, XaiCollectionsError> {
        self.check_collection(collection)?;
        let (filename, media_type, size_bytes, body) = file.into_parts();
        validate_direct_upload(&filename, &media_type, size_bytes)?;
        if fields.is_some_and(|fields| !fields.is_object()) {
            return Err(invalid("document fields must be a JSON object"));
        }
        let token = credentials.management_api_key.expose_secret();
        if token.trim().is_empty() || token.contains('\r') || token.contains('\n') {
            return Err(invalid("the required credential is empty or malformed"));
        }

        let boundary = multipart_boundary();
        let safe_filename = filename.replace(['\\', '"'], "_");
        let mut prefix = BytesMut::new();
        crate::files::append_field(&mut prefix, &boundary, "name", &filename);
        prefix.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"data\"; filename=\"{safe_filename}\"\r\nContent-Type: {media_type}\r\n\r\n"
            )
            .as_bytes(),
        );

        let mut suffix = BytesMut::new();
        suffix.extend_from_slice(b"\r\n");
        crate::files::append_field(&mut suffix, &boundary, "content_type", &media_type);
        if let Some(fields) = fields {
            let fields = serde_json::to_string(fields)
                .map_err(|_| invalid("document fields cannot be serialized"))?;
            crate::files::append_field(&mut suffix, &boundary, "fields", &fields);
        }
        suffix.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let prefix = prefix.freeze();
        let suffix = suffix.freeze();
        let content_length = u64::try_from(prefix.len())
            .ok()
            .and_then(|prefix_len| prefix_len.checked_add(size_bytes))
            .and_then(|length| {
                u64::try_from(suffix.len())
                    .ok()
                    .and_then(|suffix_len| length.checked_add(suffix_len))
            })
            .ok_or_else(|| invalid("multipart upload size overflows"))?;
        let url = self.collection_url(collection, &["documents"])?;
        let deadline = crate::runtime::Deadline::after(Some(self.config.request_timeout));
        // Keep file-source failures distinguishable from transport preflight
        // failures. A source can fail after multipart framing has been sent.
        let body = body
            .map(|chunk| {
                chunk.map_err(|error| LlmError::StreamInterrupted {
                    message: format!("document upload source failed: {error}"),
                })
            })
            .boxed();
        let request = HttpStreamRequest {
            method: "POST".into(),
            url: url.into(),
            headers: vec![
                ("authorization".into(), format!("Bearer {token}")),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: crate::files::multipart_upload_stream(prefix, body, size_bytes, suffix),
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_stream_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(|error| match error {
                LlmError::UnsupportedCapability { .. } => XaiCollectionsError::Transport(error),
                other => outcome_unknown(
                    "direct document upload",
                    XaiCollectionsError::Transport(other),
                ),
            })?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(XaiCollectionsError::Provider {
                status: response.status,
                request_id,
            });
        }
        let value = serde_json::from_slice(&response.body).map_err(|_| {
            outcome_unknown(
                "direct document upload",
                bad("upload response is not valid JSON"),
            )
        })?;
        decode_document(value, collection)
            .map_err(|error| outcome_unknown("direct document upload", error))
    }

    pub async fn add_document(
        &self,
        collection: &XaiCollectionRef,
        file: &XaiUploadedFileRef,
        fields: Option<&Value>,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiDocumentRef, XaiCollectionsError> {
        self.check_collection(collection)?;
        self.check_file(file)?;
        if fields.is_some_and(|fields| !fields.is_object()) {
            return Err(invalid("document fields must be a JSON object"));
        }
        let url = self.collection_url(collection, &["documents", &file.file_id])?;
        let body = fields.map(|fields| json!({"fields":fields}));
        self.send_json(
            url,
            "POST",
            body,
            &credentials.management_api_key,
            "add document to collection",
        )
        .await?;
        Ok(XaiDocumentRef {
            collection: collection.clone(),
            file_id: file.file_id.clone(),
        })
    }

    pub async fn list_documents(
        &self,
        collection: &XaiCollectionRef,
        request: &XaiDocumentListRequest,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiDocumentPage, XaiCollectionsError> {
        self.check_collection(collection)?;
        validate_page(request.limit, request.pagination_token.as_deref())?;
        if request
            .filter
            .as_ref()
            .is_some_and(|filter| filter.trim().is_empty())
        {
            return Err(invalid("document filter cannot be empty"));
        }
        validate_optional_query_value("order", request.order.as_deref())?;
        validate_optional_query_value("sort_by", request.sort_by.as_deref())?;
        let mut url = self.collection_url(collection, &["documents"])?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &request.limit.to_string());
            if let Some(token) = request.pagination_token.as_deref() {
                query.append_pair("pagination_token", token);
            }
            if let Some(filter) = request.filter.as_deref() {
                query.append_pair("filter", filter);
            }
            if let Some(order) = request.order.as_deref() {
                query.append_pair("order", order);
            }
            if let Some(sort_by) = request.sort_by.as_deref() {
                query.append_pair("sort_by", sort_by);
            }
        }
        let (value, _) = self
            .send_json(
                url,
                "GET",
                None,
                &credentials.management_api_key,
                "list documents",
            )
            .await?;
        let rows = value
            .get("documents")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("document list has no documents array"))?;
        let documents = rows
            .iter()
            .cloned()
            .map(|row| decode_document(row, collection))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(XaiDocumentPage {
            documents,
            pagination_token: value
                .get("pagination_token")
                .and_then(nonempty_string)
                .map(str::to_owned),
        })
    }

    pub async fn get_document(
        &self,
        document: &XaiDocumentRef,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiDocument, XaiCollectionsError> {
        self.check_document(document)?;
        let url = self.collection_url(&document.collection, &["documents", &document.file_id])?;
        let (value, _) = self
            .send_json(
                url,
                "GET",
                None,
                &credentials.management_api_key,
                "get document",
            )
            .await?;
        let decoded = decode_document(value, &document.collection)?;
        if decoded.reference.file_id != document.file_id {
            return Err(bad("document lookup returned a different file ID"));
        }
        Ok(decoded)
    }

    /// Retrieve document metadata for several documents in one collection.
    ///
    /// The response may contain a subset of the requested IDs; xAI does not
    /// document how missing IDs are represented. Returned IDs that were not
    /// requested, or duplicate returned IDs, are rejected as malformed.
    pub async fn batch_get_documents(
        &self,
        documents: &[XaiDocumentRef],
        credentials: &XaiCollectionsCredentials,
    ) -> Result<Vec<XaiDocument>, XaiCollectionsError> {
        let Some(first) = documents.first() else {
            return Err(invalid(
                "batch metadata lookup requires at least one document",
            ));
        };
        self.check_document(first)?;
        let collection = &first.collection;
        let mut requested = HashSet::with_capacity(documents.len());
        for document in documents {
            self.check_document(document)?;
            if &document.collection != collection {
                return Err(invalid(
                    "batch metadata lookup documents must belong to one collection",
                ));
            }
            if !requested.insert(document.file_id.clone()) {
                return Err(invalid("batch metadata lookup contains duplicate file IDs"));
            }
        }

        let url = self.collection_url(collection, &["documents:batchGet"])?;
        let mut url = url;
        {
            let mut query = url.query_pairs_mut();
            for document in documents {
                query.append_pair("file_ids", &document.file_id);
            }
        }
        let (value, _) = self
            .send_json(
                url,
                "GET",
                None,
                &credentials.management_api_key,
                "batch get document metadata",
            )
            .await?;
        let rows = value
            .get("documents")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("batch metadata response has no documents array"))?;
        let mut returned = HashSet::with_capacity(rows.len());
        rows.iter()
            .cloned()
            .map(|row| {
                let document = decode_document(row, collection)?;
                let file_id = document.reference.file_id.clone();
                if !requested.contains(&file_id) {
                    return Err(bad(
                        "batch metadata response contains an unrequested file ID",
                    ));
                }
                if !returned.insert(file_id) {
                    return Err(bad("batch metadata response contains a duplicate file ID"));
                }
                Ok(document)
            })
            .collect()
    }

    /// Ask xAI to regenerate the index for a document already attached to a
    /// collection. The operation uses the Management API key and returns when
    /// xAI accepts the request; callers can inspect the document status later
    /// with [`Self::get_document`].
    pub async fn reindex_document(
        &self,
        document: &XaiDocumentRef,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<(), XaiCollectionsError> {
        self.check_document(document)?;
        let url = self.collection_url(&document.collection, &["documents", &document.file_id])?;
        self.send_json(
            url,
            "PATCH",
            None,
            &credentials.management_api_key,
            "regenerate document index",
        )
        .await?;
        Ok(())
    }

    pub async fn remove_document(
        &self,
        document: &XaiDocumentRef,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<(), XaiCollectionsError> {
        self.check_document(document)?;
        let url = self.collection_url(&document.collection, &["documents", &document.file_id])?;
        self.send_json(
            url,
            "DELETE",
            None,
            &credentials.management_api_key,
            "remove document from collection",
        )
        .await?;
        Ok(())
    }

    pub async fn search(
        &self,
        request: &XaiSearchRequest,
        credentials: &XaiCollectionsCredentials,
    ) -> Result<XaiSearchResponse, XaiCollectionsError> {
        if request.query.trim().is_empty() || request.collections.is_empty() {
            return Err(invalid(
                "search requires a query and at least one collection",
            ));
        }
        let mut collection_ids = Vec::with_capacity(request.collections.len());
        for collection in &request.collections {
            self.check_collection(collection)?;
            if collection_ids.contains(&collection.collection_id) {
                return Err(invalid("search collection list contains duplicate IDs"));
            }
            collection_ids.push(collection.collection_id.clone());
        }
        if request
            .filter
            .as_ref()
            .is_some_and(|filter| filter.trim().is_empty())
        {
            return Err(invalid("search filter cannot be empty"));
        }
        let mut body = json!({
            "query": request.query,
            "source": {"collection_ids": collection_ids}
        });
        if let Some(retrieval_mode) = request.retrieval_mode {
            body["retrieval_mode"] = json!({"type": retrieval_mode.as_str()});
        }
        if let Some(filter) = &request.filter {
            body["filter"] = Value::String(filter.clone());
        }
        let mut url = parse_base(&self.config.api_base_url)?;
        push_segments(&mut url, &["documents", "search"])?;
        let (value, request_id) = self
            .send_json(
                url,
                "POST",
                Some(body),
                &credentials.api_key,
                "search documents",
            )
            .await?;
        let rows = value
            .get("matches")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("search response has no matches array"))?;
        let mut matches = Vec::with_capacity(rows.len());
        for row in rows {
            let file_id = row
                .get("file_id")
                .and_then(nonempty_string)
                .ok_or_else(|| bad("search match has no file_id"))?;
            let collection_rows = row
                .get("collection_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("search match has no collection_ids array"))?;
            let mut documents = Vec::with_capacity(collection_rows.len());
            for collection_id in collection_rows {
                let collection_id = collection_id
                    .as_str()
                    .filter(|id| !id.trim().is_empty())
                    .ok_or_else(|| bad("search match has an invalid collection ID"))?;
                let collection = request
                    .collections
                    .iter()
                    .find(|collection| collection.collection_id == collection_id)
                    .ok_or_else(|| bad("search match refers to an unrequested collection"))?;
                documents.push(XaiDocumentRef {
                    collection: collection.clone(),
                    file_id: file_id.to_owned(),
                });
            }
            matches.push(XaiSearchMatch {
                documents,
                chunk_id: row
                    .get("chunk_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                content: row
                    .get("chunk_content")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                score: row.get("score").and_then(Value::as_f64),
                native: row.clone(),
            });
        }
        Ok(XaiSearchResponse {
            matches,
            request_id,
            native: value,
        })
    }

    fn collection_url(
        &self,
        collection: &XaiCollectionRef,
        child_segments: &[&str],
    ) -> Result<Url, XaiCollectionsError> {
        let mut segments = Vec::with_capacity(child_segments.len() + 2);
        segments.push("collections");
        segments.push(collection.collection_id.as_str());
        segments.extend_from_slice(child_segments);
        let mut url = parse_base(&self.config.management_base_url)?;
        push_segments(&mut url, &segments)?;
        Ok(url)
    }

    fn management_url(&self, segments: &[&str]) -> Result<Url, XaiCollectionsError> {
        let mut url = parse_base(&self.config.management_base_url)?;
        push_segments(&mut url, segments)?;
        Ok(url)
    }

    fn check_collection(&self, reference: &XaiCollectionRef) -> Result<(), XaiCollectionsError> {
        if reference.scope != self.scope || !valid_id(&reference.collection_id) {
            return Err(invalid(
                "collection reference belongs to another profile, account, or endpoint",
            ));
        }
        Ok(())
    }

    fn check_file(&self, reference: &XaiUploadedFileRef) -> Result<(), XaiCollectionsError> {
        if reference.scope != self.scope || !valid_id(&reference.file_id) {
            return Err(invalid(
                "uploaded file reference belongs to another profile, account, or endpoint",
            ));
        }
        Ok(())
    }

    fn check_document(&self, reference: &XaiDocumentRef) -> Result<(), XaiCollectionsError> {
        self.check_collection(&reference.collection)?;
        if !valid_id(&reference.file_id) {
            return Err(invalid("document reference has an invalid file ID"));
        }
        Ok(())
    }

    async fn send_json(
        &self,
        url: Url,
        method: &str,
        body: Option<Value>,
        credential: &Secret<String>,
        operation: &str,
    ) -> Result<(Value, Option<String>), XaiCollectionsError> {
        let body = match body {
            Some(value) => Bytes::from(
                serde_json::to_vec(&value)
                    .map_err(|_| invalid("request body cannot be serialized"))?,
            ),
            None => Bytes::new(),
        };
        let content_type = (!body.is_empty()).then(|| "application/json".into());
        self.send(url, method, body, content_type, credential, operation)
            .await
    }

    async fn send(
        &self,
        url: Url,
        method: &str,
        body: Bytes,
        content_type: Option<String>,
        credential: &Secret<String>,
        operation: &str,
    ) -> Result<(Value, Option<String>), XaiCollectionsError> {
        let token = credential.expose_secret();
        if token.trim().is_empty() || token.contains('\r') || token.contains('\n') {
            return Err(invalid("the required credential is empty or malformed"));
        }
        let mut headers = vec![("authorization".into(), format!("Bearer {token}"))];
        if let Some(content_type) = content_type {
            headers.push(("content-type".into(), content_type));
        }
        let deadline = crate::runtime::Deadline::after(Some(self.config.request_timeout));
        let request = HttpRequest {
            method: method.into(),
            url: url.into(),
            headers,
            body,
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        if !(200..300).contains(&response.status) {
            return Err(XaiCollectionsError::Provider {
                status: response.status,
                request_id,
            });
        }
        let value = if response.body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&response.body)
                .map_err(|_| bad(&format!("{operation} response is not valid JSON")))?
        };
        Ok((value, request_id))
    }
}

fn decode_collection(
    value: Value,
    scope: &XaiCollectionsScope,
) -> Result<XaiCollection, XaiCollectionsError> {
    let value = value.get("collection").cloned().unwrap_or(value);
    let collection_id = value
        .get("collection_id")
        .and_then(nonempty_string)
        .ok_or_else(|| bad("collection metadata has no collection_id"))?;
    let name = value
        .get("collection_name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| bad("collection metadata has no collection_name"))?;
    Ok(XaiCollection {
        reference: XaiCollectionRef {
            scope: scope.clone(),
            collection_id: collection_id.to_owned(),
        },
        name: name.to_owned(),
        created_at: value.get("created_at").and_then(scalar_string),
        documents_count: value.get("documents_count").and_then(json_u64),
        description: value
            .get("collection_description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        native: value,
    })
}

fn decode_document(
    value: Value,
    collection: &XaiCollectionRef,
) -> Result<XaiDocument, XaiCollectionsError> {
    let file_id = value
        .get("file_metadata")
        .and_then(|metadata| metadata.get("file_id"))
        .or_else(|| value.get("file_id"))
        .and_then(nonempty_string)
        .ok_or_else(|| bad("document metadata has no file_id"))?;
    let metadata = value.get("file_metadata").unwrap_or(&value);
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .map(|status| XaiDocumentStatus::from(status.to_owned()));
    let fields = match value.get("fields") {
        Some(fields) if fields.is_object() => fields.clone(),
        Some(_) => return Err(bad("document fields are not a JSON object")),
        None => json!({}),
    };
    Ok(XaiDocument {
        reference: XaiDocumentRef {
            collection: collection.clone(),
            file_id: file_id.to_owned(),
        },
        filename: metadata
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        content_type: metadata
            .get("content_type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        size_bytes: metadata.get("size_bytes").and_then(json_u64),
        created_at: metadata.get("created_at").and_then(scalar_string),
        fields,
        status,
        error_message: value
            .get("error_message")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_owned),
        last_indexed_at: value
            .get("last_indexed_at")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_owned),
        native: value,
    })
}

fn validate_page(limit: u8, token: Option<&str>) -> Result<(), XaiCollectionsError> {
    if !(1..=100).contains(&limit) {
        return Err(invalid("page limit must be between 1 and 100"));
    }
    if token.is_some_and(|token| token.trim().is_empty()) {
        return Err(invalid("pagination token cannot be empty"));
    }
    Ok(())
}

fn validate_collection_list_options(
    request: &XaiCollectionListRequest,
) -> Result<(), XaiCollectionsError> {
    if request
        .filter
        .as_ref()
        .is_some_and(|filter| filter.trim().is_empty())
    {
        return Err(invalid("collection filter cannot be empty"));
    }
    validate_optional_query_value("order", request.order.as_deref())?;
    validate_optional_query_value("sort_by", request.sort_by.as_deref())
}

fn validate_optional_query_value(
    name: &'static str,
    value: Option<&str>,
) -> Result<(), XaiCollectionsError> {
    if value.is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control)) {
        return Err(XaiCollectionsError::InvalidRequest(format!(
            "{name} must be nonempty and contain no control characters"
        )));
    }
    Ok(())
}

fn validate_upload(file: &UploadFile) -> Result<(), XaiCollectionsError> {
    if file.bytes.is_empty() {
        return Err(invalid("file data must be nonempty"));
    }
    if file.bytes.len() as u64 > MAX_XAI_COLLECTION_FILE_BYTES {
        return Err(invalid("file exceeds the xAI Files API 50 MB upload limit"));
    }
    if file.filename.trim().is_empty() || file.filename.chars().any(char::is_control) {
        return Err(invalid(
            "file name must be nonempty and contain no line breaks",
        ));
    }
    validate_media_type(&file.media_type)?;
    Ok(())
}

fn validate_direct_upload(
    filename: &str,
    media_type: &str,
    size_bytes: u64,
) -> Result<(), XaiCollectionsError> {
    if size_bytes > MAX_XAI_COLLECTION_DOCUMENT_BYTES {
        return Err(invalid(
            "file exceeds the xAI Collections 100 MB upload limit",
        ));
    }
    if filename.trim().is_empty() || filename.chars().any(char::is_control) {
        return Err(invalid(
            "file name must be nonempty and contain no line breaks",
        ));
    }
    validate_media_type(media_type)
}

fn multipart_upload(file: &UploadFile, boundary: &str) -> Result<Bytes, XaiCollectionsError> {
    let mut body = BytesMut::with_capacity(file.bytes.len().saturating_add(512));
    // The xAI collections guide uploads with the `assistants` purpose. Keep
    // this scalar field before the file part in the multipart body.
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n"
        )
        .as_bytes(),
    );
    let filename = file.filename.replace(['\\', '"'], "_");
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
            file.media_type
        )
        .as_bytes(),
    );
    body.extend_from_slice(&file.bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok(body.freeze())
}

fn validate_media_type(media_type: &str) -> Result<(), XaiCollectionsError> {
    let Some((major, minor)) = media_type.split_once('/') else {
        return Err(invalid("media type must be a MIME type"));
    };
    let token = |part: &str| {
        !part.is_empty()
            && part.bytes().all(|byte| {
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
    };
    if media_type.matches('/').count() != 1 || !token(major) || !token(minor) {
        return Err(invalid("media type contains invalid MIME characters"));
    }
    Ok(())
}

fn normalize_base_url(value: &str) -> Result<String, XaiCollectionsError> {
    let mut url = Url::parse(value).map_err(|_| invalid("API base URL is invalid"))?;
    if url.username() != ""
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "API base URL cannot contain credentials, query parameters, or a fragment",
        ));
    }
    let host = url.host_str().unwrap_or_default();
    let local_http = url.scheme() == "http"
        && (host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback()));
    if url.scheme() != "https" && !local_http {
        return Err(invalid("API base URL must use HTTPS"));
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(if path.is_empty() { "/" } else { &path });
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn parse_base(value: &str) -> Result<Url, XaiCollectionsError> {
    Url::parse(value).map_err(|_| invalid("configured API base URL is invalid"))
}

fn push_segments(url: &mut Url, segments: &[&str]) -> Result<(), XaiCollectionsError> {
    let mut path = url
        .path_segments_mut()
        .map_err(|_| invalid("configured API base URL cannot accept resource paths"))?;
    path.pop_if_empty();
    for segment in segments {
        path.push(segment);
    }
    Ok(())
}

fn multipart_boundary() -> String {
    format!(
        "lingxi-xai-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |time| time.as_nanos())
    )
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 1024 && !value.chars().any(char::is_control)
}

fn nonempty_string(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| valid_id(value))
}

fn scalar_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_i64().map(|number| number.to_string()))
        .or_else(|| value.as_u64().map(|number| number.to_string()))
}

fn json_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

fn invalid(message: &str) -> XaiCollectionsError {
    XaiCollectionsError::InvalidRequest(message.into())
}

fn bad(message: &str) -> XaiCollectionsError {
    XaiCollectionsError::InvalidResponse(message.into())
}

fn outcome_unknown(operation: &'static str, source: XaiCollectionsError) -> XaiCollectionsError {
    XaiCollectionsError::OutcomeUnknown {
        operation,
        source: Box::new(source),
    }
}

fn is_false(value: &bool) -> bool {
    !value
}
