//! Gemini File Search store and document lifecycle operations.
//!
//! Resource references bind every store, document, and indexing operation to
//! one Google provider profile, configured endpoint, and caller-owned account
//! scope. The service imports existing Gemini Files API resources; it does
//! not upload local bytes or run a polling loop on the caller's behalf.

use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{
        LlmError, ProtocolFamily, ProviderId, ProviderProfile, ServiceAuth, ServiceSetting,
    },
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, HttpResponse},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

const MAX_RESPONSE: usize = 16 * 1024 * 1024;
const MAX_REQUEST: usize = 1024 * 1024;
const MAX_DIRECT_UPLOAD: usize = 100 * 1024 * 1024;
const MAX_PAGE_SIZE: u8 = 20;

/// Gemini File Search REST route. `endpoint` is the full
/// `.../v1beta/fileSearchStores` collection URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchRoute {
    pub endpoint: String,
    pub auth: ServiceAuth,
}

/// Opaque identity for a store created or listed through this client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchStoreRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub store_id: String,
}

/// Opaque identity for a document within a specific account-scoped store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchDocumentRef {
    pub store: GeminiFileSearchStoreRef,
    pub document_id: String,
}

/// Opaque identity for an asynchronous import/index operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchOperationRef {
    pub store: GeminiFileSearchStoreRef,
    /// Operation resource ID from `fileSearchStores/{store}/operations/{id}`.
    pub operation_id: String,
}

/// Opaque identity for an operation created by `uploadToFileSearchStore`.
/// Google exposes a separate `upload/operations` resource path for this LRO.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchUploadOperationRef {
    pub store: GeminiFileSearchStoreRef,
    pub operation_id: String,
}

/// A Google File Search store and its provider-native representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiFileSearchStore {
    pub reference: GeminiFileSearchStoreRef,
    pub display_name: Option<String>,
    pub embedding_model: Option<String>,
    pub create_time: Option<String>,
    pub update_time: Option<String>,
    pub active_documents_count: Option<u64>,
    pub pending_documents_count: Option<u64>,
    pub failed_documents_count: Option<u64>,
    pub size_bytes: Option<u64>,
    pub native: Value,
}

/// Document indexing state as reported by Gemini.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiFileSearchDocumentState {
    Unspecified,
    Pending,
    Active,
    Failed,
    Other(String),
}
impl GeminiFileSearchDocumentState {
    /// `ACTIVE` means all chunks have been processed and are available to query.
    pub fn is_ready(&self) -> bool {
        *self == Self::Active
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Active | Self::Failed)
    }
}

/// A document managed by a File Search store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiFileSearchDocument {
    pub reference: GeminiFileSearchDocumentRef,
    pub display_name: Option<String>,
    pub state: GeminiFileSearchDocumentState,
    pub create_time: Option<String>,
    pub update_time: Option<String>,
    pub size_bytes: Option<u64>,
    pub mime_type: Option<String>,
    /// Provider-native custom metadata values, retained without narrowing.
    pub custom_metadata: Vec<Value>,
    pub native: Value,
}

/// State of the asynchronous operation returned by `importFile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiFileSearchOperationState {
    Running,
    Succeeded,
    Failed,
    DoneWithoutResult,
}
impl GeminiFileSearchOperationState {
    pub fn is_terminal(self) -> bool {
        self != Self::Running
    }
}

/// The long-running import/index operation returned by Gemini.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiFileSearchOperation {
    pub reference: GeminiFileSearchOperationRef,
    pub state: GeminiFileSearchOperationState,
    pub done: bool,
    pub metadata: Option<Value>,
    pub error: Option<Value>,
    pub response: Option<Value>,
    pub native: Value,
    pub request_id: Option<String>,
}

/// Result of uploading bytes directly into a File Search store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiFileSearchUploadOperation {
    pub reference: GeminiFileSearchUploadOperationRef,
    pub state: GeminiFileSearchOperationState,
    pub done: bool,
    pub metadata: Option<Value>,
    pub error: Option<Value>,
    pub response: Option<Value>,
    pub native: Value,
    pub request_id: Option<String>,
}

/// Bytes and metadata for Google's resumable `uploadToFileSearchStore` API.
/// This client sends the supplied bytes as one final upload chunk, so payloads
/// larger than 100 MiB must use the Files API followed by `import_file`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeminiFileSearchUploadRequest {
    pub data: Bytes,
    pub mime_type: String,
    pub display_name: Option<String>,
    pub custom_metadata: Vec<GeminiFileSearchMetadata>,
    pub chunking_config: Option<GeminiFileSearchWhiteSpaceChunking>,
}

impl GeminiFileSearchUploadRequest {
    pub fn new(data: impl Into<Bytes>, mime_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            mime_type: mime_type.into(),
            display_name: None,
            custom_metadata: Vec::new(),
            chunking_config: None,
        }
    }

    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    pub fn with_custom_metadata(
        mut self,
        custom_metadata: impl IntoIterator<Item = GeminiFileSearchMetadata>,
    ) -> Self {
        self.custom_metadata = custom_metadata.into_iter().collect();
        self
    }

    pub fn with_chunking_config(
        mut self,
        chunking_config: GeminiFileSearchWhiteSpaceChunking,
    ) -> Self {
        self.chunking_config = Some(chunking_config);
        self
    }
}

/// One provider-paginated page. The next token is passed back unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiFileSearchPage<T> {
    pub items: Vec<T>,
    pub next_page_token: Option<String>,
}

/// Metadata value accepted by Google's import API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum GeminiFileSearchMetadataValue {
    String(String),
    StringList(Vec<String>),
    Number(f64),
}

/// One metadata entry associated with the imported document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchMetadata {
    pub key: String,
    pub value: GeminiFileSearchMetadataValue,
}

/// White-space chunking settings documented by the Gemini File Search API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiFileSearchWhiteSpaceChunking {
    pub max_tokens_per_chunk: u32,
    pub max_overlap_tokens: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiFileSearchError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("Gemini File Search returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid Gemini File Search response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("outcome of Gemini File Search {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
}

/// Service handle whose live configuration is captured once per operation.
#[derive(Clone, Copy)]
pub struct GeminiFileSearchService<'a> {
    source: ClientSource<'a>,
}
impl<'a> GeminiFileSearchService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn create_store(
        self,
        profile_name: &str,
        display_name: Option<&str>,
        embedding_model: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchStore, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .create_store(profile_name, display_name, embedding_model, options)
            .await
    }

    pub async fn list_stores(
        self,
        profile_name: &str,
        page_size: Option<u8>,
        page_token: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchPage<GeminiFileSearchStore>, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .list_stores(profile_name, page_size, page_token, options)
            .await
    }

    pub async fn get_store(
        self,
        reference: &GeminiFileSearchStoreRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchStore, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .get_store(reference, options)
            .await
    }

    pub async fn delete_store(
        self,
        reference: &GeminiFileSearchStoreRef,
        force: bool,
        options: &RequestOptions,
    ) -> Result<(), GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .delete_store(reference, force, options)
            .await
    }

    /// Import an existing Gemini Files API object and return its indexing LRO.
    /// The caller decides when to call `get_operation` again.
    pub async fn import_file(
        self,
        store: &GeminiFileSearchStoreRef,
        file: &ProviderFileRef,
        metadata: &[GeminiFileSearchMetadata],
        chunking: Option<GeminiFileSearchWhiteSpaceChunking>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchOperation, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .import_file(store, file, metadata, chunking, options)
            .await
    }

    pub async fn get_operation(
        self,
        reference: &GeminiFileSearchOperationRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchOperation, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .get_operation(reference, options)
            .await
    }

    /// Upload bytes directly into a File Search store through Google's
    /// resumable upload protocol. No retry or indexing poll is performed.
    pub async fn upload_to_store(
        self,
        store: &GeminiFileSearchStoreRef,
        request: &GeminiFileSearchUploadRequest,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchUploadOperation, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .upload_to_store(store, request, options)
            .await
    }

    /// Retrieve the operation returned by `upload_to_store` once.
    pub async fn get_upload_operation(
        self,
        reference: &GeminiFileSearchUploadOperationRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchUploadOperation, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .get_upload_operation(reference, options)
            .await
    }

    pub async fn list_documents(
        self,
        store: &GeminiFileSearchStoreRef,
        page_size: Option<u8>,
        page_token: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchPage<GeminiFileSearchDocument>, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .list_documents(store, page_size, page_token, options)
            .await
    }

    pub async fn get_document(
        self,
        reference: &GeminiFileSearchDocumentRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchDocument, GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .get_document(reference, options)
            .await
    }

    pub async fn delete_document(
        self,
        reference: &GeminiFileSearchDocumentRef,
        force: bool,
        options: &RequestOptions,
    ) -> Result<(), GeminiFileSearchError> {
        let snapshot = self.source.snapshot();
        Pinned { client: &snapshot }
            .delete_document(reference, force, options)
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
    ) -> Result<(&'s ProviderProfile, &'s GeminiFileSearchRoute, &'o str), GeminiFileSearchError>
    {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown Gemini File Search profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("Gemini File Search profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.gemini_file_search else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled Gemini File Search route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| invalid("Gemini File Search requires a non-secret account_scope"))?;
        Ok((profile, route, scope))
    }

    fn checked_store<'s>(
        &'s self,
        reference: &GeminiFileSearchStoreRef,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s GeminiFileSearchRoute), GeminiFileSearchError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.account_scope != scope
            || !valid_resource_id(&reference.store_id)
        {
            return Err(LlmError::PermissionDenied {
                message:
                    "Gemini File Search store belongs to another provider, profile, endpoint or account"
                        .into(),
            }
            .into());
        }
        Ok((profile, route))
    }

    async fn create_store(
        &self,
        profile_name: &str,
        display_name: Option<&str>,
        embedding_model: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchStore, GeminiFileSearchError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        if display_name.is_some_and(|name| name.len() > 512)
            || embedding_model.is_some_and(|model| !valid_model_resource(model))
        {
            return Err(
                invalid("Gemini File Search store name or embedding model is invalid").into(),
            );
        }
        let mut body = serde_json::Map::new();
        if let Some(display_name) = display_name {
            body.insert("displayName".into(), Value::String(display_name.into()));
        }
        if let Some(embedding_model) = embedding_model {
            body.insert(
                "embeddingModel".into(),
                Value::String(embedding_model.into()),
            );
        }
        let response = self
            .send(
                route,
                options,
                "POST",
                collection_url(route)?,
                encode_body(Value::Object(body))?,
                "create_store",
            )
            .await?;
        let native = success_json(response)?;
        decode_store(native, profile, route, scope, None)
    }

    async fn list_stores(
        &self,
        profile_name: &str,
        page_size: Option<u8>,
        page_token: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchPage<GeminiFileSearchStore>, GeminiFileSearchError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        validate_page(page_size, page_token)?;
        let mut url = collection_url(route)?;
        add_page_query(&mut url, page_size, page_token);
        let response = self
            .send(route, options, "GET", url, Vec::new(), "list_stores")
            .await?;
        let native = success_json(response)?;
        let rows = native
            .get("fileSearchStores")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("store list omitted fileSearchStores", native.clone()))?;
        let items = rows
            .iter()
            .cloned()
            .map(|row| decode_store(row, profile, route, scope, None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GeminiFileSearchPage {
            items,
            next_page_token: page_token_from(&native),
        })
    }

    async fn get_store(
        &self,
        reference: &GeminiFileSearchStoreRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchStore, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(reference, options)?;
        let response = self
            .send(
                route,
                options,
                "GET",
                store_url(route, reference)?,
                Vec::new(),
                "get_store",
            )
            .await?;
        decode_store(
            success_json(response)?,
            profile,
            route,
            &reference.account_scope,
            Some(&reference.store_id),
        )
    }

    async fn delete_store(
        &self,
        reference: &GeminiFileSearchStoreRef,
        force: bool,
        options: &RequestOptions,
    ) -> Result<(), GeminiFileSearchError> {
        let (_, route) = self.checked_store(reference, options)?;
        let mut url = store_url(route, reference)?;
        if force {
            url.query_pairs_mut().append_pair("force", "true");
        }
        let response = self
            .send(route, options, "DELETE", url, Vec::new(), "delete_store")
            .await?;
        ensure_success(response)
    }

    async fn import_file(
        &self,
        reference: &GeminiFileSearchStoreRef,
        file: &ProviderFileRef,
        metadata: &[GeminiFileSearchMetadata],
        chunking: Option<GeminiFileSearchWhiteSpaceChunking>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchOperation, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(reference, options)?;
        check_gemini_file(profile, reference, file, options)?;
        validate_metadata(metadata)?;
        let file_name = gemini_file_name(&file.file_id).ok_or_else(|| {
            invalid("Gemini File Search import requires a Gemini File API resource name")
        })?;
        let mut body = json!({"fileName": file_name});
        if !metadata.is_empty() {
            // The public Rust value is intentionally tagged for clarity. Convert
            // it to Google's one-of field representation before sending.
            body["customMetadata"] = Value::Array(
                metadata
                    .iter()
                    .map(metadata_to_wire)
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        if let Some(chunking) = chunking {
            body["chunkingConfig"] = json!({
                "whiteSpaceConfig": {
                    "maxTokensPerChunk": chunking.max_tokens_per_chunk,
                    "maxOverlapTokens": chunking.max_overlap_tokens
                }
            });
        }
        let url = import_url(route, reference)?;
        let response = self
            .send(
                route,
                options,
                "POST",
                url,
                encode_body(body)?,
                "import_file",
            )
            .await?;
        let request_id = request_id(&response);
        let native = success_json(response)?;
        decode_operation(native, profile, route, reference, request_id, None)
    }

    async fn get_operation(
        &self,
        reference: &GeminiFileSearchOperationRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchOperation, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(&reference.store, options)?;
        if !valid_operation_id(&reference.operation_id) {
            return Err(invalid("invalid Gemini File Search operation ID").into());
        }
        let response = self
            .send(
                route,
                options,
                "GET",
                operation_url(route, reference)?,
                Vec::new(),
                "get_operation",
            )
            .await?;
        let request_id = request_id(&response);
        let native = success_json(response)?;
        decode_operation(
            native,
            profile,
            route,
            &reference.store,
            request_id,
            Some(&reference.operation_id),
        )
    }

    async fn upload_to_store(
        &self,
        store: &GeminiFileSearchStoreRef,
        request: &GeminiFileSearchUploadRequest,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchUploadOperation, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(store, options)?;
        validate_upload_request(request)?;
        validate_metadata(&request.custom_metadata)?;
        let ServiceAuth::ApiKey { header } = &route.auth else {
            return Err(invalid("Gemini File Search requires API key authentication").into());
        };
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Gemini File Search requires a credential".into(),
            })?;

        let mut metadata = json!({"mimeType": request.mime_type});
        if let Some(display_name) = &request.display_name {
            metadata["displayName"] = Value::String(display_name.clone());
        }
        if !request.custom_metadata.is_empty() {
            metadata["customMetadata"] = Value::Array(
                request
                    .custom_metadata
                    .iter()
                    .map(metadata_to_wire)
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        if let Some(chunking) = request.chunking_config {
            metadata["chunkingConfig"] = json!({
                "whiteSpaceConfig": {
                    "maxTokensPerChunk": chunking.max_tokens_per_chunk,
                    "maxOverlapTokens": chunking.max_overlap_tokens
                }
            });
        }
        let metadata_body = encode_body(metadata)?;
        if metadata_body.len() > MAX_REQUEST {
            return Err(invalid("Gemini File Search upload metadata exceeds 1 MiB").into());
        }

        let start_url = upload_start_url(route, store)?;
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let start = HttpRequest {
            method: "POST".into(),
            url: start_url.to_string(),
            headers: vec![
                (header.clone(), credential.expose_secret().into()),
                ("x-goog-upload-protocol".into(), "resumable".into()),
                ("x-goog-upload-command".into(), "start".into()),
                (
                    "x-goog-upload-header-content-length".into(),
                    request.data.len().to_string(),
                ),
                (
                    "x-goog-upload-header-content-type".into(),
                    request.mime_type.clone(),
                ),
                ("content-type".into(), "application/json".into()),
            ],
            body: metadata_body.into(),
            timeout: deadline.remaining()?,
        };
        let start_response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(start, MAX_RESPONSE)
            .await
            .map_err(|source| GeminiFileSearchError::OutcomeUnknown {
                operation: "upload_to_store",
                source,
            })?;
        if !(200..300).contains(&start_response.status) {
            return Err(provider_error(start_response));
        }
        let response_native = parse_response_body(&start_response.body);
        let upload_url = start_response.header("x-goog-upload-url").ok_or_else(|| {
            bad(
                "resumable upload start response omitted x-goog-upload-url",
                response_native.clone(),
            )
        })?;
        let upload_url = validate_upload_url(upload_url, &start_url)
            .map_err(|message| bad(message, response_native.clone()))?;

        let finalize = HttpRequest {
            method: "POST".into(),
            url: upload_url.to_string(),
            headers: vec![
                ("content-length".into(), request.data.len().to_string()),
                ("x-goog-upload-offset".into(), "0".into()),
                ("x-goog-upload-command".into(), "upload, finalize".into()),
            ],
            body: request.data.clone(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(finalize, MAX_RESPONSE)
            .await
            .map_err(|source| GeminiFileSearchError::OutcomeUnknown {
                operation: "upload_to_store",
                source,
            })?;
        let request_id = request_id(&response);
        let native = success_json(response)?;
        decode_upload_operation(native, profile, route, store, request_id, None)
    }

    async fn get_upload_operation(
        &self,
        reference: &GeminiFileSearchUploadOperationRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchUploadOperation, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(&reference.store, options)?;
        if !valid_operation_id(&reference.operation_id) {
            return Err(invalid("invalid Gemini File Search upload operation ID").into());
        }
        let response = self
            .send(
                route,
                options,
                "GET",
                upload_operation_url(route, reference)?,
                Vec::new(),
                "get_upload_operation",
            )
            .await?;
        let request_id = request_id(&response);
        let native = success_json(response)?;
        decode_upload_operation(
            native,
            profile,
            route,
            &reference.store,
            request_id,
            Some(&reference.operation_id),
        )
    }

    async fn list_documents(
        &self,
        store: &GeminiFileSearchStoreRef,
        page_size: Option<u8>,
        page_token: Option<&str>,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchPage<GeminiFileSearchDocument>, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(store, options)?;
        validate_page(page_size, page_token)?;
        let mut url = documents_collection_url(route, store)?;
        add_page_query(&mut url, page_size, page_token);
        let response = self
            .send(route, options, "GET", url, Vec::new(), "list_documents")
            .await?;
        let native = success_json(response)?;
        let rows = native
            .get("documents")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("document list omitted documents", native.clone()))?;
        let items = rows
            .iter()
            .cloned()
            .map(|row| decode_document(row, profile, route, store, None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GeminiFileSearchPage {
            items,
            next_page_token: page_token_from(&native),
        })
    }

    async fn get_document(
        &self,
        reference: &GeminiFileSearchDocumentRef,
        options: &RequestOptions,
    ) -> Result<GeminiFileSearchDocument, GeminiFileSearchError> {
        let (profile, route) = self.checked_store(&reference.store, options)?;
        if !valid_resource_id(&reference.document_id) {
            return Err(invalid("invalid Gemini File Search document ID").into());
        }
        let response = self
            .send(
                route,
                options,
                "GET",
                document_url(route, reference)?,
                Vec::new(),
                "get_document",
            )
            .await?;
        decode_document(
            success_json(response)?,
            profile,
            route,
            &reference.store,
            Some(&reference.document_id),
        )
    }

    async fn delete_document(
        &self,
        reference: &GeminiFileSearchDocumentRef,
        force: bool,
        options: &RequestOptions,
    ) -> Result<(), GeminiFileSearchError> {
        let (_, route) = self.checked_store(&reference.store, options)?;
        if !valid_resource_id(&reference.document_id) {
            return Err(invalid("invalid Gemini File Search document ID").into());
        }
        let mut url = document_url(route, reference)?;
        if force {
            url.query_pairs_mut().append_pair("force", "true");
        }
        let response = self
            .send(route, options, "DELETE", url, Vec::new(), "delete_document")
            .await?;
        ensure_success(response)
    }

    async fn send(
        &self,
        route: &GeminiFileSearchRoute,
        options: &RequestOptions,
        method: &'static str,
        url: url::Url,
        body: Vec<u8>,
        operation: &'static str,
    ) -> Result<HttpResponse, GeminiFileSearchError> {
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Gemini File Search requires a credential".into(),
            })?;
        let ServiceAuth::ApiKey { header } = &route.auth else {
            return Err(invalid("Gemini File Search requires API key authentication").into());
        };
        if body.len() > MAX_REQUEST {
            return Err(invalid("Gemini File Search request exceeds 1 MiB").into());
        }
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let request = HttpRequest {
            method: method.into(),
            url: url.into(),
            headers: vec![
                (header.clone(), credential.expose_secret().into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: body.into(),
            timeout: deadline.remaining()?,
        };
        HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, MAX_RESPONSE)
            .await
            .map_err(|source| {
                if matches!(
                    operation,
                    "create_store" | "delete_store" | "import_file" | "delete_document"
                ) && matches!(
                    source,
                    LlmError::Transport { .. }
                        | LlmError::TransportTimeout { .. }
                        | LlmError::StreamInterrupted { .. }
                ) {
                    GeminiFileSearchError::OutcomeUnknown { operation, source }
                } else {
                    GeminiFileSearchError::Llm(source)
                }
            })
    }
}

fn decode_store(
    native: Value,
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
    scope: &str,
    expected_id: Option<&str>,
) -> Result<GeminiFileSearchStore, GeminiFileSearchError> {
    let id = native
        .get("name")
        .and_then(Value::as_str)
        .and_then(|name| name.strip_prefix("fileSearchStores/"))
        .filter(|id| valid_resource_id(id))
        .ok_or_else(|| {
            bad(
                "store response omitted a valid resource name",
                native.clone(),
            )
        })?;
    if expected_id.is_some_and(|expected| expected != id) {
        return Err(bad(
            "store response name differs from the requested resource",
            native,
        ));
    }
    Ok(GeminiFileSearchStore {
        reference: store_reference(profile, route, scope, id),
        display_name: string_field(&native, "displayName"),
        embedding_model: string_field(&native, "embeddingModel"),
        create_time: string_field(&native, "createTime"),
        update_time: string_field(&native, "updateTime"),
        active_documents_count: integer_field(&native, "activeDocumentsCount"),
        pending_documents_count: integer_field(&native, "pendingDocumentsCount"),
        failed_documents_count: integer_field(&native, "failedDocumentsCount"),
        size_bytes: integer_field(&native, "sizeBytes"),
        native,
    })
}

fn decode_document(
    native: Value,
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
    expected_id: Option<&str>,
) -> Result<GeminiFileSearchDocument, GeminiFileSearchError> {
    let resource = native
        .get("name")
        .and_then(Value::as_str)
        .and_then(|name| name.strip_prefix("fileSearchStores/"))
        .and_then(|name| name.split_once("/documents/"));
    let Some((store_id, document_id)) = resource else {
        return Err(bad(
            "document response omitted a valid resource name",
            native,
        ));
    };
    if store_id != store.store_id
        || !valid_resource_id(document_id)
        || expected_id.is_some_and(|expected| expected != document_id)
    {
        return Err(bad(
            "document response name differs from its scoped resource",
            native,
        ));
    }
    let state = match string_field(&native, "state").as_deref() {
        None | Some("STATE_UNSPECIFIED") => GeminiFileSearchDocumentState::Unspecified,
        Some("STATE_PENDING") => GeminiFileSearchDocumentState::Pending,
        Some("STATE_ACTIVE") => GeminiFileSearchDocumentState::Active,
        Some("STATE_FAILED") => GeminiFileSearchDocumentState::Failed,
        Some(other) => GeminiFileSearchDocumentState::Other(other.to_owned()),
    };
    let custom_metadata = native
        .get("customMetadata")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(GeminiFileSearchDocument {
        reference: GeminiFileSearchDocumentRef {
            store: store_reference(profile, route, &store.account_scope, &store.store_id),
            document_id: document_id.into(),
        },
        display_name: string_field(&native, "displayName"),
        state,
        create_time: string_field(&native, "createTime"),
        update_time: string_field(&native, "updateTime"),
        size_bytes: integer_field(&native, "sizeBytes"),
        mime_type: string_field(&native, "mimeType"),
        custom_metadata,
        native,
    })
}

fn decode_operation(
    native: Value,
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
    request_id: Option<String>,
    expected_id: Option<&str>,
) -> Result<GeminiFileSearchOperation, GeminiFileSearchError> {
    let name = native
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("operation response omitted name", native.clone()))?;
    let operation_id = operation_id_from_name(name, &store.store_id).ok_or_else(|| {
        bad(
            "operation response name is outside the requested store",
            native.clone(),
        )
    })?;
    if expected_id.is_some_and(|expected| expected != operation_id) {
        return Err(bad(
            "operation response name differs from the requested resource",
            native,
        ));
    }
    let done = native
        .get("done")
        .and_then(Value::as_bool)
        .ok_or_else(|| bad("operation response omitted done", native.clone()))?;
    let error = native.get("error").cloned();
    let response = native.get("response").cloned();
    if done && error.is_some() && response.is_some() {
        return Err(bad(
            "completed operation contains both error and response",
            native,
        ));
    }
    let state = if !done {
        GeminiFileSearchOperationState::Running
    } else if error.is_some() {
        GeminiFileSearchOperationState::Failed
    } else if response.is_some() {
        GeminiFileSearchOperationState::Succeeded
    } else {
        GeminiFileSearchOperationState::DoneWithoutResult
    };
    Ok(GeminiFileSearchOperation {
        reference: GeminiFileSearchOperationRef {
            store: store_reference(profile, route, &store.account_scope, &store.store_id),
            operation_id: operation_id.into(),
        },
        state,
        done,
        metadata: native.get("metadata").cloned(),
        error,
        response,
        native,
        request_id,
    })
}

fn decode_upload_operation(
    native: Value,
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
    request_id: Option<String>,
    expected_id: Option<&str>,
) -> Result<GeminiFileSearchUploadOperation, GeminiFileSearchError> {
    let name = native
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("upload operation response omitted name", native.clone()))?;
    let operation_id = upload_operation_id_from_name(name, &store.store_id).ok_or_else(|| {
        bad(
            "upload operation response name is outside the requested store",
            native.clone(),
        )
    })?;
    if expected_id.is_some_and(|expected| expected != operation_id) {
        return Err(bad(
            "upload operation response name differs from the requested resource",
            native,
        ));
    }
    let done = native
        .get("done")
        .and_then(Value::as_bool)
        .ok_or_else(|| bad("upload operation response omitted done", native.clone()))?;
    let error = native.get("error").cloned();
    let response = native.get("response").cloned();
    if (done && error.is_some() && response.is_some())
        || (!done && (error.is_some() || response.is_some()))
    {
        return Err(bad(
            "upload operation result fields conflict with its done state",
            native,
        ));
    }
    let state = operation_state(done, error.is_some(), response.is_some());
    Ok(GeminiFileSearchUploadOperation {
        reference: GeminiFileSearchUploadOperationRef {
            store: store_reference(profile, route, &store.account_scope, &store.store_id),
            operation_id: operation_id.into(),
        },
        state,
        done,
        metadata: native.get("metadata").cloned(),
        error,
        response,
        native,
        request_id,
    })
}

fn operation_state(
    done: bool,
    has_error: bool,
    has_response: bool,
) -> GeminiFileSearchOperationState {
    if !done {
        GeminiFileSearchOperationState::Running
    } else if has_error {
        GeminiFileSearchOperationState::Failed
    } else if has_response {
        GeminiFileSearchOperationState::Succeeded
    } else {
        GeminiFileSearchOperationState::DoneWithoutResult
    }
}

fn metadata_to_wire(metadata: &GeminiFileSearchMetadata) -> Result<Value, GeminiFileSearchError> {
    let mut object = serde_json::Map::new();
    object.insert("key".into(), Value::String(metadata.key.clone()));
    match &metadata.value {
        GeminiFileSearchMetadataValue::String(text) => {
            object.insert("stringValue".into(), Value::String(text.clone()));
        }
        GeminiFileSearchMetadataValue::StringList(values) => {
            object.insert("stringListValue".into(), json!({"values": values}));
        }
        GeminiFileSearchMetadataValue::Number(number) => {
            let number = serde_json::Number::from_f64(*number)
                .ok_or_else(|| invalid("metadata numeric value must be finite"))?;
            object.insert("numericValue".into(), Value::Number(number));
        }
    }
    Ok(Value::Object(object))
}

fn validate_metadata(metadata: &[GeminiFileSearchMetadata]) -> Result<(), GeminiFileSearchError> {
    if metadata.len() > 20 || metadata.iter().any(|item| item.key.trim().is_empty()) {
        return Err(invalid(
            "Gemini File Search accepts at most 20 metadata entries with nonempty keys",
        )
        .into());
    }
    for item in metadata {
        if let GeminiFileSearchMetadataValue::Number(number) = &item.value {
            if !number.is_finite() {
                return Err(invalid("metadata numeric value must be finite").into());
            }
        }
    }
    Ok(())
}

fn check_gemini_file(
    profile: &ProviderProfile,
    store: &GeminiFileSearchStoreRef,
    file: &ProviderFileRef,
    options: &RequestOptions,
) -> Result<(), GeminiFileSearchError> {
    let expected_scope = options
        .file_account_scope
        .as_deref()
        .or(options.account_scope.as_deref());
    if file.provider_id != profile.provider_id
        || file.profile_name != profile.profile_name
        || file.protocol != ProtocolFamily::GeminiGenerateContent
        || file.endpoint_fingerprint != provider_file_endpoint_fingerprint(&profile.base_url)
        || file.account_scope.as_deref() != expected_scope
        || gemini_file_name(&file.file_id).is_none()
        || store.account_scope != options.account_scope.as_deref().unwrap_or_default()
    {
        return Err(LlmError::PermissionDenied {
            message: "Gemini file belongs to another provider profile, endpoint or account scope"
                .into(),
        }
        .into());
    }
    Ok(())
}

fn store_reference(
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
    scope: &str,
    id: &str,
) -> GeminiFileSearchStoreRef {
    GeminiFileSearchStoreRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
        account_scope: scope.into(),
        store_id: id.into(),
    }
}

fn collection_url(route: &GeminiFileSearchRoute) -> Result<url::Url, GeminiFileSearchError> {
    let mut url = url::Url::parse(&route.endpoint)
        .map_err(|_| invalid("invalid Gemini File Search endpoint"))?;
    let normalized_path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&normalized_path);
    Ok(url)
}

fn store_url(
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(route, &[&store.store_id])
}

fn import_url(
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(route, &[&format!("{}:importFile", store.store_id)])
}

fn operation_url(
    route: &GeminiFileSearchRoute,
    reference: &GeminiFileSearchOperationRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(
        route,
        &[
            &reference.store.store_id,
            "operations",
            &reference.operation_id,
        ],
    )
}

fn upload_operation_url(
    route: &GeminiFileSearchRoute,
    reference: &GeminiFileSearchUploadOperationRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(
        route,
        &[
            &reference.store.store_id,
            "upload",
            "operations",
            &reference.operation_id,
        ],
    )
}

fn upload_start_url(
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
) -> Result<url::Url, GeminiFileSearchError> {
    let mut url = collection_url(route)?;
    let path = url.path().to_owned();
    let prefix = path
        .trim_end_matches('/')
        .strip_suffix("/v1beta/fileSearchStores")
        .ok_or_else(|| invalid("Gemini File Search endpoint has an invalid collection path"))?;
    url.set_path(&format!("{prefix}/upload/v1beta/fileSearchStores"));
    url.path_segments_mut()
        .map_err(|_| invalid("Gemini File Search upload URL cannot accept resource paths"))?
        .push(&format!("{}:uploadToFileSearchStore", store.store_id));
    Ok(url)
}

fn documents_collection_url(
    route: &GeminiFileSearchRoute,
    store: &GeminiFileSearchStoreRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(route, &[&store.store_id, "documents"])
}

fn document_url(
    route: &GeminiFileSearchRoute,
    reference: &GeminiFileSearchDocumentRef,
) -> Result<url::Url, GeminiFileSearchError> {
    append_path(
        route,
        &[
            &reference.store.store_id,
            "documents",
            &reference.document_id,
        ],
    )
}

fn append_path(
    route: &GeminiFileSearchRoute,
    segments: &[&str],
) -> Result<url::Url, GeminiFileSearchError> {
    let mut url = collection_url(route)?;
    let mut path = url
        .path_segments_mut()
        .map_err(|_| invalid("Gemini File Search endpoint cannot accept resource paths"))?;
    for segment in segments {
        path.push(segment);
    }
    drop(path);
    Ok(url)
}

fn validate_page(
    page_size: Option<u8>,
    page_token: Option<&str>,
) -> Result<(), GeminiFileSearchError> {
    if page_size.is_some_and(|size| !(1..=MAX_PAGE_SIZE).contains(&size))
        || page_token.is_some_and(|token| token.is_empty())
    {
        return Err(invalid("Gemini File Search page size or token is invalid").into());
    }
    Ok(())
}

fn add_page_query(url: &mut url::Url, page_size: Option<u8>, page_token: Option<&str>) {
    let mut query = url.query_pairs_mut();
    if let Some(page_size) = page_size {
        query.append_pair("pageSize", &page_size.to_string());
    }
    if let Some(page_token) = page_token {
        query.append_pair("pageToken", page_token);
    }
}

fn page_token_from(native: &Value) -> Option<String> {
    native
        .get("nextPageToken")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

fn encode_body(value: Value) -> Result<Vec<u8>, GeminiFileSearchError> {
    serde_json::to_vec(&value)
        .map_err(|_| invalid("Gemini File Search request cannot be serialized").into())
}

fn success_json(response: HttpResponse) -> Result<Value, GeminiFileSearchError> {
    let request_id = request_id(&response);
    let native = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    if !(200..300).contains(&response.status) {
        return Err(GeminiFileSearchError::Provider {
            status: response.status,
            request_id,
            body: native,
        });
    }
    if !native.is_object() {
        return Err(bad("success response is not a JSON object", native));
    }
    Ok(native)
}

fn ensure_success(response: HttpResponse) -> Result<(), GeminiFileSearchError> {
    if !(200..300).contains(&response.status) {
        return Err(provider_error(response));
    }
    Ok(())
}

fn provider_error(response: HttpResponse) -> GeminiFileSearchError {
    GeminiFileSearchError::Provider {
        status: response.status,
        request_id: request_id(&response),
        body: parse_response_body(&response.body),
    }
}

fn parse_response_body(body: &[u8]) -> Value {
    serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()))
}

fn request_id(response: &HttpResponse) -> Option<String> {
    response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .or_else(|| response.header("x-goog-request-id"))
        .map(str::to_owned)
}

fn operation_id_from_name<'a>(name: &'a str, store_id: &str) -> Option<&'a str> {
    let prefix = format!("fileSearchStores/{store_id}/operations/");
    let id = name.strip_prefix(&prefix)?;
    valid_operation_id(id).then_some(id)
}

fn upload_operation_id_from_name<'a>(name: &'a str, store_id: &str) -> Option<&'a str> {
    let prefix = format!("fileSearchStores/{store_id}/upload/operations/");
    let id = name.strip_prefix(&prefix)?;
    valid_operation_id(id).then_some(id)
}

fn validate_upload_request(
    request: &GeminiFileSearchUploadRequest,
) -> Result<(), GeminiFileSearchError> {
    if request.data.len() > MAX_DIRECT_UPLOAD
        || request.mime_type.len() > 256
        || !request.mime_type.contains('/')
        || request.mime_type.chars().any(char::is_control)
        || request
            .display_name
            .as_ref()
            .is_some_and(|name| name.chars().count() > 512)
    {
        return Err(invalid(
            "direct File Search upload requires a valid MIME type and a payload no larger than 100 MiB",
        )
        .into());
    }
    Ok(())
}

fn validate_upload_url(
    upload_url: &str,
    expected_origin: &url::Url,
) -> Result<url::Url, &'static str> {
    let url = url::Url::parse(upload_url).map_err(|_| "resumable upload URL is invalid")?;
    if url.origin() != expected_origin.origin()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.path().is_empty()
    {
        return Err("resumable upload URL has an unexpected origin or credentials");
    }
    Ok(url)
}

fn valid_resource_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 40
        && !id.starts_with('-')
        && !id.ends_with('-')
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn valid_operation_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn gemini_file_name(id: &str) -> Option<String> {
    let bare_id = id.strip_prefix("files/").unwrap_or(id);
    (!bare_id.is_empty()
        && bare_id.len() <= 256
        && bare_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')))
    .then(|| format!("files/{bare_id}"))
}

fn valid_model_resource(model: &str) -> bool {
    let Some(id) = model.strip_prefix("models/") else {
        return false;
    };
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn integer_field(native: &Value, name: &str) -> Option<u64> {
    native.get(name).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
    })
}

fn string_field(native: &Value, name: &str) -> Option<String> {
    native.get(name).and_then(Value::as_str).map(str::to_owned)
}

/// Validate a Google File Search route before publishing it in a profile.
pub fn validate_route(
    profile: &ProviderProfile,
    route: &GeminiFileSearchRoute,
) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "google"
        || profile.protocol != ProtocolFamily::GeminiGenerateContent
        || route.auth
            != (ServiceAuth::ApiKey {
                header: "x-goog-api-key".into(),
            })
    {
        return Err(invalid(
            "Gemini File Search requires Google identity and x-goog-api-key authentication",
        ));
    }
    let url = url::Url::parse(&route.endpoint)
        .map_err(|_| invalid("invalid Gemini File Search endpoint"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url
            .path()
            .trim_end_matches('/')
            .ends_with("/v1beta/fileSearchStores")
    {
        return Err(invalid(
            "Gemini File Search endpoint must be a full v1beta/fileSearchStores collection URL",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

fn bad(message: &str, native: Value) -> GeminiFileSearchError {
    GeminiFileSearchError::InvalidResponse {
        message: message.into(),
        native,
    }
}
