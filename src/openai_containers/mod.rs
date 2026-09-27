//! Scoped access to OpenAI Containers and their files.
//!
//! Each method performs one documented API operation. Container and file
//! references are bound to one OpenAI profile and account scope. Mutations are
//! never retried; uncertain outcomes are surfaced explicitly.

#![doc = concat!(
    include_str!("../../docs/openai-containers.md"),
    "\n\n",
    include_str!("../../docs/openai-containers.en.md")
)]

use crate::{
    files::{
        multipart_boundary, provider_file_endpoint_fingerprint, validate_media_type, UploadFile,
    },
    protocol::{LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use url::Url;

const API_BASE_URL: &str = "https://api.openai.com/v1";
const MAX_CONTROL_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const FILE_COPY_EXPIRY_MARGIN_SECONDS: u64 = 60;

/// Identity binding container and file references to one OpenAI connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiContainerScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
}

impl OpenAiContainerScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, OpenAiContainersError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid("profile name and account scope must be non-empty"));
        }
        Ok(Self {
            provider_id: ProviderId::from("openai"),
            profile_name,
            account_scope,
            endpoint_fingerprint: provider_file_endpoint_fingerprint(API_BASE_URL),
        })
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

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), OpenAiContainersError> {
        if self.provider_id.as_str() != "openai"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(API_BASE_URL)
        {
            return Err(invalid("OpenAI Containers scope identity is invalid"));
        }
        Ok(())
    }
}

/// Opaque identity for an OpenAI Container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiContainerRef {
    scope: OpenAiContainerScope,
    container_id: String,
}

impl OpenAiContainerRef {
    /// Bind a container ID returned by an earlier Responses call to this
    /// OpenAI profile and account scope.
    pub fn from_id(
        scope: &OpenAiContainerScope,
        container_id: impl AsRef<str>,
    ) -> Result<Self, OpenAiContainersError> {
        scope.validate()?;
        let container_id = container_id.as_ref();
        if !valid_resource_id(container_id) {
            return Err(invalid("container ID must be a non-empty identifier"));
        }
        Ok(Self {
            scope: scope.clone(),
            container_id: container_id.to_owned(),
        })
    }

    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    pub fn scope(&self) -> &OpenAiContainerScope {
        &self.scope
    }
}

/// Opaque identity for a file stored inside a scoped Container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiContainerFileRef {
    container: OpenAiContainerRef,
    file_id: String,
}

impl OpenAiContainerFileRef {
    pub fn from_id(
        container: &OpenAiContainerRef,
        file_id: impl AsRef<str>,
    ) -> Result<Self, OpenAiContainersError> {
        container.scope.validate()?;
        if !valid_resource_id(&container.container_id) {
            return Err(invalid("container ID must be a non-empty identifier"));
        }
        let file_id = file_id.as_ref();
        if !valid_resource_id(file_id) {
            return Err(invalid("container file ID must be a non-empty identifier"));
        }
        Ok(Self {
            container: container.clone(),
            file_id: file_id.to_owned(),
        })
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn container(&self) -> &OpenAiContainerRef {
        &self.container
    }
}

/// Memory tier accepted by the explicit Containers API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiContainerMemoryLimit {
    #[serde(rename = "1g")]
    G1,
    #[serde(rename = "4g")]
    G4,
    #[serde(rename = "16g")]
    G16,
    #[serde(rename = "64g")]
    G64,
}

impl OpenAiContainerMemoryLimit {
    fn as_str(self) -> &'static str {
        match self {
            Self::G1 => "1g",
            Self::G4 => "4g",
            Self::G16 => "16g",
            Self::G64 => "64g",
        }
    }
}

/// Explicit container creation settings supported by this service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiContainerCreateRequest {
    name: String,
    memory_limit: Option<OpenAiContainerMemoryLimit>,
    expiration_minutes: Option<u32>,
    file_ids: Vec<String>,
}

impl OpenAiContainerCreateRequest {
    pub fn new(name: impl Into<String>) -> Result<Self, OpenAiContainersError> {
        let name = name.into();
        if !valid_identity(&name) {
            return Err(invalid(
                "container name must be non-empty and contain no control characters",
            ));
        }
        Ok(Self {
            name,
            memory_limit: None,
            expiration_minutes: None,
            file_ids: Vec::new(),
        })
    }

    pub fn with_memory_limit(mut self, memory_limit: OpenAiContainerMemoryLimit) -> Self {
        self.memory_limit = Some(memory_limit);
        self
    }

    /// Expire the container this many minutes after its last activity.
    pub fn with_expiration_minutes(mut self, minutes: u32) -> Result<Self, OpenAiContainersError> {
        if minutes == 0 {
            return Err(invalid(
                "container expiration minutes must be greater than zero",
            ));
        }
        self.expiration_minutes = Some(minutes);
        Ok(self)
    }

    /// Copy existing OpenAI Files API resources into the new container.
    pub fn with_file_ids<I, S>(mut self, file_ids: I) -> Result<Self, OpenAiContainersError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let file_ids = file_ids.into_iter().map(Into::into).collect::<Vec<_>>();
        if file_ids.iter().any(|id| !valid_resource_id(id)) {
            return Err(invalid("file IDs must be non-empty path-safe identifiers"));
        }
        self.file_ids = file_ids;
        Ok(self)
    }

    fn encode(&self) -> Result<Bytes, OpenAiContainersError> {
        let mut body = Map::new();
        body.insert("name".into(), Value::String(self.name.clone()));
        if let Some(memory_limit) = self.memory_limit {
            body.insert(
                "memory_limit".into(),
                Value::String(memory_limit.as_str().into()),
            );
        }
        if let Some(minutes) = self.expiration_minutes {
            body.insert(
                "expires_after".into(),
                json!({ "anchor": "last_active_at", "minutes": minutes }),
            );
        }
        if !self.file_ids.is_empty() {
            body.insert("file_ids".into(), json!(self.file_ids));
        }
        serde_json::to_vec(&Value::Object(body))
            .map(Bytes::from)
            .map_err(|_| invalid("container create request could not be encoded"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiContainerOrder {
    Asc,
    Desc,
}

/// One explicit page request. Pass the returned cursor to a later call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiContainerListOptions {
    pub limit: Option<u32>,
    pub after: Option<String>,
    pub order: Option<OpenAiContainerOrder>,
    pub name: Option<String>,
}

impl OpenAiContainerListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limit(mut self, limit: u32) -> Result<Self, OpenAiContainersError> {
        if !(1..=100).contains(&limit) {
            return Err(invalid("container list limit must be between 1 and 100"));
        }
        self.limit = Some(limit);
        Ok(self)
    }

    pub fn with_after(mut self, after: impl Into<String>) -> Result<Self, OpenAiContainersError> {
        let after = after.into();
        if !valid_resource_id(&after) {
            return Err(invalid(
                "container list cursor must be a non-empty identifier",
            ));
        }
        self.after = Some(after);
        Ok(self)
    }

    pub fn with_order(mut self, order: OpenAiContainerOrder) -> Self {
        self.order = Some(order);
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, OpenAiContainersError> {
        let name = name.into();
        if !valid_identity(&name) {
            return Err(invalid("container name filter must be non-empty"));
        }
        self.name = Some(name);
        Ok(self)
    }
}

/// One page from `GET /v1/containers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiContainerPage {
    pub containers: Vec<OpenAiContainer>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
    pub native: Value,
}

/// One container's decoded fields plus its native provider object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiContainer {
    pub reference: OpenAiContainerRef,
    pub name: Option<String>,
    pub status: Option<String>,
    pub created_at: Option<u64>,
    pub last_active_at: Option<u64>,
    pub memory_limit: Option<String>,
    pub expires_after: Option<Value>,
    pub network_policy: Option<Value>,
    pub native: Value,
}

/// One page request for files in a scoped container.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiContainerFileListOptions {
    pub limit: Option<u32>,
    pub after: Option<String>,
    pub order: Option<OpenAiContainerOrder>,
}

impl OpenAiContainerFileListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limit(mut self, limit: u32) -> Result<Self, OpenAiContainersError> {
        if !(1..=100).contains(&limit) {
            return Err(invalid(
                "container file list limit must be between 1 and 100",
            ));
        }
        self.limit = Some(limit);
        Ok(self)
    }

    pub fn with_after(mut self, after: impl Into<String>) -> Result<Self, OpenAiContainersError> {
        let after = after.into();
        if !valid_resource_id(&after) {
            return Err(invalid(
                "container file cursor must be a non-empty identifier",
            ));
        }
        self.after = Some(after);
        Ok(self)
    }

    pub fn with_order(mut self, order: OpenAiContainerOrder) -> Self {
        self.order = Some(order);
        self
    }
}

/// One page from `GET /v1/containers/{container_id}/files`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiContainerFilePage {
    pub files: Vec<OpenAiContainerFile>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
    pub native: Value,
}

/// Metadata for a file stored in a container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiContainerFile {
    pub reference: OpenAiContainerFileRef,
    pub bytes: Option<u64>,
    pub created_at: Option<u64>,
    pub path: Option<String>,
    pub source: Option<String>,
    pub native: Value,
}

/// Confirmation returned by container and file delete operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiContainerDeleteReceipt {
    pub id: String,
    pub deleted: bool,
    pub native: Value,
}

/// Stream for raw container file content.
pub struct OpenAiContainerFileContent {
    pub media_type: Option<String>,
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl Stream for OpenAiContainerFileContent {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().body.as_mut().poll_next(cx)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiContainersError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid OpenAI Containers input: {0}")]
    InvalidInput(String),
    #[error("invalid OpenAI Containers response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("OpenAI Containers API returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of OpenAI Containers {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        container: Option<Box<OpenAiContainerRef>>,
        file_id: Option<String>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of OpenAI Containers {operation} is unknown: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        container: Option<Box<OpenAiContainerRef>>,
        file_id: Option<String>,
        reason: String,
    },
}

/// Direct client for OpenAI's explicit Container lifecycle and file endpoints.
/// The host supplies credentials per operation so it retains refresh ownership.
pub struct OpenAiContainersService<'a> {
    http: &'a dyn Transport,
    scope: OpenAiContainerScope,
}

impl<'a> OpenAiContainersService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        scope: OpenAiContainerScope,
    ) -> Result<Self, OpenAiContainersError> {
        scope.validate()?;
        Ok(Self { http, scope })
    }

    pub fn scope(&self) -> &OpenAiContainerScope {
        &self.scope
    }

    /// Create one explicit container. The call is sent once; an uncertain
    /// response is reported rather than repeated.
    pub async fn create_container(
        &self,
        request: &OpenAiContainerCreateRequest,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainer, OpenAiContainersError> {
        validate_credential(credential)?;
        let body = request.encode()?;
        for file_id in &request.file_ids {
            self.preflight_source_file(file_id, credential).await?;
        }
        let response = self
            .send(
                "POST",
                "/containers",
                Some(body),
                Some("application/json"),
                credential,
            )
            .await
            .map_err(|source| unknown("create_container", None, None, source))?;
        ensure_mutation_success(&response, "create_container", None, None)?;
        decode_container(
            &self.scope,
            decode_json(&response.body).map_err(|error| {
                unknown_response("create_container", None, None, error.to_string())
            })?,
        )
        .map_err(|error| unknown_response("create_container", None, None, error.to_string()))
    }

    /// Retrieve a scoped container.
    pub async fn get_container(
        &self,
        reference: &OpenAiContainerRef,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainer, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_container_ref(reference)?;
        let response = self
            .send(
                "GET",
                &format!("/containers/{}", path_segment(&reference.container_id)),
                None,
                None,
                credential,
            )
            .await?;
        ensure_success(&response)?;
        let container = decode_container(&self.scope, decode_json(&response.body)?)?;
        ensure_same_container(reference, &container.reference)?;
        Ok(container)
    }

    /// Fetch one page from the Containers list. Pagination remains caller-owned.
    pub async fn list_containers(
        &self,
        options: &OpenAiContainerListOptions,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerPage, OpenAiContainersError> {
        validate_credential(credential)?;
        let mut url = self.url("/containers")?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(limit) = options.limit {
                validate_limit(limit)?;
                query.append_pair("limit", &limit.to_string());
            }
            if let Some(after) = options.after.as_deref() {
                validate_cursor(after)?;
                query.append_pair("after", after);
            }
            if let Some(order) = options.order {
                query.append_pair("order", order.as_str());
            }
            if let Some(name) = options.name.as_deref() {
                if !valid_identity(name) {
                    return Err(invalid("container name filter must be non-empty"));
                }
                query.append_pair("name", name);
            }
        }
        let response = self.send_url("GET", url, None, None, credential).await?;
        ensure_success(&response)?;
        let native = decode_json(&response.body)?;
        let object = native.as_object().ok_or_else(|| {
            invalid_response("container list response must be an object", native.clone())
        })?;
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid_response("container list response omitted data", native.clone())
            })?;
        let containers = data
            .iter()
            .cloned()
            .map(|item| decode_container(&self.scope, item))
            .collect::<Result<Vec<_>, _>>()?;
        let (has_more, next_cursor) = decode_page_cursor(object, &native)?;
        Ok(OpenAiContainerPage {
            containers,
            has_more,
            next_cursor,
            native,
        })
    }

    /// Delete a container and verify the documented deletion receipt.
    pub async fn delete_container(
        &self,
        reference: &OpenAiContainerRef,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerDeleteReceipt, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_container_ref(reference)?;
        let response = self
            .send(
                "DELETE",
                &format!("/containers/{}", path_segment(&reference.container_id)),
                None,
                None,
                credential,
            )
            .await
            .map_err(|source| unknown("delete_container", Some(reference), None, source))?;
        ensure_mutation_success(&response, "delete_container", Some(reference), None)?;
        decode_delete_receipt(&response.body, &reference.container_id).map_err(|error| {
            unknown_response("delete_container", Some(reference), None, error.to_string())
        })
    }

    /// Upload raw bytes to a container with the documented multipart `file` part.
    pub async fn upload_file(
        &self,
        container: &OpenAiContainerRef,
        file: &UploadFile,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerFile, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_container_ref(container)?;
        validate_media_type(&file.media_type)?;
        validate_upload_filename(&file.filename)?;
        let boundary = multipart_boundary();
        let filename = quote_multipart_filename(&file.filename);
        let mut body = BytesMut::new();
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {}\r\n\r\n",
                file.media_type
            )
            .as_bytes(),
        );
        body.extend_from_slice(&file.bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let path = format!(
            "/containers/{}/files",
            path_segment(&container.container_id)
        );
        let content_type = format!("multipart/form-data; boundary={boundary}");
        let response = self
            .send(
                "POST",
                &path,
                Some(body.freeze()),
                Some(&content_type),
                credential,
            )
            .await
            .map_err(|source| unknown("upload_file", Some(container), None, source))?;
        ensure_mutation_success(&response, "upload_file", Some(container), None)?;
        decode_container_file(
            &self.scope,
            container,
            decode_json(&response.body).map_err(|error| {
                unknown_response("upload_file", Some(container), None, error.to_string())
            })?,
        )
        .map_err(|error| unknown_response("upload_file", Some(container), None, error.to_string()))
    }

    /// Add an existing OpenAI Files API object to a container by its file ID.
    pub async fn attach_file(
        &self,
        container: &OpenAiContainerRef,
        file_id: impl AsRef<str>,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerFile, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_container_ref(container)?;
        let file_id = file_id.as_ref();
        if !valid_resource_id(file_id) {
            return Err(invalid("OpenAI file ID must be a non-empty identifier"));
        }
        self.preflight_source_file(file_id, credential).await?;
        let body = Bytes::from(
            serde_json::to_vec(&json!({ "file_id": file_id }))
                .map_err(|_| invalid("container file request could not be encoded"))?,
        );
        let path = format!(
            "/containers/{}/files",
            path_segment(&container.container_id)
        );
        let response = self
            .send(
                "POST",
                &path,
                Some(body),
                Some("application/json"),
                credential,
            )
            .await
            .map_err(|source| unknown("attach_file", Some(container), None, source))?;
        ensure_mutation_success(&response, "attach_file", Some(container), None)?;
        decode_container_file(
            &self.scope,
            container,
            decode_json(&response.body).map_err(|error| {
                unknown_response("attach_file", Some(container), None, error.to_string())
            })?,
        )
        .map_err(|error| unknown_response("attach_file", Some(container), None, error.to_string()))
    }

    /// Fetch one page of files stored in a scoped container.
    pub async fn list_files(
        &self,
        container: &OpenAiContainerRef,
        options: &OpenAiContainerFileListOptions,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerFilePage, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_container_ref(container)?;
        let mut url = self.url(&format!(
            "/containers/{}/files",
            path_segment(&container.container_id)
        ))?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(limit) = options.limit {
                validate_limit(limit)?;
                query.append_pair("limit", &limit.to_string());
            }
            if let Some(after) = options.after.as_deref() {
                validate_cursor(after)?;
                query.append_pair("after", after);
            }
            if let Some(order) = options.order {
                query.append_pair("order", order.as_str());
            }
        }
        let response = self.send_url("GET", url, None, None, credential).await?;
        ensure_success(&response)?;
        let native = decode_json(&response.body)?;
        let object = native.as_object().ok_or_else(|| {
            invalid_response(
                "container file list response must be an object",
                native.clone(),
            )
        })?;
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid_response("container file list response omitted data", native.clone())
            })?;
        let files = data
            .iter()
            .cloned()
            .map(|item| decode_container_file(&self.scope, container, item))
            .collect::<Result<Vec<_>, _>>()?;
        let (has_more, next_cursor) = decode_page_cursor(object, &native)?;
        Ok(OpenAiContainerFilePage {
            files,
            has_more,
            next_cursor,
            native,
        })
    }

    /// Retrieve metadata for one file and verify it belongs to the referenced container.
    pub async fn get_file(
        &self,
        reference: &OpenAiContainerFileRef,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerFile, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_file_ref(reference)?;
        let response = self
            .send("GET", &file_path(reference), None, None, credential)
            .await?;
        ensure_success(&response)?;
        decode_container_file(
            &self.scope,
            &reference.container,
            decode_json(&response.body)?,
        )
        .and_then(|file| {
            if file.reference.file_id == reference.file_id {
                Ok(file)
            } else {
                Err(invalid_response(
                    "OpenAI returned a different container file ID",
                    file.native,
                ))
            }
        })
    }

    /// Delete one file and verify its documented deletion receipt.
    pub async fn delete_file(
        &self,
        reference: &OpenAiContainerFileRef,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerDeleteReceipt, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_file_ref(reference)?;
        let response = self
            .send("DELETE", &file_path(reference), None, None, credential)
            .await
            .map_err(|source| {
                unknown(
                    "delete_file",
                    Some(&reference.container),
                    Some(&reference.file_id),
                    source,
                )
            })?;
        ensure_mutation_success(
            &response,
            "delete_file",
            Some(&reference.container),
            Some(&reference.file_id),
        )?;
        decode_delete_receipt(&response.body, &reference.file_id).map_err(|error| {
            unknown_response(
                "delete_file",
                Some(&reference.container),
                Some(&reference.file_id),
                error.to_string(),
            )
        })
    }

    /// Stream raw bytes from the documented container file content endpoint.
    pub async fn download_file(
        &self,
        reference: &OpenAiContainerFileRef,
        credential: &Secret<String>,
    ) -> Result<OpenAiContainerFileContent, OpenAiContainersError> {
        validate_credential(credential)?;
        self.validate_file_ref(reference)?;
        let request = HttpRequest {
            method: "GET".into(),
            url: self
                .url(&format!("{}/content", file_path(reference)))?
                .into(),
            headers: Self::authorization_headers(credential),
            body: Bytes::new(),
            timeout: None,
        };
        let response = HttpExecutor::new(self.http).send(request).await?;
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_CONTROL_RESPONSE_BYTES)).await?;
            ensure_success(&response)?;
            return Err(invalid_response(
                "non-success content response was not rejected",
                Value::Null,
            ));
        }
        Ok(OpenAiContainerFileContent {
            media_type: response
                .header("content-type")
                .map(|value| value.split(';').next().unwrap_or(value).trim().to_owned()),
            body: response.body,
        })
    }

    fn validate_container_ref(
        &self,
        reference: &OpenAiContainerRef,
    ) -> Result<(), OpenAiContainersError> {
        if reference.scope != self.scope || !valid_resource_id(&reference.container_id) {
            return Err(scope_mismatch("container"));
        }
        Ok(())
    }

    fn validate_file_ref(
        &self,
        reference: &OpenAiContainerFileRef,
    ) -> Result<(), OpenAiContainersError> {
        self.validate_container_ref(&reference.container)?;
        if !valid_resource_id(&reference.file_id) {
            return Err(scope_mismatch("container file"));
        }
        Ok(())
    }

    /// Confirm an existing Files API object is visible to the current
    /// credential and has not reached its configured expiry before copying it
    /// into a container. This read happens before the container mutation.
    async fn preflight_source_file(
        &self,
        file_id: &str,
        credential: &Secret<String>,
    ) -> Result<(), OpenAiContainersError> {
        if !valid_resource_id(file_id) {
            return Err(invalid("OpenAI file ID must be a non-empty identifier"));
        }
        let response = self
            .send(
                "GET",
                &format!("/files/{}", path_segment(file_id)),
                None,
                None,
                credential,
            )
            .await?;
        ensure_success(&response)?;
        let native = decode_json(&response.body)?;
        let object = native.as_object().ok_or_else(|| {
            invalid_response("Files API metadata must be a JSON object", native.clone())
        })?;
        if object.get("id").and_then(Value::as_str) != Some(file_id) {
            return Err(invalid_response(
                "Files API metadata returned a different file ID",
                native,
            ));
        }
        if let Some(expires_at) = object.get("expires_at").filter(|value| !value.is_null()) {
            let expires_at = expires_at.as_u64().ok_or_else(|| {
                invalid_response(
                    "Files API expires_at must be a Unix timestamp",
                    native.clone(),
                )
            })?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| invalid("system clock is before the Unix epoch"))?
                .as_secs();
            if expires_at <= now.saturating_add(FILE_COPY_EXPIRY_MARGIN_SECONDS) {
                return Err(invalid(
                    "source Files API object is expired or expires too soon to copy",
                ));
            }
        }
        Ok(())
    }

    async fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Bytes>,
        content_type: Option<&str>,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, LlmError> {
        let url = self.url(path)?;
        self.send_url(method, url, body, content_type, credential)
            .await
    }

    async fn send_url(
        &self,
        method: &str,
        url: Url,
        body: Option<Bytes>,
        content_type: Option<&str>,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, LlmError> {
        let mut headers = Self::authorization_headers(credential);
        if let Some(content_type) = content_type {
            headers.push(("Content-Type".into(), content_type.to_owned()));
        }
        HttpExecutor::new(self.http)
            .execute_bounded(
                HttpRequest {
                    method: method.into(),
                    url: url.into(),
                    headers,
                    body: body.unwrap_or_default(),
                    timeout: None,
                },
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
    }

    fn authorization_headers(credential: &Secret<String>) -> Vec<(String, String)> {
        vec![(
            "Authorization".into(),
            format!("Bearer {}", credential.expose_secret()),
        )]
    }

    fn url(&self, path: &str) -> Result<Url, LlmError> {
        Url::parse(&format!("{API_BASE_URL}{path}")).map_err(|_| LlmError::InvalidRequest {
            message: "OpenAI Containers URL could not be constructed".into(),
        })
    }
}

fn decode_container(
    scope: &OpenAiContainerScope,
    native: Value,
) -> Result<OpenAiContainer, OpenAiContainersError> {
    let object = native
        .as_object()
        .ok_or_else(|| invalid_response("container must be a JSON object", native.clone()))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| valid_resource_id(id))
        .ok_or_else(|| invalid_response("container response omitted a valid id", native.clone()))?;
    Ok(OpenAiContainer {
        reference: OpenAiContainerRef {
            scope: scope.clone(),
            container_id: id.to_owned(),
        },
        name: object
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        status: object
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_owned),
        created_at: object.get("created_at").and_then(Value::as_u64),
        last_active_at: object.get("last_active_at").and_then(Value::as_u64),
        memory_limit: object
            .get("memory_limit")
            .and_then(Value::as_str)
            .map(str::to_owned),
        expires_after: object.get("expires_after").cloned(),
        network_policy: object.get("network_policy").cloned(),
        native,
    })
}

fn decode_container_file(
    scope: &OpenAiContainerScope,
    expected_container: &OpenAiContainerRef,
    native: Value,
) -> Result<OpenAiContainerFile, OpenAiContainersError> {
    let object = native
        .as_object()
        .ok_or_else(|| invalid_response("container file must be a JSON object", native.clone()))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| valid_resource_id(id))
        .ok_or_else(|| {
            invalid_response("container file response omitted a valid id", native.clone())
        })?;
    let container_id = object
        .get("container_id")
        .and_then(Value::as_str)
        .filter(|id| valid_resource_id(id))
        .ok_or_else(|| {
            invalid_response(
                "container file response omitted container_id",
                native.clone(),
            )
        })?;
    if container_id != expected_container.container_id || &expected_container.scope != scope {
        return Err(invalid_response(
            "container file response belongs to a different container or scope",
            native,
        ));
    }
    Ok(OpenAiContainerFile {
        reference: OpenAiContainerFileRef {
            container: expected_container.clone(),
            file_id: id.to_owned(),
        },
        bytes: object.get("bytes").and_then(Value::as_u64),
        created_at: object.get("created_at").and_then(Value::as_u64),
        path: object
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_owned),
        source: object
            .get("source")
            .and_then(Value::as_str)
            .map(str::to_owned),
        native,
    })
}

fn decode_delete_receipt(
    body: &[u8],
    expected_id: &str,
) -> Result<OpenAiContainerDeleteReceipt, OpenAiContainersError> {
    let native = decode_json(body)?;
    let object = native
        .as_object()
        .ok_or_else(|| invalid_response("delete response must be a JSON object", native.clone()))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| *id == expected_id)
        .ok_or_else(|| {
            invalid_response(
                "delete response ID differed from the requested resource",
                native.clone(),
            )
        })?;
    let deleted = object
        .get("deleted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !deleted {
        return Err(invalid_response(
            "delete response did not confirm deletion",
            native,
        ));
    }
    Ok(OpenAiContainerDeleteReceipt {
        id: id.to_owned(),
        deleted,
        native,
    })
}

fn ensure_success(response: &HttpResponse) -> Result<(), OpenAiContainersError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let body = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    Err(OpenAiContainersError::Provider {
        status: response.status,
        request_id: response
            .header("x-request-id")
            .or_else(|| response.header("openai-request-id"))
            .map(str::to_owned),
        body,
    })
}

fn ensure_mutation_success(
    response: &HttpResponse,
    operation: &'static str,
    container: Option<&OpenAiContainerRef>,
    file_id: Option<&str>,
) -> Result<(), OpenAiContainersError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    if response.status >= 500 {
        return Err(unknown_response(
            operation,
            container,
            file_id,
            format!("OpenAI returned HTTP {} during a mutation", response.status),
        ));
    }
    ensure_success(response)
}

fn decode_json(bytes: &[u8]) -> Result<Value, OpenAiContainersError> {
    serde_json::from_slice(bytes).map_err(|error| {
        invalid_response(format!("response JSON is invalid: {error}"), Value::Null)
    })
}

fn decode_page_cursor(
    object: &Map<String, Value>,
    native: &Value,
) -> Result<(bool, Option<String>), OpenAiContainersError> {
    let has_more = object
        .get("has_more")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid_response("list response omitted has_more", native.clone()))?;
    let next_cursor = if has_more {
        let cursor = object
            .get("last_id")
            .and_then(Value::as_str)
            .filter(|cursor| valid_resource_id(cursor))
            .ok_or_else(|| {
                invalid_response("list response omitted a valid last_id", native.clone())
            })?;
        Some(cursor.to_owned())
    } else {
        None
    };
    Ok((has_more, next_cursor))
}

fn file_path(reference: &OpenAiContainerFileRef) -> String {
    format!(
        "/containers/{}/files/{}",
        path_segment(&reference.container.container_id),
        path_segment(&reference.file_id)
    )
}

fn path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_resource_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= 512
        && !value.chars().any(char::is_control)
        && !value.contains('/')
        && !value.contains('\\')
}

fn validate_cursor(cursor: &str) -> Result<(), OpenAiContainersError> {
    if valid_resource_id(cursor) {
        Ok(())
    } else {
        Err(invalid("pagination cursor must be a non-empty identifier"))
    }
}

fn validate_limit(limit: u32) -> Result<(), OpenAiContainersError> {
    if (1..=100).contains(&limit) {
        Ok(())
    } else {
        Err(invalid("list limit must be between 1 and 100"))
    }
}

fn validate_upload_filename(filename: &str) -> Result<(), OpenAiContainersError> {
    if filename.trim().is_empty() || filename.chars().any(char::is_control) {
        return Err(invalid(
            "upload filename must be non-empty and contain no control characters",
        ));
    }
    Ok(())
}

fn quote_multipart_filename(filename: &str) -> String {
    filename.replace('\\', "\\\\").replace('"', "\\\"")
}

fn scope_mismatch(resource: &str) -> OpenAiContainersError {
    LlmError::PermissionDenied {
        message: format!(
            "OpenAI {resource} reference belongs to another profile, endpoint, or account scope"
        ),
    }
    .into()
}

fn ensure_same_container(
    expected: &OpenAiContainerRef,
    actual: &OpenAiContainerRef,
) -> Result<(), OpenAiContainersError> {
    if expected == actual {
        Ok(())
    } else {
        Err(invalid_response(
            "OpenAI returned a different container ID",
            Value::Null,
        ))
    }
}

fn unknown(
    operation: &'static str,
    container: Option<&OpenAiContainerRef>,
    file_id: Option<&str>,
    source: LlmError,
) -> OpenAiContainersError {
    OpenAiContainersError::OutcomeUnknown {
        operation,
        container: container.cloned().map(Box::new),
        file_id: file_id.map(str::to_owned),
        source,
    }
}

fn unknown_response(
    operation: &'static str,
    container: Option<&OpenAiContainerRef>,
    file_id: Option<&str>,
    reason: String,
) -> OpenAiContainersError {
    OpenAiContainersError::OutcomeUnknownResponse {
        operation,
        container: container.cloned().map(Box::new),
        file_id: file_id.map(str::to_owned),
        reason,
    }
}

fn invalid(message: impl Into<String>) -> OpenAiContainersError {
    OpenAiContainersError::InvalidInput(message.into())
}

fn validate_credential(credential: &Secret<String>) -> Result<(), OpenAiContainersError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || value.contains('\r') || value.contains('\n') {
        return Err(invalid("OpenAI API key must be non-empty and well-formed"));
    }
    Ok(())
}

fn invalid_response(message: impl Into<String>, native: Value) -> OpenAiContainersError {
    OpenAiContainersError::InvalidResponse {
        message: message.into(),
        native,
    }
}

impl OpenAiContainerOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}
