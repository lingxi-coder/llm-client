//! Provider file APIs, kept separate from durable application attachments.
//!
//! A [`ProviderFileRef`] belongs to one concrete profile/account. It may be
//! useful as model input for a particular wire, but it is never a display URL
//! and must not replace the app-owned attachment in conversation history.

use crate::auth::Authenticator;
use crate::transport::{HttpRequest, HttpResponse, Transport};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use lingxi_agent_api::protocol::{
    AuthStrategy, LlmError, ModelProfile, ProtocolFamily, ProviderFileSource, ProviderId,
    ProviderProfile, Secret,
};
use serde_json::Value;
use std::time::{Duration, Instant};

/// Maximum response body retained by [`FileService::download`].
pub const MAX_PROVIDER_FILE_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;

const FILE_TIMEOUT: Duration = Duration::from_secs(120);
const GEMINI_FILE_POLL_INTERVAL: Duration = Duration::from_secs(1);
const GEMINI_PDF_MAX_UPLOAD_BYTES: u64 = 50_000_000;
const OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES: u64 = 50_000_000;
/// Automatic uploads do not expose provider IDs to callers. Providers that
/// support an upload TTL receive one, and cached references expire earlier.
pub(crate) const AUTOMATIC_FILE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub(crate) const AUTOMATIC_FILE_CACHE_TTL: Duration = Duration::from_secs(23 * 60 * 60);
/// The Anthropic request preflight reserves this many encoded bytes per
/// automatic file ID before the upload happens.
pub(crate) const MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES: usize = 512;

pub(crate) fn automatic_file_cache_ttl(profile: &ProviderProfile) -> Option<Duration> {
    matches!(
        adapter(profile),
        Some(Adapter::Anthropic | Adapter::OpenAi | Adapter::Xai)
    )
    .then_some(AUTOMATIC_FILE_CACHE_TTL)
}

/// Return the stable, non-secret endpoint fingerprint carried by provider
/// file references. It covers the full configured base URL without retaining
/// userinfo, query parameters, or any other URL text in the reference.
#[must_use]
pub fn provider_file_endpoint_fingerprint(base_url: &str) -> String {
    // FNV-1a 128 keeps the binding deterministic across Rust versions without
    // another dependency. This is an endpoint identifier, not an auth token.
    let hash =
        base_url
            .as_bytes()
            .iter()
            .fold(0x6c62272e07bb014262b821756295c58du128, |hash, byte| {
                (hash ^ u128::from(*byte)).wrapping_mul(0x0000000001000000000000000000013bu128)
            });
    format!("fnv1a128:{hash:032x}")
}

/// A payload resolved by the host application before provider preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadFile {
    pub filename: String,
    pub media_type: String,
    pub bytes: Bytes,
}

/// Why a file is being uploaded. The provider-specific purpose field is
/// selected by this library; callers do not need to know wire spellings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FilePurpose {
    /// Attach this file to a model request where the provider documents it.
    ModelInput,
    /// Upload a file for Moonshot's text/OCR extraction service.
    Extraction,
    /// Upload an auxiliary file for a documented provider agent service.
    Auxiliary,
    /// Store a file in a provider workspace for shell or sandbox use.
    Workspace,
    /// Clone a voice on MiniMax.
    VoiceClone,
    /// Supply prompt audio to MiniMax speech endpoints.
    PromptAudio,
    /// Supply input for asynchronous MiniMax text-to-audio generation.
    AsyncTtsInput,
    /// Supply a video for MiniMax video understanding.
    VideoUnderstanding,
    /// Supply an input video for MiniMax video generation.
    VideoGenerationInput,
}

/// Operations whose support differs by provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOperation {
    Upload,
    RetrieveMetadata,
    List,
    Delete,
    Download,
    ExtractText,
}

/// How an uploaded provider file can be attached to a completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFileReference {
    Unsupported,
    FileId,
    FileUri,
}

/// Whether the provider API can return original bytes for a stored file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadSupport {
    Unsupported,
    /// Download is available for files the provider marks downloadable; user
    /// uploads are commonly not downloadable.
    ProviderMarkedDownloadable,
    /// The documented endpoint returns only files generated server-side.
    GeneratedFilesOnly,
    /// Raw content download is documented for stored user uploads.
    UploadedFiles,
}

/// Explicit provider and model-specific file support. File management support
/// does not imply that a file ID can be sent as a chat input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileCapabilities {
    pub upload: bool,
    pub retrieve_metadata: bool,
    pub list: bool,
    pub delete: bool,
    pub download: DownloadSupport,
    pub extract_text: bool,
    pub model_input: ModelFileReference,
    pub max_upload_bytes: Option<u64>,
    pub retention: Option<Duration>,
}

impl FileCapabilities {
    fn unsupported() -> Self {
        Self {
            upload: false,
            retrieve_metadata: false,
            list: false,
            delete: false,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: None,
            retention: None,
        }
    }

    #[must_use]
    pub fn supports(&self, operation: FileOperation) -> bool {
        match operation {
            FileOperation::Upload => self.upload,
            FileOperation::RetrieveMetadata => self.retrieve_metadata,
            FileOperation::List => self.list,
            FileOperation::Delete => self.delete,
            FileOperation::Download => self.download != DownloadSupport::Unsupported,
            FileOperation::ExtractText => self.extract_text,
        }
    }
}

/// Provider-owned file identity and metadata. The profile name, endpoint
/// fingerprint, and account scope make accidental cross-connection reuse detectable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFileRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    /// Deterministic, non-secret fingerprint of the full configured base URL.
    /// The URL itself, including any userinfo or query parameters, is not kept.
    pub endpoint_fingerprint: String,
    pub account_scope: Option<String>,
    pub protocol: ProtocolFamily,
    pub file_id: String,
    /// A provider model-input URI, when the API requires one (Gemini).
    pub uri: Option<String>,
    pub filename: Option<String>,
    pub media_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub expires_at: Option<String>,
    pub downloadable: Option<bool>,
    /// Provider purpose used when creating the file. Some APIs require it for
    /// later list/delete calls.
    pub purpose: Option<String>,
}

impl ProviderFileRef {
    /// Build the protocol-level reference consumed by a wire codec.
    #[must_use]
    pub fn model_reference(&self) -> ProviderFileSource {
        ProviderFileSource {
            protocol: self.protocol,
            provider_id: self.provider_id.clone(),
            profile_name: self.profile_name.clone(),
            endpoint_fingerprint: self.endpoint_fingerprint.clone(),
            account_scope: self.account_scope.clone(),
            file_id: self.file_id.clone(),
            uri: self.uri.clone(),
            media_type: self.media_type.clone(),
            purpose: self.purpose.clone(),
        }
    }
}

/// Normalized metadata returned by provider file APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFileMetadata {
    pub file: ProviderFileRef,
    pub created_at: Option<String>,
    pub status: Option<String>,
}

/// One paginated page. Cursors are opaque and may only be passed back to
/// [`FileService::list`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFilePage {
    pub files: Vec<ProviderFileMetadata>,
    pub next_cursor: Option<String>,
}

/// Download result with the response's content type when the provider sends it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFileContent {
    pub media_type: Option<String>,
    pub bytes: Bytes,
}

/// File API adapter bound to one profile and one credential scope.
///
/// Construct this from the same profile, authenticator and per-attempt
/// credential used for a model request. File IDs are therefore never silently
/// reused across failover connections.
pub struct FileService<'a> {
    http: &'a dyn Transport,
    profile: &'a ProviderProfile,
    authenticator: Option<&'a dyn Authenticator>,
    credential: Option<&'a Secret<String>>,
    account_scope: Option<&'a str>,
}

impl<'a> FileService<'a> {
    #[must_use]
    pub fn new(
        http: &'a dyn Transport,
        profile: &'a ProviderProfile,
        authenticator: Option<&'a dyn Authenticator>,
        credential: Option<&'a Secret<String>>,
        account_scope: Option<&'a str>,
    ) -> Self {
        Self {
            http,
            profile,
            authenticator,
            credential,
            account_scope,
        }
    }

    /// Return capabilities for this concrete profile, model and media type.
    /// The caller must still check model vision/file modality when present.
    #[must_use]
    pub fn capabilities(&self, model: &str, media_type: &str) -> FileCapabilities {
        capabilities(self.profile, model, media_type)
    }

    /// Return upload and model-input support for a specific purpose. This
    /// separates extraction or agent auxiliary files from chat references.
    #[must_use]
    pub fn capabilities_for_purpose(
        &self,
        model: &str,
        media_type: &str,
        purpose: FilePurpose,
    ) -> FileCapabilities {
        capabilities_for_purpose(self.profile, model, media_type, purpose)
    }

    /// Upload bytes using a provider-supported purpose.
    pub async fn upload(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        self.upload_with_expiration(file, purpose, None).await
    }

    /// Automatic uploads receive a bounded lifetime where supported. Explicit
    /// FileService uploads retain their existing caller-managed lifetime.
    pub(crate) async fn upload_automatic(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        let expiration = matches!(
            adapter(self.profile),
            Some(Adapter::Anthropic | Adapter::OpenAi | Adapter::Xai)
        )
        .then_some(AUTOMATIC_FILE_TTL.as_secs());
        let uploaded = self
            .upload_with_expiration(file, purpose, expiration)
            .await?;
        if adapter(self.profile) == Some(Adapter::Anthropic)
            && serde_json::to_string(&uploaded.file_id).map_or(true, |encoded| {
                encoded.len() > MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES
            })
        {
            // A provider ID longer than the preflight bound cannot safely be
            // substituted into a request that was checked before upload.
            let _ = self.delete(&uploaded).await;
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic returned a file ID too long for automatic request preparation"
                    .into(),
            });
        }
        Ok(uploaded)
    }

    async fn upload_with_expiration(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
        expires_in_seconds: Option<u64>,
    ) -> Result<ProviderFileRef, LlmError> {
        validate_media_type(&file.media_type)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("upload"))?;
        let caps = capabilities_for_media_type(
            purpose_capabilities(adapter, purpose, &file.media_type),
            adapter,
            &file.media_type,
        );
        if !caps.upload {
            return Err(unsupported("upload for this purpose"));
        }
        if matches!(
            purpose,
            FilePurpose::ModelInput | FilePurpose::VideoUnderstanding
        ) && self
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "model-input file uploads require a nonempty account scope".into(),
            });
        }
        if let Some(limit) = caps.max_upload_bytes {
            if u64::try_from(file.bytes.len()).unwrap_or(u64::MAX) > limit {
                return Err(LlmError::RequestTooLarge {
                    message: format!("file exceeds the provider limit of {limit} bytes"),
                });
            }
        }

        if adapter == Adapter::Gemini {
            return self.upload_gemini(file).await;
        }

        let boundary = multipart_boundary();
        let body = multipart_body(adapter, purpose, file, &boundary, expires_in_seconds)?;
        let url = if adapter == Adapter::MiniMax {
            format!("{}/upload", files_url(self.profile, adapter))
        } else {
            files_url(self.profile, adapter)
        };
        let req = self
            .request(
                "POST",
                url,
                body,
                Some(format!("multipart/form-data; boundary={boundary}")),
            )
            .await?;
        let response = self.http.execute_no_follow(req).await?;
        let value = adapter_json_success(adapter, &response, "file upload")?;
        let mut metadata = decode_metadata(self.profile, self.account_scope, &value)?;
        // The API may omit MIME metadata on upload, but the input is known here.
        if metadata.file.media_type.is_none() {
            metadata.file.media_type = Some(file.media_type.clone());
        }
        if metadata.file.filename.is_none() {
            metadata.file.filename = Some(file.filename.clone());
        }
        if metadata.file.purpose.is_none() {
            metadata.file.purpose = purpose_name(adapter, purpose).map(str::to_owned);
        }
        Ok(metadata.file)
    }

    /// Retrieve metadata for a provider-owned file.
    pub async fn get(&self, file: &ProviderFileRef) -> Result<ProviderFileMetadata, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("metadata retrieval"))?;
        if !capabilities_for_adapter(adapter).retrieve_metadata {
            return Err(unsupported("metadata retrieval"));
        }
        let url = if adapter == Adapter::MiniMax {
            format!(
                "{}?file_id={}",
                file_url(self.profile, adapter, &file.file_id),
                query_value(&file.file_id)
            )
        } else {
            file_url(self.profile, adapter, &file.file_id)
        };
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let response = self.http.execute_no_follow(req).await?;
        let value = adapter_json_success(adapter, &response, "file metadata")?;
        decode_metadata(self.profile, self.account_scope, &value)
    }

    /// List provider-owned files. The cursor is opaque.
    pub async fn list(&self, cursor: Option<&str>) -> Result<ProviderFilePage, LlmError> {
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file listing"))?;
        if adapter == Adapter::MiniMax {
            return Err(LlmError::InvalidRequest {
                message: "MiniMax file listing requires list_for_purpose with an explicit purpose"
                    .into(),
            });
        }
        self.list_inner(adapter, None, cursor).await
    }

    /// List files for an API that requires a purpose filter (MiniMax), or
    /// constrain a provider's normal list to one purpose.
    pub async fn list_for_purpose(
        &self,
        purpose: FilePurpose,
        cursor: Option<&str>,
    ) -> Result<ProviderFilePage, LlmError> {
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file listing"))?;
        if adapter == Adapter::MiniMax && minimax_list_purpose(purpose).is_none() {
            return Err(unsupported("MiniMax file listing for this purpose"));
        }
        if adapter == Adapter::MiniMax && cursor.is_some() {
            return Err(unsupported("MiniMax file-list pagination"));
        }
        self.list_inner(adapter, Some(purpose), cursor).await
    }

    async fn list_inner(
        &self,
        adapter: Adapter,
        purpose: Option<FilePurpose>,
        cursor: Option<&str>,
    ) -> Result<ProviderFilePage, LlmError> {
        if !capabilities_for_adapter(adapter).list {
            return Err(unsupported("file listing"));
        }
        let (url, cursor_field) = list_url(self.profile, adapter, purpose, cursor);
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let response = self.http.execute_no_follow(req).await?;
        let value = adapter_json_success(adapter, &response, "file listing")?;
        let rows = value
            .pointer("/data/files")
            .or_else(|| value.pointer("/data/file_list"))
            .or_else(|| value.pointer("/data/data"))
            .or_else(|| value.get("data").filter(|data| data.is_array()))
            .or_else(|| value.get("files"))
            .or_else(|| value.get("file_list"))
            .and_then(Value::as_array)
            .ok_or_else(|| provider_shape("file list has no array of files"))?;
        let files = rows
            .iter()
            .map(|row| decode_metadata(self.profile, self.account_scope, row))
            .collect::<Result<Vec<_>, _>>()?;
        let next_cursor = match cursor_field {
            CursorField::OpenAi => value
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then(|| {
                    value
                        .get("last_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .flatten(),
            CursorField::Anthropic => value.get("next_page").and_then(nonempty_string),
            CursorField::Gemini => value.get("nextPageToken").and_then(nonempty_string),
            CursorField::Xai if rows.len() >= 100 => {
                value.get("pagination_token").and_then(nonempty_string)
            }
            CursorField::Xai => None,
            CursorField::OpenRouter => value
                .get("next_cursor")
                .or_else(|| value.get("cursor"))
                .and_then(nonempty_string),
            CursorField::MiniMax => None,
        };
        Ok(ProviderFilePage { files, next_cursor })
    }

    /// Delete provider-owned storage. This does not delete the app's original
    /// attachment.
    pub async fn delete(&self, file: &ProviderFileRef) -> Result<(), LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file deletion"))?;
        if !capabilities_for_adapter(adapter).delete {
            return Err(unsupported("file deletion"));
        }
        let (method, url, body, content_type) = if adapter == Adapter::MiniMax {
            let purpose = file
                .purpose
                .as_deref()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "MiniMax file deletion requires the original provider purpose".into(),
                })?;
            let purpose = minimax_delete_purpose(purpose)
                .ok_or_else(|| unsupported("MiniMax deletion for this purpose"))?;
            (
                "POST",
                format!("{}/delete", files_url(self.profile, adapter)),
                Bytes::from(
                    serde_json::to_vec(
                        &serde_json::json!({"file_id": file.file_id, "purpose": purpose}),
                    )
                    .map_err(|error| provider_shape(&error.to_string()))?,
                ),
                Some("application/json".into()),
            )
        } else {
            (
                "DELETE",
                file_url(self.profile, adapter, &file.file_id),
                Bytes::new(),
                None,
            )
        };
        let req = self.request(method, url, body, content_type).await?;
        let response = self.http.execute_no_follow(req).await?;
        adapter_status_result(adapter, &response, "file deletion")?;
        Ok(())
    }

    /// Download original provider bytes where the provider documents that
    /// uploaded files are retrievable. The accumulated body is capped at 64 MiB.
    pub async fn download(&self, file: &ProviderFileRef) -> Result<ProviderFileContent, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file download"))?;
        let support = capabilities_for_adapter(adapter).download;
        if support == DownloadSupport::Unsupported {
            return Err(unsupported("file download"));
        }
        if matches!(
            support,
            DownloadSupport::GeneratedFilesOnly | DownloadSupport::ProviderMarkedDownloadable
        ) {
            let metadata = self.get(file).await?;
            if metadata.file.downloadable != Some(true) {
                return Err(unsupported("download of this uploaded file"));
            }
        }
        if adapter == Adapter::Moonshot {
            return Err(unsupported(
                "raw download; use extract_text for Moonshot files",
            ));
        }
        let url = if adapter == Adapter::MiniMax {
            format!(
                "{}?file_id={}",
                content_url(self.profile, adapter, &file.file_id),
                query_value(&file.file_id)
            )
        } else {
            content_url(self.profile, adapter, &file.file_id)
        };
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let mut response = self.http.open_stream_no_follow(req).await?;
        if !(200..300).contains(&response.status) {
            let mut body = BytesMut::new();
            while let Some(frame) = response.body.next().await {
                if let Ok(frame) = frame {
                    let remaining =
                        crate::transport::MAX_ERROR_BODY_SIZE.saturating_sub(body.len());
                    if remaining == 0 {
                        break;
                    }
                    body.extend_from_slice(&frame[..frame.len().min(remaining)]);
                }
            }
            return Err(status_error(
                response.status,
                &String::from_utf8_lossy(&body),
                "file download",
            ));
        }
        let media_type = response
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            .map(|(_, value)| value.split(';').next().unwrap_or(value).trim().to_owned());
        let mut body = BytesMut::new();
        while let Some(frame) = response.body.next().await {
            let frame = frame?;
            if body.len().saturating_add(frame.len()) > MAX_PROVIDER_FILE_DOWNLOAD_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "provider file exceeds the {} byte download cap",
                        MAX_PROVIDER_FILE_DOWNLOAD_BYTES
                    ),
                });
            }
            body.extend_from_slice(&frame);
        }
        Ok(ProviderFileContent {
            media_type,
            bytes: body.freeze(),
        })
    }

    /// Extract the provider's text representation of an uploaded file. This is
    /// distinct from downloading the original binary; currently only
    /// Moonshot/Kimi documents this operation in the supported adapter set.
    pub async fn extract_text(&self, file: &ProviderFileRef) -> Result<String, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("text extraction"))?;
        if !capabilities_for_adapter(adapter).extract_text {
            return Err(unsupported("text extraction"));
        }
        let url = format!(
            "{}/{}/content",
            files_url(self.profile, adapter),
            path_segment(&file.file_id)
        );
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let response = self.http.execute_no_follow(req).await?;
        status_result(&response, "file text extraction")?;
        String::from_utf8(response.body.to_vec())
            .map_err(|error| provider_shape(&format!("extracted file text is not UTF-8: {error}")))
    }

    async fn upload_gemini(&self, file: &UploadFile) -> Result<ProviderFileRef, LlmError> {
        let deadline = Instant::now() + FILE_TIMEOUT;
        let start_url = gemini_upload_url(self.profile);
        let body = serde_json::json!({"file": {"display_name": file.filename}});
        let mut req = self
            .request(
                "POST",
                start_url,
                Bytes::from(
                    serde_json::to_vec(&body)
                        .map_err(|error| provider_shape(&error.to_string()))?,
                ),
                Some("application/json".to_owned()),
            )
            .await?;
        req.timeout = Some(gemini_timeout_remaining(deadline)?);
        req.headers.extend([
            ("x-goog-upload-protocol".to_owned(), "resumable".to_owned()),
            ("x-goog-upload-command".to_owned(), "start".to_owned()),
            (
                "x-goog-upload-header-content-length".to_owned(),
                file.bytes.len().to_string(),
            ),
            (
                "x-goog-upload-header-content-type".to_owned(),
                file.media_type.clone(),
            ),
        ]);
        let start = self.http.execute_no_follow(req).await?;
        status_result(&start, "Gemini file upload start")?;
        let upload_url = start
            .header("x-goog-upload-url")
            .ok_or_else(|| provider_shape("Gemini upload start omitted x-goog-upload-url"))?;
        if !same_origin(upload_url, &self.profile.base_url) {
            return Err(LlmError::PermissionDenied {
                message: "Gemini returned an upload URL outside the configured API origin".into(),
            });
        }
        // The resumable upload URL is a temporary bearer capability; Google's
        // documented second request does not send the API key again.
        let upload_req = HttpRequest {
            method: "POST".into(),
            url: upload_url.to_owned(),
            headers: vec![
                ("content-length".into(), file.bytes.len().to_string()),
                ("x-goog-upload-offset".into(), "0".into()),
                ("x-goog-upload-command".into(), "upload, finalize".into()),
            ],
            body: file.bytes.clone(),
            timeout: Some(gemini_timeout_remaining(deadline)?),
        };
        let response = self.http.execute_no_follow(upload_req).await?;
        let value = json_success(&response, "Gemini file upload")?;
        let file_value = value.get("file").unwrap_or(&value);
        let mut metadata = decode_metadata(self.profile, self.account_scope, file_value)?;
        metadata.file.media_type = Some(file.media_type.clone());
        metadata.file.filename = Some(file.filename.clone());
        match file_value.get("state").and_then(Value::as_str) {
            Some("FAILED") => {
                return Err(provider_shape("Gemini file processing failed"));
            }
            Some("PROCESSING") => {
                let file_id = metadata.file.file_id.clone();
                let initial_uri = metadata.file.uri.clone();
                metadata.file = self
                    .wait_for_gemini_file_active(&file_id, initial_uri.as_deref(), deadline)
                    .await?;
                metadata.file.media_type = Some(file.media_type.clone());
                metadata.file.filename = Some(file.filename.clone());
            }
            Some("ACTIVE") | None => {} // Older APIs and transports may omit the state.
            Some(_) => {
                return Err(provider_shape(
                    "Gemini file upload returned an unknown processing state",
                ));
            }
        }
        if metadata.file.uri.is_none() {
            return Err(provider_shape(
                "Gemini file upload omitted the model-input URI",
            ));
        }
        Ok(metadata.file)
    }

    async fn wait_for_gemini_file_active(
        &self,
        file_id: &str,
        initial_uri: Option<&str>,
        deadline: Instant,
    ) -> Result<ProviderFileRef, LlmError> {
        loop {
            let remaining = gemini_timeout_remaining(deadline)?;
            async_delay(GEMINI_FILE_POLL_INTERVAL.min(remaining)).await;

            let remaining = gemini_timeout_remaining(deadline)?;
            let mut request = self
                .request(
                    "GET",
                    file_url(self.profile, Adapter::Gemini, file_id),
                    Bytes::new(),
                    None,
                )
                .await?;
            request.timeout = Some(remaining);
            let response = self.http.execute_no_follow(request).await?;
            let value = json_success(&response, "Gemini file processing status")?;
            let file_value = value.get("file").unwrap_or(&value);
            match file_value.get("state").and_then(Value::as_str) {
                Some("ACTIVE") => {
                    let mut metadata =
                        decode_metadata(self.profile, self.account_scope, file_value)?.file;
                    if metadata.uri.is_none() {
                        metadata.uri = initial_uri.map(str::to_owned);
                    }
                    return Ok(metadata);
                }
                Some("FAILED") => {
                    return Err(provider_shape("Gemini file processing failed"));
                }
                Some("PROCESSING") => {}
                _ => {
                    return Err(provider_shape(
                        "Gemini file processing returned an unknown state",
                    ));
                }
            }
        }
    }

    async fn request(
        &self,
        method: &str,
        url: String,
        body: Bytes,
        content_type: Option<String>,
    ) -> Result<HttpRequest, LlmError> {
        let mut headers = vec![("accept".into(), "application/json".into())];
        if let Some(content_type) = content_type {
            headers.push(("content-type".into(), content_type));
        }
        crate::codecs::extras::merge_headers(self.profile, &mut headers);
        if adapter(self.profile) == Some(Adapter::Anthropic) {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-version"));
            let version = self
                .profile
                .extra
                .get("api_version")
                .and_then(Value::as_str)
                .unwrap_or(crate::codecs::anthropic::DEFAULT_API_VERSION);
            headers.push(("anthropic-version".into(), version.to_owned()));
        }
        let mut request = HttpRequest {
            method: method.to_owned(),
            url,
            headers,
            body,
            timeout: Some(FILE_TIMEOUT),
        };
        if self.profile.auth != AuthStrategy::None {
            let authenticator =
                self.authenticator
                    .ok_or_else(|| LlmError::UnsupportedCapability {
                        message: format!(
                            "no authenticator is registered for file operations on profile {:?}",
                            self.profile.profile_name
                        ),
                    })?;
            authenticator
                .apply(&mut request, self.profile, self.credential)
                .await?;
        }
        Ok(request)
    }

    fn check_ref(&self, file: &ProviderFileRef) -> Result<(), LlmError> {
        if file.profile_name != self.profile.profile_name
            || file.provider_id != self.profile.provider_id
            || file.protocol != self.profile.protocol
            || file.endpoint_fingerprint
                != provider_file_endpoint_fingerprint(&self.profile.base_url)
            || file.account_scope.as_deref() != self.account_scope
        {
            return Err(LlmError::PermissionDenied {
                message:
                    "provider file reference belongs to another profile, endpoint, or account scope"
                        .into(),
            });
        }
        Ok(())
    }
}

/// Capability matrix keyed by provider identity and the concrete protocol.
/// OpenAI-compatible protocols do not opt providers into Files APIs by themselves.
#[must_use]
pub fn capabilities(profile: &ProviderProfile, model: &str, media_type: &str) -> FileCapabilities {
    let Some(adapter) = adapter(profile) else {
        return FileCapabilities::unsupported();
    };
    let mut result =
        capabilities_for_media_type(capabilities_for_adapter(adapter), adapter, media_type);
    result.model_input = model_reference_for(profile, model, media_type, adapter);
    result
}

/// Provider capabilities for one operation purpose plus the profile's model
/// reference format, if that purpose is a normal model input.
#[must_use]
pub fn capabilities_for_purpose(
    profile: &ProviderProfile,
    model: &str,
    media_type: &str,
    purpose: FilePurpose,
) -> FileCapabilities {
    let Some(adapter) = adapter(profile) else {
        return FileCapabilities::unsupported();
    };
    let mut result = capabilities_for_media_type(
        purpose_capabilities(adapter, purpose, media_type),
        adapter,
        media_type,
    );
    result.model_input = if matches!(
        purpose,
        FilePurpose::ModelInput | FilePurpose::VideoUnderstanding
    ) {
        model_reference_for(profile, model, media_type, adapter)
    } else {
        ModelFileReference::Unsupported
    };
    result
}

fn model_reference_for(
    profile: &ProviderProfile,
    model: &str,
    media_type: &str,
    adapter: Adapter,
) -> ModelFileReference {
    let model_profile = profile
        .models
        .iter()
        .find(|candidate| candidate.request_model == model);
    let Some(model_profile) = model_profile else {
        return ModelFileReference::Unsupported;
    };
    let media_type = media_type.to_ascii_lowercase();
    let is_image = media_type.starts_with("image/");
    // OpenAI accepts non-animated GIF input; this MIME-only capability check
    // cannot inspect file bytes to distinguish animated GIFs.
    let is_openai_image = is_openai_image_type(&media_type);
    let is_anthropic_image = matches!(
        media_type.as_str(),
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    );
    // Gemini's image-input guide documents these five MIME types.
    let is_gemini_image = matches!(
        media_type.as_str(),
        "image/jpeg" | "image/png" | "image/webp" | "image/heic" | "image/heif"
    );
    let is_pdf = media_type == "application/pdf";
    let has_file_modality = model_declares_file_input(model_profile);
    match adapter {
        Adapter::Qwen
            if is_qwen_region_supported(profile)
                && profile.protocol == ProtocolFamily::OpenAiChat
                && model.eq_ignore_ascii_case("qwen-long")
                && is_qwen_file_type(&media_type) =>
        {
            ModelFileReference::FileUri
        }
        Adapter::MiniMax
            if profile.protocol == ProtocolFamily::AnthropicMessages
                && model.eq_ignore_ascii_case("minimax-m3")
                && media_type.starts_with("video/") =>
        {
            ModelFileReference::FileUri
        }
        Adapter::OpenAi => match profile.protocol {
            ProtocolFamily::OpenAiResponses
                if (is_openai_image && model_declares_image_input(model_profile))
                    || (is_openai_responses_file_type(&media_type) && has_file_modality) =>
            {
                ModelFileReference::FileId
            }
            ProtocolFamily::OpenAiChat if is_pdf && has_file_modality => ModelFileReference::FileId,
            _ => ModelFileReference::Unsupported,
        },
        Adapter::Anthropic if profile.protocol == ProtocolFamily::AnthropicMessages => {
            if (is_anthropic_image && model_declares_image_input(model_profile))
                || ((is_pdf || media_type == "text/plain") && has_file_modality)
            {
                ModelFileReference::FileId
            } else {
                ModelFileReference::Unsupported
            }
        }
        Adapter::Gemini if profile.protocol == ProtocolFamily::GeminiGenerateContent => {
            if (is_gemini_image && model_declares_image_input(model_profile))
                || (!is_image && has_file_modality)
            {
                ModelFileReference::FileUri
            } else {
                ModelFileReference::Unsupported
            }
        }
        Adapter::Xai if profile.protocol == ProtocolFamily::OpenAiResponses => {
            if is_xai_document_type(&media_type) && has_file_modality {
                ModelFileReference::FileId
            } else {
                ModelFileReference::Unsupported
            }
        }
        // OpenRouter Files are workspace/shell storage, not a confirmed
        // ordinary-chat attachment reference. Moonshot file_id context is
        // explicitly unsupported. Z.AI uploads are auxiliary Agent API files.
        _ => ModelFileReference::Unsupported,
    }
}

fn is_qwen_region_supported(profile: &ProviderProfile) -> bool {
    reqwest::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            matches!(
                host.as_str(),
                "dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com"
            )
        })
}

fn is_qwen_file_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf"
                | "application/json"
                | "application/epub+zip"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.oasis.opendocument.text"
                | "application/msword"
                | "application/vnd.ms-excel"
                | "image/bmp"
                | "image/png"
                | "image/jpeg"
                | "image/gif"
        )
}

fn purpose_name(adapter: Adapter, purpose: FilePurpose) -> Option<&'static str> {
    match (adapter, purpose) {
        (Adapter::OpenAi, FilePurpose::ModelInput) => Some("user_data"),
        (Adapter::Xai, FilePurpose::ModelInput) => Some("assistants"),
        (Adapter::Moonshot | Adapter::Qwen, FilePurpose::Extraction | FilePurpose::ModelInput) => {
            Some("file-extract")
        }
        (Adapter::Zai | Adapter::Zhipu, FilePurpose::Auxiliary) => Some("agent"),
        (Adapter::MiniMax, FilePurpose::VoiceClone) => Some("voice_clone"),
        (Adapter::MiniMax, FilePurpose::PromptAudio) => Some("prompt_audio"),
        (Adapter::MiniMax, FilePurpose::AsyncTtsInput) => Some("t2a_async_input"),
        (Adapter::MiniMax, FilePurpose::VideoUnderstanding) => Some("video_understanding"),
        (Adapter::MiniMax, FilePurpose::VideoGenerationInput) => Some("video_generation_input"),
        _ => None,
    }
}

fn minimax_list_purpose(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::VoiceClone => Some("voice_clone"),
        FilePurpose::PromptAudio => Some("prompt_audio"),
        FilePurpose::AsyncTtsInput => Some("t2a_async_input"),
        FilePurpose::VideoGenerationInput => Some("video_generation_input"),
        _ => None,
    }
}

fn minimax_delete_purpose(purpose: &str) -> Option<&'static str> {
    match purpose {
        "voice_clone" => Some("voice_clone"),
        "prompt_audio" => Some("prompt_audio"),
        "t2a_async" | "t2a_async_input" => Some("t2a_async"),
        "video_generation_input" | "video_generation" => Some("video_generation"),
        _ => None,
    }
}

fn is_xai_document_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf" | "application/json" | "application/x-ndjson"
        )
}

fn is_openai_image_type(media_type: &str) -> bool {
    ["image/jpeg", "image/png", "image/webp", "image/gif"]
        .iter()
        .any(|supported| media_type.eq_ignore_ascii_case(supported))
}

fn is_openai_responses_file_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf"
                | "application/json"
                | "application/graphql"
                | "application/javascript"
                | "application/typescript"
                | "application/csv"
                | "application/x-iif"
                | "application/x-sql"
                | "application/x-scala"
                | "application/x-rust"
                | "application/x-powershell"
                | "application/x-patch"
                | "application/x-php"
                | "application/x-httpd-php"
                | "application/x-httpd-php-source"
                | "application/x-bash"
                | "application/x-awk"
                | "application/x-protobuf"
                | "application/x-terraform"
                | "application/x-graphql"
                | "application/x-ndjson"
                | "application/json5"
                | "application/x-json5"
                | "application/x-toml"
                | "application/toml"
                | "application/x-yaml"
                | "application/yaml"
                | "application/x-subrip"
                | "application/msword"
                | "application/rtf"
                | "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.google-apps.spreadsheet"
                | "application/vnd.apple.pages"
                | "application/vnd.apple.iwork"
                | "application/vnd.oasis.opendocument.text"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.google-apps.document"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.ms-powerpoint"
                | "application/vnd.apple.keynote"
                | "application/vnd.google-apps.presentation"
                | "message/rfc822"
        )
}

fn model_declares_file_input(model: &ModelProfile) -> bool {
    model.metadata.input_modalities.iter().any(|modality| {
        matches!(
            modality.to_ascii_lowercase().as_str(),
            "file" | "files" | "document" | "pdf"
        )
    }) || model.capabilities.documents
        || model.metadata.attachments == Some(true)
}

fn model_declares_image_input(model: &ModelProfile) -> bool {
    model
        .metadata
        .input_modalities
        .iter()
        .any(|modality| modality.eq_ignore_ascii_case("image"))
        || model.capabilities.vision
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Adapter {
    OpenAi,
    Anthropic,
    Gemini,
    Xai,
    OpenRouter,
    Moonshot,
    Zai,
    Zhipu,
    Qwen,
    MiniMax,
}

fn adapter(profile: &ProviderProfile) -> Option<Adapter> {
    let host = reqwest::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))?;
    match (profile.provider_id.as_str(), host.as_str()) {
        ("openai", "api.openai.com") => Some(Adapter::OpenAi),
        ("anthropic", "api.anthropic.com") => Some(Adapter::Anthropic),
        ("google", "generativelanguage.googleapis.com")
            if profile.protocol == ProtocolFamily::GeminiGenerateContent =>
        {
            Some(Adapter::Gemini)
        }
        ("xai", "api.x.ai") => Some(Adapter::Xai),
        ("openrouter", "openrouter.ai") => Some(Adapter::OpenRouter),
        ("kimi" | "moonshot", "api.moonshot.cn" | "api.moonshot.ai") => Some(Adapter::Moonshot),
        ("zhipu", "api.z.ai") => Some(Adapter::Zai),
        ("zhipu", "open.bigmodel.cn") => Some(Adapter::Zhipu),
        ("qwen", "dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com") => Some(Adapter::Qwen),
        ("minimax", "api.minimaxi.com" | "api.minimax.io") => Some(Adapter::MiniMax),
        _ => None,
    }
}

fn capabilities_for_adapter(adapter: Adapter) -> FileCapabilities {
    match adapter {
        Adapter::OpenAi => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::UploadedFiles,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(512 * 1024 * 1024),
            retention: None,
        },
        Adapter::Anthropic => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(500 * 1024 * 1024),
            retention: None,
        },
        Adapter::Gemini => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(2 * 1024 * 1024 * 1024),
            retention: Some(Duration::from_secs(48 * 60 * 60)),
        },
        Adapter::Xai => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::UploadedFiles,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(512 * 1024 * 1024),
            retention: None,
        },
        Adapter::OpenRouter => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: None,
        },
        Adapter::Moonshot => FileCapabilities {
            upload: false,
            retrieve_metadata: false,
            list: false,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: true,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: None,
        },
        // Z.AI's documented /files API is limited to Agent API auxiliary files;
        // it is not a reference format for ordinary chat completions.
        Adapter::Zai => FileCapabilities {
            upload: false,
            retrieve_metadata: false,
            list: false,
            delete: false,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: Some(Duration::from_secs(180 * 24 * 60 * 60)),
        },
        Adapter::Zhipu => FileCapabilities {
            upload: false,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(20 * 1024 * 1024),
            retention: None,
        },
        Adapter::Qwen => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(150_000_000),
            retention: None,
        },
        Adapter::MiniMax => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: None,
            retention: None,
        },
    }
}

fn purpose_capabilities(
    adapter: Adapter,
    purpose: FilePurpose,
    media_type: &str,
) -> FileCapabilities {
    match (adapter, purpose) {
        (Adapter::Moonshot, FilePurpose::Extraction) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Zai, FilePurpose::Auxiliary) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Zhipu, FilePurpose::Auxiliary) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Qwen, FilePurpose::ModelInput | FilePurpose::Extraction) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::MiniMax, purpose)
            if matches!(
                purpose,
                FilePurpose::VoiceClone
                    | FilePurpose::PromptAudio
                    | FilePurpose::AsyncTtsInput
                    | FilePurpose::VideoUnderstanding
                    | FilePurpose::VideoGenerationInput
            ) =>
        {
            let mut capabilities = capabilities_for_adapter(adapter);
            capabilities.upload = true;
            capabilities.max_upload_bytes = None;
            if matches!(
                purpose,
                FilePurpose::VideoUnderstanding | FilePurpose::VideoGenerationInput
            ) {
                capabilities.retention = Some(Duration::from_secs(7 * 24 * 60 * 60));
            }
            if purpose == FilePurpose::VideoUnderstanding {
                capabilities.list = false;
                capabilities.delete = false;
                capabilities.download = DownloadSupport::Unsupported;
            }
            capabilities
        }
        (Adapter::OpenRouter, FilePurpose::Workspace) => capabilities_for_adapter(adapter),
        (Adapter::OpenAi, FilePurpose::ModelInput) => {
            let mut capabilities = capabilities_for_adapter(adapter);
            if !is_openai_image_type(media_type) {
                capabilities.max_upload_bytes = Some(OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES);
            }
            capabilities
        }
        (Adapter::Anthropic | Adapter::Gemini | Adapter::Xai, FilePurpose::ModelInput) => {
            capabilities_for_adapter(adapter)
        }
        _ => FileCapabilities::unsupported(),
    }
}

fn capabilities_for_media_type(
    mut capabilities: FileCapabilities,
    adapter: Adapter,
    media_type: &str,
) -> FileCapabilities {
    if adapter == Adapter::Gemini
        && capabilities.upload
        && media_type.eq_ignore_ascii_case("application/pdf")
    {
        capabilities.max_upload_bytes = Some(
            capabilities
                .max_upload_bytes
                .map_or(GEMINI_PDF_MAX_UPLOAD_BYTES, |limit| {
                    limit.min(GEMINI_PDF_MAX_UPLOAD_BYTES)
                }),
        );
    }
    capabilities
}

fn multipart_body(
    adapter: Adapter,
    purpose: FilePurpose,
    file: &UploadFile,
    boundary: &str,
    expires_in_seconds: Option<u64>,
) -> Result<Bytes, LlmError> {
    let mut body = BytesMut::new();
    if let Some(seconds) = expires_in_seconds {
        match adapter {
            Adapter::Anthropic => {
                append_field(
                    &mut body,
                    boundary,
                    "expires_in_seconds",
                    &seconds.to_string(),
                );
            }
            Adapter::OpenAi => {
                append_field(&mut body, boundary, "expires_after[anchor]", "created_at");
                append_field(
                    &mut body,
                    boundary,
                    "expires_after[seconds]",
                    &seconds.to_string(),
                );
            }
            // xAI's raw multipart API takes a scalar number of seconds, and
            // requires this field to precede the file part.
            Adapter::Xai => {
                append_field(&mut body, boundary, "expires_after", &seconds.to_string());
            }
            _ => {}
        }
    }
    match (adapter, purpose) {
        (Adapter::OpenAi, FilePurpose::ModelInput) => {
            append_field(&mut body, boundary, "purpose", "user_data")
        }
        (Adapter::Xai, FilePurpose::ModelInput) => {
            append_field(&mut body, boundary, "purpose", "assistants")
        }
        (Adapter::Moonshot, FilePurpose::Extraction) => {
            append_field(&mut body, boundary, "purpose", "file-extract")
        }
        (Adapter::Qwen, FilePurpose::ModelInput | FilePurpose::Extraction) => {
            append_field(&mut body, boundary, "purpose", "file-extract")
        }
        (Adapter::MiniMax, purpose) => {
            if let Some(value) = purpose_name(adapter, purpose) {
                append_field(&mut body, boundary, "purpose", value);
            }
        }
        (Adapter::Zai, FilePurpose::Auxiliary) => {
            append_field(&mut body, boundary, "purpose", "agent")
        }
        (Adapter::Zhipu, FilePurpose::Auxiliary) => {
            append_field(&mut body, boundary, "purpose", "agent")
        }
        _ => {}
    }
    let safe_filename = sanitize_filename(&file.filename);
    body.extend_from_slice(format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_filename}\"\r\nContent-Type: {}\r\n\r\n",
        file.media_type
    ).as_bytes());
    body.extend_from_slice(&file.bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok(body.freeze())
}

fn validate_media_type(value: &str) -> Result<(), LlmError> {
    let Some((top_level, subtype)) = value.split_once('/') else {
        return Err(LlmError::InvalidRequest {
            message: "attachment media_type must be a MIME type".into(),
        });
    };
    let is_token = |part: &str| {
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
    if value.matches('/').count() != 1 || !is_token(top_level) || !is_token(subtype) {
        return Err(LlmError::InvalidRequest {
            message: "attachment media_type contains invalid MIME token characters".into(),
        });
    }
    Ok(())
}

fn append_field(body: &mut BytesMut, boundary: &str, name: &str, value: &str) {
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .as_bytes(),
    );
}

fn multipart_boundary() -> String {
    // Unique enough to avoid accidental content collision; no random dependency
    // is needed because the bytes are locally generated and not user-facing.
    format!(
        "lingxi-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |time| time.as_nanos())
    )
}

fn files_url(profile: &ProviderProfile, adapter: Adapter) -> String {
    let base = profile.base_url.trim_end_matches('/');
    match adapter {
        Adapter::Anthropic if base.ends_with("/v1") => format!("{base}/files"),
        Adapter::Anthropic => format!("{base}/v1/files"),
        Adapter::Gemini if base.ends_with("/v1beta") => format!("{base}/files"),
        Adapter::Gemini => format!("{base}/v1beta/files"),
        Adapter::MiniMax => minimax_files_root(profile),
        _ => format!("{base}/files"),
    }
}

fn minimax_files_root(profile: &ProviderProfile) -> String {
    let url = reqwest::Url::parse(&profile.base_url).expect("adapter validates a URL");
    format!(
        "{}://{}/v1/files",
        url.scheme(),
        url.host_str().unwrap_or_default()
    )
}

fn gemini_upload_url(profile: &ProviderProfile) -> String {
    let base = profile.base_url.trim_end_matches('/');
    if base.ends_with("/v1beta") {
        format!("{}/upload/v1beta/files", base.trim_end_matches("/v1beta"))
    } else {
        format!("{base}/upload/v1beta/files")
    }
}

fn content_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    let encoded = path_segment(file_id);
    match adapter {
        Adapter::OpenRouter | Adapter::Xai | Adapter::OpenAi => {
            format!("{}/{encoded}/content", files_url(profile, adapter))
        }
        Adapter::Anthropic => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::Gemini => format!("{}:download?alt=media", file_url(profile, adapter, file_id)),
        Adapter::Moonshot => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::MiniMax => format!("{}/retrieve_content", files_url(profile, adapter)),
        Adapter::Qwen => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::Zai | Adapter::Zhipu => {
            format!("{}/{encoded}/content", files_url(profile, adapter))
        }
    }
}

fn file_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    if adapter == Adapter::MiniMax {
        let _ = file_id;
        return format!("{}/retrieve", files_url(profile, adapter));
    }
    if adapter == Adapter::Gemini {
        let resource_name = if file_id.starts_with("files/") {
            file_id.to_owned()
        } else {
            format!("files/{file_id}")
        };
        let base = profile.base_url.trim_end_matches('/');
        let base = if base.ends_with("/v1beta") {
            base.to_owned()
        } else {
            format!("{base}/v1beta")
        };
        return format!("{base}/{}", encoded_path(&resource_name));
    }
    format!("{}/{}", files_url(profile, adapter), path_segment(file_id))
}

#[derive(Clone, Copy)]
enum CursorField {
    OpenAi,
    Anthropic,
    Gemini,
    Xai,
    OpenRouter,
    MiniMax,
}

fn list_url(
    profile: &ProviderProfile,
    adapter: Adapter,
    purpose: Option<FilePurpose>,
    cursor: Option<&str>,
) -> (String, CursorField) {
    let base = files_url(profile, adapter);
    if adapter == Adapter::MiniMax {
        let Some(purpose) = purpose.and_then(minimax_list_purpose) else {
            return (format!("{base}/list"), CursorField::MiniMax);
        };
        let params = vec![("purpose".to_owned(), purpose.to_owned())];
        return (
            format!(
                "{base}/list?{}",
                params
                    .into_iter()
                    .map(|(key, value)| format!("{key}={}", query_value(&value)))
                    .collect::<Vec<_>>()
                    .join("&")
            ),
            CursorField::MiniMax,
        );
    }
    let (field, limit_field, limit) = match adapter {
        Adapter::OpenAi => ("after", "limit", "100"),
        Adapter::Qwen => ("after", "limit", "100"),
        Adapter::Anthropic => ("page", "limit", "1000"),
        Adapter::Gemini => ("pageToken", "pageSize", "100"),
        Adapter::Xai => ("pagination_token", "limit", "100"),
        Adapter::OpenRouter => ("cursor", "limit", "100"),
        _ => return (base, CursorField::OpenAi),
    };
    let cursor_field = match adapter {
        Adapter::OpenAi => CursorField::OpenAi,
        Adapter::Anthropic => CursorField::Anthropic,
        Adapter::Gemini => CursorField::Gemini,
        Adapter::Xai => CursorField::Xai,
        Adapter::OpenRouter => CursorField::OpenRouter,
        Adapter::Qwen => CursorField::OpenAi,
        _ => CursorField::OpenAi,
    };
    let mut params = vec![(limit_field.to_owned(), limit.to_owned())];
    if let Some(cursor) = cursor {
        params.push((field.to_owned(), cursor.to_owned()));
    }
    let query = params
        .into_iter()
        .map(|(key, value)| format!("{key}={}", query_value(&value)))
        .collect::<Vec<_>>()
        .join("&");
    let mut query = query;
    if adapter == Adapter::Qwen {
        if let Some(purpose) = purpose.and_then(|purpose| purpose_name(adapter, purpose)) {
            query.push_str("&purpose=");
            query.push_str(&query_value(purpose));
        }
    }
    (format!("{base}?{query}"), cursor_field)
}

fn query_value(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(byte))
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn path_segment(value: &str) -> String {
    query_value(value)
}

fn encoded_path(value: &str) -> String {
    value
        .split('/')
        .map(path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn json_success(response: &HttpResponse, operation: &str) -> Result<Value, LlmError> {
    status_result(response, operation)?;
    serde_json::from_slice(&response.body)
        .map_err(|error| provider_shape(&format!("{operation} response was not JSON: {error}")))
}

fn adapter_json_success(
    adapter: Adapter,
    response: &HttpResponse,
    operation: &str,
) -> Result<Value, LlmError> {
    let value = json_success(response, operation)?;
    if adapter == Adapter::MiniMax {
        check_minimax_base_response(&value, operation)?;
    }
    Ok(value)
}

fn adapter_status_result(
    adapter: Adapter,
    response: &HttpResponse,
    operation: &str,
) -> Result<(), LlmError> {
    status_result(response, operation)?;
    if adapter == Adapter::MiniMax {
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            provider_shape(&format!("{operation} response was not JSON: {error}"))
        })?;
        check_minimax_base_response(&value, operation)?;
    }
    Ok(())
}

fn check_minimax_base_response(value: &Value, operation: &str) -> Result<(), LlmError> {
    let base = value
        .get("base_resp")
        .ok_or_else(|| provider_shape(&format!("{operation} response omitted base_resp")))?;
    let code = base
        .get("status_code")
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.parse().ok()))
        .ok_or_else(|| {
            provider_shape(&format!(
                "{operation} response has invalid base_resp.status_code"
            ))
        })?;
    if code == 0 {
        Ok(())
    } else {
        let message = base
            .get("status_msg")
            .and_then(Value::as_str)
            .unwrap_or("provider returned an error")
            .chars()
            .take(512)
            .collect::<String>();
        Err(LlmError::ProviderInternal {
            message: format!("{operation} failed with MiniMax status {code}: {message}"),
        })
    }
}

fn status_result(response: &HttpResponse, operation: &str) -> Result<(), LlmError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let body = String::from_utf8_lossy(&response.body);
    Err(status_error(response.status, &body, operation))
}

fn status_error(status: u16, body: &str, operation: &str) -> LlmError {
    let message = format!(
        "{operation} failed with HTTP {status}: {}",
        body.chars().take(512).collect::<String>()
    );
    match status {
        401 => LlmError::Authentication { message },
        403 => LlmError::PermissionDenied { message },
        404 => LlmError::InvalidRequest { message },
        413 => LlmError::RequestTooLarge { message },
        429 => LlmError::RateLimited {
            message,
            retry_after: None,
        },
        500..=599 => LlmError::ProviderInternal { message },
        _ => LlmError::InvalidRequest { message },
    }
}

fn decode_metadata(
    profile: &ProviderProfile,
    account_scope: Option<&str>,
    value: &Value,
) -> Result<ProviderFileMetadata, LlmError> {
    let value = value.get("file").unwrap_or(value);
    let adapter = adapter(profile).ok_or_else(|| unsupported("file metadata decoding"))?;
    let file_id = value
        .get("id")
        .or_else(|| value.get("file_id"))
        .or_else(|| value.get("name"))
        .and_then(json_optional_scalar_string)
        .ok_or_else(|| provider_shape("file metadata has no id or name"))?
        .to_owned();
    let protocol = profile.protocol;
    let purpose = value
        .get("purpose")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let uri = value
        .get("uri")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| match (adapter, purpose.as_deref()) {
            (Adapter::Qwen, Some("file-extract")) => Some(format!("fileid://{file_id}")),
            (Adapter::MiniMax, Some("video_understanding")) => Some(format!("mm_file://{file_id}")),
            _ => None,
        });
    if adapter == Adapter::Gemini && uri.is_none() {
        // Some metadata/list responses omit URI, but uploaded model refs must
        // retain the URI from the upload response.
    }
    let file = ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: account_scope.map(str::to_owned),
        protocol,
        file_id,
        uri,
        filename: value
            .get("filename")
            .or_else(|| value.get("display_name"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        media_type: value
            .get("mime_type")
            .or_else(|| value.get("mimeType"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        size_bytes: value
            .get("bytes")
            .or_else(|| value.get("size_bytes"))
            .or_else(|| value.get("sizeBytes"))
            .and_then(json_u64),
        expires_at: value
            .get("expires_at")
            .or_else(|| value.get("expirationTime"))
            .and_then(json_optional_scalar_string),
        // Gemini only returns content for generated files with a download URI.
        // `download` uses the canonical API endpoint, so credentials are never
        // forwarded to the metadata-provided URI.
        downloadable: if adapter == Adapter::Gemini {
            Some(
                value.get("source").and_then(Value::as_str) == Some("GENERATED")
                    && value
                        .get("downloadUri")
                        .and_then(nonempty_string)
                        .is_some_and(|uri| valid_download_uri(&uri)),
            )
        } else if adapter == Adapter::MiniMax {
            value
                .get("download_url")
                .or_else(|| value.get("downloadable"))
                .and_then(|v| v.as_bool().or_else(|| nonempty_string(v).map(|_| true)))
        } else {
            value.get("downloadable").and_then(Value::as_bool)
        },
        purpose,
    };
    Ok(ProviderFileMetadata {
        file,
        created_at: value
            .get("created_at")
            .or_else(|| value.get("createTime"))
            .and_then(json_optional_scalar_string),
        status: value
            .get("status")
            .or_else(|| value.get("state"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

fn json_optional_scalar_string(value: &Value) -> Option<String> {
    (!value.is_null()).then(|| {
        value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string())
    })
}

fn json_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

async fn async_delay(duration: Duration) {
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::time::sleep(duration).await;
        return;
    }
    let (send, receive) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(duration);
        let _ = send.send(());
    });
    let _ = receive.await;
}

fn gemini_timeout_remaining(deadline: Instant) -> Result<Duration, LlmError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        Err(gemini_file_timeout())
    } else {
        Ok(remaining)
    }
}

fn gemini_file_timeout() -> LlmError {
    LlmError::TransportTimeout {
        message: format!("Gemini file upload or processing timed out after {FILE_TIMEOUT:?}"),
    }
}

fn nonempty_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn valid_download_uri(uri: &str) -> bool {
    let Ok(uri) = reqwest::Url::parse(uri) else {
        return false;
    };
    uri.scheme() == "https"
        && uri.host_str().is_some()
        && uri.username().is_empty()
        && uri.password().is_none()
}

fn same_origin(left: &str, right: &str) -> bool {
    let (Ok(left), Ok(right)) = (reqwest::Url::parse(left), reqwest::Url::parse(right)) else {
        return false;
    };
    left.scheme() == "https"
        && left.username().is_empty()
        && left.password().is_none()
        && left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn sanitize_filename(value: &str) -> String {
    let leaf = value.rsplit(['/', '\\']).next().unwrap_or(value);
    leaf.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ' ') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches(|ch: char| ch == '.' || ch.is_ascii_whitespace())
        .chars()
        .take(180)
        .collect::<String>()
        .pipe_nonempty("attachment")
}

trait PipeNonempty {
    fn pipe_nonempty(self, fallback: &str) -> String;
}
impl PipeNonempty for String {
    fn pipe_nonempty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_owned()
        } else {
            self
        }
    }
}

fn unsupported(operation: &str) -> LlmError {
    LlmError::UnsupportedCapability {
        message: format!("provider file {operation} is not supported for this profile"),
    }
}

fn provider_shape(message: &str) -> LlmError {
    LlmError::ProviderInternal {
        message: message.to_owned(),
    }
}
