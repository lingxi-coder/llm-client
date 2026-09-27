//! Provider file inputs and results.
use super::*;

/// A one-shot file input whose bytes are yielded directly to the HTTP
/// transport. The declared size is checked against the exact stream length.
pub struct UploadFileStream {
    filename: String,
    media_type: String,
    size_bytes: u64,
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl UploadFileStream {
    /// Construct an upload stream. The stream is consumed once by
    /// [`FileService::upload_stream`](super::FileService::upload_stream).
    pub fn new<S>(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        size_bytes: u64,
        body: S,
    ) -> Self
    where
        S: futures::Stream<Item = Result<Bytes, LlmError>> + Send + 'static,
    {
        Self {
            filename: filename.into(),
            media_type: media_type.into(),
            size_bytes,
            body: body.boxed(),
        }
    }

    /// Build a one-chunk stream from bytes for callers that already have the
    /// entire file in memory.
    pub fn from_bytes(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        bytes: impl Into<Bytes>,
    ) -> Self {
        let bytes = bytes.into();
        let size_bytes = bytes.len() as u64;
        Self::new(
            filename,
            media_type,
            size_bytes,
            stream::once(async move { Ok(bytes) }),
        )
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        String,
        String,
        u64,
        BoxStream<'static, Result<Bytes, LlmError>>,
    ) {
        (self.filename, self.media_type, self.size_bytes, self.body)
    }
}

/// Result of a streamed upload. A transport interruption after dispatch is
/// represented separately because the provider may have accepted the file.
#[derive(Debug, thiserror::Error)]
pub enum FileUploadError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("file upload outcome is unknown during {operation}: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: Box<LlmError>,
        /// A scoped reference when the provider supplied one before the
        /// interruption; currently absent when upload bytes are in flight.
        reference: Option<Box<ProviderFileRef>>,
    },
}

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
    /// JSONL input for a provider Batch API job. The wire representation is
    /// adapter-specific; xAI's Batch upload omits the optional `purpose` field.
    Batch,
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
    pub(crate) fn unsupported() -> Self {
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
    /// Deterministic, non-secret fingerprint of the configured file endpoint.
    /// Foundry uses its canonical resource base. The URL itself, including
    /// userinfo or query parameters, is not kept.
    pub endpoint_fingerprint: String,
    pub account_scope: Option<String>,
    pub protocol: ProtocolFamily,
    pub file_id: String,
    /// A provider model-input URI, when the API requires one (Gemini).
    pub uri: Option<String>,
    pub filename: Option<String>,
    pub media_type: Option<String>,
    pub size_bytes: Option<u64>,
    /// Provider-reported expiry, retained in its original timestamp format.
    pub expires_at: Option<String>,
    /// Provider-reported processing status, retained verbatim for local
    /// readiness checks. It is not sent as model-input data.
    pub processing_status: Option<String>,
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
            expires_at: self.expires_at.clone(),
            processing_status: self.processing_status.clone(),
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
