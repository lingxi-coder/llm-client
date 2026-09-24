//! Provider file inputs and results.
use super::*;

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
