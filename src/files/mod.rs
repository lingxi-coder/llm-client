//! Provider file APIs, kept separate from durable application attachments.
//!
//! A [`ProviderFileRef`] belongs to one concrete profile/account. It may be
//! useful as model input for a particular wire, but it is never a display URL
//! and must not replace the app-owned attachment in conversation history.

pub(crate) use crate::auth::Authenticator;
pub(crate) use crate::protocol::{
    AuthStrategy, ChatRequest, ContentBlock, DocumentSource, ImageSource, LlmError, ModelProfile,
    ProtocolFamily, ProviderFileSource, ProviderId, ProviderProfile, Secret, VideoSource,
};
pub(crate) use crate::transport::{HttpRequest, HttpResponse, Transport};
pub(crate) use bytes::{Bytes, BytesMut};
pub(crate) use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
pub(crate) use serde_json::Value;
pub(crate) use std::future::Future;
pub(crate) use std::sync::{Arc, Mutex};
pub(crate) use std::time::{Duration, Instant};

mod adapters;
mod http;
mod lifecycle;
mod policy;
mod service;
mod types;
pub(crate) use adapters::*;
pub use http::provider_file_endpoint_fingerprint;
pub(crate) use http::*;
pub(crate) use lifecycle::*;
pub(crate) use policy::*;
pub use policy::{capabilities, capabilities_for_purpose};
pub use service::FileService;
pub(crate) use service::{
    buffered_upload_error, unknown_buffered_upload_outcome, unknown_upload_outcome,
};
pub use types::{
    DownloadSupport, FileCapabilities, FileOperation, FilePurpose, FileUploadError,
    ModelFileReference, ProviderFileContent, ProviderFileMetadata, ProviderFilePage,
    ProviderFileRef, UploadFile, UploadFileStream,
};

/// Maximum response body retained by [`FileService::download`].
pub const MAX_PROVIDER_FILE_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;

pub(crate) const FILE_TIMEOUT: Duration = Duration::from_secs(120);
/// Automatic uploads do not expose provider IDs to callers. Providers that
/// support an upload TTL receive one, and cached references expire earlier.
pub(crate) const AUTOMATIC_FILE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub(crate) const AUTOMATIC_FILE_CACHE_TTL: Duration = Duration::from_secs(23 * 60 * 60);

pub(crate) use crate::providers::anthropic::files::MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES;
pub(crate) use crate::providers::google::files::{is_gemini_video_type, GEMINI_VIDEO_FILE_TIMEOUT};
pub(crate) use crate::providers::minimax::files::{
    check_minimax_base_response, minimax_base_response_code, minimax_delete_purpose,
    minimax_list_purpose,
};
pub(crate) use crate::providers::qwen::files::{
    is_qwen_long_model, qwen_long_region_supported, valid_qwen_file_id, validate_qwen_long_blocks,
    validate_qwen_long_inputs, QwenFileRateLimiter, QWEN_CLEANUP_DEFAULT_TIMEOUT,
    QWEN_LONG_MAX_FILE_REFERENCES,
};
