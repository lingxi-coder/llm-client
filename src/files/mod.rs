//! Provider file APIs, kept separate from durable application attachments.
//!
//! A [`ProviderFileRef`] belongs to one concrete profile/account. It may be
//! useful as model input for a particular wire, but it is never a display URL
//! and must not replace the app-owned attachment in conversation history.

use crate::auth::Authenticator;
use crate::protocol::{
    AuthStrategy, CompletionRequest, ContentBlock, DocumentSource, ImageSource, LlmError,
    ModelProfile, ProtocolFamily, ProviderFileSource, ProviderId, ProviderProfile, Secret,
    VideoSource,
};
use crate::transport::{HttpRequest, HttpResponse, Transport};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
pub use types::{
    DownloadSupport, FileCapabilities, FileOperation, FilePurpose, ModelFileReference,
    ProviderFileContent, ProviderFileMetadata, ProviderFilePage, ProviderFileRef, UploadFile,
};

/// Maximum response body retained by [`FileService::download`].
pub const MAX_PROVIDER_FILE_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;

const FILE_TIMEOUT: Duration = Duration::from_secs(120);
const GEMINI_FILE_POLL_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_GEMINI_PROCESSING_TIMEOUT: Duration = Duration::from_secs(10 * 60);
pub(crate) const GEMINI_VIDEO_FILE_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);
const GEMINI_FILE_RETENTION: Duration = Duration::from_secs(48 * 60 * 60);
const QWEN_FILE_POLL_INTERVAL: Duration = Duration::from_secs(1);
const QWEN_FILE_PROCESSING_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const QWEN_CLEANUP_DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const QWEN_UPLOAD_INTERVAL: Duration = Duration::from_millis(350);
const QWEN_METADATA_INTERVAL: Duration = Duration::from_millis(110);
const QWEN_IMAGE_MAX_UPLOAD_BYTES: u64 = 20_000_000;
pub(crate) const QWEN_LONG_MAX_FILE_REFERENCES: usize = 100;
const GEMINI_PDF_MAX_UPLOAD_BYTES: u64 = 50_000_000;
const OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES: u64 = 50_000_000;
/// Automatic uploads do not expose provider IDs to callers. Providers that
/// support an upload TTL receive one, and cached references expire earlier.
pub(crate) const AUTOMATIC_FILE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub(crate) const AUTOMATIC_FILE_CACHE_TTL: Duration = Duration::from_secs(23 * 60 * 60);
/// The Anthropic request preflight reserves this many encoded bytes per
/// automatic file ID before the upload happens.
pub(crate) const MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES: usize = 512;
