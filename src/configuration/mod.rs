//! Layered provider configuration and local persistence.
mod coordinator;
mod merge;
mod model;
mod repository;
mod sync;
use crate::{client::BuildError, presets::PresetError, protocol::LlmError};
pub(crate) use coordinator::*;
pub(crate) use merge::*;
pub(crate) use model::{empty_model, values, SavedConfig, SavedProfile};
pub use model::{ConfiguredModel, DefinitionRef, FieldOverride, ModelField};
pub use sync::{ProviderSyncOperation, ProviderSyncResult};
use thiserror::Error;
const MAX_PAGES: usize = 100;
#[derive(Debug, Error)]
pub enum ProviderStoreError {
    #[error("invalid model override: {0}")]
    InvalidModelOverride(String),
    #[error("provider configuration directory has not been set")]
    NotConfigured,
    #[error("provider profile {0:?} was not found")]
    UnknownProfile(String),
    #[error("provider profile {0:?} is not a built-in preset")]
    NotBuiltin(String),
    #[error("provider profile {0:?} changed while its model directory was being synced")]
    ProfileChanged(String),
    #[error("model {model:?} was not found on provider profile {profile_name:?}")]
    UnknownModel { profile_name: String, model: String },
    #[error("model {model:?} is not tracked for provider {provider_id:?}")]
    UntrackedModel { provider_id: String, model: String },
    #[error("provider profile {0:?} does not publish a supported model directory")]
    NoDirectory(String),
    #[error("provider profile {0:?} contains a static credential that cannot be saved")]
    StaticCredential(String),
    #[error("provider profile {0:?} has a credential-bearing header in extra.headers")]
    CredentialHeader(String),
    #[error("provider profile {0:?} is duplicated in the saved configuration")]
    DuplicateProfile(String),
    #[error("unsupported provider configuration version {0}")]
    UnsupportedVersion(u32),
    #[error("model directory pagination exceeded {MAX_PAGES} pages or repeated a cursor")]
    InvalidPagination,
    #[error("provider configuration worker failed: {0}")]
    Worker(String),
    #[error("provider configuration I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("provider configuration JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Build(#[from] BuildError),
    #[error(transparent)]
    Preset(#[from] PresetError),
    #[error(transparent)]
    Directory(#[from] LlmError),
}

pub(crate) use sync::CancelCommit;
