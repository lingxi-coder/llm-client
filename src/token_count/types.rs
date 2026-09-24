use crate::{client::ResolveError, protocol::ProviderId};
use serde::{Deserialize, Serialize};
use thiserror::Error;
/// A local estimate of the input tokens for a completion request.
///
/// Text is counted with the selected model's tokenizer. Message framing and
/// provider-specific prompt serialization are estimates, so `is_estimate` is
/// always true. `is_partial` is true when server-side or multimodal input could
/// not be counted locally; those parts are listed in `uncounted_components`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTokenEstimate {
    /// Estimated input tokens represented by locally available request data.
    pub input_tokens: u64,
    /// Profile selected by route resolution. A failover route is estimated for
    /// its first profile only.
    pub profile_name: String,
    pub provider_id: ProviderId,
    pub request_model: String,
    /// Tokenizer and pinned asset family used for the text counts.
    pub tokenizer: String,
    /// True because local message framing can differ from the provider's wire.
    pub is_estimate: bool,
    /// True when one or more request components were unavailable locally.
    pub is_partial: bool,
    /// Components omitted from the estimate. Entries may repeat when the
    /// request contains multiple items of the same kind.
    pub uncounted_components: Vec<LocalTokenEstimateOmission>,
}

/// A part of a request that cannot be counted from local text alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTokenEstimateOmission {
    ImageInput,
    DocumentInput,
    VideoInput,
    ProviderFileInput,
    HostedWebSearchContext,
    HostedFileSearchContext,
    PreviousResponseState,
    ProviderOpaqueContent,
    ProviderSignature,
    ProviderMetadata,
}

/// Why the client could not return a local token estimate.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LocalTokenCountError {
    #[error("local token estimation requires the {feature:?} feature")]
    FeatureDisabled { feature: String },
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    #[error(
        "no bundled local tokenizer is mapped for provider {provider_id:?}, model {request_model:?} on profile {profile_name:?}"
    )]
    UnsupportedModel {
        profile_name: String,
        provider_id: ProviderId,
        request_model: String,
    },
    #[error("failed to load local tokenizer {tokenizer:?}: {message}")]
    TokenizerInitialization { tokenizer: String, message: String },
    #[error("failed to tokenize request content: {0}")]
    Tokenization(String),
    #[error("failed to serialize structured request content: {0}")]
    Serialization(String),
}
