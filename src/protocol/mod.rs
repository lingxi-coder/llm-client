//! Provider-neutral data for LLM communication.
//!
//! Requests, responses, messages, provider configuration, and errors belong to
//! the client. Callers own tool execution, conversation storage, permissions,
//! credential refresh, and context-compaction decisions.

pub mod ids;
pub mod image;
mod inference;
pub mod llm;
pub mod message;
mod price_quote;
pub mod provider;
pub mod secret;
pub use inference::*;
pub use price_quote::*;

pub use ids::{ProviderId, ResponseId, ToolUseId};
pub use image::*;
pub use llm::{
    CapabilitySupport, CompletionRequest, CompletionResponse, FileSearchConfig, FileSearchHit,
    FileSearchResult, LlmError, LlmErrorKind, ModelCapability, ModelCapabilitySupport,
    ModelListing, ProviderListing, ReportedCost, ServerToolUsage, StopReason, StreamEvent,
    SystemBlock, ToolChoice, ToolSpec, Usage, WebCitation, WebSearchConfig, WebSearchResult,
};
pub use message::{
    AttachmentRef, ContentBlock, ConversationMessage, DocumentSource, ImageSource, MessageRole,
    ProviderFileSource, VideoSource,
};
pub use provider::{
    AuthStrategy, AzureConfig, BatchPricing, BillingMode, ConnectionSpec, CredentialConfig,
    DirectoryRoute, FailoverTriggers, ModelMetadata, ModelProfile, PeakSchedule, PricingConfig,
    ProtocolFamily, ProviderInfo, ProviderProfile, Region, SigningConfig, Submission, TokenPricing,
};
pub use secret::Secret;

mod usage_report;
pub use usage_report::{UsageReport, UsageState};
