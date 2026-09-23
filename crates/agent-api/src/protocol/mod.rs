//! Shared protocol data model.
//!
//! Messages, LLM requests and responses, provider profiles, permissions, and
//! settings are plain serializable data. This module does not depend on the
//! client crate or any agent runtime.

pub mod ids;
pub mod llm;
pub mod message;
pub mod provider;
pub mod secret;

#[cfg(feature = "agent")]
pub mod compaction;
#[cfg(feature = "agent")]
pub mod error;
#[cfg(feature = "agent")]
pub mod events;
#[cfg(feature = "agent")]
pub mod hook_payload;
#[cfg(feature = "agent")]
pub mod hooks;
#[cfg(feature = "agent")]
pub mod origin;
#[cfg(feature = "agent")]
pub mod permission;
#[cfg(feature = "agent")]
pub mod queue;
#[cfg(feature = "agent")]
pub mod scope;
#[cfg(feature = "agent")]
pub mod settings;
#[cfg(feature = "agent")]
pub mod task;
#[cfg(feature = "agent")]
pub mod tool;
#[cfg(feature = "agent")]
pub mod transcript;

#[cfg(feature = "agent")]
pub use compaction::{CompactError, CompactionResult};
#[cfg(feature = "agent")]
pub use error::TurnError;
#[cfg(feature = "agent")]
pub use events::{AgentEvent, AgentEventKind, ModeSwitchBy, ToolResultSummary};
#[cfg(feature = "agent")]
pub use hooks::{HookDecision, HookEventType, HookResponse};
#[cfg(feature = "agent")]
pub use ids::{AgentId, AskId, BatchId, CompId, EffectId, ModeName, PluginId, SessionId, TurnId};
pub use ids::{ProviderId, ResponseId, ToolUseId};
pub use llm::{
    CapabilitySupport, CompactTrigger, CompletionRequest, CompletionResponse, FileSearchConfig,
    FileSearchHit, FileSearchResult, LlmError, LlmErrorKind, ModelCapabilities, ModelCapability,
    ModelCapabilitySupport, ModelListing, ProviderListing, ReportedCost, ServerToolUsage,
    StopReason, StreamEvent, SystemBlock, ThinkingConfig, TokenEstimate, TokenEstimateSource,
    ToolChoice, ToolSpec, Usage, WebCitation, WebSearchConfig, WebSearchResult,
};
pub use message::{
    AttachmentRef, ContentBlock, ConversationMessage, DocumentSource, ImageSource, MessageRole,
    ProviderFileSource, VideoSource,
};
#[cfg(feature = "agent")]
pub use origin::{Origin, OriginSource};
#[cfg(feature = "agent")]
pub use permission::{
    Adjudication, DecisionReason, ElicitationRequest, ElicitationResponse, ElicitationSource,
    HookVerdict, PermissionDecision, PermissionMode, PermissionPrompt, PermissionRequest,
    PermissionResult, PermissionRule, PermissionSuggestion, RuleBehavior, RuleSetView,
};
pub use provider::{
    AuthStrategy, AzureConfig, BatchPricing, BillingMode, ConnectionSpec, CredentialConfig,
    DirectoryRoute, FailoverTriggers, ModelMetadata, ModelProfile, PeakSchedule, PricingConfig,
    ProtocolFamily, ProviderInfo, ProviderProfile, SigningConfig, Submission, TokenPricing,
};
#[cfg(feature = "agent")]
pub use queue::{Priority, QueuedCommand, QueuedCommandKind};
#[cfg(feature = "agent")]
pub use scope::{ManifestKind, Scope};
pub use secret::Secret;
#[cfg(feature = "agent")]
pub use settings::AgentSettings;
#[cfg(feature = "agent")]
pub use task::{
    ShutdownId, TaskActivity, TaskAssociation, TaskCapabilities, TaskCapability, TaskEpoch,
    TaskGeneration, TaskId, TaskIdentity, TaskMessage, TaskMessageSource, TaskOutcome, TaskOwner,
    TaskSnapshot, TaskState,
};
#[cfg(feature = "agent")]
pub use tool::{InterruptBehavior, ToolError, ToolResultData, ValidatedContextUpdate};
#[cfg(feature = "agent")]
pub use transcript::{Backup, SessionMetadata, SnapshotRecord, ToolUseSummary, TranscriptEntry};
