//! Shared protocol data model.
//!
//! Messages, LLM requests and responses, provider profiles, permissions, and
//! settings are plain serializable data. This module does not depend on the
//! client crate or any agent runtime.

pub mod compaction;
pub mod error;
pub mod events;
pub mod hook_payload;
pub mod hooks;
pub mod ids;
pub mod llm;
pub mod message;
pub mod origin;
pub mod permission;
pub mod provider;
pub mod queue;
pub mod scope;
pub mod secret;
pub mod settings;
pub mod task;
pub mod tool;
pub mod transcript;

pub use compaction::{CompactError, CompactionResult};
pub use error::TurnError;
pub use events::{AgentEvent, AgentEventKind, ModeSwitchBy, ToolResultSummary};
pub use hooks::{HookDecision, HookEventType, HookResponse};
pub use ids::{
    AgentId, AskId, BatchId, CompId, EffectId, ModeName, PluginId, ProviderId, ResponseId,
    SessionId, ToolUseId, TurnId,
};
pub use llm::{
    CompactTrigger, CompletionRequest, CompletionResponse, LlmError, LlmErrorKind,
    ModelCapabilities, ModelListing, ProviderListing, ReportedCost, StopReason, StreamEvent,
    SystemBlock, ThinkingConfig, TokenEstimate, TokenEstimateSource, ToolChoice, ToolSpec, Usage,
    WebCitation, WebSearchConfig, WebSearchResult,
};
pub use message::{ContentBlock, ConversationMessage, DocumentSource, ImageSource, MessageRole};
pub use origin::{Origin, OriginSource};
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
pub use queue::{Priority, QueuedCommand, QueuedCommandKind};
pub use scope::{ManifestKind, Scope};
pub use secret::Secret;
pub use settings::AgentSettings;
pub use task::{
    ShutdownId, TaskActivity, TaskAssociation, TaskCapabilities, TaskCapability, TaskEpoch,
    TaskGeneration, TaskId, TaskIdentity, TaskMessage, TaskMessageSource, TaskOutcome, TaskOwner,
    TaskSnapshot, TaskState,
};
pub use tool::{InterruptBehavior, ToolError, ToolResultData, ValidatedContextUpdate};
pub use transcript::{Backup, SessionMetadata, SnapshotRecord, ToolUseSummary, TranscriptEntry};
