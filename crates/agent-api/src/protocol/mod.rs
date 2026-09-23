//! The data model (design §5 layer `wire`, now this module).
//!
//! Plain data only: no behaviour traits, no I/O, no dependency on any workspace
//! crate. Everything a host's capability contract, the frontend
//! (`frontend-api`)
//! and the LLM layer (`llm-client`) exchange lives here, below all three.
//! `scripts/check-protocol-is-data.sh` is the gate: no `dyn`, `BoxFuture`,
//! `BoxStream`, `Mutex`, `Fn*(` or `impl Future` in this directory.
//!
//! Layering note against §5's one-line description ("对话数据模型 + Scope +
//! Origin"): the LLM request/response/error/event types, the permission
//! request/decision types, `ProviderProfile` and `AgentSettings` also live
//! here. The host's `Compactor`, `ToolUseContext` and `Frontend`
//! (in `frontend-api`) need them, and neither may depend on `llm-client`; a
//! provider profile is consumed by the codec capabilities, `llm-client` and
//! `config` alike, and exists once (review C2). Moving the *data* down keeps
//! the §5 direction.

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
