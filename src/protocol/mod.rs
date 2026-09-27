//! Provider-neutral data for LLM communication.
//!
//! Requests, responses, messages, provider configuration, and errors belong to
//! the client. Callers own tool execution, conversation storage, permissions,
//! credential refresh, and context-compaction decisions.

mod anthropic_client_toolsets;
mod anthropic_fetch_sources;
mod continuation;
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

pub use anthropic_client_toolsets::{
    AnthropicBrowserMember, AnthropicBrowserToolsetConfig, AnthropicClientToolConfig,
    AnthropicClientToolset, AnthropicComputerMember, AnthropicComputerToolsetConfig,
};
pub use anthropic_fetch_sources::{
    AnthropicFetchToolReference, AnthropicFetchToolResultsSources, AnthropicFetchUrlSources,
    AnthropicFetchUserInputSources,
};
pub use continuation::ContinuationRef;
pub use ids::{ProviderId, ResponseId, ToolUseId};
pub use image::*;
pub use llm::{
    AnthropicCodeExecutionConfig, AnthropicContainerMetadata, AnthropicContainerRef,
    AnthropicContainerScope, AnthropicFetchCaller, AnthropicFetchResponseInclusion,
    AnthropicMcpCacheControl, AnthropicMcpCacheTtl, AnthropicMcpConfig, AnthropicMcpTool,
    AnthropicMcpToolConfig, AnthropicSkillRef, AnthropicSkillScope, AnthropicSkillType,
    AnthropicToolCaller, AnthropicToolChange, AnthropicToolReference, AnthropicToolSearchConfig,
    AnthropicToolSearchStrategy, AnthropicWebFetchConfig, AnthropicWebFetchVersion,
    CapabilitySupport, ChatRequest, ChatResponse, CodeInterpreterConfig,
    CodeInterpreterMemoryLimit, FileSearchConfig, FileSearchHit, FileSearchResult, GeminiLatLng,
    GeminiMapsGroundingConfig, HostedTool, LlmError, LlmErrorKind, McpApprovalPolicy,
    McpApprovalRequest, McpApprovalResponse, ModelCapability, ModelCapabilitySupport, ModelListing,
    OpenAiToolSearchConfig, OpenAiToolSearchExecution, OpenRouterContainerMetadata,
    OpenRouterContainerRef, OpenRouterContainerScope, OpenRouterShellConfig, OpenRouterShellEngine,
    OpenRouterShellEnvironment, OpenRouterShellNetworkPolicy, OpenRouterToolSearchConfig,
    ProviderListing, RemoteMcpConfig, ReportedCost, ResponseCacheObservation, ResponseCacheStatus,
    ServerToolUsage, StopReason, StreamEvent, SystemBlock, ToolChoice, ToolSpec, Usage,
    WebCitation, WebSearchConfig, WebSearchResult, XaiRemoteMcpConfig,
};
pub use message::{
    AnthropicClearAt, AnthropicMessageEffort, AnthropicMessageOptions, AttachmentRef, ContentBlock,
    ConversationMessage, DocumentSource, ImageSource, MessageRole, ProviderFileSource, VideoSource,
};
pub use provider::{
    AuthStrategy, AzureConfig, BatchPricing, BillingMode, ConnectionSpec, CredentialConfig,
    DirectoryRoute, FailoverTriggers, FoundryDeployment, FoundryHosting, ModelMetadata,
    ModelProfile, PeakSchedule, PricingConfig, ProtocolFamily, ProviderInfo, ProviderProfile,
    Region, SigningConfig, Submission, TokenPricing,
};
pub use secret::Secret;

mod usage_report;
pub use usage_report::{UsageReport, UsageState};

pub mod structured;
pub use structured::{OutputFormat, StructuredOutputError, StructuredOutputErrorKind};

pub mod cache;
pub use cache::{
    CacheBreakpoint, CachePosition, CacheTtl, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
    OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, PromptCachePolicy,
};

pub mod service;
pub use service::{ServiceAuth, ServiceSetting};
