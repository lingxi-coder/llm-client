//! LLM-facing data: request, response, stream events, model listings, and the
//! error taxonomy used by the client and its codecs.

use crate::protocol::ids::{ProviderId, ResponseId, ToolUseId};
use crate::protocol::message::{
    ContentBlock, ConversationMessage, MessageRole, ProviderFileSource,
};
use crate::protocol::provider::{
    AuthStrategy, BillingMode, FoundryDeployment, FoundryHosting, ProtocolFamily, ProviderInfo,
    ProviderProfile, Region, TokenPricing,
};
use crate::protocol::ContinuationRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;

/// Provider and client communication failures. HTTP status details, when
/// available, are included in `message`; callers can classify failures by
/// variant or [`LlmError::kind`]. Recovery decisions belong to the caller.
///
/// Deliberately not `#[non_exhaustive]`: downstream `match`es must fail to
/// compile when a variant is added, so no branch dies silently.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmError {
    #[error("authentication failed: {message}")]
    Authentication { message: String },
    #[error("permission denied by provider: {message}")]
    PermissionDenied { message: String },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("quota exceeded: {message}")]
    QuotaExceeded { message: String },
    /// The input no longer fits the model's context window.
    #[error("context window exceeded: {message}")]
    ContextOverflow {
        message: String,
        limit: Option<u64>,
        actual: Option<u64>,
    },
    /// The request body itself is over the provider's byte limit.
    #[error("request too large: {message}")]
    RequestTooLarge { message: String },
    #[error("model unavailable: {message}")]
    ModelUnavailable { message: String },
    #[error("provider internal error: {message}")]
    ProviderInternal { message: String },
    #[error("provider overloaded: {message}")]
    Overloaded { message: String },
    #[error("transport error: {message}")]
    Transport { message: String },
    #[error("transport timeout: {message}")]
    TransportTimeout { message: String },
    /// Upload succeeded, but readiness is pending or could not be confirmed.
    /// Keep this scoped reference so the caller can resume polling.
    #[error("provider file processing is unresolved: {message}")]
    ProviderFileProcessing {
        message: String,
        file: Box<ProviderFileSource>,
    },
    #[error("TLS certificate error: {message}")]
    TlsCert { message: String },
    #[error("stream interrupted: {message}")]
    StreamInterrupted { message: String },
    #[error("cost unavailable: {message}")]
    CostUnavailable { message: String },
    #[error("unsupported capability: {message}")]
    UnsupportedCapability { message: String },
}

/// Field-less mirror of [`LlmError`] for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmErrorKind {
    Authentication,
    PermissionDenied,
    InvalidRequest,
    RateLimited,
    QuotaExceeded,
    ContextOverflow,
    RequestTooLarge,
    ModelUnavailable,
    ProviderInternal,
    Overloaded,
    Transport,
    TransportTimeout,
    ProviderFileProcessing,
    TlsCert,
    StreamInterrupted,
    CostUnavailable,
    UnsupportedCapability,
}

impl LlmErrorKind {
    pub const ALL: [LlmErrorKind; 17] = [
        LlmErrorKind::Authentication,
        LlmErrorKind::PermissionDenied,
        LlmErrorKind::InvalidRequest,
        LlmErrorKind::RateLimited,
        LlmErrorKind::QuotaExceeded,
        LlmErrorKind::ContextOverflow,
        LlmErrorKind::RequestTooLarge,
        LlmErrorKind::ModelUnavailable,
        LlmErrorKind::ProviderInternal,
        LlmErrorKind::Overloaded,
        LlmErrorKind::Transport,
        LlmErrorKind::TransportTimeout,
        LlmErrorKind::ProviderFileProcessing,
        LlmErrorKind::TlsCert,
        LlmErrorKind::StreamInterrupted,
        LlmErrorKind::CostUnavailable,
        LlmErrorKind::UnsupportedCapability,
    ];
}

impl LlmError {
    pub fn kind(&self) -> LlmErrorKind {
        match self {
            LlmError::Authentication { .. } => LlmErrorKind::Authentication,
            LlmError::PermissionDenied { .. } => LlmErrorKind::PermissionDenied,
            LlmError::InvalidRequest { .. } => LlmErrorKind::InvalidRequest,
            LlmError::RateLimited { .. } => LlmErrorKind::RateLimited,
            LlmError::QuotaExceeded { .. } => LlmErrorKind::QuotaExceeded,
            LlmError::ContextOverflow { .. } => LlmErrorKind::ContextOverflow,
            LlmError::RequestTooLarge { .. } => LlmErrorKind::RequestTooLarge,
            LlmError::ModelUnavailable { .. } => LlmErrorKind::ModelUnavailable,
            LlmError::ProviderInternal { .. } => LlmErrorKind::ProviderInternal,
            LlmError::Overloaded { .. } => LlmErrorKind::Overloaded,
            LlmError::Transport { .. } => LlmErrorKind::Transport,
            LlmError::TransportTimeout { .. } => LlmErrorKind::TransportTimeout,
            LlmError::ProviderFileProcessing { .. } => LlmErrorKind::ProviderFileProcessing,
            LlmError::TlsCert { .. } => LlmErrorKind::TlsCert,
            LlmError::StreamInterrupted { .. } => LlmErrorKind::StreamInterrupted,
            LlmError::CostUnavailable { .. } => LlmErrorKind::CostUnavailable,
            LlmError::UnsupportedCapability { .. } => LlmErrorKind::UnsupportedCapability,
        }
    }
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    Other(String),
}

/// One response's token counts, normalized across wires.
///
/// **The four billable buckets are disjoint.** Every wire reports cached and
/// reasoning tokens differently — some as separate counters, most as subsets
/// already folded into a larger one — so each codec subtracts until these four
/// partition the total. Without that, `total()` counts the same token twice on
/// the wires that fold (see `Usage::total`).
/// The one-hour cache-write count below is a subset of `cache_write_tokens`.
///
/// `reasoning_tokens` is the exception and is **not** billable on its own: it is
/// a breakdown *of* `output_tokens`, carried because providers price thinking
/// separately and the count is otherwise unrecoverable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input. Excludes `cache_read_tokens` and `cache_write_tokens`.
    pub input_tokens: u64,
    /// Generated tokens, including the reasoning ones broken out below.
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// One-hour cache-write tokens, already included in `cache_write_tokens`.
    /// Separate for pricing because this TTL has a different rate from the
    /// ordinary cache-write rate. `total()` must not add this subset again.
    #[serde(default)]
    pub cache_write_1h_tokens: u64,
    /// How many of `output_tokens` were reasoning. A subset, never an addend.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// What the provider said this cost, when it says so at all.
    ///
    /// `None` means unknown, never free. Most first-party providers report token
    /// counts and no money at all, so this is absent far more often than it is
    /// present; in practice only an aggregator, which knows which upstream it
    /// routed to, can say. Nothing here derives the figure from a price list: a
    /// number the provider did not send is a guess, and a guess is not a bill.
    #[serde(default)]
    pub cost: Option<ReportedCost>,
    /// Provider-reported hosted tool calls. These are kept separate from
    /// token totals because providers bill and count them independently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_usage: Option<ServerToolUsage>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerToolUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u64>,
    /// Native Web Fetch calls; a reported count does not imply a per-call fee.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_fetch_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_extractor_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_search_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_interpreter_requests: Option<u64>,
}

/// A cost a provider reported, in nano-USD.
///
/// Integer, not a float, for two reasons: money should not accumulate rounding
/// error over a long session, and `Usage` stays `Copy + Eq` so callers can
/// still compare whole states exactly. Nano rather than micro because a single
/// call can cost well under a micro-dollar, which micros would round to zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReportedCost {
    pub nano_usd: u64,
}

impl ReportedCost {
    pub const ZERO: Self = Self { nano_usd: 0 };

    /// From a provider's decimal USD amount.
    ///
    /// A negative, NaN or absurd amount is not a cost, so it is rejected rather
    /// than wrapped into a nonsense integer.
    #[must_use]
    pub fn from_usd(amount: f64) -> Option<Self> {
        if !amount.is_finite() || amount < 0.0 {
            return None;
        }
        let nanos = amount * 1e9;
        if nanos > u64::MAX as f64 {
            return None;
        }
        Some(Self {
            nano_usd: nanos.round() as u64,
        })
    }

    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            nano_usd: self.nano_usd.saturating_add(other.nano_usd),
        }
    }
}

impl Usage {
    /// Every token the request was billed for, counted once.
    ///
    /// `reasoning_tokens` is deliberately absent: it is already inside
    /// `output_tokens`, and adding it would over-count every thinking turn.
    /// If a provider reports counters whose sum exceeds `u64::MAX`, the result
    /// saturates rather than panicking in debug builds or wrapping in release.
    pub fn total(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.output_tokens)
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }
}

/// Provider-neutral stream events. Block indices are per response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Provider-reported inference settings; may arrive before final usage.
    Inference {
        report: super::InferenceReport,
    },
    /// Complete native replay data for this output block. Preserve it as a
    /// `ContentBlock::ProviderContent` in the assistant transcript. Reasoning
    /// deltas at the same index are display text, not a replacement for this
    /// payload. Replaying on another protocol is rejected.
    /// Chat reasoning envelopes carry metadata for the enclosing assistant
    /// message and are emitted once before the terminal event.
    ProviderContent {
        block: usize,
        protocol: ProtocolFamily,
        value: Value,
    },
    /// Raw provider stream frame that the codec does not interpret. This is
    /// observational data, not a content block: do not replay it as message
    /// content or execute it as a client tool call.
    ProviderEvent {
        protocol: ProtocolFamily,
        payload: Value,
    },
    Start {
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<ResponseId>,
    },
    TextDelta {
        block: usize,
        text: String,
    },
    ReasoningDelta {
        block: usize,
        text: String,
    },
    ThoughtSignature {
        block: usize,
        signature: String,
    },
    RedactedThinking {
        block: usize,
        data: String,
    },
    ToolCallDelta {
        block: usize,
        id: ToolUseId,
        /// Provider-issued call ID, when one was present on the wire.
        /// Absent for locally generated IDs and protocols that do not need it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_id: Option<String>,
        /// Native caller metadata when the provider routes a tool call through
        /// another hosted tool, as Anthropic does for programmatic calls.
        /// Keep the original object so callers can distinguish direct calls
        /// and retain future provider fields.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
        /// Anthropic client-toolset family for Browser/Computer member calls.
        /// Carried from the opening content-block event to each input delta.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
        name: String,
        arguments_fragment: String,
    },
    /// Search attribution records received in this frame. These are not
    /// client-executed tool calls. Native metadata retains citation locations.
    WebSearch {
        result: WebSearchResult,
    },
    /// Provider-hosted knowledge-base retrieval. This is attribution data,
    /// not a client tool invocation.
    FileSearch {
        result: FileSearchResult,
    },
    End {
        stop_reason: StopReason,
        usage: super::UsageReport,
        #[serde(default)]
        inference: super::InferenceReport,
    },
}

/// A system-prompt block. Cache breakpoints belong to `ChatRequest.prompt_cache`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemBlock {
    pub text: String,
}

/// Anthropic Messages caller classes for a user-defined tool.
///
/// Other codecs reject non-empty [`ToolSpec::allowed_callers`] rather than
/// silently dropping this provider-specific policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnthropicToolCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521,
}

/// A tool as advertised on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default)]
    pub strict: bool,
    /// Ask supported hosted tool search to load this definition only after
    /// discovery. Anthropic Messages, OpenAI Responses and OpenRouter support it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
    /// Anthropic Messages API or Anthropic-hosted Foundry caller classes for
    /// this tool. Programmatic callers require Code Execution on the same request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<AnthropicToolCaller>,
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Enable provider-hosted web search. The provider decides when to search.
/// Unsupported controls are rejected instead of silently discarded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_domains: Vec<String>,
    /// Maximum searches per request, currently supported by Anthropic only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
}

/// A cited web source. Native citation spans remain in search metadata because
/// providers use different block indices and offset units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebCitation {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Search attribution, including native annotations, grounding supports,
/// search suggestions and server-side search errors. Metadata is provider
/// specific; it must not be treated as executable client tool instructions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResult {
    #[serde(default)]
    pub citations: Vec<WebCitation>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FileSearchResult {
    #[serde(default)]
    pub queries: Vec<String>,
    #[serde(default)]
    pub hits: Vec<FileSearchHit>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileSearchHit {
    pub file_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchConfig {
    /// One Qwen Model Studio knowledge base ID. The service currently accepts
    /// a single ID per request.
    pub knowledge_base_id: String,
    /// Model Studio workspace used to construct the regional dedicated API
    /// host required by the knowledge retrieval endpoint.
    pub workspace_id: String,
}

/// Live Anthropic Messages Web Fetch tool variants. Newer variants add
/// provider capabilities; basic fetch remains selectable without dynamic filtering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicWebFetchVersion {
    #[serde(rename = "20250910")]
    V20250910,
    #[serde(rename = "20260209")]
    V20260209,
    #[serde(rename = "20260309")]
    V20260309,
    #[serde(rename = "20260318")]
    #[default]
    V20260318,
}

impl AnthropicWebFetchVersion {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V20250910 => "20250910",
            Self::V20260209 => "20260209",
            Self::V20260309 => "20260309",
            Self::V20260318 => "20260318",
        }
    }

    pub const fn supports_dynamic_filtering(self) -> bool {
        !matches!(self, Self::V20250910)
    }

    pub const fn supports_cache_bypass(self) -> bool {
        matches!(self, Self::V20260309 | Self::V20260318)
    }

    pub const fn supports_response_inclusion(self) -> bool {
        matches!(self, Self::V20260318)
    }
}

/// How Web Fetch results consumed by completed Code Execution calls appear in
/// the response. Direct calls and paused calls are always returned in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicFetchResponseInclusion {
    Full,
    Excluded,
}

/// Callers Anthropic may use to invoke Web Fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicFetchCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521,
}

/// Typed configuration for the first-party Anthropic Messages Web Fetch
/// server tool. The provider performs retrieval; the client never fetches URLs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicWebFetchConfig {
    pub version: AnthropicWebFetchVersion,
    /// Define Web Fetch in a mid-conversation `tool_addition` instead of the
    /// top-level `tools` array. The addition must exactly match this typed
    /// configuration and is supported only on the first-party Messages API.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inline_definition: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<AnthropicFetchCaller>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::protocol::cache::CacheTtl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_sources: Option<crate::protocol::AnthropicFetchUrlSources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_domains: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_cache: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_inclusion: Option<AnthropicFetchResponseInclusion>,
}

impl AnthropicWebFetchConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if !self.allowed_domains.is_empty() && !self.blocked_domains.is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic Web Fetch accepts allowed_domains or blocked_domains, not both"
                    .into(),
            });
        }
        if self.use_cache.is_some() && !self.version.supports_cache_bypass() {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch use_cache requires version 20260309 or later".into(),
            });
        }
        if self.response_inclusion.is_some() && !self.version.supports_response_inclusion() {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch response_inclusion requires version 20260318".into(),
            });
        }
        if self.cache_control == Some(crate::protocol::cache::CacheTtl::ThirtyMinutes) {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch cache_control supports only 5m or 1h".into(),
            });
        }
        if self.defer_loading && self.cache_control.is_some() {
            return Err(LlmError::InvalidRequest {
                message: "deferred Anthropic Web Fetch tools cannot have cache_control".into(),
            });
        }
        for domain in self.allowed_domains.iter().chain(&self.blocked_domains) {
            if domain.trim().is_empty()
                || domain.contains("://")
                || domain.chars().any(char::is_whitespace)
                || domain.chars().any(char::is_control)
                || match domain.split_once('/') {
                    Some((host, _)) => host.is_empty() || host.contains('*'),
                    None => domain.contains('*'),
                }
            {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Web Fetch domain filters must omit schemes and whitespace; wildcards are allowed only in paths".into(),
                });
            }
        }
        Ok(())
    }
}

/// A tool executed by the provider. Ordinary caller-executed functions remain
/// in `ChatRequest.tools` and require the host to return their results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum HostedTool {
    WebSearch(WebSearchConfig),
    /// Qwen Responses web extraction. Qwen requires a hosted web-search tool
    /// in the same request; this marker has no provider-side configuration.
    WebExtractor,
    FileSearch(FileSearchConfig),
    CodeInterpreter(CodeInterpreterConfig),
    /// OpenRouter-hosted regex tool search on Responses and Messages APIs.
    OpenRouterToolSearch(OpenRouterToolSearchConfig),
    /// OpenAI Responses function and MCP tool discovery.
    OpenAiToolSearch(OpenAiToolSearchConfig),
    /// OpenRouter-hosted shell execution on Responses and Messages APIs.
    OpenRouterShell(OpenRouterShellConfig),
    /// Gemini Developer API server-side Python execution for GenerateContent.
    GeminiCodeExecution,
    /// Gemini Developer API URL Context retrieval for GenerateContent.
    GeminiUrlContext,
    /// Gemini Developer API Google Maps grounding for GenerateContent.
    GeminiMapsGrounding(GeminiMapsGroundingConfig),
    AnthropicToolSearch(AnthropicToolSearchConfig),
    /// First-party Anthropic Messages server-side Web Fetch.
    AnthropicWebFetch(AnthropicWebFetchConfig),
    /// Anthropic Messages sandbox execution on the first-party API or an
    /// Anthropic-hosted Foundry deployment, optionally reusing a scoped container.
    AnthropicCodeExecution(AnthropicCodeExecutionConfig),
    /// First-party Anthropic Messages remote MCP connector. Each value
    /// configures one URL server and its matching toolset.
    AnthropicMcp(AnthropicMcpConfig),
    /// An OpenAI Responses remote MCP server. The provider connects to this
    /// server and executes its tools; it does not create host-side calls.
    RemoteMcp(RemoteMcpConfig),
    /// An xAI Responses remote MCP server. This is intentionally separate from
    /// OpenAI's variant because xAI does not support OpenAI's approval fields.
    XaiRemoteMcp(XaiRemoteMcpConfig),
}

/// Optional widget-context support for Gemini Google Maps grounding.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiMapsGroundingConfig {
    /// Request the Maps widget context token when the caller renders the
    /// provider's map widget. `None` leaves the API default unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_widget: Option<bool>,
    /// Optional user location supplied to Maps retrieval. The provider API
    /// encodes this at `toolConfig.retrievalConfig.latLng`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lat_lng: Option<GeminiLatLng>,
}

/// Latitude and longitude used as context for Gemini Maps grounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiLatLng {
    pub latitude: f64,
    pub longitude: f64,
}

impl GeminiLatLng {
    pub fn new(latitude: f64, longitude: f64) -> Result<Self, LlmError> {
        let location = Self {
            latitude,
            longitude,
        };
        location.validate()?;
        Ok(location)
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if !self.latitude.is_finite() || !(-90.0..=90.0).contains(&self.latitude) {
            return Err(LlmError::InvalidRequest {
                message: "Gemini Maps latitude must be finite and within -90..=90".into(),
            });
        }
        if !self.longitude.is_finite() || !(-180.0..=180.0).contains(&self.longitude) {
            return Err(LlmError::InvalidRequest {
                message: "Gemini Maps longitude must be finite and within -180..=180".into(),
            });
        }
        Ok(())
    }
}

impl PartialEq for GeminiLatLng {
    fn eq(&self, other: &Self) -> bool {
        self.latitude.to_bits() == other.latitude.to_bits()
            && self.longitude.to_bits() == other.longitude.to_bits()
    }
}

impl Eq for GeminiLatLng {}

impl GeminiMapsGroundingConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if let Some(location) = &self.lat_lng {
            location.validate()?;
        }
        Ok(())
    }
}

/// Whether OpenAI must pause for host approval before invoking remote MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpApprovalPolicy {
    Always,
    Never,
}

/// Public, non-secret connection settings for an OpenAI Responses remote MCP server.
///
/// OAuth authorization is deliberately not part of this serializable type.
/// A caller injects a fresh token through request-scoped credentials in the
/// client layer; it must be included on every Responses creation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteMcpConfig {
    server_label: String,
    server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    server_description: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    defer_loading: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    allowed_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    require_approval: Option<McpApprovalPolicy>,
}

impl RemoteMcpConfig {
    pub fn new(
        server_label: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let config = Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            server_description: None,
            defer_loading: false,
            allowed_tools: Vec::new(),
            require_approval: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.server_description = Some(description.into());
        self
    }

    /// Ask OpenAI Responses tool search to defer discovery of this MCP server.
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }

    pub fn with_allowed_tools(
        mut self,
        allowed_tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, LlmError> {
        self.allowed_tools = allowed_tools.into_iter().map(Into::into).collect();
        self.validate()?;
        Ok(self)
    }

    pub fn with_require_approval(mut self, policy: McpApprovalPolicy) -> Self {
        self.require_approval = Some(policy);
        self
    }

    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    pub fn server_description(&self) -> Option<&str> {
        self.server_description.as_deref()
    }

    pub fn defer_loading(&self) -> bool {
        self.defer_loading
    }

    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    pub fn require_approval(&self) -> Option<McpApprovalPolicy> {
        self.require_approval
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let valid_name =
            |name: &str| !name.trim().is_empty() && !name.chars().any(char::is_control);
        let url = url::Url::parse(&self.server_url).ok();
        let valid_url = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none()
        });
        let mut names = std::collections::BTreeSet::new();
        if !valid_name(&self.server_label)
            || !valid_url
            || self
                .server_description
                .as_deref()
                .is_some_and(|value| value.chars().any(char::is_control))
            || self
                .allowed_tools
                .iter()
                .any(|name| !valid_name(name) || !names.insert(name.as_str()))
        {
            return Err(LlmError::InvalidRequest {
                message: "remote MCP label, HTTPS server URL, description, or allowed tool names are invalid".into(),
            });
        }
        Ok(())
    }
}

/// Public, non-secret connection settings for an xAI Responses remote MCP
/// server. xAI's documented Responses contract does not support
/// `require_approval` or `connector_id`; neither is represented here.
/// Authorization is supplied separately for each request through
/// `RequestOptions::mcp_authorizations`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiRemoteMcpConfig {
    server_label: String,
    server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    server_description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    allowed_tools: Vec<String>,
}

impl XaiRemoteMcpConfig {
    pub fn new(
        server_label: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let config = Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            server_description: None,
            allowed_tools: Vec::new(),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.server_description = Some(description.into());
        self
    }

    pub fn with_allowed_tools(
        mut self,
        allowed_tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, LlmError> {
        self.allowed_tools = allowed_tools.into_iter().map(Into::into).collect();
        self.validate()?;
        Ok(self)
    }

    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    pub fn server_description(&self) -> Option<&str> {
        self.server_description.as_deref()
    }

    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let valid_name =
            |name: &str| !name.trim().is_empty() && !name.chars().any(char::is_control);
        let url = url::Url::parse(&self.server_url).ok();
        let valid_url = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        });
        let mut names = std::collections::BTreeSet::new();
        if !valid_name(&self.server_label)
            || !valid_url
            || self
                .server_description
                .as_deref()
                .is_some_and(|value| value.chars().any(char::is_control))
            || self
                .allowed_tools
                .iter()
                .any(|name| !valid_name(name) || !names.insert(name.as_str()))
        {
            return Err(LlmError::InvalidRequest {
                message: "xAI remote MCP label, HTTPS server URL, description, or allowed tool names are invalid".into(),
            });
        }
        Ok(())
    }
}

/// An MCP tool invocation awaiting a host-owned approval decision.
///
/// `arguments` is retained as the provider's original JSON string. The host
/// decides whether to approve; this type does not execute or authorize tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpApprovalRequest {
    pub approval_request_id: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
}

impl McpApprovalRequest {
    /// Parse a native Responses output item retained by the codec.
    pub fn from_native_item(item: &Value) -> Option<Self> {
        (item.get("type").and_then(Value::as_str) == Some("mcp_approval_request")).then_some(())?;
        let nonempty = |key: &str| {
            item.get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty() && !value.chars().any(char::is_control))
                .map(str::to_owned)
        };
        Some(Self {
            approval_request_id: nonempty("id")?,
            server_label: nonempty("server_label")?,
            name: nonempty("name")?,
            arguments: item.get("arguments")?.as_str()?.to_owned(),
        })
    }

    pub fn respond(self, approve: bool) -> McpApprovalResponse {
        McpApprovalResponse {
            approval_request_id: self.approval_request_id,
            approve,
        }
    }
}

/// Caller-supplied answer to an OpenAI Responses MCP approval request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpApprovalResponse {
    pub approval_request_id: String,
    pub approve: bool,
}

impl McpApprovalResponse {
    /// Build a native Responses input item inside an assistant transcript
    /// message. The Responses codec replays it as a sibling input item.
    pub fn into_assistant_message(self) -> ConversationMessage {
        ConversationMessage::assistant(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: serde_json::json!({
                "type": "mcp_approval_response",
                "approval_request_id": self.approval_request_id,
                "approve": self.approve
            }),
        }])
    }
}

/// Select Anthropic's hosted catalog search implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicToolSearchStrategy {
    /// Search tool names and descriptions with Python-compatible regular expressions.
    Regex,
    /// Search the tool catalog with natural-language BM25 queries.
    Bm25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicToolSearchConfig {
    pub strategy: AnthropicToolSearchStrategy,
}

/// Per-tool MCP settings accepted by Anthropic's `mcp_toolset`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicMcpToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

/// A caller-pinned MCP tool definition accepted by the
/// `mcp-client-2026-09-15` beta. The schema is retained as provider JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Prompt-cache lifetime accepted on an Anthropic MCP toolset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicMcpCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Prompt-cache breakpoint placed on the final expanded tool in an MCP set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpCacheControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<AnthropicMcpCacheTtl>,
}

/// Non-secret definition of one Anthropic remote MCP server and
/// its one-to-one `mcp_toolset`. A pinned `tools: Some(vec![])` is distinct
/// from `None`: the empty list explicitly pins a server with no tools.
/// Foundry uses the 2025 connector and rejects pinned lists and inline toolsets;
/// the first-party Claude API supports the 2026 listing/pinning extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpConfig {
    name: String,
    url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_config: Option<AnthropicMcpToolConfig>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    configs: std::collections::BTreeMap<String, AnthropicMcpToolConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicMcpTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicMcpCacheControl>,
    /// Keep this server in `mcp_servers`, but add its toolset at a later
    /// `tool_addition` position inside a system message.
    #[serde(default, skip_serializing_if = "is_false")]
    inline_toolset: bool,
}

impl AnthropicMcpConfig {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Result<Self, LlmError> {
        let config = Self {
            name: name.into(),
            url: url.into(),
            default_config: None,
            configs: std::collections::BTreeMap::new(),
            tools: None,
            cache_control: None,
            inline_toolset: false,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_default_config(mut self, config: AnthropicMcpToolConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    /// Set or replace one per-tool override. Unknown tool names are retained;
    /// Anthropic documents that the server may expose dynamic tool names.
    pub fn with_tool_config(
        mut self,
        name: impl Into<String>,
        config: AnthropicMcpToolConfig,
    ) -> Result<Self, LlmError> {
        self.configs.insert(name.into(), config);
        self.validate()?;
        Ok(self)
    }

    /// Pin the server's exact tool list. Passing an empty vector deliberately
    /// pins an empty toolset; omit this method to let Anthropic fetch tools.
    pub fn with_tools(
        mut self,
        tools: impl IntoIterator<Item = AnthropicMcpTool>,
    ) -> Result<Self, LlmError> {
        self.tools = Some(tools.into_iter().collect());
        self.validate()?;
        Ok(self)
    }

    pub fn with_cache_control(mut self, cache_control: AnthropicMcpCacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Place this server's `mcp_toolset` at its `inline_tool_addition` point in
    /// message history. The connection remains in top-level `mcp_servers` so
    /// its URL and request-scoped credential are available to Anthropic.
    pub fn with_inline_toolset(mut self, enabled: bool) -> Self {
        self.inline_toolset = enabled;
        self
    }

    pub fn inline_toolset(&self) -> bool {
        self.inline_toolset
    }

    /// Build the native Anthropic `tool_addition` block for this MCP toolset.
    /// Add it to a `MessageRole::System` message at the point the server's
    /// tools should become available.
    pub fn inline_tool_addition(&self) -> Result<ContentBlock, LlmError> {
        self.validate()?;
        if !self.inline_toolset {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP inline_tool_addition requires with_inline_toolset(true)"
                    .into(),
            });
        }
        Ok(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: serde_json::json!({
                "type":"tool_addition",
                "tool":{"type":"tool_definition","definition":self.toolset_value()}
            }),
        })
    }

    pub fn inline_tool_addition_message(&self) -> Result<ConversationMessage, LlmError> {
        Ok(ConversationMessage {
            anthropic: None,
            role: MessageRole::System,
            content: vec![self.inline_tool_addition()?],
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn default_config(&self) -> Option<&AnthropicMcpToolConfig> {
        self.default_config.as_ref()
    }

    pub fn configs(&self) -> &std::collections::BTreeMap<String, AnthropicMcpToolConfig> {
        &self.configs
    }

    pub fn tools(&self) -> Option<&[AnthropicMcpTool]> {
        self.tools.as_deref()
    }

    pub fn cache_control(&self) -> Option<AnthropicMcpCacheControl> {
        self.cache_control
    }

    pub(crate) fn toolset_value(&self) -> Value {
        let mut toolset = serde_json::json!({
            "type":"mcp_toolset",
            "mcp_server_name":self.name,
        });
        if let Some(default_config) = &self.default_config {
            toolset["default_config"] = serde_json::to_value(default_config).unwrap_or(Value::Null);
        }
        if !self.configs.is_empty() {
            toolset["configs"] = serde_json::to_value(&self.configs).unwrap_or(Value::Null);
        }
        if let Some(pinned_tools) = &self.tools {
            toolset["tools"] = serde_json::to_value(pinned_tools).unwrap_or(Value::Null);
        }
        if let Some(cache_control) = self.cache_control {
            let mut value = serde_json::json!({"type":"ephemeral"});
            if let Some(ttl) = cache_control.ttl {
                value["ttl"] = serde_json::Value::String(
                    match ttl {
                        AnthropicMcpCacheTtl::FiveMinutes => "5m",
                        AnthropicMcpCacheTtl::OneHour => "1h",
                    }
                    .into(),
                );
            }
            toolset["cache_control"] = value;
        }
        toolset
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let url = url::Url::parse(&self.url).ok();
        let valid_url = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none()
        });
        let valid_name =
            |name: &str| !name.trim().is_empty() && !name.chars().any(char::is_control);
        let pinned_names = self.tools.as_ref().map(|tools| {
            let mut names = std::collections::BTreeSet::new();
            tools.iter().all(|tool| {
                valid_name(&tool.name)
                    && tool.input_schema.is_object()
                    && names.insert(tool.name.as_str())
            })
        });
        if !valid_name(&self.name)
            || !valid_url
            || self.configs.keys().any(|name| !valid_name(name))
            || pinned_names == Some(false)
        {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP server name, HTTPS URL, tool config, or pinned tool list is invalid".into(),
            });
        }
        Ok(())
    }
}

/// A tool reference accepted by Anthropic's mid-conversation tool changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolReference {
    Tool { name: String },
    McpTool { server_name: String, name: String },
    McpToolset { server_name: String },
}

impl AnthropicToolReference {
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }

    pub fn mcp_tool(server_name: impl Into<String>, name: impl Into<String>) -> Self {
        Self::McpTool {
            server_name: server_name.into(),
            name: name.into(),
        }
    }

    pub fn mcp_toolset(server_name: impl Into<String>) -> Self {
        Self::McpToolset {
            server_name: server_name.into(),
        }
    }

    fn wire_value(&self) -> Value {
        match self {
            Self::Tool { name } => serde_json::json!({"type":"tool_reference","name":name}),
            Self::McpTool { server_name, name } => {
                serde_json::json!({"type":"mcp_tool_reference","server_name":server_name,"name":name})
            }
            Self::McpToolset { server_name } => {
                serde_json::json!({"type":"mcp_toolset_reference","server_name":server_name})
            }
        }
    }
}

/// One Anthropic inline tool change. A removal always requires a reference;
/// custom definitions are additions or same-name updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", content = "value", rename_all = "snake_case")]
pub enum AnthropicToolChange {
    AddReference(AnthropicToolReference),
    Remove(AnthropicToolReference),
    DefineTool(ToolSpec),
}

impl AnthropicToolChange {
    pub fn add_reference(reference: AnthropicToolReference) -> Self {
        Self::AddReference(reference)
    }

    pub fn remove(reference: AnthropicToolReference) -> Self {
        Self::Remove(reference)
    }

    pub fn define_tool(tool: ToolSpec) -> Self {
        Self::DefineTool(tool)
    }

    pub fn into_content_block(self) -> ContentBlock {
        let value = match self {
            Self::AddReference(reference) => serde_json::json!({
                "type":"tool_addition", "tool":reference.wire_value()
            }),
            Self::Remove(reference) => serde_json::json!({
                "type":"tool_removal", "tool":reference.wire_value()
            }),
            Self::DefineTool(tool) => {
                let mut definition = serde_json::json!({
                    "name":tool.name,
                    "description":tool.description,
                    "input_schema":tool.input_schema,
                });
                if tool.strict {
                    definition["strict"] = Value::Bool(true);
                }
                if tool.defer_loading {
                    definition["defer_loading"] = Value::Bool(true);
                }
                if !tool.allowed_callers.is_empty() {
                    definition["allowed_callers"] =
                        serde_json::to_value(tool.allowed_callers).unwrap_or(Value::Null);
                }
                serde_json::json!({
                    "type":"tool_addition",
                    "tool":{"type":"tool_definition","definition":definition}
                })
            }
        };
        ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value,
        }
    }

    pub fn into_system_message(self) -> ConversationMessage {
        ConversationMessage {
            anthropic: None,
            role: MessageRole::System,
            content: vec![self.into_content_block()],
        }
    }
}

/// Anthropic Messages `code_execution_20260521` settings. The first-party API
/// and supported Anthropic-hosted Foundry deployments run commands and return
/// native server-tool blocks; the host never executes them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicCodeExecutionConfig {
    /// Define Code Execution in a mid-conversation `tool_addition` instead of
    /// the top-level `tools` array. The addition must use the typed tool value
    /// and is supported only on the first-party Messages API.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inline_definition: bool,
    /// Reuse this account-bound container. `None` asks the API for a new one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerRef>,
    /// Anthropic-managed or workspace-custom Skills made available in the
    /// Code Execution container. Built-in Skills work on Anthropic-hosted
    /// Foundry. Custom Skill references require the explicit first-party or
    /// Foundry Skills service scope; Foundry version-content download is unsupported.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<AnthropicSkillRef>,
    /// Already-uploaded first-party Anthropic or Foundry files added as
    /// `container_upload` blocks to the final user message. Foundry references
    /// must come from a Foundry Files service and match the resource/account;
    /// they are not bound to a deployment or model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ProviderFileSource>,
}

/// Skill source accepted by the Anthropic Messages container parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicSkillType {
    Anthropic,
    Custom,
}

/// Local scope for a custom Anthropic Skill ID. On the Claude API, `account_scope`
/// identifies the owning workspace. On Foundry, it identifies the resource/account
/// credentials. Neither value is sent on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSkillScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workspace_id: Option<String>,
    /// Explicitly distinguishes Anthropic-hosted Foundry from the direct API.
    /// `false` identifies the direct Claude API route.
    #[serde(default, skip_serializing_if = "is_false")]
    foundry: bool,
}

impl AnthropicSkillScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
            workspace_id: None,
            foundry: false,
        };
        scope.validate()?;
        scope.endpoint = "https://api.anthropic.com".into();
        Ok(scope)
    }

    /// Bind a custom Skill reference to an Anthropic-hosted Microsoft Foundry
    /// resource. This scope is resource/account-scoped and deliberately does
    /// not capture a chat deployment or model.
    pub fn new_foundry(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        hosting: FoundryHosting,
    ) -> Result<Self, LlmError> {
        if hosting != FoundryHosting::Anthropic {
            return Err(LlmError::UnsupportedCapability {
                message: "custom Anthropic Skills require a Foundry deployment hosted on Anthropic"
                    .into(),
            });
        }
        let endpoint = endpoint.into();
        let canonical_endpoint = AnthropicContainerScope::normalize_foundry_endpoint(&endpoint)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Foundry Skill scope requires the HTTPS Anthropic resource endpoint"
                    .into(),
            })?;
        let scope = Self {
            profile_name: profile_name.into(),
            endpoint: canonical_endpoint,
            account_scope: account_scope.into(),
            workspace_id: None,
            foundry: true,
        };
        scope.validate()?;
        Ok(scope)
    }

    /// Bind a direct Claude API reference to an explicit Anthropic Workspace.
    /// The same value must be configured as
    /// `extra.headers.anthropic-workspace-id` on Messages profiles using it.
    /// Foundry scopes are resource/account-bound and cannot set this header.
    pub fn with_workspace_id(mut self, workspace_id: impl Into<String>) -> Result<Self, LlmError> {
        self.workspace_id = Some(workspace_id.into());
        self.validate()?;
        Ok(self)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn workspace_id(&self) -> Option<&str> {
        self.workspace_id.as_deref()
    }

    pub(crate) fn is_foundry(&self) -> bool {
        self.foundry
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if [&self.profile_name, &self.account_scope]
            .into_iter()
            .any(|value| value.trim().is_empty() || value.chars().any(char::is_control))
            || self
                .workspace_id
                .as_deref()
                .is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control))
            || (self.foundry
                && (self.workspace_id.is_some()
                    || AnthropicContainerScope::normalize_foundry_endpoint(&self.endpoint)
                        .is_none()))
            || (!self.foundry && !AnthropicContainerScope::is_official_endpoint(&self.endpoint))
        {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic custom Skill scope requires a profile, supported endpoint, and workspace/resource-bound account identity".into(),
            });
        }
        Ok(())
    }
}

fn anthropic_profile_workspace_id(profile: &ProviderProfile) -> Result<Option<&str>, LlmError> {
    let Some(headers) = profile.extra.get("headers").and_then(Value::as_object) else {
        return Ok(None);
    };
    let mut matching = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-workspace-id"));
    let Some((_, value)) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() {
        return Err(LlmError::InvalidRequest {
            message: "Anthropic Skill profile contains duplicate anthropic-workspace-id headers"
                .into(),
        });
    }
    let workspace_id = value.as_str().ok_or_else(|| LlmError::InvalidRequest {
        message: "Anthropic Skill profile anthropic-workspace-id must be a string".into(),
    })?;
    if workspace_id.trim().is_empty() || workspace_id.chars().any(char::is_control) {
        return Err(LlmError::InvalidRequest {
            message: "Anthropic Skill profile anthropic-workspace-id is empty or invalid".into(),
        });
    }
    Ok(Some(workspace_id))
}

/// A Skill to make available in an Anthropic Code Execution container.
///
/// `version` is optional; when supplied, Anthropic Skills accept a date such
/// as `20251013` or `latest`, and custom Skills accept `skver_…`, the legacy
/// `skill_version_…` form, or `latest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSkillRef {
    #[serde(rename = "type")]
    skill_type: AnthropicSkillType,
    skill_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<AnthropicSkillScope>,
}

impl AnthropicSkillRef {
    /// Reference one of Anthropic's built-in Skills, such as `pptx` or `pdf`.
    pub fn anthropic(skill_id: impl Into<String>) -> Self {
        Self {
            skill_type: AnthropicSkillType::Anthropic,
            skill_id: skill_id.into(),
            version: None,
            scope: None,
        }
    }

    /// Reference a custom Skill already uploaded to the scoped Anthropic
    /// workspace or Anthropic-hosted Foundry resource.
    pub fn custom(skill_id: impl Into<String>, scope: AnthropicSkillScope) -> Self {
        Self {
            skill_type: AnthropicSkillType::Custom,
            skill_id: skill_id.into(),
            version: None,
            scope: Some(scope),
        }
    }

    /// Pin a documented version or use `latest`.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn skill_type(&self) -> AnthropicSkillType {
        self.skill_type
    }

    pub fn skill_id(&self) -> &str {
        &self.skill_id
    }

    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    pub fn scope(&self) -> Option<&AnthropicSkillScope> {
        self.scope.as_ref()
    }

    pub(crate) fn validate_for(
        &self,
        profile: &ProviderProfile,
        account_scope: Option<&str>,
        foundry_deployment: Option<&FoundryDeployment>,
    ) -> Result<(), LlmError> {
        if self.skill_id.trim().is_empty() || self.skill_id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic Skill references require a nonempty valid skill_id".into(),
            });
        }
        match (self.skill_type, self.scope.as_ref()) {
            (AnthropicSkillType::Anthropic, None) => {}
            (AnthropicSkillType::Anthropic, Some(_)) => {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic-managed Skill references do not take a custom scope".into(),
                });
            }
            (AnthropicSkillType::Custom, Some(scope)) => {
                scope.validate()?;
                let scope_matches_route = if scope.foundry {
                    profile.protocol == ProtocolFamily::FoundryClaude
                        && foundry_deployment.is_some_and(|deployment| {
                            deployment.hosting == FoundryHosting::Anthropic
                        })
                        && AnthropicContainerScope::normalize_foundry_endpoint(&profile.base_url)
                            == AnthropicContainerScope::normalize_foundry_endpoint(&scope.endpoint)
                        && scope.workspace_id.is_none()
                } else {
                    profile.protocol == ProtocolFamily::AnthropicMessages
                        && foundry_deployment.is_none()
                        && AnthropicContainerScope::is_official_endpoint(&profile.base_url)
                        && scope.endpoint == "https://api.anthropic.com"
                        && scope.workspace_id.as_deref() == anthropic_profile_workspace_id(profile)?
                };
                if scope.profile_name != profile.profile_name
                    || !scope_matches_route
                    || Some(scope.account_scope.as_str()) != account_scope
                {
                    return Err(LlmError::InvalidRequest {
                        message: "custom Anthropic Skill scope does not match the selected profile, supported endpoint and hosting, account_scope, or configured anthropic-workspace-id".into(),
                    });
                }
            }
            (AnthropicSkillType::Custom, None) => {
                return Err(LlmError::InvalidRequest {
                    message: "custom Anthropic Skill references require an explicit workspace/resource-bound scope".into(),
                });
            }
        }
        if let Some(version) = &self.version {
            if version.trim().is_empty() || version.chars().any(char::is_control) {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Skill version must be a nonempty valid identifier".into(),
                });
            }
            let valid_version = if version == "latest" {
                true
            } else {
                match self.skill_type {
                    AnthropicSkillType::Anthropic => {
                        version.len() == 8 && version.bytes().all(|byte| byte.is_ascii_digit())
                    }
                    AnthropicSkillType::Custom => {
                        ["skver_", "skill_version_"].iter().any(|prefix| {
                            version
                                .strip_prefix(prefix)
                                .is_some_and(|id| !id.is_empty())
                        })
                    }
                }
            };
            if !valid_version {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Skill version must be `latest`, an Anthropic date version, or a custom Skill version ID".into(),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn wire_value(&self) -> Value {
        let mut value = serde_json::json!({
            "type": match self.skill_type {
                AnthropicSkillType::Anthropic => "anthropic",
                AnthropicSkillType::Custom => "custom",
            },
            "skill_id": self.skill_id,
        });
        if let Some(version) = &self.version {
            value["version"] = Value::String(version.clone());
        }
        value
    }
}

/// Non-secret local identity for an Anthropic execution container. The exact
/// request model is pinned by client policy; aliases are resolved before checking.
/// Foundry scopes also capture the typed hosting and underlying-model identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
    request_model: String,
    /// Foundry's request model is a caller-chosen deployment name, so the
    /// underlying model and hosting choice must travel with a scoped ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    foundry_deployment: Option<FoundryDeployment>,
}

impl AnthropicContainerScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        request_model: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
            request_model: request_model.into(),
            foundry_deployment: None,
        };
        scope.validate()?;
        scope.endpoint = "https://api.anthropic.com".into();
        Ok(scope)
    }

    /// Bind a container returned by a Foundry deployment hosted on Anthropic.
    /// The endpoint is the Foundry resource base URL, and `deployment` must be
    /// copied from the selected model row rather than inferred from its name.
    pub fn new_foundry(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        request_model: impl Into<String>,
        deployment: FoundryDeployment,
    ) -> Result<Self, LlmError> {
        let endpoint = endpoint.into();
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.clone(),
            account_scope: account_scope.into(),
            request_model: request_model.into(),
            foundry_deployment: Some(deployment),
        };
        scope.validate()?;
        scope.endpoint = Self::normalize_foundry_endpoint(&endpoint).ok_or_else(|| {
            LlmError::InvalidRequest {
                message: "Foundry container scope requires the HTTPS resource Anthropic Messages endpoint".into(),
            }
        })?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn request_model(&self) -> &str {
        &self.request_model
    }

    /// Underlying hosting and model identity captured for a Foundry container.
    pub fn foundry_deployment(&self) -> Option<&FoundryDeployment> {
        self.foundry_deployment.as_ref()
    }

    pub(crate) fn is_official_endpoint(endpoint: &str) -> bool {
        let Ok(url) = url::Url::parse(endpoint) else {
            return false;
        };
        url.scheme() == "https"
            && url.host_str() == Some("api.anthropic.com")
            && url.port().is_none()
            && url.path() == "/"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
    }

    /// Canonicalize only the Foundry Anthropic resource base documented for
    /// Messages and other Claude API resources. This intentionally does not
    /// accept arbitrary Azure hosts, custom paths, or proxy URLs.
    pub(crate) fn normalize_foundry_endpoint(endpoint: &str) -> Option<String> {
        let url = url::Url::parse(endpoint).ok()?;
        if url.scheme() != "https"
            || url.port().is_some()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.path(), "/anthropic" | "/anthropic/")
        {
            return None;
        }
        let host = url.host_str()?;
        let resource = host.strip_suffix(".services.ai.azure.com")?;
        if resource.is_empty()
            || resource.contains('.')
            || resource.starts_with('-')
            || resource.ends_with('-')
            || !resource
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return None;
        }
        Some(format!("https://{host}/anthropic"))
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let identity_fields_are_valid =
            [&self.profile_name, &self.account_scope, &self.request_model]
                .into_iter()
                .all(|value| !value.trim().is_empty() && !value.chars().any(char::is_control));
        let valid_route = match &self.foundry_deployment {
            None => Self::is_official_endpoint(&self.endpoint),
            Some(deployment) => {
                deployment.hosting == FoundryHosting::Anthropic
                    && !deployment.model_id.trim().is_empty()
                    && !deployment.model_id.chars().any(char::is_control)
                    && Self::normalize_foundry_endpoint(&self.endpoint).is_some()
            }
        };
        if !identity_fields_are_valid || !valid_route {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic container scope requires a profile, supported endpoint and hosting identity, stable account identity, and request model".into(),
            });
        }
        Ok(())
    }
}

/// A caller-imported Anthropic container ID. Scope stays local and is checked
/// before dispatch; only the ID is sent as the request's top-level `container`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerRef {
    id: String,
    scope: AnthropicContainerScope,
}

impl AnthropicContainerRef {
    pub fn new(id: impl Into<String>, scope: AnthropicContainerScope) -> Result<Self, LlmError> {
        let reference = Self {
            id: id.into(),
            scope,
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn scope(&self) -> &AnthropicContainerScope {
        &self.scope
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        self.scope.validate()?;
        if self.id.trim().is_empty() || self.id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic container reference requires a nonempty container ID".into(),
            });
        }
        Ok(())
    }
}

/// First-party Anthropic or Anthropic-hosted Foundry's complete native
/// container envelope, including unknown metadata. The provider's rolling
/// `expires_at` does not describe the full container lifetime, so it is retained
/// without enforcing local expiry.
/// Native execution container metadata returned by the first-party Anthropic
/// API or a supported Anthropic-hosted Foundry deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerMetadata {
    pub envelope: Value,
}

impl AnthropicContainerMetadata {
    /// Explicitly bind an observed ID to the caller's route and account.
    pub fn reference_for(
        &self,
        scope: AnthropicContainerScope,
    ) -> Result<AnthropicContainerRef, LlmError> {
        let id = self
            .envelope
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Anthropic container metadata has no string ID to import".into(),
            })?;
        AnthropicContainerRef::new(id, scope)
    }
}

/// OpenRouter's hosted regex tool-search server tool. The BM25 variant is not
/// included because OpenRouter currently documents only regex search.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenRouterToolSearchConfig {
    /// Maximum discovered tools returned by one search; OpenRouter defaults
    /// this to five and caps it at fifty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
}

/// Execution mode for OpenAI Responses tool search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiToolSearchExecution {
    /// OpenAI searches the deferred function and MCP definitions in this request.
    #[default]
    Server,
    /// The caller performs discovery and returns a native `tool_search_output` item.
    Client,
}

/// OpenAI Responses tool search configuration.
///
/// Server execution needs only the `tool_search` tool marker. Client execution
/// also requires a description and JSON Schema for the search call arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiToolSearchConfig {
    #[serde(skip_serializing_if = "is_server_tool_search")]
    pub execution: OpenAiToolSearchExecution,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

fn is_server_tool_search(value: &OpenAiToolSearchExecution) -> bool {
    *value == OpenAiToolSearchExecution::Server
}

impl OpenAiToolSearchConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        match self.execution {
            OpenAiToolSearchExecution::Server
                if self.description.is_some() || self.parameters.is_some() =>
            {
                Err(LlmError::InvalidRequest {
                    message: "OpenAI server tool search does not accept client search fields".into(),
                })
            }
            OpenAiToolSearchExecution::Client
                if self
                    .description
                    .as_deref()
                    .is_none_or(|description| description.trim().is_empty())
                    || !self.parameters.as_ref().is_some_and(Value::is_object) =>
            {
                Err(LlmError::InvalidRequest {
                    message: "OpenAI client tool search requires a description and object parameters schema".into(),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Execution engine and container settings for OpenRouter's hosted shell.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenRouterShellConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<OpenRouterShellEngine>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<OpenRouterShellEnvironment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterShellEngine {
    /// Use a provider-native hosted shell when available, otherwise the
    /// OpenRouter sandbox. Commands remain server-executed.
    Auto,
    /// Always use the OpenRouter sandbox.
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterShellEnvironment {
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<OpenRouterShellNetworkPolicy>,
    },
    ContainerReference {
        container: OpenRouterContainerRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<OpenRouterShellNetworkPolicy>,
    },
}

/// OpenRouter's container network policy. Allowlist hosts follow the
/// provider's lowercase hostname/glob syntax and fifty-entry cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterShellNetworkPolicy {
    Disabled,
    Allowlist { allowed_domains: Vec<String> },
}

/// Local scope attached to a caller-imported OpenRouter shell container ID.
/// The reference is bound to the exact profile, endpoint, and stable account
/// identity selected by the application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
}

impl OpenRouterContainerScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
        };
        if scope.profile_name.trim().is_empty()
            || scope.endpoint.trim().is_empty()
            || scope.account_scope.trim().is_empty()
            || scope.profile_name.chars().any(char::is_control)
            || scope.endpoint.chars().any(char::is_control)
            || scope.account_scope.chars().any(char::is_control)
        {
            return Err(LlmError::InvalidRequest {
                message: "OpenRouter container scope requires a profile, endpoint, and stable account identity".into(),
            });
        }
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }
}

/// A caller-imported container ID. The encoder sends only `id`; this local
/// scope is checked before dispatch and never appears in the provider request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerRef {
    id: String,
    scope: OpenRouterContainerScope,
}

impl OpenRouterContainerRef {
    pub fn new(id: impl Into<String>, scope: OpenRouterContainerScope) -> Result<Self, LlmError> {
        let id = id.into();
        if id.trim().is_empty() || id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "OpenRouter container reference requires a nonempty container ID".into(),
            });
        }
        Ok(Self { id, scope })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn scope(&self) -> &OpenRouterContainerScope {
        &self.scope
    }
}

/// Raw OpenRouter Messages container metadata. Direct codec decoding has no
/// account identity, so callers explicitly supply a local scope when turning
/// its ID into a reusable shell reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerMetadata {
    pub envelope: Value,
}

impl OpenRouterContainerMetadata {
    pub fn reference_for(
        &self,
        scope: OpenRouterContainerScope,
    ) -> Result<OpenRouterContainerRef, LlmError> {
        let id = self
            .envelope
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "OpenRouter container metadata has no string ID to import".into(),
            })?;
        OpenRouterContainerRef::new(id, scope)
    }
}

/// Code Interpreter settings for OpenAI Responses.
///
/// An absent `container` uses OpenAI's automatic container. In automatic mode,
/// `files` are scope-bound Files API references mounted through the documented
/// `file_ids` property. An explicit `container` reuses an existing scoped
/// OpenAI container; its memory tier and files are managed by the Containers
/// service rather than configured on the Responses request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeInterpreterConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<CodeInterpreterMemoryLimit>,
    /// A scope-bound existing OpenAI container. `None` selects automatic mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::openai_containers::OpenAiContainerRef>,
    /// Existing OpenAI Files API references to mount into an automatic
    /// container. They are encoded as `container.file_ids` after scope,
    /// readiness, and expiry checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ProviderFileSource>,
}

impl CodeInterpreterConfig {
    /// Reuse an existing container returned by the OpenAI Containers service
    /// or explicitly imported with its account scope.
    #[must_use]
    pub fn with_container(
        mut self,
        container: crate::openai_containers::OpenAiContainerRef,
    ) -> Self {
        self.container = Some(container);
        self
    }

    /// Mount existing Files API references when using an automatic container.
    #[must_use]
    pub fn with_files(mut self, files: impl IntoIterator<Item = ProviderFileSource>) -> Self {
        self.files.extend(files);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeInterpreterMemoryLimit {
    #[serde(rename = "1g")]
    OneG,
    #[serde(rename = "4g")]
    FourG,
    #[serde(rename = "16g")]
    SixteenG,
    #[serde(rename = "64g")]
    SixtyFourG,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    #[serde(default)]
    pub prompt_cache: super::PromptCachePolicy,
    /// Requested output contract, separate from strict function tools.
    #[serde(default)]
    pub output_format: super::OutputFormat,
    pub model: String,
    /// Provider-executed tools. These do not create host tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hosted_tools: Vec<HostedTool>,
    /// Anthropic-defined Browser and Computer toolsets whose member calls are
    /// executed by the host application.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anthropic_client_toolsets: Vec<super::AnthropicClientToolset>,
    /// The immediately preceding response, scoped to its original account,
    /// endpoint, profile and model. Supply only new input in `messages`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<ContinuationRef>,
    #[serde(default)]
    pub system: Vec<SystemBlock>,
    pub messages: Vec<ConversationMessage>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default = "ToolChoice::auto")]
    pub tool_choice: ToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<super::ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<super::ServiceTier>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Provider-opaque extras a codec may forward.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl ChatRequest {
    pub fn hosted_web_search(&self) -> Option<&WebSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::WebSearch(config) => Some(config),
            HostedTool::WebExtractor
            | HostedTool::FileSearch(_)
            | HostedTool::CodeInterpreter(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::AnthropicToolSearch(_)
            | HostedTool::AnthropicWebFetch(_)
            | HostedTool::AnthropicCodeExecution(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => None,
        })
    }

    pub fn hosted_web_search_mut(&mut self) -> Option<&mut WebSearchConfig> {
        self.hosted_tools.iter_mut().find_map(|tool| match tool {
            HostedTool::WebSearch(config) => Some(config),
            HostedTool::WebExtractor
            | HostedTool::FileSearch(_)
            | HostedTool::CodeInterpreter(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::AnthropicToolSearch(_)
            | HostedTool::AnthropicWebFetch(_)
            | HostedTool::AnthropicCodeExecution(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => None,
        })
    }

    pub fn hosted_file_search(&self) -> Option<&FileSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::FileSearch(config) => Some(config),
            HostedTool::WebSearch(_)
            | HostedTool::WebExtractor
            | HostedTool::CodeInterpreter(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::AnthropicToolSearch(_)
            | HostedTool::AnthropicWebFetch(_)
            | HostedTool::AnthropicCodeExecution(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => None,
        })
    }

    pub fn hosted_code_interpreter(&self) -> Option<&CodeInterpreterConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::CodeInterpreter(config) => Some(config),
            HostedTool::WebSearch(_)
            | HostedTool::WebExtractor
            | HostedTool::FileSearch(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::AnthropicToolSearch(_)
            | HostedTool::AnthropicWebFetch(_)
            | HostedTool::AnthropicCodeExecution(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => None,
        })
    }

    pub fn hosted_anthropic_tool_search(&self) -> Option<&AnthropicToolSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::AnthropicToolSearch(config) => Some(config),
            HostedTool::AnthropicCodeExecution(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::AnthropicWebFetch(_)
            | HostedTool::WebSearch(_)
            | HostedTool::WebExtractor
            | HostedTool::FileSearch(_)
            | HostedTool::CodeInterpreter(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => None,
        })
    }

    pub fn hosted_anthropic_code_execution(&self) -> Option<&AnthropicCodeExecutionConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::AnthropicCodeExecution(config) => Some(config),
            _ => None,
        })
    }

    pub fn hosted_anthropic_web_fetch(&self) -> Option<&AnthropicWebFetchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::AnthropicWebFetch(config) => Some(config),
            _ => None,
        })
    }

    pub fn hosted_openrouter_tool_search(&self) -> Option<&OpenRouterToolSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::OpenRouterToolSearch(config) => Some(config),
            _ => None,
        })
    }

    pub fn hosted_openai_tool_search(&self) -> Option<&OpenAiToolSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::OpenAiToolSearch(config) => Some(config),
            _ => None,
        })
    }

    pub fn hosted_openrouter_shell(&self) -> Option<&OpenRouterShellConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::OpenRouterShell(config) => Some(config),
            _ => None,
        })
    }

    pub fn remote_mcp_servers(&self) -> impl Iterator<Item = &RemoteMcpConfig> {
        self.hosted_tools.iter().filter_map(|tool| match tool {
            HostedTool::RemoteMcp(config) => Some(config),
            _ => None,
        })
    }

    pub fn xai_remote_mcp_servers(&self) -> impl Iterator<Item = &XaiRemoteMcpConfig> {
        self.hosted_tools.iter().filter_map(|tool| match tool {
            HostedTool::XaiRemoteMcp(config) => Some(config),
            _ => None,
        })
    }

    pub fn anthropic_mcp_servers(&self) -> impl Iterator<Item = &AnthropicMcpConfig> {
        self.hosted_tools.iter().filter_map(|tool| match tool {
            HostedTool::AnthropicMcp(config) => Some(config),
            _ => None,
        })
    }

    pub fn set_hosted_web_search(&mut self, config: Option<WebSearchConfig>) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::WebSearch(_)));
        if let Some(config) = config {
            self.hosted_tools.push(HostedTool::WebSearch(config));
        }
    }

    pub fn set_hosted_file_search(&mut self, config: Option<FileSearchConfig>) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::FileSearch(_)));
        if let Some(config) = config {
            self.hosted_tools.push(HostedTool::FileSearch(config));
        }
    }

    pub fn set_hosted_web_extractor(&mut self, enabled: bool) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::WebExtractor));
        if enabled {
            self.hosted_tools.push(HostedTool::WebExtractor);
        }
    }

    pub fn has_hosted_web_extractor(&self) -> bool {
        self.hosted_tools
            .iter()
            .any(|tool| matches!(tool, HostedTool::WebExtractor))
    }

    pub fn set_hosted_anthropic_tool_search(&mut self, config: Option<AnthropicToolSearchConfig>) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::AnthropicToolSearch(_)));
        if let Some(config) = config {
            self.hosted_tools
                .push(HostedTool::AnthropicToolSearch(config));
        }
    }

    pub fn set_hosted_anthropic_web_fetch(&mut self, config: Option<AnthropicWebFetchConfig>) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::AnthropicWebFetch(_)));
        if let Some(config) = config {
            self.hosted_tools
                .push(HostedTool::AnthropicWebFetch(config));
        }
    }

    pub fn set_hosted_openai_tool_search(&mut self, config: Option<OpenAiToolSearchConfig>) {
        self.hosted_tools
            .retain(|tool| !matches!(tool, HostedTool::OpenAiToolSearch(_)));
        if let Some(config) = config {
            self.hosted_tools.push(HostedTool::OpenAiToolSearch(config));
        }
    }

    pub fn validate_hosted_tools(&self) -> Result<(), LlmError> {
        let mut web_search = false;
        let mut web_extractor = false;
        let mut file_search = false;
        let mut code_interpreter = false;
        let mut gemini_code_execution = false;
        let mut gemini_url_context = false;
        let mut gemini_maps_grounding = false;
        let mut anthropic_tool_search = false;
        let mut anthropic_web_fetch = false;
        let mut anthropic_code_execution = false;
        let mut openrouter_tool_search = false;
        let mut openai_tool_search = false;
        let mut openrouter_shell = false;
        let mut remote_mcp_labels = std::collections::BTreeSet::new();
        let mut anthropic_mcp_names = std::collections::BTreeSet::new();
        for tool in &self.hosted_tools {
            let seen = match tool {
                HostedTool::WebSearch(_) => &mut web_search,
                HostedTool::WebExtractor => &mut web_extractor,
                HostedTool::FileSearch(_) => &mut file_search,
                HostedTool::CodeInterpreter(_) => &mut code_interpreter,
                HostedTool::OpenRouterToolSearch(_) => &mut openrouter_tool_search,
                HostedTool::OpenAiToolSearch(config) => {
                    config.validate()?;
                    &mut openai_tool_search
                }
                HostedTool::OpenRouterShell(_) => &mut openrouter_shell,
                HostedTool::GeminiCodeExecution => &mut gemini_code_execution,
                HostedTool::GeminiUrlContext => &mut gemini_url_context,
                HostedTool::GeminiMapsGrounding(_) => &mut gemini_maps_grounding,
                HostedTool::AnthropicToolSearch(_) => &mut anthropic_tool_search,
                HostedTool::AnthropicWebFetch(config) => {
                    config.validate()?;
                    &mut anthropic_web_fetch
                }
                HostedTool::AnthropicCodeExecution(_) => &mut anthropic_code_execution,
                HostedTool::AnthropicMcp(config) => {
                    config.validate()?;
                    if !anthropic_mcp_names.insert(config.name.as_str()) {
                        return Err(LlmError::InvalidRequest {
                            message: "Anthropic MCP server names must be unique per request".into(),
                        });
                    }
                    continue;
                }
                HostedTool::RemoteMcp(config) => {
                    config.validate()?;
                    if !remote_mcp_labels.insert(config.server_label.as_str()) {
                        return Err(LlmError::InvalidRequest {
                            message: "remote MCP server labels must be unique per request".into(),
                        });
                    }
                    continue;
                }
                HostedTool::XaiRemoteMcp(config) => {
                    config.validate()?;
                    if !remote_mcp_labels.insert(config.server_label.as_str()) {
                        return Err(LlmError::InvalidRequest {
                            message: "remote MCP server labels must be unique per request".into(),
                        });
                    }
                    continue;
                }
            };
            if *seen {
                return Err(LlmError::InvalidRequest {
                    message: "the same hosted tool may appear only once per request".into(),
                });
            }
            *seen = true;
        }
        if anthropic_mcp_names.len() > 20 {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP supports at most 20 servers per request".into(),
            });
        }
        Ok(())
    }
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Auto
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub inference: super::InferenceReport,
    /// A response-cache observation explicitly reported by the serving gateway.
    /// Missing or unrecognized signals remain `None`; usage is never used to
    /// infer a cache hit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_cache: Option<ResponseCacheObservation>,
    pub message: ConversationMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<WebSearchResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_search: Option<FileSearchResult>,
    /// OpenRouter Messages' top-level container envelope, retained verbatim.
    /// Direct codecs do not know the caller's account identity; import the ID
    /// as an [`OpenRouterContainerRef`] with an explicit local scope before reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter_container: Option<OpenRouterContainerMetadata>,
    /// First-party Anthropic or Anthropic-hosted Foundry execution container
    /// metadata. Import its ID with an explicit [`AnthropicContainerScope`] to
    /// reuse it on a later request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_container: Option<AnthropicContainerMetadata>,
    /// Unmodified Anthropic API or Foundry usage JSON. Hosted execution
    /// counters do not report execution duration or establish the total billed
    /// amount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_usage: Option<Value>,
    pub stop_reason: StopReason,
    pub usage: super::UsageReport,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<ResponseId>,
    /// A reusable state reference when `RequestOptions.account_scope` was set
    /// and the provider returned a response ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<ContinuationRef>,
    /// Profile that completed a high-level client request. Direct codec
    /// decoding leaves this absent because the wire does not identify it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_profile: Option<String>,
}

/// Whether a gateway explicitly reports that it served this response from a
/// response cache or missed the cache. This is distinct from provider prompt
/// cache token usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseCacheStatus {
    Hit,
    Miss,
}

/// Explicit response-cache metadata returned with a completed response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseCacheObservation {
    pub status: ResponseCacheStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_generation_id: Option<String>,
}

/// Whether the available model metadata establishes a capability.
///
/// `Unknown` is deliberately distinct from `Unsupported`: directory listings
/// often omit capability facts entirely. A caller may use `Unsupported` as an
/// advisory preflight result, but absence of a fact must not be treated as a
/// rejection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySupport {
    /// No authoritative fact is available.
    #[default]
    Unknown,
    /// The provider or explicit configuration says the capability is present.
    Supported,
    /// The provider or explicit configuration says the capability is absent.
    Unsupported,
}

impl CapabilitySupport {
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// A selector for one of the capability facts represented by
/// [`ModelCapabilitySupport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Vision,
    Documents,
    Tools,
    Reasoning,
    SignedReasoning,
    Streaming,
    StructuredOutput,
}

/// Explicit model capability facts.
///
/// Missing fields default to [`CapabilitySupport::Unknown`] and are omitted
/// when serialized. This lets a partial provider listing state one fact
/// without turning every omitted fact into a negative one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelCapabilitySupport {
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub vision: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub documents: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub tools: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub reasoning: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub signed_reasoning: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub streaming: CapabilitySupport,
    #[serde(default, skip_serializing_if = "CapabilitySupport::is_unknown")]
    pub structured_output: CapabilitySupport,
}

impl ModelCapabilitySupport {
    #[must_use]
    pub const fn get(&self, capability: ModelCapability) -> CapabilitySupport {
        match capability {
            ModelCapability::Vision => self.vision,
            ModelCapability::Documents => self.documents,
            ModelCapability::Tools => self.tools,
            ModelCapability::Reasoning => self.reasoning,
            ModelCapability::SignedReasoning => self.signed_reasoning,
            ModelCapability::Streaming => self.streaming,
            ModelCapability::StructuredOutput => self.structured_output,
        }
    }
}

/// One configured provider, as the client lists it.
///
/// Every profile in the selected region is listed, whether or not a credential exists — that is
/// the point of `info.api_key_url`. This crate holds no credentials,
/// so it reports the variable one is conventionally read from and leaves
/// "is it actually set" to the host, which is the only side that can know.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderListing {
    /// Usage regions declared by this connection.
    #[serde(default = "Region::all")]
    pub regions: Vec<Region>,
    pub provider_id: ProviderId,
    /// The connection's identity, and the half of a `profile/model` ref.
    pub profile_name: String,
    /// The group this connection belongs to. Connections in one group serve
    /// the same models and fail over to each other.
    pub group: String,
    #[serde(default)]
    pub info: ProviderInfo,
    pub protocol: ProtocolFamily,
    pub auth: AuthStrategy,
    /// Where a credential is conventionally read from, when the profile says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_env: Option<String>,
    pub billing_mode: BillingMode,
    pub model_count: usize,
    /// A spare connection, reachable by failover but never offered in a picker.
    pub hidden: bool,
}

/// One model as the client lists it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelListing {
    #[serde(default)]
    pub info: super::ModelInfo,
    /// Usage regions declared by this connection.
    #[serde(default = "Region::all")]
    pub regions: Vec<Region>,
    pub id: String,
    /// What to show a user in place of the id. A name, not a sentence.
    pub display_name: String,
    /// The vendor's own blurb, when the catalog carries one. Prose, and long
    /// enough that it belongs nowhere a name is expected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub provider_id: ProviderId,
    /// Which connection offers it. With `id`, this forms the `profile/model`
    /// ref that `resolve_in` accepts — without it a caller cannot say which of
    /// a group's connections it meant.
    pub profile_name: String,
    /// The id this model is called by on the wire, which is often not `id`.
    pub request_model: String,
    /// How this model is billed, the connection's mode unless it overrides it.
    pub billing_mode: BillingMode,
    /// Published prices. Absent is unpriced, which is not free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<TokenPricing>,
    /// The model's context window, when it is published.
    ///
    /// `None` means unpublished, and is deliberately not zero: a caller
    /// divides an estimate by this, and the two answers "no room left" and "we
    /// do not know the size of the room" call for opposite behaviour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub capability_support: ModelCapabilitySupport,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_errors_preserve_provider_details() {
        let e = LlmError::ContextOverflow {
            message: "m".into(),
            limit: Some(1),
            actual: Some(2),
        };
        assert_eq!(e.kind(), LlmErrorKind::ContextOverflow);
        let encoded = serde_json::to_value(&e).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({
                "kind": "context_overflow", "message": "m", "limit": 1, "actual": 2
            })
        );
        assert_eq!(serde_json::from_value::<LlmError>(encoded).unwrap(), e);
        let too_large = LlmError::RequestTooLarge {
            message: "body limit".into(),
        };
        assert_eq!(too_large.kind(), LlmErrorKind::RequestTooLarge);
        assert_eq!(
            serde_json::to_value(&too_large).unwrap(),
            serde_json::json!({
                "kind": "request_too_large", "message": "body limit"
            })
        );
    }

    #[test]
    fn stream_event_round_trips() {
        let ev = StreamEvent::ToolCallDelta {
            block: 1,
            id: ToolUseId::new("call_1"),
            provider_id: None,
            caller: None,
            toolset_name: Some("browser".into()),
            name: "Read".into(),
            arguments_fragment: "{\"file".into(),
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert_eq!(serde_json::from_str::<StreamEvent>(&s).unwrap(), ev);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&s).unwrap()["toolset_name"],
            "browser"
        );
        let old = serde_json::json!({
            "event": "tool_call_delta",
            "block": 1,
            "id": "call_1",
            "name": "Read",
            "arguments_fragment": "{}"
        });
        assert!(matches!(
            serde_json::from_value::<StreamEvent>(old).unwrap(),
            StreamEvent::ToolCallDelta {
                provider_id: None,
                toolset_name: None,
                ..
            }
        ));
    }

    /// A request that is not continuing anything must not say so with a null.
    ///
    /// `Option<T>` already deserializes from an absent key without help, so the
    /// decision the attribute actually carries is on the way out: an endpoint
    /// that accepts the key but rejects a null for it would refuse every first
    /// turn, and the shape this project promises has no null-valued keys in it.
    #[test]
    fn an_absent_continuation_is_left_out_rather_than_sent_as_null() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": []
        }))
        .unwrap();
        let wire = serde_json::to_value(&req).unwrap();
        assert!(
            wire.get("continuation").is_none(),
            "an absent continuation became {:?}",
            wire.get("continuation")
        );
    }

    #[test]
    fn response_ids_are_optional_in_current_requests_and_reports() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": []
        }))
        .unwrap();
        assert!(req.continuation.is_none());

        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "message": {"role": "assistant", "content": []},
            "stop_reason": "end_turn",
            "usage": {"usage": null, "state": "missing"},
            "model": "m"
        }))
        .unwrap();
        assert!(resp.response_id.is_none());
        assert!(resp.executed_profile.is_none());

        let start: StreamEvent = serde_json::from_value(serde_json::json!({
            "event": "start",
            "model": "m"
        }))
        .unwrap();
        assert!(matches!(
            start,
            StreamEvent::Start {
                response_id: None,
                ..
            }
        ));
    }

    #[test]
    fn response_ids_round_trip_as_opaque_strings() {
        let id = ResponseId::new("resp_abc");
        let reference = ContinuationRef {
            response_id: id.clone(),
            provider_id: ProviderId::new("acme"),
            profile_name: "acme:one".into(),
            endpoint_fingerprint: "endpoint".into(),
            account_scope: "account-1".into(),
            request_model: "m".into(),
            workspace_id: None,
        };
        let req = ChatRequest {
            prompt_cache: Default::default(),
            output_format: Default::default(),
            service_tier: None,
            model: "m".to_owned(),
            anthropic_client_toolsets: Vec::new(),
            hosted_tools: vec![],
            continuation: Some(reference.clone()),
            system: vec![],
            messages: vec![],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            max_tokens: None,
            temperature: None,
            thinking: None,
            stop_sequences: vec![],
            metadata: Value::Null,
        };
        let encoded = serde_json::to_value(&req).unwrap();
        assert_eq!(encoded["continuation"]["response_id"], "resp_abc");
        assert_eq!(
            serde_json::from_value::<ChatRequest>(encoded)
                .unwrap()
                .continuation,
            Some(reference)
        );
    }

    #[test]
    fn reasoning_tokens_are_a_subset_of_output_not_another_bucket() {
        // They are already inside `output_tokens`. Adding them to the total
        // would over-bill every thinking turn by the size of its thinking.
        let u = Usage {
            input_tokens: 10,
            output_tokens: 100,
            cache_read_tokens: 5,
            cache_write_tokens: 2,
            cache_write_1h_tokens: 1,
            reasoning_tokens: 80,
            cost: None,
            server_tool_usage: None,
        };
        assert_eq!(u.total(), 117, "80 reasoning tokens are inside the 100");
        let without = Usage {
            reasoning_tokens: 0,
            ..u
        };
        assert_eq!(
            u.total(),
            without.total(),
            "how much of the output was thinking cannot change what is owed"
        );
    }

    #[test]
    fn a_reported_cost_survives_the_trip_through_nanos() {
        assert_eq!(
            ReportedCost::from_usd(0.95),
            Some(ReportedCost {
                nano_usd: 950_000_000
            })
        );
        assert_eq!(
            ReportedCost::from_usd(0.000_000_1),
            Some(ReportedCost { nano_usd: 100 }),
            "a tenth of a micro-dollar is a real per-call cost; micros would \
             have rounded it to nothing"
        );
        assert_eq!(ReportedCost::from_usd(0.0), Some(ReportedCost::ZERO));
    }

    #[test]
    fn an_amount_that_is_not_a_cost_is_rejected_rather_than_wrapped() {
        for bad in [-1.0, f64::NAN, f64::INFINITY, 1e30] {
            assert_eq!(
                ReportedCost::from_usd(bad),
                None,
                "{bad} is not a cost, and a nonsense integer would read as one"
            );
        }
    }

    #[test]
    fn an_older_usage_without_the_reasoning_field_still_parses() {
        // New usage fields default because transcripts written before they
        // existed must keep loading; a hard error there would strand sessions.
        let u: Usage = serde_json::from_str(
            r#"{"input_tokens":1,"output_tokens":2,"cache_read_tokens":3,"cache_write_tokens":4}"#,
        )
        .expect("a usage record without the newer field must still load");
        assert_eq!(u.reasoning_tokens, 0);
        assert_eq!(u.cache_write_1h_tokens, 0);
        assert_eq!(u.total(), 10);
    }

    #[test]
    fn total_saturates_when_provider_token_counters_overflow() {
        let usage = Usage {
            input_tokens: u64::MAX,
            output_tokens: 1,
            ..Usage::default()
        };
        assert_eq!(usage.total(), u64::MAX);
    }
}
