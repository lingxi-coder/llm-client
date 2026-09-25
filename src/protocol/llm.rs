//! LLM-facing data: request, response, stream events, model listings, and the
//! error taxonomy used by the client and its codecs.

use crate::protocol::ids::{ProviderId, ResponseId, ToolUseId};
use crate::protocol::message::{ConversationMessage, ProviderFileSource};
use crate::protocol::provider::{
    AuthStrategy, BillingMode, ProtocolFamily, ProviderInfo, Region, TokenPricing,
};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_search_requests: Option<u64>,
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
    /// The provider completed this output block.
    BlockEnd {
        block: usize,
    },
    /// Provider-owned annotation or extension delta, preserved for host rendering.
    NativeDelta {
        block: usize,
        protocol: ProtocolFamily,
        delta: serde_json::Value,
    },
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

/// A system-prompt block. `cacheable` partitions the prompt so the stable part
/// can be prefix-cached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    pub cacheable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<super::CacheControl>,
}

/// A tool as advertised on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default)]
    pub strict: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub extra: Value,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(default)]
    pub controls: super::RequestControls,
    /// Absent by default. Requires an explicit search adapter on the profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<WebSearchConfig>,
    /// Enable Qwen hosted knowledge-base retrieval on a Responses profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_search: Option<FileSearchConfig>,
    /// The immediately preceding response to continue on a stateful Responses
    /// endpoint. The caller owns the chain and supplies only the desired new
    /// input in `messages`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<ResponseId>,
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

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Auto
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub inference: super::InferenceReport,
    pub message: ConversationMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<WebSearchResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_search: Option<FileSearchResult>,
    pub stop_reason: StopReason,
    pub usage: super::UsageReport,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<ResponseId>,
    /// Profile that completed a high-level client request. Direct codec
    /// decoding leaves this absent because the wire does not identify it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_profile: Option<String>,
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
            name: "Read".into(),
            arguments_fragment: "{\"file".into(),
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert_eq!(serde_json::from_str::<StreamEvent>(&s).unwrap(), ev);
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
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": []
        }))
        .unwrap();
        let wire = serde_json::to_value(&req).unwrap();
        assert!(
            wire.get("previous_response_id").is_none(),
            "an absent continuation became {:?}",
            wire.get("previous_response_id")
        );
    }

    #[test]
    fn response_ids_are_optional_in_current_requests_and_reports() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": []
        }))
        .unwrap();
        assert!(req.previous_response_id.is_none());

        let resp: CompletionResponse = serde_json::from_value(serde_json::json!({
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
        let req = CompletionRequest {
            controls: Default::default(),
            service_tier: None,
            model: "m".to_owned(),
            web_search: None,
            file_search: None,
            previous_response_id: Some(id.clone()),
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
        assert_eq!(encoded["previous_response_id"], "resp_abc");
        assert_eq!(
            serde_json::from_value::<CompletionRequest>(encoded)
                .unwrap()
                .previous_response_id,
            Some(id)
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
