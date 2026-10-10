//! LLM-facing data: request, response, stream events, model listings, and the
//! error taxonomy used by the client and its codecs.

use crate::protocol::ids::{ProviderId, ResponseId, ToolUseId};
use crate::protocol::provider::{
    AuthStrategy, BillingMode, ProtocolFamily, ProviderInfo, Region, TokenPricing,
};
use crate::protocol::ContinuationRef;
use crate::protocol::{ConversationMessage, ProviderFileSource};
use crate::protocol::{JsonPayload, JsonPayloadFragment};
use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{value::RawValue, Value};
use std::time::Duration;
use thiserror::Error;

/// Provider and client communication failures. Most variants include HTTP
/// details in `message`; [`LlmError::ProviderResponse`] retains structured
/// provider diagnostics. Callers can classify failures with [`LlmError::kind`].
/// Recovery decisions belong to the caller.
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
    /// A provider failure whose exact response is needed for recovery.
    /// `classification` retains the usual error category for generic callers.
    #[error("provider response HTTP {status} ({classification:?}): {body}")]
    ProviderResponse {
        status: u16,
        request_id: Option<String>,
        body: Box<Value>,
        classification: LlmErrorKind,
        retry_after: Option<Duration>,
    },
    /// A provider's typed `timeout_error`, distinct from a local transport timer.
    #[error("provider timeout: {message}")]
    ProviderTimeout {
        message: String,
        status: Option<u16>,
    },
    #[error("provider overloaded: {message}")]
    Overloaded {
        message: String,
        /// SDK-derived Native `vU(error)` result for the Anthropic `Rt` route.
        /// Other provider overloads do not inherit this route-specific fact.
        native_retry_base_eligible: bool,
    },
    #[error("transport error: {message}")]
    Transport { message: String },
    #[error("transport timeout: {message}")]
    TransportTimeout { message: String },
    /// An upload was dispatched but its acceptance could not be confirmed.
    /// Retrying it may create another provider-owned file.
    #[error("file upload outcome is unknown: {message}")]
    FileUploadOutcomeUnknown { message: String },
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
    /// Selected provider response text, while keeping its original taxonomy for
    /// retry, timeout, and recovery decisions.
    #[error("{source}")]
    ProviderErrorText {
        source: Box<LlmError>,
        text: ProviderErrorText,
        #[serde(skip)]
        native_anthropic_facts: Option<Box<NativeAnthropicErrorFacts>>,
    },
}

/// Source-backed Native Anthropic error facts attached to the selected route.
/// These remain process-local and are never serialized with the error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAnthropicErrorFacts {
    pub provider_id: ProviderId,
    pub protocol: ProtocolFamily,
    /// Authority of the finalized request URL (host plus any explicit
    /// non-default port). This process-local route evidence omits userinfo,
    /// scheme, path, query, and fragment.
    pub request_authority: Option<String>,
    /// Current Native Hqo/gVr per-response resumable-query refusal.
    pub thinking_resumption_refused: bool,
    /// Whether the exact resumption beta was present on the dispatched request.
    pub thinking_resumption_beta_requested: Option<bool>,
    /// Native Aet matched an HTTP 400 rejection of an actually sent beta.
    pub thinking_resumption_beta_refused: bool,
}

/// Provider error text with the exact JavaScript UTF-16 projection.
/// `display` is the lossy UTF-8 view; `utf16_code_units` is authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderErrorText {
    display: String,
    utf16_code_units: Vec<u16>,
}

impl std::fmt::Display for ProviderErrorText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display)
    }
}

impl ProviderErrorText {
    pub fn new(display: String, utf16_code_units: Vec<u16>) -> Option<Self> {
        (String::from_utf16_lossy(&utf16_code_units) == display).then_some(Self {
            display,
            utf16_code_units,
        })
    }

    pub fn display(&self) -> &str {
        &self.display
    }

    pub fn utf16_code_units(&self) -> &[u16] {
        &self.utf16_code_units
    }

    #[must_use]
    pub fn prefixed(&self, prefix: &str) -> Self {
        let mut display = String::with_capacity(prefix.len() + self.display.len());
        display.push_str(prefix);
        display.push_str(&self.display);
        let mut units = prefix.encode_utf16().collect::<Vec<_>>();
        units.extend_from_slice(&self.utf16_code_units);
        Self::new(display, units).expect("prefixing preserves the UTF-16 display projection")
    }
}

impl From<String> for ProviderErrorText {
    fn from(display: String) -> Self {
        let units = display.encode_utf16().collect();
        Self::new(display, units).expect("Rust strings have valid UTF-16")
    }
}

impl From<&str> for ProviderErrorText {
    fn from(display: &str) -> Self {
        display.to_owned().into()
    }
}

impl<'de> Deserialize<'de> for ProviderErrorText {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            display: String,
            utf16_code_units: Vec<u16>,
        }
        let wire = Wire::deserialize(deserializer)?;
        ProviderErrorText::new(wire.display, wire.utf16_code_units).ok_or_else(|| {
            serde::de::Error::custom("provider error UTF-16 units do not match display text")
        })
    }
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
    ProviderTimeout,
    Overloaded,
    Transport,
    TransportTimeout,
    FileUploadOutcomeUnknown,
    ProviderFileProcessing,
    TlsCert,
    StreamInterrupted,
    CostUnavailable,
    UnsupportedCapability,
}

impl LlmErrorKind {
    pub const ALL: [LlmErrorKind; 19] = [
        LlmErrorKind::Authentication,
        LlmErrorKind::PermissionDenied,
        LlmErrorKind::InvalidRequest,
        LlmErrorKind::RateLimited,
        LlmErrorKind::QuotaExceeded,
        LlmErrorKind::ContextOverflow,
        LlmErrorKind::RequestTooLarge,
        LlmErrorKind::ModelUnavailable,
        LlmErrorKind::ProviderInternal,
        LlmErrorKind::ProviderTimeout,
        LlmErrorKind::Overloaded,
        LlmErrorKind::Transport,
        LlmErrorKind::TransportTimeout,
        LlmErrorKind::FileUploadOutcomeUnknown,
        LlmErrorKind::ProviderFileProcessing,
        LlmErrorKind::TlsCert,
        LlmErrorKind::StreamInterrupted,
        LlmErrorKind::CostUnavailable,
        LlmErrorKind::UnsupportedCapability,
    ];
}

impl LlmError {
    /// Return the original classification error beneath the response text.
    #[must_use]
    pub fn classification_source(&self) -> &Self {
        match self {
            Self::ProviderErrorText { source, .. } => source.classification_source(),
            other => other,
        }
    }

    pub(crate) fn classification_source_mut(&mut self) -> &mut Self {
        match self {
            Self::ProviderErrorText { source, .. } => source.classification_source_mut(),
            other => other,
        }
    }

    pub(crate) fn into_classification_source(self) -> Self {
        match self {
            Self::ProviderErrorText { source, .. } => source.into_classification_source(),
            other => other,
        }
    }

    /// Return the provider-selected response text, when present.
    #[must_use]
    pub fn provider_error_text(&self) -> Option<&ProviderErrorText> {
        match self {
            Self::ProviderErrorText { text, .. } => Some(text),
            _ => None,
        }
    }

    /// Current Native Anthropic facts when this error came from that SDK route.
    #[must_use]
    pub fn native_anthropic_error_facts(&self) -> Option<&NativeAnthropicErrorFacts> {
        match self {
            Self::ProviderErrorText {
                native_anthropic_facts,
                ..
            } => native_anthropic_facts.as_deref(),
            _ => None,
        }
    }

    /// Attach provider text without changing the error taxonomy or previously
    /// attached Native facts.
    pub fn with_provider_error_text(self, text: ProviderErrorText) -> Self {
        let (source, native_anthropic_facts) = match self {
            Self::ProviderErrorText {
                source,
                native_anthropic_facts,
                ..
            } => (*source, native_anthropic_facts),
            other => (other, None),
        };
        Self::ProviderErrorText {
            source: Box::new(source.into_classification_source()),
            text,
            native_anthropic_facts,
        }
    }

    /// Attach route/error facts while preserving the original classification
    /// and authoritative provider text.
    pub fn with_native_anthropic_error(
        self,
        text: ProviderErrorText,
        facts: NativeAnthropicErrorFacts,
    ) -> Self {
        let source = self.into_classification_source();
        Self::ProviderErrorText {
            source: Box::new(source),
            text,
            native_anthropic_facts: Some(Box::new(facts)),
        }
    }

    /// Attach the selected HTTP request authority to process-local Native
    /// Anthropic error facts, without replacing the provider text or taxonomy.
    pub fn with_native_anthropic_request_authority(mut self, authority: Option<String>) -> Self {
        if let Self::ProviderErrorText {
            native_anthropic_facts: Some(facts),
            ..
        } = &mut self
        {
            facts.request_authority = authority;
        }
        self
    }

    pub fn with_optional_provider_error_text(self, text: Option<ProviderErrorText>) -> Self {
        match text {
            Some(text) => self.with_provider_error_text(text),
            None => self,
        }
    }

    pub fn kind(&self) -> LlmErrorKind {
        match self.classification_source() {
            LlmError::Authentication { .. } => LlmErrorKind::Authentication,
            LlmError::PermissionDenied { .. } => LlmErrorKind::PermissionDenied,
            LlmError::InvalidRequest { .. } => LlmErrorKind::InvalidRequest,
            LlmError::RateLimited { .. } => LlmErrorKind::RateLimited,
            LlmError::QuotaExceeded { .. } => LlmErrorKind::QuotaExceeded,
            LlmError::ContextOverflow { .. } => LlmErrorKind::ContextOverflow,
            LlmError::RequestTooLarge { .. } => LlmErrorKind::RequestTooLarge,
            LlmError::ModelUnavailable { .. } => LlmErrorKind::ModelUnavailable,
            LlmError::ProviderInternal { .. } => LlmErrorKind::ProviderInternal,
            LlmError::ProviderResponse { classification, .. } => *classification,
            LlmError::ProviderTimeout { .. } => LlmErrorKind::ProviderTimeout,
            LlmError::Overloaded { .. } => LlmErrorKind::Overloaded,
            LlmError::Transport { .. } => LlmErrorKind::Transport,
            LlmError::TransportTimeout { .. } => LlmErrorKind::TransportTimeout,
            LlmError::FileUploadOutcomeUnknown { .. } => LlmErrorKind::FileUploadOutcomeUnknown,
            LlmError::ProviderFileProcessing { .. } => LlmErrorKind::ProviderFileProcessing,
            LlmError::TlsCert { .. } => LlmErrorKind::TlsCert,
            LlmError::StreamInterrupted { .. } => LlmErrorKind::StreamInterrupted,
            LlmError::CostUnavailable { .. } => LlmErrorKind::CostUnavailable,
            LlmError::UnsupportedCapability { .. } => LlmErrorKind::UnsupportedCapability,
            LlmError::ProviderErrorText { .. } => {
                unreachable!("classification_source removes carriers")
            }
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
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A complete provider-owned typed item. Preserve its native format and
    /// require a successful terminal response before caller-owned execution.
    Native {
        block: usize,
        value: super::NativeExtension,
    },
    /// SDK-decoded provider control data, never replayable content.
    NativeControl {
        protocol: ProtocolFamily,
        control: super::NativeExtension,
    },
    /// The provider completed this output block.
    BlockEnd {
        block: usize,
    },
    /// Provider-owned annotation or extension delta, preserved for host rendering.
    NativeDelta {
        block: usize,
        protocol: ProtocolFamily,
        delta: JsonPayload,
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
        value: JsonPayload,
    },
    /// Raw provider stream frame that the codec does not interpret. This is
    /// observational data, not a content block: do not replay it as message
    /// content or execute it as a client tool call.
    ProviderEvent {
        protocol: ProtocolFamily,
        payload: JsonPayload,
    },
    Start {
        model: String,
        response_id: Option<ResponseId>,
    },
    TextDelta {
        block: usize,
        text: String,
    },
    /// Text delta decoded from a provider JSON string containing unpaired
    /// JavaScript UTF-16 units. Exact units must survive block accumulation.
    TextDeltaJsUtf16 {
        block: usize,
        text: String,
        utf16_code_units: Vec<u16>,
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
        provider_id: Option<String>,
        /// Native caller metadata when the provider routes a tool call through
        /// another hosted tool, as Anthropic does for programmatic calls.
        /// Keep the original object so callers can distinguish direct calls
        /// and retain future provider fields.
        caller: Option<JsonPayload>,
        /// Anthropic client-toolset family for Browser/Computer member calls.
        /// Carried from the opening content-block event to each input delta.
        toolset_name: Option<String>,
        name: String,
        arguments_fragment: JsonPayloadFragment,
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
        inference: super::InferenceReport,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StreamEventWithoutRawPayload {
    Native {
        block: usize,
        value: super::NativeExtension,
    },
    NativeControl {
        protocol: ProtocolFamily,
        control: super::NativeExtension,
    },
    BlockEnd {
        block: usize,
    },
    Inference {
        report: super::InferenceReport,
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
    TextDeltaJsUtf16 {
        block: usize,
        text: String,
        #[serde(skip)]
        utf16_code_units: Vec<u16>,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<JsonPayload>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
        name: String,
        arguments_fragment: JsonPayloadFragment,
    },
    WebSearch {
        result: WebSearchResult,
    },
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

impl From<StreamEventWithoutRawPayload> for StreamEvent {
    fn from(event: StreamEventWithoutRawPayload) -> Self {
        match event {
            StreamEventWithoutRawPayload::Native { block, value } => Self::Native { block, value },
            StreamEventWithoutRawPayload::NativeControl { protocol, control } => {
                Self::NativeControl { protocol, control }
            }
            StreamEventWithoutRawPayload::BlockEnd { block } => Self::BlockEnd { block },
            StreamEventWithoutRawPayload::Inference { report } => Self::Inference { report },
            StreamEventWithoutRawPayload::Start { model, response_id } => {
                Self::Start { model, response_id }
            }
            StreamEventWithoutRawPayload::TextDelta { block, text } => {
                Self::TextDelta { block, text }
            }
            StreamEventWithoutRawPayload::TextDeltaJsUtf16 {
                block,
                text,
                utf16_code_units,
            } => Self::TextDeltaJsUtf16 {
                block,
                text,
                utf16_code_units,
            },
            StreamEventWithoutRawPayload::ReasoningDelta { block, text } => {
                Self::ReasoningDelta { block, text }
            }
            StreamEventWithoutRawPayload::ThoughtSignature { block, signature } => {
                Self::ThoughtSignature { block, signature }
            }
            StreamEventWithoutRawPayload::RedactedThinking { block, data } => {
                Self::RedactedThinking { block, data }
            }
            StreamEventWithoutRawPayload::ToolCallDelta {
                block,
                id,
                provider_id,
                caller,
                toolset_name,
                name,
                arguments_fragment,
            } => Self::ToolCallDelta {
                block,
                id,
                provider_id,
                caller,
                toolset_name,
                name,
                arguments_fragment,
            },
            StreamEventWithoutRawPayload::WebSearch { result } => Self::WebSearch { result },
            StreamEventWithoutRawPayload::FileSearch { result } => Self::FileSearch { result },
            StreamEventWithoutRawPayload::End {
                stop_reason,
                usage,
                inference,
            } => Self::End {
                stop_reason,
                usage,
                inference,
            },
        }
    }
}

impl TryFrom<&StreamEvent> for StreamEventWithoutRawPayload {
    type Error = ();

    fn try_from(event: &StreamEvent) -> Result<Self, Self::Error> {
        Ok(match event {
            StreamEvent::Native { block, value } => Self::Native {
                block: *block,
                value: value.clone(),
            },
            StreamEvent::NativeControl { protocol, control } => Self::NativeControl {
                protocol: *protocol,
                control: control.clone(),
            },
            StreamEvent::BlockEnd { block } => Self::BlockEnd { block: *block },
            StreamEvent::Inference { report } => Self::Inference {
                report: report.clone(),
            },
            StreamEvent::Start { model, response_id } => Self::Start {
                model: model.clone(),
                response_id: response_id.clone(),
            },
            StreamEvent::TextDelta { block, text } => Self::TextDelta {
                block: *block,
                text: text.clone(),
            },
            StreamEvent::TextDeltaJsUtf16 {
                block,
                text,
                utf16_code_units,
            } => Self::TextDeltaJsUtf16 {
                block: *block,
                text: text.clone(),
                utf16_code_units: utf16_code_units.clone(),
            },
            StreamEvent::ReasoningDelta { block, text } => Self::ReasoningDelta {
                block: *block,
                text: text.clone(),
            },
            StreamEvent::ThoughtSignature { block, signature } => Self::ThoughtSignature {
                block: *block,
                signature: signature.clone(),
            },
            StreamEvent::RedactedThinking { block, data } => Self::RedactedThinking {
                block: *block,
                data: data.clone(),
            },
            StreamEvent::ToolCallDelta {
                block,
                id,
                provider_id,
                caller,
                toolset_name,
                name,
                arguments_fragment,
            } => Self::ToolCallDelta {
                block: *block,
                id: id.clone(),
                provider_id: provider_id.clone(),
                caller: caller.clone(),
                toolset_name: toolset_name.clone(),
                name: name.clone(),
                arguments_fragment: arguments_fragment.clone(),
            },
            StreamEvent::WebSearch { result } => Self::WebSearch {
                result: result.clone(),
            },
            StreamEvent::FileSearch { result } => Self::FileSearch {
                result: result.clone(),
            },
            StreamEvent::End {
                stop_reason,
                usage,
                inference,
            } => Self::End {
                stop_reason: stop_reason.clone(),
                usage: usage.clone(),
                inference: inference.clone(),
            },
            StreamEvent::NativeDelta { .. }
            | StreamEvent::ProviderContent { .. }
            | StreamEvent::ProviderEvent { .. } => return Err(()),
        })
    }
}

impl Serialize for StreamEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::NativeDelta {
                block,
                protocol,
                delta,
            } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("event", "native_delta")?;
                map.serialize_entry("block", block)?;
                map.serialize_entry("protocol", protocol)?;
                map.serialize_entry("delta", delta)?;
                map.end()
            }
            Self::ProviderContent {
                block,
                protocol,
                value,
            } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("event", "provider_content")?;
                map.serialize_entry("block", block)?;
                map.serialize_entry("protocol", protocol)?;
                map.serialize_entry("value", value)?;
                map.end()
            }
            Self::ProviderEvent { protocol, payload } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("event", "provider_event")?;
                map.serialize_entry("protocol", protocol)?;
                map.serialize_entry("payload", payload)?;
                map.end()
            }
            other => StreamEventWithoutRawPayload::try_from(other)
                .expect("non-raw stream event")
                .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for StreamEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let scanner = crate::response_json::ResponseJsonScanner::parse(
            raw.get().as_bytes(),
            "stream event is not valid JSON",
        )
        .map_err(serde::de::Error::custom)?;
        let event = scanner
            .string_at("/event")
            .ok_or_else(|| serde::de::Error::custom("stream event has no string event tag"))?;
        match event {
            "native_delta" => Ok(Self::NativeDelta {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                protocol: required_stream_field(&scanner, "/protocol")
                    .map_err(serde::de::Error::custom)?,
                delta: required_stream_payload(&scanner, "/delta")
                    .map_err(serde::de::Error::custom)?,
            }),
            "provider_content" => Ok(Self::ProviderContent {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                protocol: required_stream_field(&scanner, "/protocol")
                    .map_err(serde::de::Error::custom)?,
                value: required_stream_payload(&scanner, "/value")
                    .map_err(serde::de::Error::custom)?,
            }),
            "provider_event" => Ok(Self::ProviderEvent {
                protocol: required_stream_field(&scanner, "/protocol")
                    .map_err(serde::de::Error::custom)?,
                payload: required_stream_payload(&scanner, "/payload")
                    .map_err(serde::de::Error::custom)?,
            }),
            "web_search" => Ok(Self::WebSearch {
                result: required_stream_field(&scanner, "/result")
                    .map_err(serde::de::Error::custom)?,
            }),
            "file_search" => Ok(Self::FileSearch {
                result: required_stream_field(&scanner, "/result")
                    .map_err(serde::de::Error::custom)?,
            }),
            "native" => Ok(Self::Native {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                value: required_stream_field(&scanner, "/value")
                    .map_err(serde::de::Error::custom)?,
            }),
            "native_control" => Ok(Self::NativeControl {
                protocol: required_stream_field(&scanner, "/protocol")
                    .map_err(serde::de::Error::custom)?,
                control: required_stream_field(&scanner, "/control")
                    .map_err(serde::de::Error::custom)?,
            }),
            "block_end" => Ok(Self::BlockEnd {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
            }),
            "inference" => Ok(Self::Inference {
                report: required_stream_field(&scanner, "/report")
                    .map_err(serde::de::Error::custom)?,
            }),
            "start" => Ok(Self::Start {
                model: required_stream_string(&scanner, "/model")
                    .map_err(serde::de::Error::custom)?,
                response_id: optional_stream_field(&scanner, "/response_id")
                    .map_err(serde::de::Error::custom)?,
            }),
            "text_delta" => Ok(Self::TextDelta {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                text: required_stream_string(&scanner, "/text")
                    .map_err(serde::de::Error::custom)?,
            }),
            "text_delta_js_utf16" => {
                let block =
                    required_stream_field(&scanner, "/block").map_err(serde::de::Error::custom)?;
                let text = scanner
                    .string_at("/text")
                    .ok_or_else(|| serde::de::Error::custom("stream text is missing"))?
                    .to_owned();
                let utf16_code_units = scanner
                    .string_units_at("/text")
                    .ok_or_else(|| serde::de::Error::custom("stream text units are missing"))?;
                Ok(Self::TextDeltaJsUtf16 {
                    block,
                    text,
                    utf16_code_units,
                })
            }
            "reasoning_delta" => Ok(Self::ReasoningDelta {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                text: required_stream_string(&scanner, "/text")
                    .map_err(serde::de::Error::custom)?,
            }),
            "thought_signature" => Ok(Self::ThoughtSignature {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                signature: required_stream_string(&scanner, "/signature")
                    .map_err(serde::de::Error::custom)?,
            }),
            "redacted_thinking" => Ok(Self::RedactedThinking {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                data: required_stream_string(&scanner, "/data")
                    .map_err(serde::de::Error::custom)?,
            }),
            "tool_call_delta" => Ok(Self::ToolCallDelta {
                block: required_stream_field(&scanner, "/block")
                    .map_err(serde::de::Error::custom)?,
                id: required_stream_field(&scanner, "/id").map_err(serde::de::Error::custom)?,
                provider_id: optional_stream_field(&scanner, "/provider_id")
                    .map_err(serde::de::Error::custom)?,
                caller: optional_stream_payload(&scanner, "/caller")
                    .map_err(serde::de::Error::custom)?,
                toolset_name: optional_stream_field(&scanner, "/toolset_name")
                    .map_err(serde::de::Error::custom)?,
                name: required_stream_string(&scanner, "/name")
                    .map_err(serde::de::Error::custom)?,
                arguments_fragment: required_stream_field(&scanner, "/arguments_fragment")
                    .map_err(serde::de::Error::custom)?,
            }),
            "end" => Ok(Self::End {
                stop_reason: required_stream_field(&scanner, "/stop_reason")
                    .map_err(serde::de::Error::custom)?,
                usage: required_stream_field(&scanner, "/usage")
                    .map_err(serde::de::Error::custom)?,
                inference: optional_stream_field(&scanner, "/inference")
                    .map_err(serde::de::Error::custom)?
                    .unwrap_or_default(),
            }),
            _ => Err(serde::de::Error::custom("unknown stream event type")),
        }
    }
}

fn required_stream_field<T: DeserializeOwned>(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<T, String> {
    let source = scanner
        .raw_source_at(pointer)
        .ok_or_else(|| format!("required stream field {pointer} is missing"))?;
    serde_json::from_str(source).map_err(|error| error.to_string())
}

fn optional_stream_field<T: DeserializeOwned>(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<T>, String> {
    let Some(source) = scanner.raw_source_at(pointer) else {
        return Ok(None);
    };
    serde_json::from_str::<Option<T>>(source).map_err(|error| error.to_string())
}

fn required_stream_string(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<String, String> {
    let units = scanner
        .string_units_at(pointer)
        .ok_or_else(|| format!("required stream string {pointer} is missing"))?;
    String::from_utf16(&units)
        .map_err(|_| format!("stream string {pointer} contains unpaired UTF-16 units"))
}

fn required_stream_payload(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<JsonPayload, String> {
    let source = scanner
        .raw_source_at(pointer)
        .ok_or_else(|| format!("required raw stream field {pointer} is missing"))?;
    JsonPayload::parse(source).map_err(|error| error.to_string())
}

fn optional_stream_payload(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<JsonPayload>, String> {
    scanner
        .raw_source_at(pointer)
        .map(JsonPayload::parse)
        .transpose()
        .map_err(|error| error.to_string())
}

/// A system-prompt block. Cache breakpoints belong to `ChatRequest.prompt_cache`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemBlock {
    text: JsonPayload,
}

impl SystemBlock {
    /// Construct a provider system-text block from valid Unicode text.
    pub fn new(text: impl AsRef<str>) -> Self {
        Self {
            text: JsonPayload::string(text.as_ref()),
        }
    }

    /// Construct a provider system-text block while retaining exact JS UTF-16 units.
    pub fn from_utf16_code_units(code_units: &[u16]) -> Self {
        Self {
            text: JsonPayload::string_from_utf16_code_units(code_units),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text
            .root_string_code_units()
            .is_some_and(|units| units.is_empty())
    }

    pub fn text_payload(&self) -> &JsonPayload {
        &self.text
    }

    #[cfg(test)]
    pub(crate) fn display_text_lossy(&self) -> String {
        self.text
            .display_root_string_lossy()
            .expect("SystemBlock constructors and deserializer enforce root strings")
    }
}

impl<'de> Deserialize<'de> for SystemBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireSystemBlock {
            text: JsonPayload,
        }

        let wire = WireSystemBlock::deserialize(deserializer)?;
        if !wire.text.is_root_string() {
            return Err(serde::de::Error::custom(
                "system block text must be a JSON string value",
            ));
        }
        Ok(Self { text: wire.text })
    }
}

/// A tool as advertised on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolSpec {
    pub name: String,
    /// Root JSON string retained as source, including lone JavaScript UTF-16 units.
    #[serde(deserialize_with = "deserialize_tool_description")]
    pub description: JsonPayload,
    pub input_schema: JsonPayload,
    #[serde(default)]
    pub strict: bool,
    /// Ask supported hosted tool search to load this definition only after
    /// discovery. Anthropic Messages, OpenAI Responses and OpenRouter support it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
    /// Provider-owned tool policy, validated before encoding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_options: Vec<super::NativeExtension>,
    /// Native tool type for protocols that distinguish function, custom, or
    /// provider-specific tools. Ordinary function tools leave this unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// Provider-specific tool fields preserved for codecs that explicitly
    /// support them. Unsupported codecs must reject rather than drop these.
    #[serde(default, skip_serializing_if = "JsonPayload::is_root_null")]
    pub extra: JsonPayload,
}

fn deserialize_tool_description<'de, D>(deserializer: D) -> Result<JsonPayload, D::Error>
where
    D: Deserializer<'de>,
{
    let description = JsonPayload::deserialize(deserializer)?;
    if description.is_root_string() {
        Ok(description)
    } else {
        Err(serde::de::Error::custom(
            "tool description must be a JSON string",
        ))
    }
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
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_search_domains"
    )]
    pub allowed_domains: Vec<JsonPayload>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_search_domains"
    )]
    pub blocked_domains: Vec<JsonPayload>,
    /// Maximum searches per request, currently supported by Anthropic only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
}

fn deserialize_search_domains<'de, D>(deserializer: D) -> Result<Vec<JsonPayload>, D::Error>
where
    D: Deserializer<'de>,
{
    let domains = Vec::<JsonPayload>::deserialize(deserializer)?;
    if domains.iter().all(JsonPayload::is_root_string) {
        Ok(domains)
    } else {
        Err(serde::de::Error::custom(
            "web search domain filters must contain only JSON strings",
        ))
    }
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
    pub metadata: JsonPayload,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FileSearchResult {
    #[serde(default)]
    pub queries: Vec<String>,
    #[serde(default)]
    pub hits: Vec<FileSearchHit>,
    #[serde(default)]
    pub metadata: JsonPayload,
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

/// A tool executed by the provider. Ordinary caller-executed functions remain
/// in `ChatRequest.tools` and require the host to return their results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum HostedTool {
    WebSearch(WebSearchConfig),
    Native(super::NativeExtension),
}

impl<'de> Deserialize<'de> for HostedTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let payload = JsonPayload::parse(raw.get()).map_err(serde::de::Error::custom)?;
        let kind = payload
            .string_at("/type")
            .ok_or_else(|| serde::de::Error::custom("hosted tool type is missing"))?;
        let config = payload
            .payload_at("/config")
            .ok_or_else(|| serde::de::Error::custom("hosted tool config is missing"))?;
        match kind.as_str() {
            "web_search" => serde_json::from_str::<WebSearchConfig>(config.as_json_source())
                .map(Self::WebSearch)
                .map_err(serde::de::Error::custom),
            "native" => serde_json::from_str::<super::NativeExtension>(config.as_json_source())
                .map(Self::Native)
                .map_err(serde::de::Error::custom),
            _ => Err(serde::de::Error::custom("unknown hosted tool type")),
        }
    }
}

impl HostedTool {
    /// Borrow a provider-owned view. Execution validates malformed native
    /// payloads before using these optional capability helpers.
    pub fn native<T: super::NativeType>(&self) -> Option<&T> {
        match self {
            Self::Native(extension) if extension.is::<T>() => extension.decode::<T>().ok(),
            _ => None,
        }
    }
    pub fn edit_native<T: super::NativeType + Clone, R>(
        &mut self,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, LlmError> {
        match self {
            Self::Native(extension) => extension.edit::<T, R>(edit),
            _ => Err(LlmError::InvalidRequest {
                message: "this hosted tool has no native extension".into(),
            }),
        }
    }
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
    /// Explicit wire-level controls. These are validated against the selected
    /// protocol and never change shared profile defaults.
    #[serde(default)]
    pub controls: super::RequestControls,
    /// Provider-executed tools. These do not create host tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hosted_tools: Vec<HostedTool>,
    /// Format-tagged options validated by the selected provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_options: Vec<super::NativeExtension>,
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
    #[serde(default, skip_serializing_if = "JsonPayload::is_root_null")]
    pub metadata: JsonPayload,
}

impl ChatRequest {
    /// Construct model input with no conversation or optional overrides.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt_cache: super::PromptCachePolicy::default(),
            output_format: super::OutputFormat::default(),
            controls: super::RequestControls::default(),
            hosted_tools: Vec::new(),
            native_options: Vec::new(),
            continuation: None,
            system: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            max_tokens: None,
            temperature: None,
            thinking: None,
            service_tier: None,
            stop_sequences: Vec::new(),
            metadata: JsonPayload::default(),
        }
    }

    /// Both scoped and explicitly supplied legacy response IDs pin execution.
    pub fn has_response_continuation(&self) -> bool {
        self.continuation.is_some() || self.controls.responses.previous_response_id.is_some()
    }
    pub fn hosted_web_search(&self) -> Option<&WebSearchConfig> {
        self.hosted_tools.iter().find_map(|tool| match tool {
            HostedTool::WebSearch(config) => Some(config),
            _ => None,
        })
    }
    pub fn hosted_web_search_mut(&mut self) -> Option<&mut WebSearchConfig> {
        self.hosted_tools.iter_mut().find_map(|tool| match tool {
            HostedTool::WebSearch(config) => Some(config),
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
    /// Lossless provider metadata. Import resource identifiers with the
    /// corresponding provider's explicit account and endpoint scope before reuse.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_metadata: Vec<super::NativeExtension>,
    pub stop_reason: StopReason,
    pub usage: super::UsageReport,
    /// SDK-computed Native Anthropic resumable-query context decision. It is
    /// execution metadata, not provider wire or model-visible data.
    #[serde(skip)]
    pub native_resume_context_exhausted: Option<bool>,
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
    fn provider_response_body_box_keeps_tagged_wire_and_classification() {
        let body = serde_json::json!({
            "error": {
                "code": "subscription_sharing_usage_limit_exceeded",
                "param": "model"
            }
        });
        let error = LlmError::ProviderResponse {
            status: 429,
            request_id: Some("req-plan-limit".into()),
            body: Box::new(body.clone()),
            classification: LlmErrorKind::QuotaExceeded,
            retry_after: None,
        };

        let encoded = serde_json::to_value(&error).unwrap();
        assert_eq!(encoded["kind"], "provider_response");
        assert_eq!(encoded["status"], 429);
        assert_eq!(encoded["request_id"], "req-plan-limit");
        assert_eq!(encoded["body"], body);
        assert_eq!(error.kind(), LlmErrorKind::QuotaExceeded);

        let decoded: LlmError = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, error);
        assert_eq!(decoded.kind(), LlmErrorKind::QuotaExceeded);
        assert_eq!(decoded.to_string(), error.to_string());
    }

    #[test]
    fn exact_provider_error_carrier_roundtrips_through_tagged_llm_error() {
        let error = LlmError::InvalidRequest {
            message: "400 bad �".into(),
        }
        .with_provider_error_text(
            ProviderErrorText::new(
                "400 bad �".into(),
                "400 bad ".encode_utf16().chain([0xd800]).collect(),
            )
            .unwrap(),
        );
        let encoded = serde_json::to_string(&error).unwrap();
        let decoded: LlmError = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, error);
        assert_eq!(decoded.kind(), LlmErrorKind::InvalidRequest);
        assert_eq!(
            decoded.provider_error_text().unwrap().utf16_code_units(),
            &[52, 48, 48, 32, 98, 97, 100, 32, 0xd800]
        );
    }

    #[test]
    fn attaching_provider_text_flattens_carriers_without_changing_classification() {
        let first = ProviderErrorText::from("first");
        let second = ProviderErrorText::new(
            "second�".into(),
            vec![
                b's' as u16,
                b'e' as u16,
                b'c' as u16,
                b'o' as u16,
                b'n' as u16,
                b'd' as u16,
                0xd800,
            ],
        )
        .unwrap();
        let error = LlmError::InvalidRequest {
            message: "first".into(),
        }
        .with_provider_error_text(first)
        .with_provider_error_text(second.clone());
        assert_eq!(error.provider_error_text(), Some(&second));
        assert!(matches!(
            error.classification_source(),
            LlmError::InvalidRequest { .. }
        ));
        let LlmError::ProviderErrorText { source, .. } = &error else {
            panic!("one outer response-text carrier is expected");
        };
        assert!(matches!(source.as_ref(), LlmError::InvalidRequest { .. }));
    }

    #[test]
    fn finalized_request_authority_attaches_to_native_error_facts() {
        let facts = NativeAnthropicErrorFacts {
            provider_id: ProviderId::from("gateway"),
            protocol: ProtocolFamily::AnthropicMessages,
            request_authority: None,
            thinking_resumption_refused: false,
            thinking_resumption_beta_requested: None,
            thinking_resumption_beta_refused: false,
        };
        let error = LlmError::InvalidRequest {
            message: "500 provider error".into(),
        }
        .with_native_anthropic_error(ProviderErrorText::from("500 provider error"), facts)
        .with_native_anthropic_request_authority(Some("127.0.0.1:43123".into()));

        assert_eq!(
            error
                .native_anthropic_error_facts()
                .and_then(|facts| facts.request_authority.as_deref()),
            Some("127.0.0.1:43123")
        );
        assert_eq!(
            error.provider_error_text().map(ProviderErrorText::display),
            Some("500 provider error")
        );
    }

    #[test]
    fn system_blocks_are_validated_exact_root_string_carriers() {
        let source = r#"{"text":"before\ud800after"}"#;
        let block: SystemBlock = serde_json::from_str(source).unwrap();
        assert_eq!(
            block.text_payload().root_string_code_units(),
            Some(
                "before"
                    .encode_utf16()
                    .chain([0xd800])
                    .chain("after".encode_utf16())
                    .collect()
            )
        );
        assert_eq!(serde_json::to_string(&block).unwrap(), source);
        assert!(serde_json::from_str::<SystemBlock>(r#"{"text":null}"#).is_err());
        assert!(serde_json::from_str::<SystemBlock>(r#"{"text":"valid","extra":true}"#).is_err());
    }

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

    #[test]
    fn tagged_tool_call_event_keeps_deep_lone_caller_source() {
        let depth = 16_384;
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let caller_source = format!(r#"{{"deep":{nested},"future":"\ud800"}}"#);
        let source = format!(
            r#"{{"event":"tool_call_delta","block":1,"id":"call_1","caller":{caller_source},"name":"Read","arguments_fragment":"{{}}"}}"#
        );
        let event: StreamEvent = serde_json::from_str(&source).unwrap();
        let StreamEvent::ToolCallDelta {
            caller: Some(caller),
            ..
        } = &event
        else {
            panic!("caller carrier must be retained")
        };
        assert_eq!(caller.as_json_source(), caller_source);
        let encoded = serde_json::to_string(&event).unwrap();
        assert!(encoded.contains(&format!("\"caller\":{caller_source}")));
        assert_eq!(
            serde_json::from_str::<StreamEvent>(&encoded).unwrap(),
            event
        );
    }

    #[test]
    fn request_metadata_and_tool_json_fields_keep_deep_and_lone_source() {
        let depth = 16_384;
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let metadata = format!(r#"{{"future":{nested},"lone":"\ud800"}}"#);
        let schema = format!(r#"{{"type":"object","future":{nested},"lone":"\udfff"}}"#);
        let extra = format!(r#"{{"provider_field":{nested},"lone":"\ud800"}}"#);
        let tool = format!(
            r#"{{"name":"read","description":"Read\ud800","input_schema":{schema},"extra":{extra}}}"#
        );
        let request_source =
            format!(r#"{{"model":"m","messages":[],"metadata":{metadata},"tools":[{tool}]}}"#);
        let request: ChatRequest = serde_json::from_str(&request_source).unwrap();
        assert_eq!(request.metadata.as_json_source(), metadata);
        assert_eq!(
            request.tools[0].description.as_json_source(),
            r#""Read\ud800""#
        );
        assert_eq!(request.tools[0].input_schema.as_json_source(), schema);
        assert_eq!(request.tools[0].extra.as_json_source(), extra);
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.contains(&format!("\"metadata\":{metadata}")));
        assert!(encoded.contains("\"description\":\"Read\\ud800\""));
        assert!(encoded.contains(&format!("\"input_schema\":{schema}")));
        assert!(encoded.contains(&format!("\"extra\":{extra}")));
        assert!(serde_json::from_str::<ToolSpec>(
            r#"{"name":"tool","description":{},"input_schema":{}}"#
        )
        .is_err());
    }

    #[test]
    fn hosted_native_tool_enum_keeps_deep_and_lone_payload_source() {
        let depth = 16_384;
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let data = format!(r#"{{"deep":{nested},"lone":"\ud800"}}"#);
        let source =
            format!(r#"{{"type":"native","config":{{"format":"test.native.v1","data":{data}}}}}"#);
        let tool: HostedTool = serde_json::from_str(&source).unwrap();
        let HostedTool::Native(extension) = &tool else {
            panic!("expected a native hosted tool")
        };
        assert_eq!(extension.data().as_json_source(), data);
        assert_eq!(serde_json::to_string(&tool).unwrap(), source);
    }

    #[test]
    fn search_event_deserialization_keeps_deep_and_lone_utf16_metadata() {
        let depth = 16_384;
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let cases = [
            format!(r#"{{"deep":{nested},"valid":"café"}}"#),
            r#"{"lone_key":"\ud800","\udfff":"value"}"#.to_owned(),
        ];
        for metadata in cases {
            let source = format!(
                r#"{{"event":"web_search","result":{{"citations":[],"metadata":{metadata}}}}}"#
            );
            let event: StreamEvent = serde_json::from_str(&source).unwrap();
            let StreamEvent::WebSearch { result } = event else {
                panic!("expected a web search event")
            };
            assert_eq!(result.metadata.as_json_source(), metadata);
            let encoded = serde_json::to_string(&StreamEvent::WebSearch { result }).unwrap();
            assert!(encoded.contains(&format!("\"metadata\":{metadata}")));
        }
        let metadata = r#"{"provider":{"extra":[1,2,3]}}"#;
        let source = format!(r#"{{"event":"file_search","result":{{"metadata":{metadata}}}}}"#);
        let event: StreamEvent = serde_json::from_str(&source).unwrap();
        let StreamEvent::FileSearch { result } = event else {
            panic!("expected a file search event")
        };
        assert_eq!(result.metadata.as_json_source(), metadata);
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
            protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
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
            controls: Default::default(),
            service_tier: None,
            model: "m".to_owned(),
            native_options: Vec::new(),
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
            metadata: JsonPayload::default(),
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
