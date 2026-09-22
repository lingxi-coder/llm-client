//! LLM-facing data: request, response, stream events, model listings, and the
//! error taxonomy. The traits that produce these live in `llm-client`.

use crate::protocol::ids::{ProviderId, ResponseId, ToolUseId};
use crate::protocol::message::ConversationMessage;
use crate::protocol::provider::{
    AuthStrategy, BillingMode, ProtocolFamily, ProviderInfo, TokenPricing,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;

/// 19 variants — the previous project's `llm-client/src/error.rs` at
/// `57c60a67a` and at HEAD. The HTTP status, when there is one, lives in
/// `message`, never in a field: a status field invites callers to branch on the
/// number instead of the variant, and the variant is what the turn loop
/// branches on (`ContextOverflow` → reactive compaction, §13).
///
/// Deliberately not `#[non_exhaustive]`: downstream `match`es must fail to
/// compile when a variant is added, so no branch dies silently.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmError {
    #[error("authentication failed: {message}")]
    Authentication { message: String },
    #[error("OAuth refresh is dead: {message}")]
    OAuthRefreshDead { message: String },
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
    /// The transcript no longer fits the model's context window.
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
    #[error("TLS certificate error: {message}")]
    TlsCert { message: String },
    #[error("stream interrupted: {message}")]
    StreamInterrupted { message: String },
    #[error("cost unavailable: {message}")]
    CostUnavailable { message: String },
    #[error("unsupported capability: {message}")]
    UnsupportedCapability { message: String },
    #[error("media delegation unavailable: {message}")]
    MediaDelegationUnavailable { message: String },
    #[error("media delegation partial: {message}")]
    MediaDelegationPartial { message: String },
}

/// Field-less mirror of [`LlmError`], for tables (failover triggers, gate 31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmErrorKind {
    Authentication,
    OAuthRefreshDead,
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
    TlsCert,
    StreamInterrupted,
    CostUnavailable,
    UnsupportedCapability,
    MediaDelegationUnavailable,
    MediaDelegationPartial,
}

impl LlmErrorKind {
    pub const ALL: [LlmErrorKind; 19] = [
        LlmErrorKind::Authentication,
        LlmErrorKind::OAuthRefreshDead,
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
        LlmErrorKind::TlsCert,
        LlmErrorKind::StreamInterrupted,
        LlmErrorKind::CostUnavailable,
        LlmErrorKind::UnsupportedCapability,
        LlmErrorKind::MediaDelegationUnavailable,
        LlmErrorKind::MediaDelegationPartial,
    ];
}

impl LlmError {
    pub fn kind(&self) -> LlmErrorKind {
        match self {
            LlmError::Authentication { .. } => LlmErrorKind::Authentication,
            LlmError::OAuthRefreshDead { .. } => LlmErrorKind::OAuthRefreshDead,
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
            LlmError::TlsCert { .. } => LlmErrorKind::TlsCert,
            LlmError::StreamInterrupted { .. } => LlmErrorKind::StreamInterrupted,
            LlmError::CostUnavailable { .. } => LlmErrorKind::CostUnavailable,
            LlmError::UnsupportedCapability { .. } => LlmErrorKind::UnsupportedCapability,
            LlmError::MediaDelegationUnavailable { .. } => LlmErrorKind::MediaDelegationUnavailable,
            LlmError::MediaDelegationPartial { .. } => LlmErrorKind::MediaDelegationPartial,
        }
    }

    /// §13 step 3: these two, and only these two, send the turn into
    /// `Draining → Compact(Reactive)` and one retry.
    pub fn triggers_reactive_compaction(&self) -> bool {
        matches!(
            self,
            LlmError::ContextOverflow { .. } | LlmError::RequestTooLarge { .. }
        )
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
}

/// A cost a provider reported, in nano-USD.
///
/// Integer, not a float, for two reasons: money should not accumulate rounding
/// error over a long session, and `Usage` stays `Copy + Eq` so the reducer can
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
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }
}

/// Provider-neutral stream events (§7.2). Block indices are per response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
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
        name: String,
        arguments_fragment: String,
    },
    End {
        stop_reason: StopReason,
        usage: Usage,
    },
}

/// A system-prompt block. `cacheable` partitions the prompt so the stable part
/// can be prefix-cached (§13 step 2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    pub cacheable: bool,
}

/// A tool as advertised on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
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
    pub thinking: Option<ThinkingConfig>,
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
    pub message: ConversationMessage,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<ResponseId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub vision: bool,
    /// Document input (PDF, files). Separate from `vision`: a model that reads
    /// an image may still refuse a PDF, and routing a document to it wastes a
    /// turn discovering that.
    pub documents: bool,
    pub tools: bool,
    pub reasoning: bool,
    /// Reasoning blocks carry a signature that must round-trip.
    pub signed_reasoning: bool,
    pub streaming: bool,
    pub structured_output: bool,
}

impl ModelCapabilities {
    /// Whether this model accepts a non-text input this project can represent.
    #[must_use]
    pub const fn multimodal(&self) -> bool {
        self.vision || self.documents
    }
}

/// One configured provider, as the client lists it.
///
/// Every profile is listed, whether or not a credential exists for it — that is
/// the point of `info.api_key_url`. This crate holds no credentials (gate 64),
/// so it reports the variable one is conventionally read from and leaves
/// "is it actually set" to the host, which is the only side that can know.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderListing {
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
    /// `None` means unpublished, and is deliberately not zero: a compactor
    /// divides an estimate by this, and the two answers "no room left" and "we
    /// do not know the size of the room" call for opposite behaviour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

/// Who counted. Provider counts win; a local estimate is the fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenEstimateSource {
    Provider,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEstimate {
    pub tokens: u64,
    pub source: TokenEstimateSource,
}

/// Why a compaction ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactTrigger {
    /// `should_compact` said so before the request.
    Proactive,
    /// The provider said `ContextOverflow` / `RequestTooLarge`.
    Reactive,
    /// The user asked (`/compact`), between turns.
    Manual,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nineteen_kinds_and_kind_is_total() {
        assert_eq!(LlmErrorKind::ALL.len(), 19);
        let e = LlmError::ContextOverflow {
            message: "m".into(),
            limit: Some(1),
            actual: Some(2),
        };
        assert_eq!(e.kind(), LlmErrorKind::ContextOverflow);
        assert!(e.triggers_reactive_compaction());
        assert!(LlmError::RequestTooLarge {
            message: "m".into()
        }
        .triggers_reactive_compaction());
        assert!(!LlmError::Overloaded {
            message: "m".into()
        }
        .triggers_reactive_compaction());
    }

    #[test]
    fn stream_event_round_trips() {
        let ev = StreamEvent::ToolCallDelta {
            block: 1,
            id: ToolUseId::new("call_1"),
            name: "Read".into(),
            arguments_fragment: "{\"file".into(),
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert_eq!(serde_json::from_str::<StreamEvent>(&s).unwrap(), ev);
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
    fn response_ids_are_optional_for_older_serialized_requests_and_responses() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": []
        }))
        .unwrap();
        assert!(req.previous_response_id.is_none());

        let resp: CompletionResponse = serde_json::from_value(serde_json::json!({
            "message": {"role": "assistant", "content": []},
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0
            },
            "model": "m"
        }))
        .unwrap();
        assert!(resp.response_id.is_none());

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
            model: "m".to_owned(),
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
            reasoning_tokens: 80,
            cost: None,
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
        // The field is `#[serde(default)]` because transcripts written before it
        // existed must keep loading; a hard error there would strand sessions.
        let u: Usage = serde_json::from_str(
            r#"{"input_tokens":1,"output_tokens":2,"cache_read_tokens":3,"cache_write_tokens":4}"#,
        )
        .expect("a usage record without the newer field must still load");
        assert_eq!(u.reasoning_tokens, 0);
        assert_eq!(u.total(), 10);
    }
}
