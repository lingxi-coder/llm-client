//! `AgentSettings`: the knobs the turn loop reads. Pure data, so it sits in
//! `protocol` and both `config` (which loads it) and `agent-runtime` (which
//! obeys it) name the same type instead of converting between two (review C7).
//! The layered managed/user/project/local merge with schema validation and
//! error attribution (gate 40) is M1; here is the shape and the defaults.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct AgentSettings {
    /// The model a turn is sent to: a catalog id, or `profile/model` to pin
    /// the connection. `None` until configured; a run without one is refused
    /// at boot rather than guessed at.
    pub model: Option<String>,
    /// Upper bound on concurrency-safe tool calls running in one batch.
    pub max_concurrent_tools: usize,
    pub hook_timeout_ms: u64,
    /// How long an `AskHost` may stay unanswered before it counts as `Deny`.
    pub ask_timeout_ms: u64,
    /// The `<total_tokens>` reminder: `off`, `infinite`, `fixed`,
    /// `countdown` or `padded-countdown`. Absent means off.
    pub total_tokens_reminder: Option<String>,
    /// The padded countdown's budget; absent means fifteen million.
    pub total_tokens_reminder_budget: Option<u64>,
    /// Whether a user prompt re-anchors the countdown; absent means yes.
    pub total_tokens_reminder_after_user_turn: Option<bool>,
    /// The model a `refusal` stop swaps to, once per session.
    pub refusal_fallback_model: Option<String>,
    /// An ordered chain to try instead, each hop as the previous refuses.
    pub refusal_fallback_chain: Vec<String>,
    /// Offer the `EndConversation` tool.
    pub end_conversation_tool: bool,
    /// Also read the ecosystem's plugin manifest directory (the one compat
    /// switch the design allows; default off).
    pub compat_plugin_manifest: bool,
    /// Withdraw the Workflow tool (the managed `disableWorkflows`).
    pub disable_workflows: bool,
    /// `unrestricted`, `small`, `medium` or `large`: how many agents a
    /// workflow should fan out to. Absent means `medium`.
    pub workflow_size_guideline: Option<String>,
    /// Let a model pick the memories relevant to a turn (the deterministic
    /// ranker otherwise).
    pub memory_selector: bool,
    /// The model the memory selector asks; absent means the session's.
    pub memory_selector_model: Option<String>,
    /// Distil the session's durable notes into the session-memory
    /// directory as the conversation grows.
    pub session_memory: bool,
    /// The model that distils; absent means the session's.
    pub session_memory_model: Option<String>,
}

impl AgentSettings {
    /// The refusal chain: the explicit chain, else the single fallback
    /// model as a one-hop chain.
    #[must_use]
    pub fn refusal_chain(&self) -> Vec<String> {
        if !self.refusal_fallback_chain.is_empty() {
            return self.refusal_fallback_chain.clone();
        }
        self.refusal_fallback_model.iter().cloned().collect()
    }
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            model: None,
            max_concurrent_tools: 10,
            hook_timeout_ms: 60_000,
            ask_timeout_ms: 600_000,
            total_tokens_reminder: None,
            total_tokens_reminder_budget: None,
            total_tokens_reminder_after_user_turn: None,
            refusal_fallback_model: None,
            refusal_fallback_chain: Vec::new(),
            end_conversation_tool: false,
            compat_plugin_manifest: false,
            disable_workflows: false,
            workflow_size_guideline: None,
            memory_selector: false,
            memory_selector_model: None,
            session_memory: false,
            session_memory_model: None,
        }
    }
}
