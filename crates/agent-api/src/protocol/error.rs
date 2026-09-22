//! Errors the agent runtime reports to the frontend.

use crate::protocol::ids::{EffectId, TurnId};
use crate::protocol::llm::LlmError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum TurnError {
    /// A completion event arrived for an effect no longer outstanding, or for a
    /// previous turn. Dropped without touching state (§13 rule 1).
    #[error("late result for {effect} in {turn}")]
    LateResult { effect: EffectId, turn: TurnId },
    #[error(transparent)]
    Llm(LlmError),
    #[error("turn cancelled")]
    Cancelled,
    #[error("budget exceeded: {message}")]
    BudgetExceeded { message: String },
    #[error("compaction failed: {message}")]
    Compaction { message: String },
    #[error("internal: {message}")]
    Internal { message: String },
}
