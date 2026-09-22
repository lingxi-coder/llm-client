//! Pure compaction results and errors.

use crate::protocol::{CompactTrigger, ConversationMessage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionResult {
    pub messages: Vec<ConversationMessage>,
    pub summary: String,
    pub dropped: usize,
    pub trigger: CompactTrigger,
}

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompactError {
    #[error("compaction circuit breaker open after {attempts} attempts")]
    CircuitOpen { attempts: u32 },
    #[error("nothing left to compact")]
    NothingToCompact,
    #[error("summary model failed: {message}")]
    Llm { message: String },
    #[error("cancelled")]
    Cancelled,
}
