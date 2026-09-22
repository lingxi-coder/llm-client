//! Commands queued for the turn loop (adopted from the previous project's
//! `msgqueue`). The reducer takes them by priority at `Idle` (§13 step 1).

use crate::protocol::ids::ModeName;
use crate::protocol::message::ContentBlock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Interrupts the current turn.
    Now,
    /// Runs before anything else queued.
    Next,
    /// Runs in arrival order.
    Later,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueuedCommandKind {
    UserInput {
        text: String,
        #[serde(default)]
        attachments: Vec<ContentBlock>,
        /// The message was committed as part of an external delivery batch
        /// before this command entered the turn reducer.  The reducer must
        /// not append it a second time at UserPromptSubmit.
        #[serde(default)]
        history_committed: bool,
    },
    SlashCommand {
        name: String,
        args: String,
    },
    ModeSwitch {
        to: ModeName,
    },
    /// `/compact`: compact now, with the user's focus if any; not a turn.
    Compact {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
    Interrupt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueuedCommand {
    pub priority: Priority,
    pub kind: QueuedCommandKind,
}
