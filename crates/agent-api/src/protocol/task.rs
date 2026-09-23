//! Data-only task identity, lifecycle, and authorization facts.
//!
//! Runtime-owned futures, cancellation scopes, locks, and driver traits stay
//! in `lingxi-agent-runtime::tasks`. These values are safe to persist or send
//! across a supervisor boundary and contain no live execution capability.

use crate::protocol::ids::{AgentId, SessionId, ToolUseId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

macro_rules! numeric_id {
    ($name:ident, $first:expr) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl $name {
            pub const FIRST: Self = Self($first);
            pub const fn new(value: u64) -> Self {
                Self(value)
            }
            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

numeric_id!(TaskEpoch, 1);
numeric_id!(TaskGeneration, 1);
numeric_id!(ShutdownId, 1);

/// Opaque task identity used by handles, events, and persistence records.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TaskId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// The authenticated owner namespace used by runtime handles.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskOwner {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<AgentId>,
}

/// Conversation/agent association for task registration and notification
/// routing.  This is data-only; authorization still comes from the live
/// owner-bound controller and task identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskAssociation {
    pub session_id: SessionId,
    pub generation: TaskGeneration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<AgentId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<ToolUseId>,
}

impl TaskOwner {
    pub fn root(session_id: impl Into<SessionId>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: None,
        }
    }

    #[must_use]
    pub fn agent(session_id: impl Into<SessionId>, agent_id: impl Into<AgentId>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: Some(agent_id.into()),
        }
    }
}

/// Lifecycle state published by the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Registered,
    Running,
    Parked,
    Stopping,
    Completed,
    Failed,
    Cancelled,
    Stopped,
}

/// Observable work state for a live task. Waiting for external input keeps
/// the task Running and supervisor-owned while allowing controllers to treat
/// it as idle for close/drain decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskActivity {
    #[default]
    Running,
    WaitingForInput,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Stopped
        )
    }
}

/// Fine-grained rights carried by a host's owner-bound `TaskHandle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCapability {
    Observe,
    Activate,
    SendMessage,
    Interrupt,
    Stop,
    Resume,
    Background,
    ManageChildren,
    Wait,
}

pub type TaskCapabilities = BTreeSet<TaskCapability>;

/// Identity facts checked on every control or result path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskIdentity {
    pub task_id: TaskId,
    pub owner: TaskOwner,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<TaskId>,
    pub generation: TaskGeneration,
    pub epoch: TaskEpoch,
}

/// A protocol-level result. The supervisor is responsible for exactly-once
/// publication of the result for a `(task_id, generation, epoch)` tuple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum TaskOutcome {
    Completed { value: Value },
    Failed { message: String },
    Cancelled,
    Stopped,
}

/// A message queued for a running or parked task. Interrupt-and-send is a
/// distinct runtime command and is never represented as an ordinary message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskMessage {
    pub message_id: String,
    pub body: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskMessageSource {
    User,
    Model,
    Parent,
    System,
}

/// Read-only task view returned to callers; it contains no live handles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub identity: TaskIdentity,
    /// Monotonically increasing task-view revision for race-free watches.
    #[serde(default)]
    pub revision: u64,
    pub state: TaskState,
    #[serde(default)]
    pub activity: TaskActivity,
    pub capabilities: TaskCapabilities,
    pub aliases: Vec<String>,
    pub background: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<TaskOutcome>,
}
