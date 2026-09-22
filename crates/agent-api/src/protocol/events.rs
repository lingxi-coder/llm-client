//! Frontend event data emitted by the runtime.

use crate::protocol::{
    AgentId, AskId, CompId, CompactTrigger, ModeName, PermissionDecision, Priority, StopReason,
    ToolUseId, TurnError, TurnId, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentEvent {
    pub agent_id: Option<AgentId>,
    #[serde(flatten)]
    pub kind: AgentEventKind,
}

impl AgentEvent {
    pub fn main(kind: AgentEventKind) -> Self {
        Self {
            agent_id: None,
            kind,
        }
    }
    pub fn for_agent(agent_id: AgentId, kind: AgentEventKind) -> Self {
        Self {
            agent_id: Some(agent_id),
            kind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeSwitchBy {
    User,
    Model,
    Profile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultSummary {
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_content: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEventKind {
    TurnStarted {
        turn: TurnId,
    },
    TextDelta {
        turn: TurnId,
        block: usize,
        text: String,
    },
    ReasoningDelta {
        turn: TurnId,
        block: usize,
        text: String,
    },
    ToolCallStarted {
        id: ToolUseId,
        name: String,
        input: Value,
        is_concurrency_safe: bool,
    },
    ToolProgress {
        id: ToolUseId,
        data: Value,
    },
    ToolCallFinished {
        id: ToolUseId,
        result: ToolResultSummary,
    },
    PermissionDecided {
        ask: AskId,
        id: ToolUseId,
        decision: PermissionDecision,
    },
    CompactionApplied {
        comp: CompId,
        trigger: CompactTrigger,
        dropped: usize,
    },
    ModeChanged {
        from: ModeName,
        to: ModeName,
        by: ModeSwitchBy,
    },
    ModelChanged {
        from: String,
        to: String,
        reason: String,
    },
    CapabilityDegraded {
        name: String,
        reason: String,
    },
    QueueChanged {
        len: usize,
        next: Option<Priority>,
    },
    TurnFinished {
        turn: TurnId,
        stop_reason: StopReason,
        usage: Usage,
    },
    CommandOutput {
        command: String,
        text: String,
    },
    ScheduledFire {
        id: String,
        wakeup: bool,
    },
    Error(TurnError),
}
