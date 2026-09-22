//! What the runtime asks the frontend, and what the frontend answers (§7.7 / §7.8).

use crate::protocol::hooks::HookDecision;
use crate::protocol::ids::{AgentId, ToolUseId};
use crate::protocol::scope::Scope;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    #[default]
    Default,
    Plan,
    AcceptEdits,
    BypassPermissions,
    DontAsk,
}

impl PermissionMode {
    pub fn strictness(&self) -> u8 {
        match self {
            Self::Plan => 0,
            Self::Default => 1,
            Self::AcceptEdits => 2,
            Self::DontAsk => 3,
            Self::BypassPermissions => 4,
        }
    }

    pub fn stricter(self, other: Self) -> Self {
        if self.strictness() <= other.strictness() {
            self
        } else {
            other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleBehavior {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    pub behavior: RuleBehavior,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    pub scope: Scope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum DecisionReason {
    Rule { rule: String, scope: Scope },
    Mode { mode: PermissionMode },
    ToolDefault,
    Hook { name: String },
    User,
    Classifier { message: String },
    Sandbox,
    HardDeny { message: String },
    Other { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionPrompt {
    pub message: String,
    #[serde(default)]
    pub suggestions: Vec<PermissionSuggestion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum PermissionResult {
    Allow {
        reason: DecisionReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        updated_input: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        update_destination: Option<Scope>,
    },
    Deny {
        reason: DecisionReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        explanation: Option<String>,
    },
    Ask {
        reason: DecisionReason,
        prompt: PermissionPrompt,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookVerdict {
    pub decision: HookDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_permissions: Option<Vec<Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "adjudication", rename_all = "snake_case")]
pub enum Adjudication {
    Allow,
    Deny { reason: String },
    Ask { protected: bool },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleSetView {
    pub rules: Vec<PermissionRule>,
}

/// A rule the user could accept to avoid being asked again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionSuggestion {
    /// Rule text in the `Tool(content)` grammar, produced by the tool's matcher.
    pub rule: String,
    pub scope: Scope,
}

/// The question put to the frontend. Built by the runtime from the tool's
/// classification of the (hook-rewritten) input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub tool_use_id: ToolUseId,
    pub tool_name: String,
    pub input: Value,
    /// Tool-provided human description of this call.
    pub description: String,
    pub agent_id: Option<AgentId>,
    pub destructive: bool,
    pub open_world: bool,
    /// A protected ask (MCP server cap, `requires_user_interaction`): no hook
    /// may answer it (§7.8 guarantee b).
    pub protected: bool,
    #[serde(default)]
    pub suggestions: Vec<PermissionSuggestion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow {
        /// Remember as a rule at this scope (`Session`, `Local`, `Project`, `User`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remember: Option<Scope>,
        /// Optional final candidate returned by the approving user.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        updated_input: Option<Value>,
    },
    Deny {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remember: Option<Scope>,
    },
    /// The kernel withdrew the question (cancel, timeout) before an answer.
    Withdrawn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ElicitationSource {
    McpServer { name: String },
    Tool { name: String },
    Hook,
}

/// A structured question from a server, tool or hook to the user.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElicitationRequest {
    pub message: String,
    /// JSON schema of the expected answer.
    pub schema: Value,
    pub source: ElicitationSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ElicitationResponse {
    Accept { content: Value },
    Decline,
    Cancel,
}
