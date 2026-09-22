//! Pure hook event and response data.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEventType {
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    SessionStart,
    SessionEnd,
    Setup,
    UserPromptSubmit,
    Stop,
    StopFailure,
    AgentSpawn,
    SubagentStart,
    SubagentStop,
    PreCompact,
    PostCompact,
    PreModelSwitch,
    PostModelSwitch,
    PermissionRequest,
    PermissionDenied,
    TeammateIdle,
    TaskCreated,
    TaskCompleted,
    Elicitation,
    ElicitationResult,
    ConfigChange,
    WorktreeCreate,
    WorktreeRemove,
    InstructionsLoaded,
    CwdChanged,
    FileChanged,
    Notification,
    PostToolBatch,
    UserPromptExpansion,
    MessageDisplay,
    DirectoryAdded,
    ModeChanged,
}

impl HookEventType {
    pub const ALL: [Self; 35] = [
        Self::PreToolUse,
        Self::PostToolUse,
        Self::PostToolUseFailure,
        Self::SessionStart,
        Self::SessionEnd,
        Self::Setup,
        Self::UserPromptSubmit,
        Self::Stop,
        Self::StopFailure,
        Self::AgentSpawn,
        Self::SubagentStart,
        Self::SubagentStop,
        Self::PreCompact,
        Self::PostCompact,
        Self::PreModelSwitch,
        Self::PostModelSwitch,
        Self::PermissionRequest,
        Self::PermissionDenied,
        Self::TeammateIdle,
        Self::TaskCreated,
        Self::TaskCompleted,
        Self::Elicitation,
        Self::ElicitationResult,
        Self::ConfigChange,
        Self::WorktreeCreate,
        Self::WorktreeRemove,
        Self::InstructionsLoaded,
        Self::CwdChanged,
        Self::FileChanged,
        Self::Notification,
        Self::PostToolBatch,
        Self::UserPromptExpansion,
        Self::MessageDisplay,
        Self::DirectoryAdded,
        Self::ModeChanged,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::SessionStart => "SessionStart",
            Self::SessionEnd => "SessionEnd",
            Self::Setup => "Setup",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::Stop => "Stop",
            Self::StopFailure => "StopFailure",
            Self::AgentSpawn => "AgentSpawn",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
            Self::PreCompact => "PreCompact",
            Self::PostCompact => "PostCompact",
            Self::PreModelSwitch => "PreModelSwitch",
            Self::PostModelSwitch => "PostModelSwitch",
            Self::PermissionRequest => "PermissionRequest",
            Self::PermissionDenied => "PermissionDenied",
            Self::TeammateIdle => "TeammateIdle",
            Self::TaskCreated => "TaskCreated",
            Self::TaskCompleted => "TaskCompleted",
            Self::Elicitation => "Elicitation",
            Self::ElicitationResult => "ElicitationResult",
            Self::ConfigChange => "ConfigChange",
            Self::WorktreeCreate => "WorktreeCreate",
            Self::WorktreeRemove => "WorktreeRemove",
            Self::InstructionsLoaded => "InstructionsLoaded",
            Self::CwdChanged => "CwdChanged",
            Self::FileChanged => "FileChanged",
            Self::Notification => "Notification",
            Self::PostToolBatch => "PostToolBatch",
            Self::UserPromptExpansion => "UserPromptExpansion",
            Self::MessageDisplay => "MessageDisplay",
            Self::DirectoryAdded => "DirectoryAdded",
            Self::ModeChanged => "ModeChanged",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|event| event.as_str() == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookDecision {
    Allow,
    Approve,
    Block,
    Continue,
    Ask,
    Defer,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HookResponse {
    pub impossible: bool,
    pub decision: Option<HookDecision>,
    pub reason: Option<String>,
    pub block_command: Option<String>,
    pub updated_input: Option<Value>,
    pub updated_permissions: Option<Vec<Value>>,
    pub interrupt: Option<bool>,
    pub permission_request_result: Option<Value>,
    pub system_message: Option<String>,
    pub additional_context: Option<String>,
    pub attachments: Vec<Value>,
    pub suppress_output: bool,
    pub prevent_continuation: bool,
    pub structured_content: Option<Value>,
    pub updated_mcp_tool_output: Option<Value>,
    pub updated_tool_output: Option<Option<Value>>,
    pub classifier_context: Option<String>,
    pub elicitation_response: Option<Value>,
    pub retry: Option<bool>,
    pub terminal_sequence: Option<String>,
    pub session_title: Option<String>,
    pub suppress_original_prompt: bool,
    pub display_content: Option<String>,
    pub watch_paths: Option<Vec<String>>,
    pub initial_user_message: Option<String>,
    pub reload_skills: Option<bool>,
    pub async_rewake: bool,
    pub async_backgrounded: bool,
}
