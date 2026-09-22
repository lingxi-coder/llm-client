//! Serializable tool outcome facts and execution policy data.

use crate::protocol::{ConversationMessage, ModeName};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptBehavior {
    Cancel,
    Block,
}

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolError {
    #[error("tool not found: {name}")]
    NotFound { name: String },
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
    #[error("permission denied: {message}")]
    PermissionDenied { message: String },
    #[error("io: {message}")]
    Io { message: String },
    #[error("aborted")]
    Aborted,
    #[error("internal: {message}")]
    Internal { message: String },
    #[error("interaction required: {message}")]
    InteractionRequired { message: String },
    #[error("{path} was not read before editing")]
    EditWithoutRead { path: PathBuf },
    #[error("{path} was modified externally since it was read")]
    FileModifiedExternally { path: PathBuf },
    #[error("{path} was only partially read; read it again before editing")]
    PartialViewMustReread { path: PathBuf },
    #[error("{path} does not contain the expected text")]
    FileContentMismatch { path: PathBuf },
    #[error("{path} is too large ({size} > {limit} bytes)")]
    FileTooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    #[error("{path} is blocked")]
    PathBlocked { path: PathBuf },
    #[error("{path} is a binary file")]
    BinaryFile { path: PathBuf },
    #[error("timed out after {after_ms} ms")]
    Timeout { after_ms: u64 },
    #[error("output truncated at {limit} chars")]
    OutputTruncated { limit: usize },
    #[error("subagent failed: {message}")]
    SubagentFailed { message: String },
    #[error("MCP failure: {message}")]
    McpFailure { message: String },
    #[error("LSP failure: {message}")]
    LspFailure { message: String },
    #[error("transport: {message}")]
    Transport { message: String },
}

/// A context update after runtime authorization and revision checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidatedContextUpdate {
    WorkingDirectory(PathBuf),
    Worktree(PathBuf),
    Mode {
        mode: ModeName,
        permission_mode: crate::protocol::PermissionMode,
    },
    TightenPermissions(crate::protocol::PermissionMode),
}

/// Runtime-validated, serializable result facts delivered to the kernel/task journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultData {
    pub data: Value,
    pub model_content: Option<String>,
    pub new_messages: Vec<ConversationMessage>,
    pub context_update: Option<ValidatedContextUpdate>,
    pub mcp_meta: Option<Value>,
    pub is_error: bool,
}

impl ToolResultData {
    pub fn error(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            data: Value::String(message.clone()),
            model_content: Some(message),
            new_messages: Vec::new(),
            context_update: None,
            mcp_meta: None,
            is_error: true,
        }
    }
}
