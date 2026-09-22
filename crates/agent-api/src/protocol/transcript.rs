//! Pure transcript, session metadata, and rewind snapshot records.

use crate::protocol::{
    ConversationMessage, HookEventType, HookResponse, ModeName, SessionId, ToolUseId, TurnId,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUseSummary {
    pub id: ToolUseId,
    pub name: String,
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_content: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptEntry {
    Message(ConversationMessage),
    ToolUseSummary(ToolUseSummary),
    CompactBoundary {
        summary: String,
        dropped: usize,
        trigger: crate::protocol::CompactTrigger,
    },
    Tombstone {
        entry: usize,
    },
    HookResult {
        event: HookEventType,
        response: Box<HookResponse>,
    },
    ModeChanged {
        from: ModeName,
        to: ModeName,
    },
    TurnStarted {
        turn: TurnId,
    },
    Snapshot(SnapshotRecord),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<SessionId>,
    pub created_at: SystemTime,
    pub project_dir: PathBuf,
    pub cwd: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
    pub model: String,
    pub mode: ModeName,
    #[serde(default)]
    pub enabled_plugins: Vec<String>,
    #[serde(default)]
    pub mcp_servers_enabled: Vec<String>,
    #[serde(default)]
    pub working_directories: Vec<PathBuf>,
    #[serde(default)]
    pub agents_md_paths: Vec<PathBuf>,
    pub last_modified: SystemTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Backup {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backup_file_name: Option<String>,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRecord {
    pub turn: TurnId,
    pub files: BTreeMap<String, Backup>,
}
