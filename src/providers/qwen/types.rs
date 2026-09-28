//! Provider-specific request, tool, and replay types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchConfig {
    /// One Qwen Model Studio knowledge base ID. The service currently accepts
    /// a single ID per request.
    pub knowledge_base_id: String,
    /// Model Studio workspace used to construct the regional dedicated API
    /// host required by the knowledge retrieval endpoint.
    pub workspace_id: String,
}
