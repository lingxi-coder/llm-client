//! Provider-specific request, tool, and replay types.

use crate::protocol::{
    ContentBlock, ConversationMessage, LlmError, ProtocolFamily, ProviderFileSource,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn is_false(value: &bool) -> bool {
    !value
}

/// Whether OpenAI must pause for host approval before invoking remote MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpApprovalPolicy {
    Always,
    Never,
}

/// Public, non-secret connection settings for an OpenAI Responses remote MCP server.
///
/// OAuth authorization is deliberately not part of this serializable type.
/// A caller injects a fresh token through request-scoped credentials in the
/// client layer; it must be included on every Responses creation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteMcpConfig {
    server_label: String,
    server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    server_description: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    defer_loading: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    allowed_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    require_approval: Option<McpApprovalPolicy>,
}

impl RemoteMcpConfig {
    pub fn new(
        server_label: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let config = Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            server_description: None,
            defer_loading: false,
            allowed_tools: Vec::new(),
            require_approval: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.server_description = Some(description.into());
        self
    }

    /// Ask OpenAI Responses tool search to defer discovery of this MCP server.
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }

    pub fn with_allowed_tools(
        mut self,
        allowed_tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, LlmError> {
        self.allowed_tools = allowed_tools.into_iter().map(Into::into).collect();
        self.validate()?;
        Ok(self)
    }

    pub fn with_require_approval(mut self, policy: McpApprovalPolicy) -> Self {
        self.require_approval = Some(policy);
        self
    }

    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    pub fn server_description(&self) -> Option<&str> {
        self.server_description.as_deref()
    }

    pub fn defer_loading(&self) -> bool {
        self.defer_loading
    }

    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    pub fn require_approval(&self) -> Option<McpApprovalPolicy> {
        self.require_approval
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let valid_name =
            |name: &str| !name.trim().is_empty() && !name.chars().any(char::is_control);
        let url = url::Url::parse(&self.server_url).ok();
        let valid_url = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none()
        });
        let mut names = std::collections::BTreeSet::new();
        if !valid_name(&self.server_label)
            || !valid_url
            || self
                .server_description
                .as_deref()
                .is_some_and(|value| value.chars().any(char::is_control))
            || self
                .allowed_tools
                .iter()
                .any(|name| !valid_name(name) || !names.insert(name.as_str()))
        {
            return Err(LlmError::InvalidRequest {
                message: "remote MCP label, HTTPS server URL, description, or allowed tool names are invalid".into(),
            });
        }
        Ok(())
    }
}

/// An MCP tool invocation awaiting a host-owned approval decision.
///
/// `arguments` is retained as the provider's original JSON string. The host
/// decides whether to approve; this type does not execute or authorize tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpApprovalRequest {
    pub approval_request_id: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
}

impl McpApprovalRequest {
    /// Parse a native Responses output item retained by the codec.
    pub fn from_native_item(item: &Value) -> Option<Self> {
        (item.get("type").and_then(Value::as_str) == Some("mcp_approval_request")).then_some(())?;
        let nonempty = |key: &str| {
            item.get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty() && !value.chars().any(char::is_control))
                .map(str::to_owned)
        };
        Some(Self {
            approval_request_id: nonempty("id")?,
            server_label: nonempty("server_label")?,
            name: nonempty("name")?,
            arguments: item.get("arguments")?.as_str()?.to_owned(),
        })
    }

    pub fn respond(self, approve: bool) -> McpApprovalResponse {
        McpApprovalResponse {
            approval_request_id: self.approval_request_id,
            approve,
        }
    }
}

/// Caller-supplied answer to an OpenAI Responses MCP approval request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpApprovalResponse {
    pub approval_request_id: String,
    pub approve: bool,
}

impl McpApprovalResponse {
    /// Build a native Responses input item inside an assistant transcript
    /// message. The Responses codec replays it as a sibling input item.
    pub fn into_assistant_message(self) -> ConversationMessage {
        ConversationMessage::assistant(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: serde_json::json!({
                "type": "mcp_approval_response",
                "approval_request_id": self.approval_request_id,
                "approve": self.approve
            }),
        }])
    }
}

/// Execution mode for OpenAI Responses tool search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiToolSearchExecution {
    /// OpenAI searches the deferred function and MCP definitions in this request.
    #[default]
    Server,
    /// The caller performs discovery and returns a native `tool_search_output` item.
    Client,
}

/// OpenAI Responses tool search configuration.
///
/// Server execution needs only the `tool_search` tool marker. Client execution
/// also requires a description and JSON Schema for the search call arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiToolSearchConfig {
    #[serde(skip_serializing_if = "is_server_tool_search")]
    pub execution: OpenAiToolSearchExecution,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

fn is_server_tool_search(value: &OpenAiToolSearchExecution) -> bool {
    *value == OpenAiToolSearchExecution::Server
}

impl OpenAiToolSearchConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        match self.execution {
            OpenAiToolSearchExecution::Server
                if self.description.is_some() || self.parameters.is_some() =>
            {
                Err(LlmError::InvalidRequest {
                    message: "OpenAI server tool search does not accept client search fields".into(),
                })
            }
            OpenAiToolSearchExecution::Client
                if self
                    .description
                    .as_deref()
                    .is_none_or(|description| description.trim().is_empty())
                    || !self.parameters.as_ref().is_some_and(Value::is_object) =>
            {
                Err(LlmError::InvalidRequest {
                    message: "OpenAI client tool search requires a description and object parameters schema".into(),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Code Interpreter settings for OpenAI Responses.
///
/// An absent `container` uses OpenAI's automatic container. In automatic mode,
/// `files` are scope-bound Files API references mounted through the documented
/// `file_ids` property. An explicit `container` reuses an existing scoped
/// OpenAI container; its memory tier and files are managed by the Containers
/// service rather than configured on the Responses request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeInterpreterConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<CodeInterpreterMemoryLimit>,
    /// A scope-bound existing OpenAI container. `None` selects automatic mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::providers::openai::containers::OpenAiContainerRef>,
    /// Existing OpenAI Files API references to mount into an automatic
    /// container. They are encoded as `container.file_ids` after scope,
    /// readiness, and expiry checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ProviderFileSource>,
}

impl CodeInterpreterConfig {
    /// Reuse an existing container returned by the OpenAI Containers service
    /// or explicitly imported with its account scope.
    #[must_use]
    pub fn with_container(
        mut self,
        container: crate::providers::openai::containers::OpenAiContainerRef,
    ) -> Self {
        self.container = Some(container);
        self
    }

    /// Mount existing Files API references when using an automatic container.
    #[must_use]
    pub fn with_files(mut self, files: impl IntoIterator<Item = ProviderFileSource>) -> Self {
        self.files.extend(files);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeInterpreterMemoryLimit {
    #[serde(rename = "1g")]
    OneG,
    #[serde(rename = "4g")]
    FourG,
    #[serde(rename = "16g")]
    SixteenG,
    #[serde(rename = "64g")]
    SixtyFourG,
}
