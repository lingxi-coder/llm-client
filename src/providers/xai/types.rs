//! Provider-specific request, tool, and replay types.

use crate::protocol::LlmError;
use serde::{Deserialize, Serialize};

/// Public, non-secret connection settings for an xAI Responses remote MCP
/// server. xAI's documented Responses contract does not support
/// `require_approval` or `connector_id`; neither is represented here.
/// Authorization is supplied separately for each request through
/// `RequestOptions::mcp_authorizations`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XaiRemoteMcpConfig {
    server_label: String,
    server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    server_description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    allowed_tools: Vec<String>,
}

impl XaiRemoteMcpConfig {
    pub fn new(
        server_label: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let config = Self {
            server_label: server_label.into(),
            server_url: server_url.into(),
            server_description: None,
            allowed_tools: Vec::new(),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.server_description = Some(description.into());
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

    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    pub fn server_description(&self) -> Option<&str> {
        self.server_description.as_deref()
    }

    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
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
                && url.query().is_none()
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
                message: "xAI remote MCP label, HTTPS server URL, description, or allowed tool names are invalid".into(),
            });
        }
        Ok(())
    }
}
