//! Typed native request options owned by anthropic.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum AnthropicHostedTool {
    ToolSearch(AnthropicToolSearchConfig),
    WebFetch(AnthropicWebFetchConfig),
    CodeExecution(AnthropicCodeExecutionConfig),
    Mcp(AnthropicMcpConfig),
}
impl NativeType for AnthropicHostedTool {
    const FORMAT: &'static str = "anthropic.hosted_tool.v1";
}
impl From<AnthropicHostedTool> for HostedTool {
    fn from(value: AnthropicHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_anthropic_tool_search(&self) -> Option<&AnthropicToolSearchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::ToolSearch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_anthropic_tool_search(&mut self, config: Option<AnthropicToolSearchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<AnthropicHostedTool>(),
                Some(AnthropicHostedTool::ToolSearch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(AnthropicHostedTool::ToolSearch(config).into());
        }
    }
    pub fn hosted_anthropic_web_fetch(&self) -> Option<&AnthropicWebFetchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::WebFetch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_anthropic_web_fetch(&mut self, config: Option<AnthropicWebFetchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<AnthropicHostedTool>(),
                Some(AnthropicHostedTool::WebFetch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(AnthropicHostedTool::WebFetch(config).into());
        }
    }
    pub fn hosted_anthropic_code_execution(&self) -> Option<&AnthropicCodeExecutionConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::CodeExecution(config)) => Some(config),
                _ => None,
            })
    }
    pub fn anthropic_mcp_servers(&self) -> impl Iterator<Item = &AnthropicMcpConfig> {
        self.hosted_tools
            .iter()
            .filter_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::Mcp(config)) => Some(config),
                _ => None,
            })
    }
}

/// Anthropic message-level policy, stored in a format-tagged extension.
impl NativeType for AnthropicMessageOptions {
    const FORMAT: &'static str = "anthropic.message_options.v1";
}

/// Anthropic request options beyond provider-executed tools.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicRequestOptions {
    pub client_toolsets: Vec<AnthropicClientToolset>,
}
impl NativeType for AnthropicRequestOptions {
    const FORMAT: &'static str = "anthropic.request_options.v1";
}
impl ChatRequest {
    pub fn anthropic_client_toolsets(&self) -> &[AnthropicClientToolset] {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicRequestOptions>())
            .and_then(|extension| extension.decode::<AnthropicRequestOptions>().ok())
            .map_or(&[], |options| options.client_toolsets.as_slice())
    }
    pub fn set_anthropic_client_toolsets(&mut self, client_toolsets: Vec<AnthropicClientToolset>) {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicRequestOptions>());
        if !client_toolsets.is_empty() {
            self.native_options.push(
                NativeExtension::from_typed(AnthropicRequestOptions { client_toolsets })
                    .expect("toolset JSON is serializable"),
            );
        }
    }
}
impl crate::protocol::ConversationMessage {
    pub fn anthropic_options(&self) -> Option<&AnthropicMessageOptions> {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicMessageOptions>())
            .and_then(|extension| extension.decode::<AnthropicMessageOptions>().ok())
    }
    pub fn with_anthropic_options(mut self, options: AnthropicMessageOptions) -> Self {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicMessageOptions>());
        self.native_options.push(
            NativeExtension::from_typed(options).expect("message options JSON is serializable"),
        );
        self
    }
}

/// Provider-owned policy for a caller-executed function declaration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicToolOptions {
    pub allowed_callers: Vec<AnthropicToolCaller>,
}
impl NativeType for AnthropicToolOptions {
    const FORMAT: &'static str = "anthropic.tool_options.v1";
}
impl crate::protocol::ToolSpec {
    pub fn anthropic_allowed_callers(&self) -> &[AnthropicToolCaller] {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicToolOptions>())
            .and_then(|extension| extension.decode::<AnthropicToolOptions>().ok())
            .map_or(&[], |options| options.allowed_callers.as_slice())
    }
    pub fn set_anthropic_allowed_callers(&mut self, allowed_callers: Vec<AnthropicToolCaller>) {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicToolOptions>());
        if !allowed_callers.is_empty() {
            self.native_options.push(
                NativeExtension::from_typed(AnthropicToolOptions { allowed_callers })
                    .expect("caller enum is serializable"),
            );
        }
    }
    pub fn with_anthropic_allowed_callers(
        mut self,
        allowed_callers: Vec<AnthropicToolCaller>,
    ) -> Self {
        self.set_anthropic_allowed_callers(allowed_callers);
        self
    }
}

/// Unmodified native container and usage observations from Messages.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicResponseMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    /// Provider stop diagnostics, preserved without interpreting their schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<serde_json::Value>,
}
impl NativeType for AnthropicResponseMetadata {
    const FORMAT: &'static str = "anthropic.response_metadata.v1";
}
impl crate::protocol::ChatResponse {
    pub fn anthropic_container(&self) -> Option<&AnthropicContainerMetadata> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.container.as_ref())
    }
    pub fn anthropic_usage(&self) -> Option<&serde_json::Value> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.usage.as_ref())
    }
    pub fn anthropic_stop_details(&self) -> Option<&serde_json::Value> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.stop_details.as_ref())
    }
    pub fn set_anthropic_stop_details(&mut self, stop_details: Option<serde_json::Value>) {
        if let Some(extension) = self
            .native_metadata
            .iter_mut()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
        {
            extension
                .edit::<AnthropicResponseMetadata, _>(|metadata| {
                    metadata.stop_details = stop_details
                })
                .expect("native response metadata is JSON");
        } else if stop_details.is_some() {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    stop_details,
                    ..Default::default()
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
    pub fn set_anthropic_metadata(
        &mut self,
        container: Option<AnthropicContainerMetadata>,
        usage: Option<serde_json::Value>,
    ) {
        let stop_details = self.anthropic_stop_details().cloned();
        self.native_metadata
            .retain(|extension| !extension.is::<AnthropicResponseMetadata>());
        if container.is_some() || usage.is_some() || stop_details.is_some() {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    container,
                    usage,
                    stop_details,
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
}
