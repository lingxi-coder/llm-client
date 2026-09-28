//! Typed native request options owned by openrouter.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum OpenRouterHostedTool {
    ToolSearch(OpenRouterToolSearchConfig),
    Shell(OpenRouterShellConfig),
}
impl NativeType for OpenRouterHostedTool {
    const FORMAT: &'static str = "openrouter.hosted_tool.v1";
}
impl From<OpenRouterHostedTool> for HostedTool {
    fn from(value: OpenRouterHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_openrouter_tool_search(&self) -> Option<&OpenRouterToolSearchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<OpenRouterHostedTool>() {
                Some(OpenRouterHostedTool::ToolSearch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn hosted_openrouter_shell(&self) -> Option<&OpenRouterShellConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<OpenRouterHostedTool>() {
                Some(OpenRouterHostedTool::Shell(config)) => Some(config),
                _ => None,
            })
    }
}

impl NativeType for OpenRouterContainerMetadata {
    const FORMAT: &'static str = "openrouter.container_metadata.v1";
}
impl crate::protocol::ChatResponse {
    pub fn openrouter_container(&self) -> Option<&OpenRouterContainerMetadata> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<OpenRouterContainerMetadata>())
            .and_then(|extension| extension.decode::<OpenRouterContainerMetadata>().ok())
    }
    pub fn set_openrouter_container(&mut self, container: Option<OpenRouterContainerMetadata>) {
        self.native_metadata
            .retain(|extension| !extension.is::<OpenRouterContainerMetadata>());
        if let Some(container) = container {
            self.native_metadata
                .push(NativeExtension::from_typed(container).expect("container metadata is JSON"));
        }
    }
}
