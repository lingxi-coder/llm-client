//! Typed native request options owned by openai.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum OpenAiHostedTool {
    ToolSearch(OpenAiToolSearchConfig),
    CodeInterpreter(CodeInterpreterConfig),
    RemoteMcp(RemoteMcpConfig),
}
impl NativeType for OpenAiHostedTool {
    const FORMAT: &'static str = "openai.hosted_tool.v1";
}
impl From<OpenAiHostedTool> for HostedTool {
    fn from(value: OpenAiHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_openai_tool_search(&self) -> Option<&OpenAiToolSearchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<OpenAiHostedTool>() {
                Some(OpenAiHostedTool::ToolSearch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_openai_tool_search(&mut self, config: Option<OpenAiToolSearchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<OpenAiHostedTool>(),
                Some(OpenAiHostedTool::ToolSearch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(OpenAiHostedTool::ToolSearch(config).into());
        }
    }
    pub fn hosted_code_interpreter(&self) -> Option<&CodeInterpreterConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<OpenAiHostedTool>() {
                Some(OpenAiHostedTool::CodeInterpreter(config)) => Some(config),
                _ => None,
            })
    }
    pub fn remote_mcp_servers(&self) -> impl Iterator<Item = &RemoteMcpConfig> {
        self.hosted_tools
            .iter()
            .filter_map(|tool| match tool.native::<OpenAiHostedTool>() {
                Some(OpenAiHostedTool::RemoteMcp(config)) => Some(config),
                _ => None,
            })
    }
}
