//! Typed native request options owned by openai.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum OpenAiHostedTool {
    ToolSearch(OpenAiToolSearchConfig),
    CodeInterpreter(CodeInterpreterConfig),
    RemoteMcp(RemoteMcpConfig),
}
impl<'de> Deserialize<'de> for OpenAiHostedTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let scanner = crate::response_json::ResponseJsonScanner::parse(
            raw.get().as_bytes(),
            "OpenAI hosted tool is not valid JSON",
        )
        .map_err(serde::de::Error::custom)?;
        let kind = scanner
            .string_at("/type")
            .ok_or_else(|| serde::de::Error::custom("OpenAI hosted tool has no string type"))?;
        let config = scanner
            .raw_source_at("/config")
            .ok_or_else(|| serde::de::Error::custom("OpenAI hosted tool has no config"))?;
        match kind {
            "tool_search" => serde_json::from_str(config)
                .map(Self::ToolSearch)
                .map_err(serde::de::Error::custom),
            "code_interpreter" => serde_json::from_str(config)
                .map(Self::CodeInterpreter)
                .map_err(serde::de::Error::custom),
            "remote_mcp" => serde_json::from_str(config)
                .map(Self::RemoteMcp)
                .map_err(serde::de::Error::custom),
            _ => Err(serde::de::Error::custom(
                "OpenAI hosted tool has an unknown type",
            )),
        }
    }
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
