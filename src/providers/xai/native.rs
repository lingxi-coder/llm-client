//! Typed native request options owned by xai.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum XaiHostedTool {
    RemoteMcp(XaiRemoteMcpConfig),
}
impl NativeType for XaiHostedTool {
    const FORMAT: &'static str = "xai.hosted_tool.v1";
}
impl From<XaiHostedTool> for HostedTool {
    fn from(value: XaiHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn xai_remote_mcp_servers(&self) -> impl Iterator<Item = &XaiRemoteMcpConfig> {
        self.hosted_tools
            .iter()
            .filter_map(|tool| match tool.native::<XaiHostedTool>() {
                Some(XaiHostedTool::RemoteMcp(config)) => Some(config),
                _ => None,
            })
    }
}
