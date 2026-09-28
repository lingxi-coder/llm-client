//! Typed native request options owned by qwen.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum QwenHostedTool {
    WebExtractor,
    FileSearch(FileSearchConfig),
}
impl NativeType for QwenHostedTool {
    const FORMAT: &'static str = "qwen.hosted_tool.v1";
}
impl From<QwenHostedTool> for HostedTool {
    fn from(value: QwenHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_file_search(&self) -> Option<&FileSearchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<QwenHostedTool>() {
                Some(QwenHostedTool::FileSearch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_file_search(&mut self, config: Option<FileSearchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<QwenHostedTool>(),
                Some(QwenHostedTool::FileSearch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(QwenHostedTool::FileSearch(config).into());
        }
    }
    pub fn has_hosted_web_extractor(&self) -> bool {
        self.hosted_tools.iter().any(|tool| {
            matches!(
                tool.native::<QwenHostedTool>(),
                Some(QwenHostedTool::WebExtractor)
            )
        })
    }
    pub fn set_hosted_web_extractor(&mut self, enabled: bool) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<QwenHostedTool>(),
                Some(QwenHostedTool::WebExtractor)
            )
        });
        if enabled {
            self.hosted_tools.push(QwenHostedTool::WebExtractor.into());
        }
    }
}
