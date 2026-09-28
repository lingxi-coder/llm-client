//! Typed native request options owned by google.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, NativeExtension, NativeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum GoogleHostedTool {
    CodeExecution,
    UrlContext,
    MapsGrounding(GeminiMapsGroundingConfig),
}
impl NativeType for GoogleHostedTool {
    const FORMAT: &'static str = "google.hosted_tool.v1";
}
impl From<GoogleHostedTool> for HostedTool {
    fn from(value: GoogleHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_google_maps_grounding(&self) -> Option<&GeminiMapsGroundingConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<GoogleHostedTool>() {
                Some(GoogleHostedTool::MapsGrounding(config)) => Some(config),
                _ => None,
            })
    }
}
