//! Typed source filters for Anthropic Messages Web Fetch.
//!
//! These values are provider policy: this crate validates references to names
//! that the request actually declares, then sends the filters unchanged. It
//! does not infer URLs from tool arguments or fetch any URLs itself.

use crate::protocol::{ChatRequest, HostedTool, LlmError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

/// Which URL sources Web Fetch may use for the current request.
///
/// The optional fields preserve Anthropic's defaults when omitted. `None` is
/// therefore omitted during serialization instead of being rewritten to an
/// explicit `all` or `none` policy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicFetchUrlSources {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_input: Option<AnthropicFetchUserInputSources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_tool_results: Option<AnthropicFetchToolResultsSources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_results: Option<AnthropicFetchToolResultsSources>,
}

/// Whether URLs written directly in user input are eligible for Web Fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnthropicFetchUserInputSources {
    All,
    None,
}

/// Which declared tools' results can contribute URLs to Web Fetch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnthropicFetchToolResultsSources {
    All,
    None,
    Only {
        tools: Vec<AnthropicFetchToolReference>,
    },
    Except {
        tools: Vec<AnthropicFetchToolReference>,
    },
}

/// A tool name reference in a Web Fetch URL source filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnthropicFetchToolReference {
    ToolReference { name: String },
}

impl AnthropicFetchToolReference {
    pub fn new(name: impl Into<String>) -> Self {
        Self::ToolReference { name: name.into() }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::ToolReference { name } => name,
        }
    }
}

impl AnthropicFetchUrlSources {
    /// Check each reference against a name actually emitted in Anthropic's
    /// top-level `tools` array for this request. In particular, MCP toolsets
    /// have no top-level tool `name`, so this does not invent a server/tool
    /// naming convention for them.
    pub(crate) fn validate(&self, request: &ChatRequest) -> Result<(), LlmError> {
        let declared = declared_tool_names(request);
        for filter in [
            self.client_tool_results.as_ref(),
            self.server_tool_results.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let references = match filter {
                AnthropicFetchToolResultsSources::Only { tools }
                | AnthropicFetchToolResultsSources::Except { tools } => tools,
                AnthropicFetchToolResultsSources::All | AnthropicFetchToolResultsSources::None => {
                    continue
                }
            };
            for reference in references {
                if !declared.contains(reference.name()) {
                    return Err(LlmError::InvalidRequest {
                        message: format!(
                            "Anthropic Web Fetch url_sources references tool {:?}, which is not declared in this request's tools array",
                            reference.name()
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Serialize to the wire shape after request-local name validation.
    pub(crate) fn to_value(&self) -> Result<Value, LlmError> {
        serde_json::to_value(self).map_err(|error| LlmError::InvalidRequest {
            message: format!("could not serialize Anthropic Web Fetch url_sources: {error}"),
        })
    }
}

fn declared_tool_names(request: &ChatRequest) -> BTreeSet<String> {
    let mut names = request
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();

    // Keep these names in sync with the first-party Anthropic codec's
    // `tools[]` encoders. Other hosted tools are rejected or belong to another
    // provider and therefore are not declarations on this route.
    for tool in &request.hosted_tools {
        match tool {
            HostedTool::WebSearch(_) => {
                names.insert("web_search".into());
            }
            HostedTool::AnthropicWebFetch(_) => {
                names.insert("web_fetch".into());
            }
            HostedTool::AnthropicToolSearch(config) => {
                names.insert(
                    match config.strategy {
                        crate::protocol::AnthropicToolSearchStrategy::Regex => {
                            "tool_search_tool_regex"
                        }
                        crate::protocol::AnthropicToolSearchStrategy::Bm25 => {
                            "tool_search_tool_bm25"
                        }
                    }
                    .into(),
                );
            }
            HostedTool::AnthropicCodeExecution(_) => {
                names.insert("code_execution".into());
            }
            HostedTool::WebExtractor
            | HostedTool::FileSearch(_)
            | HostedTool::CodeInterpreter(_)
            | HostedTool::OpenRouterToolSearch(_)
            | HostedTool::OpenAiToolSearch(_)
            | HostedTool::OpenRouterShell(_)
            | HostedTool::GeminiCodeExecution
            | HostedTool::GeminiUrlContext
            | HostedTool::GeminiMapsGrounding(_)
            | HostedTool::AnthropicMcp(_)
            | HostedTool::RemoteMcp(_)
            | HostedTool::XaiRemoteMcp(_) => {}
        }
    }
    names
}
