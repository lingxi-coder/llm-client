//! Provider-specific request, tool, and replay types.

use crate::protocol::{
    ContentBlock, ConversationMessage, FoundryDeployment, FoundryHosting, LlmError, MessageRole,
    ProtocolFamily, ProviderFileSource, ProviderProfile, ToolSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn is_false(value: &bool) -> bool {
    !value
}

/// Anthropic Messages caller classes for a user-defined tool.
///
/// Other codecs reject non-empty [`ToolSpec::anthropic_allowed_callers`] rather than
/// silently dropping this provider-specific policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnthropicToolCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521,
}

/// Live Anthropic Messages Web Fetch tool variants. Newer variants add
/// provider capabilities; basic fetch remains selectable without dynamic filtering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicWebFetchVersion {
    #[serde(rename = "20250910")]
    V20250910,
    #[serde(rename = "20260209")]
    V20260209,
    #[serde(rename = "20260309")]
    V20260309,
    #[serde(rename = "20260318")]
    #[default]
    V20260318,
}

impl AnthropicWebFetchVersion {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V20250910 => "20250910",
            Self::V20260209 => "20260209",
            Self::V20260309 => "20260309",
            Self::V20260318 => "20260318",
        }
    }

    pub const fn supports_dynamic_filtering(self) -> bool {
        !matches!(self, Self::V20250910)
    }

    pub const fn supports_cache_bypass(self) -> bool {
        matches!(self, Self::V20260309 | Self::V20260318)
    }

    pub const fn supports_response_inclusion(self) -> bool {
        matches!(self, Self::V20260318)
    }
}

/// How Web Fetch results consumed by completed Code Execution calls appear in
/// the response. Direct calls and paused calls are always returned in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicFetchResponseInclusion {
    Full,
    Excluded,
}

/// Callers Anthropic may use to invoke Web Fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicFetchCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521,
}

/// Typed configuration for the first-party Anthropic Messages Web Fetch
/// server tool. The provider performs retrieval; the client never fetches URLs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicWebFetchConfig {
    pub version: AnthropicWebFetchVersion,
    /// Define Web Fetch in a mid-conversation `tool_addition` instead of the
    /// top-level `tools` array. The addition must exactly match this typed
    /// configuration and is supported only on the first-party Messages API.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inline_definition: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<AnthropicFetchCaller>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::protocol::cache::CacheTtl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_sources: Option<crate::providers::anthropic::types::AnthropicFetchUrlSources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_domains: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_cache: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_inclusion: Option<AnthropicFetchResponseInclusion>,
}

impl AnthropicWebFetchConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if !self.allowed_domains.is_empty() && !self.blocked_domains.is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic Web Fetch accepts allowed_domains or blocked_domains, not both"
                    .into(),
            });
        }
        if self.use_cache.is_some() && !self.version.supports_cache_bypass() {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch use_cache requires version 20260309 or later".into(),
            });
        }
        if self.response_inclusion.is_some() && !self.version.supports_response_inclusion() {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch response_inclusion requires version 20260318".into(),
            });
        }
        if self.cache_control == Some(crate::protocol::cache::CacheTtl::ThirtyMinutes) {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Web Fetch cache_control supports only 5m or 1h".into(),
            });
        }
        if self.defer_loading && self.cache_control.is_some() {
            return Err(LlmError::InvalidRequest {
                message: "deferred Anthropic Web Fetch tools cannot have cache_control".into(),
            });
        }
        for domain in self.allowed_domains.iter().chain(&self.blocked_domains) {
            if domain.trim().is_empty()
                || domain.contains("://")
                || domain.chars().any(char::is_whitespace)
                || domain.chars().any(char::is_control)
                || match domain.split_once('/') {
                    Some((host, _)) => host.is_empty() || host.contains('*'),
                    None => domain.contains('*'),
                }
            {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Web Fetch domain filters must omit schemes and whitespace; wildcards are allowed only in paths".into(),
                });
            }
        }
        Ok(())
    }
}

/// Select Anthropic's hosted catalog search implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicToolSearchStrategy {
    /// Search tool names and descriptions with Python-compatible regular expressions.
    Regex,
    /// Search the tool catalog with natural-language BM25 queries.
    Bm25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicToolSearchConfig {
    pub strategy: AnthropicToolSearchStrategy,
}

/// Per-tool MCP settings accepted by Anthropic's `mcp_toolset`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicMcpToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

/// A caller-pinned MCP tool definition accepted by the
/// `mcp-client-2026-09-15` beta. The schema is retained as provider JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Prompt-cache lifetime accepted on an Anthropic MCP toolset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicMcpCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Prompt-cache breakpoint placed on the final expanded tool in an MCP set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpCacheControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<AnthropicMcpCacheTtl>,
}

/// Non-secret definition of one Anthropic remote MCP server and
/// its one-to-one `mcp_toolset`. A pinned `tools: Some(vec![])` is distinct
/// from `None`: the empty list explicitly pins a server with no tools.
/// Foundry uses the 2025 connector and rejects pinned lists and inline toolsets;
/// the first-party Claude API supports the 2026 listing/pinning extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMcpConfig {
    name: String,
    url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_config: Option<AnthropicMcpToolConfig>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    configs: std::collections::BTreeMap<String, AnthropicMcpToolConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicMcpTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicMcpCacheControl>,
    /// Keep this server in `mcp_servers`, but add its toolset at a later
    /// `tool_addition` position inside a system message.
    #[serde(default, skip_serializing_if = "is_false")]
    inline_toolset: bool,
}

impl AnthropicMcpConfig {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Result<Self, LlmError> {
        let config = Self {
            name: name.into(),
            url: url.into(),
            default_config: None,
            configs: std::collections::BTreeMap::new(),
            tools: None,
            cache_control: None,
            inline_toolset: false,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_default_config(mut self, config: AnthropicMcpToolConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    /// Set or replace one per-tool override. Unknown tool names are retained;
    /// Anthropic documents that the server may expose dynamic tool names.
    pub fn with_tool_config(
        mut self,
        name: impl Into<String>,
        config: AnthropicMcpToolConfig,
    ) -> Result<Self, LlmError> {
        self.configs.insert(name.into(), config);
        self.validate()?;
        Ok(self)
    }

    /// Pin the server's exact tool list. Passing an empty vector deliberately
    /// pins an empty toolset; omit this method to let Anthropic fetch tools.
    pub fn with_tools(
        mut self,
        tools: impl IntoIterator<Item = AnthropicMcpTool>,
    ) -> Result<Self, LlmError> {
        self.tools = Some(tools.into_iter().collect());
        self.validate()?;
        Ok(self)
    }

    pub fn with_cache_control(mut self, cache_control: AnthropicMcpCacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Place this server's `mcp_toolset` at its `inline_tool_addition` point in
    /// message history. The connection remains in top-level `mcp_servers` so
    /// its URL and request-scoped credential are available to Anthropic.
    pub fn with_inline_toolset(mut self, enabled: bool) -> Self {
        self.inline_toolset = enabled;
        self
    }

    pub fn inline_toolset(&self) -> bool {
        self.inline_toolset
    }

    /// Build the native Anthropic `tool_addition` block for this MCP toolset.
    /// Add it to a `MessageRole::System` message at the point the server's
    /// tools should become available.
    pub fn inline_tool_addition(&self) -> Result<ContentBlock, LlmError> {
        self.validate()?;
        if !self.inline_toolset {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP inline_tool_addition requires with_inline_toolset(true)"
                    .into(),
            });
        }
        Ok(ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: serde_json::json!({
                "type":"tool_addition",
                "tool":{"type":"tool_definition","definition":self.toolset_value()}
            }),
        })
    }

    pub fn inline_tool_addition_message(&self) -> Result<ConversationMessage, LlmError> {
        Ok(ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::System,
            content: vec![self.inline_tool_addition()?],
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn default_config(&self) -> Option<&AnthropicMcpToolConfig> {
        self.default_config.as_ref()
    }

    pub fn configs(&self) -> &std::collections::BTreeMap<String, AnthropicMcpToolConfig> {
        &self.configs
    }

    pub fn tools(&self) -> Option<&[AnthropicMcpTool]> {
        self.tools.as_deref()
    }

    pub fn cache_control(&self) -> Option<AnthropicMcpCacheControl> {
        self.cache_control
    }

    pub(crate) fn toolset_value(&self) -> Value {
        let mut toolset = serde_json::json!({
            "type":"mcp_toolset",
            "mcp_server_name":self.name,
        });
        if let Some(default_config) = &self.default_config {
            toolset["default_config"] = serde_json::to_value(default_config).unwrap_or(Value::Null);
        }
        if !self.configs.is_empty() {
            toolset["configs"] = serde_json::to_value(&self.configs).unwrap_or(Value::Null);
        }
        if let Some(pinned_tools) = &self.tools {
            toolset["tools"] = serde_json::to_value(pinned_tools).unwrap_or(Value::Null);
        }
        if let Some(cache_control) = self.cache_control {
            let mut value = serde_json::json!({"type":"ephemeral"});
            if let Some(ttl) = cache_control.ttl {
                value["ttl"] = serde_json::Value::String(
                    match ttl {
                        AnthropicMcpCacheTtl::FiveMinutes => "5m",
                        AnthropicMcpCacheTtl::OneHour => "1h",
                    }
                    .into(),
                );
            }
            toolset["cache_control"] = value;
        }
        toolset
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let url = url::Url::parse(&self.url).ok();
        let valid_url = url.as_ref().is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none()
        });
        let valid_name =
            |name: &str| !name.trim().is_empty() && !name.chars().any(char::is_control);
        let pinned_names = self.tools.as_ref().map(|tools| {
            let mut names = std::collections::BTreeSet::new();
            tools.iter().all(|tool| {
                valid_name(&tool.name)
                    && tool.input_schema.is_object()
                    && names.insert(tool.name.as_str())
            })
        });
        if !valid_name(&self.name)
            || !valid_url
            || self.configs.keys().any(|name| !valid_name(name))
            || pinned_names == Some(false)
        {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP server name, HTTPS URL, tool config, or pinned tool list is invalid".into(),
            });
        }
        Ok(())
    }
}

/// A tool reference accepted by Anthropic's mid-conversation tool changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolReference {
    Tool { name: String },
    McpTool { server_name: String, name: String },
    McpToolset { server_name: String },
}

impl AnthropicToolReference {
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }

    pub fn mcp_tool(server_name: impl Into<String>, name: impl Into<String>) -> Self {
        Self::McpTool {
            server_name: server_name.into(),
            name: name.into(),
        }
    }

    pub fn mcp_toolset(server_name: impl Into<String>) -> Self {
        Self::McpToolset {
            server_name: server_name.into(),
        }
    }

    fn wire_value(&self) -> Value {
        match self {
            Self::Tool { name } => serde_json::json!({"type":"tool_reference","name":name}),
            Self::McpTool { server_name, name } => {
                serde_json::json!({"type":"mcp_tool_reference","server_name":server_name,"name":name})
            }
            Self::McpToolset { server_name } => {
                serde_json::json!({"type":"mcp_toolset_reference","server_name":server_name})
            }
        }
    }
}

/// One Anthropic inline tool change. A removal always requires a reference;
/// custom definitions are additions or same-name updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", content = "value", rename_all = "snake_case")]
pub enum AnthropicToolChange {
    AddReference(AnthropicToolReference),
    Remove(AnthropicToolReference),
    DefineTool(ToolSpec),
}

impl AnthropicToolChange {
    pub fn add_reference(reference: AnthropicToolReference) -> Self {
        Self::AddReference(reference)
    }

    pub fn remove(reference: AnthropicToolReference) -> Self {
        Self::Remove(reference)
    }

    pub fn define_tool(tool: ToolSpec) -> Self {
        Self::DefineTool(tool)
    }

    pub fn into_content_block(self) -> ContentBlock {
        let value = match self {
            Self::AddReference(reference) => serde_json::json!({
                "type":"tool_addition", "tool":reference.wire_value()
            }),
            Self::Remove(reference) => serde_json::json!({
                "type":"tool_removal", "tool":reference.wire_value()
            }),
            Self::DefineTool(tool) => {
                let mut definition = serde_json::json!({
                    "name":tool.name,
                    "description":tool.description,
                    "input_schema":tool.input_schema,
                });
                if tool.strict {
                    definition["strict"] = Value::Bool(true);
                }
                if tool.defer_loading {
                    definition["defer_loading"] = Value::Bool(true);
                }
                if !tool.anthropic_allowed_callers().is_empty() {
                    definition["allowed_callers"] =
                        serde_json::to_value(tool.anthropic_allowed_callers())
                            .unwrap_or(Value::Null);
                }
                serde_json::json!({
                    "type":"tool_addition",
                    "tool":{"type":"tool_definition","definition":definition}
                })
            }
        };
        ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value,
        }
    }

    pub fn into_system_message(self) -> ConversationMessage {
        ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::System,
            content: vec![self.into_content_block()],
        }
    }
}

/// Anthropic Messages `code_execution_20260521` settings. The first-party API
/// and supported Anthropic-hosted Foundry deployments run commands and return
/// native server-tool blocks; the host never executes them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicCodeExecutionConfig {
    /// Define Code Execution in a mid-conversation `tool_addition` instead of
    /// the top-level `tools` array. The addition must use the typed tool value
    /// and is supported only on the first-party Messages API.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inline_definition: bool,
    /// Reuse this account-bound container. `None` asks the API for a new one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerRef>,
    /// Anthropic-managed or workspace-custom Skills made available in the
    /// Code Execution container. Built-in Skills work on Anthropic-hosted
    /// Foundry. Custom Skill references require the explicit first-party or
    /// Foundry Skills service scope; Foundry version-content download is unsupported.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<AnthropicSkillRef>,
    /// Already-uploaded first-party Anthropic or Foundry files added as
    /// `container_upload` blocks to the final user message. Foundry references
    /// must come from a Foundry Files service and match the resource/account;
    /// they are not bound to a deployment or model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ProviderFileSource>,
}

/// Skill source accepted by the Anthropic Messages container parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicSkillType {
    Anthropic,
    Custom,
}

/// Local scope for a custom Anthropic Skill ID. On the Claude API, `account_scope`
/// identifies the owning workspace. On Foundry, it identifies the resource/account
/// credentials. Neither value is sent on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSkillScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workspace_id: Option<String>,
    /// Explicitly distinguishes Anthropic-hosted Foundry from the direct API.
    /// `false` identifies the direct Claude API route.
    #[serde(default, skip_serializing_if = "is_false")]
    foundry: bool,
}

impl AnthropicSkillScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
            workspace_id: None,
            foundry: false,
        };
        scope.validate()?;
        scope.endpoint = "https://api.anthropic.com".into();
        Ok(scope)
    }

    /// Bind a custom Skill reference to an Anthropic-hosted Microsoft Foundry
    /// resource. This scope is resource/account-scoped and deliberately does
    /// not capture a chat deployment or model.
    pub fn new_foundry(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        hosting: FoundryHosting,
    ) -> Result<Self, LlmError> {
        if hosting != FoundryHosting::Anthropic {
            return Err(LlmError::UnsupportedCapability {
                message: "custom Anthropic Skills require a Foundry deployment hosted on Anthropic"
                    .into(),
            });
        }
        let endpoint = endpoint.into();
        let canonical_endpoint = AnthropicContainerScope::normalize_foundry_endpoint(&endpoint)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Foundry Skill scope requires the HTTPS Anthropic resource endpoint"
                    .into(),
            })?;
        let scope = Self {
            profile_name: profile_name.into(),
            endpoint: canonical_endpoint,
            account_scope: account_scope.into(),
            workspace_id: None,
            foundry: true,
        };
        scope.validate()?;
        Ok(scope)
    }

    /// Bind a direct Claude API reference to an explicit Anthropic Workspace.
    /// The same value must be configured as
    /// `extra.headers.anthropic-workspace-id` on Messages profiles using it.
    /// Foundry scopes are resource/account-bound and cannot set this header.
    pub fn with_workspace_id(mut self, workspace_id: impl Into<String>) -> Result<Self, LlmError> {
        self.workspace_id = Some(workspace_id.into());
        self.validate()?;
        Ok(self)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn workspace_id(&self) -> Option<&str> {
        self.workspace_id.as_deref()
    }

    pub(crate) fn is_foundry(&self) -> bool {
        self.foundry
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if [&self.profile_name, &self.account_scope]
            .into_iter()
            .any(|value| value.trim().is_empty() || value.chars().any(char::is_control))
            || self
                .workspace_id
                .as_deref()
                .is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control))
            || (self.foundry
                && (self.workspace_id.is_some()
                    || AnthropicContainerScope::normalize_foundry_endpoint(&self.endpoint)
                        .is_none()))
            || (!self.foundry && !AnthropicContainerScope::is_official_endpoint(&self.endpoint))
        {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic custom Skill scope requires a profile, supported endpoint, and workspace/resource-bound account identity".into(),
            });
        }
        Ok(())
    }
}

fn anthropic_profile_workspace_id(profile: &ProviderProfile) -> Result<Option<&str>, LlmError> {
    let Some(headers) = profile.extra.get("headers").and_then(Value::as_object) else {
        return Ok(None);
    };
    let mut matching = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-workspace-id"));
    let Some((_, value)) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() {
        return Err(LlmError::InvalidRequest {
            message: "Anthropic Skill profile contains duplicate anthropic-workspace-id headers"
                .into(),
        });
    }
    let workspace_id = value.as_str().ok_or_else(|| LlmError::InvalidRequest {
        message: "Anthropic Skill profile anthropic-workspace-id must be a string".into(),
    })?;
    if workspace_id.trim().is_empty() || workspace_id.chars().any(char::is_control) {
        return Err(LlmError::InvalidRequest {
            message: "Anthropic Skill profile anthropic-workspace-id is empty or invalid".into(),
        });
    }
    Ok(Some(workspace_id))
}

/// A Skill to make available in an Anthropic Code Execution container.
///
/// `version` is optional; when supplied, Anthropic Skills accept a date such
/// as `20251013` or `latest`, and custom Skills accept `skver_…`, the legacy
/// `skill_version_…` form, or `latest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSkillRef {
    #[serde(rename = "type")]
    skill_type: AnthropicSkillType,
    skill_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<AnthropicSkillScope>,
}

impl AnthropicSkillRef {
    /// Reference one of Anthropic's built-in Skills, such as `pptx` or `pdf`.
    pub fn anthropic(skill_id: impl Into<String>) -> Self {
        Self {
            skill_type: AnthropicSkillType::Anthropic,
            skill_id: skill_id.into(),
            version: None,
            scope: None,
        }
    }

    /// Reference a custom Skill already uploaded to the scoped Anthropic
    /// workspace or Anthropic-hosted Foundry resource.
    pub fn custom(skill_id: impl Into<String>, scope: AnthropicSkillScope) -> Self {
        Self {
            skill_type: AnthropicSkillType::Custom,
            skill_id: skill_id.into(),
            version: None,
            scope: Some(scope),
        }
    }

    /// Pin a documented version or use `latest`.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn skill_type(&self) -> AnthropicSkillType {
        self.skill_type
    }

    pub fn skill_id(&self) -> &str {
        &self.skill_id
    }

    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    pub fn scope(&self) -> Option<&AnthropicSkillScope> {
        self.scope.as_ref()
    }

    pub(crate) fn validate_for(
        &self,
        profile: &ProviderProfile,
        account_scope: Option<&str>,
        foundry_deployment: Option<&FoundryDeployment>,
    ) -> Result<(), LlmError> {
        if self.skill_id.trim().is_empty() || self.skill_id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic Skill references require a nonempty valid skill_id".into(),
            });
        }
        match (self.skill_type, self.scope.as_ref()) {
            (AnthropicSkillType::Anthropic, None) => {}
            (AnthropicSkillType::Anthropic, Some(_)) => {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic-managed Skill references do not take a custom scope".into(),
                });
            }
            (AnthropicSkillType::Custom, Some(scope)) => {
                scope.validate()?;
                let scope_matches_route = if scope.foundry {
                    profile.protocol == ProtocolFamily::FoundryClaude
                        && foundry_deployment.is_some_and(|deployment| {
                            deployment.hosting == FoundryHosting::Anthropic
                        })
                        && AnthropicContainerScope::normalize_foundry_endpoint(&profile.base_url)
                            == AnthropicContainerScope::normalize_foundry_endpoint(&scope.endpoint)
                        && scope.workspace_id.is_none()
                } else {
                    profile.protocol == ProtocolFamily::AnthropicMessages
                        && foundry_deployment.is_none()
                        && AnthropicContainerScope::is_official_endpoint(&profile.base_url)
                        && scope.endpoint == "https://api.anthropic.com"
                        && scope.workspace_id.as_deref() == anthropic_profile_workspace_id(profile)?
                };
                if scope.profile_name != profile.profile_name
                    || !scope_matches_route
                    || Some(scope.account_scope.as_str()) != account_scope
                {
                    return Err(LlmError::InvalidRequest {
                        message: "custom Anthropic Skill scope does not match the selected profile, supported endpoint and hosting, account_scope, or configured anthropic-workspace-id".into(),
                    });
                }
            }
            (AnthropicSkillType::Custom, None) => {
                return Err(LlmError::InvalidRequest {
                    message: "custom Anthropic Skill references require an explicit workspace/resource-bound scope".into(),
                });
            }
        }
        if let Some(version) = &self.version {
            if version.trim().is_empty() || version.chars().any(char::is_control) {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Skill version must be a nonempty valid identifier".into(),
                });
            }
            let valid_version = if version == "latest" {
                true
            } else {
                match self.skill_type {
                    AnthropicSkillType::Anthropic => {
                        version.len() == 8 && version.bytes().all(|byte| byte.is_ascii_digit())
                    }
                    AnthropicSkillType::Custom => {
                        ["skver_", "skill_version_"].iter().any(|prefix| {
                            version
                                .strip_prefix(prefix)
                                .is_some_and(|id| !id.is_empty())
                        })
                    }
                }
            };
            if !valid_version {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic Skill version must be `latest`, an Anthropic date version, or a custom Skill version ID".into(),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn wire_value(&self) -> Value {
        let mut value = serde_json::json!({
            "type": match self.skill_type {
                AnthropicSkillType::Anthropic => "anthropic",
                AnthropicSkillType::Custom => "custom",
            },
            "skill_id": self.skill_id,
        });
        if let Some(version) = &self.version {
            value["version"] = Value::String(version.clone());
        }
        value
    }
}

/// Non-secret local identity for an Anthropic execution container. The exact
/// request model is pinned by client policy; aliases are resolved before checking.
/// Foundry scopes also capture the typed hosting and underlying-model identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
    request_model: String,
    /// Foundry's request model is a caller-chosen deployment name, so the
    /// underlying model and hosting choice must travel with a scoped ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    foundry_deployment: Option<FoundryDeployment>,
}

impl AnthropicContainerScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        request_model: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
            request_model: request_model.into(),
            foundry_deployment: None,
        };
        scope.validate()?;
        scope.endpoint = "https://api.anthropic.com".into();
        Ok(scope)
    }

    /// Bind a container returned by a Foundry deployment hosted on Anthropic.
    /// The endpoint is the Foundry resource base URL, and `deployment` must be
    /// copied from the selected model row rather than inferred from its name.
    pub fn new_foundry(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
        request_model: impl Into<String>,
        deployment: FoundryDeployment,
    ) -> Result<Self, LlmError> {
        let endpoint = endpoint.into();
        let mut scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.clone(),
            account_scope: account_scope.into(),
            request_model: request_model.into(),
            foundry_deployment: Some(deployment),
        };
        scope.validate()?;
        scope.endpoint = Self::normalize_foundry_endpoint(&endpoint).ok_or_else(|| {
            LlmError::InvalidRequest {
                message: "Foundry container scope requires the HTTPS resource Anthropic Messages endpoint".into(),
            }
        })?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn request_model(&self) -> &str {
        &self.request_model
    }

    /// Underlying hosting and model identity captured for a Foundry container.
    pub fn foundry_deployment(&self) -> Option<&FoundryDeployment> {
        self.foundry_deployment.as_ref()
    }

    pub(crate) fn is_official_endpoint(endpoint: &str) -> bool {
        let Ok(url) = url::Url::parse(endpoint) else {
            return false;
        };
        url.scheme() == "https"
            && url.host_str() == Some("api.anthropic.com")
            && url.port().is_none()
            && url.path() == "/"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
    }

    /// Canonicalize only the Foundry Anthropic resource base documented for
    /// Messages and other Claude API resources. This intentionally does not
    /// accept arbitrary Azure hosts, custom paths, or proxy URLs.
    pub(crate) fn normalize_foundry_endpoint(endpoint: &str) -> Option<String> {
        let url = url::Url::parse(endpoint).ok()?;
        if url.scheme() != "https"
            || url.port().is_some()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.path(), "/anthropic" | "/anthropic/")
        {
            return None;
        }
        let host = url.host_str()?;
        let resource = host.strip_suffix(".services.ai.azure.com")?;
        if resource.is_empty()
            || resource.contains('.')
            || resource.starts_with('-')
            || resource.ends_with('-')
            || !resource
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return None;
        }
        Some(format!("https://{host}/anthropic"))
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        let identity_fields_are_valid =
            [&self.profile_name, &self.account_scope, &self.request_model]
                .into_iter()
                .all(|value| !value.trim().is_empty() && !value.chars().any(char::is_control));
        let valid_route = match &self.foundry_deployment {
            None => Self::is_official_endpoint(&self.endpoint),
            Some(deployment) => {
                deployment.hosting == FoundryHosting::Anthropic
                    && !deployment.model_id.trim().is_empty()
                    && !deployment.model_id.chars().any(char::is_control)
                    && Self::normalize_foundry_endpoint(&self.endpoint).is_some()
            }
        };
        if !identity_fields_are_valid || !valid_route {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic container scope requires a profile, supported endpoint and hosting identity, stable account identity, and request model".into(),
            });
        }
        Ok(())
    }
}

/// A caller-imported Anthropic container ID. Scope stays local and is checked
/// before dispatch; only the ID is sent as the request's top-level `container`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerRef {
    id: String,
    scope: AnthropicContainerScope,
}

impl AnthropicContainerRef {
    pub fn new(id: impl Into<String>, scope: AnthropicContainerScope) -> Result<Self, LlmError> {
        let reference = Self {
            id: id.into(),
            scope,
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn scope(&self) -> &AnthropicContainerScope {
        &self.scope
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        self.scope.validate()?;
        if self.id.trim().is_empty() || self.id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic container reference requires a nonempty container ID".into(),
            });
        }
        Ok(())
    }
}

/// First-party Anthropic or Anthropic-hosted Foundry's complete native
/// container envelope, including unknown metadata. The provider's rolling
/// `expires_at` does not describe the full container lifetime, so it is retained
/// without enforcing local expiry.
/// Native execution container metadata returned by the first-party Anthropic
/// API or a supported Anthropic-hosted Foundry deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicContainerMetadata {
    pub envelope: Value,
}

impl AnthropicContainerMetadata {
    /// Explicitly bind an observed ID to the caller's route and account.
    pub fn reference_for(
        &self,
        scope: AnthropicContainerScope,
    ) -> Result<AnthropicContainerRef, LlmError> {
        let id = self
            .envelope
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Anthropic container metadata has no string ID to import".into(),
            })?;
        AnthropicContainerRef::new(id, scope)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicClearAt {
    Never,
    NextUserMessage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicMessageEffort {
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

/// Anthropic-only metadata associated with one message. Codecs map these
/// options to the provider's supported message-level fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicMessageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_at: Option<AnthropicClearAt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<AnthropicMessageEffort>,
}

pub use super::fetch_sources::*;
pub use super::toolsets::*;
