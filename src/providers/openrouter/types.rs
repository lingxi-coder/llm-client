//! Provider-specific request, tool, and replay types.

use crate::protocol::LlmError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenRouter's hosted regex tool-search server tool. The BM25 variant is not
/// included because OpenRouter currently documents only regex search.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenRouterToolSearchConfig {
    /// Maximum discovered tools returned by one search; OpenRouter defaults
    /// this to five and caps it at fifty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
}

/// Execution engine and container settings for OpenRouter's hosted shell.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenRouterShellConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<OpenRouterShellEngine>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<OpenRouterShellEnvironment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterShellEngine {
    /// Use a provider-native hosted shell when available, otherwise the
    /// OpenRouter sandbox. Commands remain server-executed.
    Auto,
    /// Always use the OpenRouter sandbox.
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterShellEnvironment {
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<OpenRouterShellNetworkPolicy>,
    },
    ContainerReference {
        container: OpenRouterContainerRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<OpenRouterShellNetworkPolicy>,
    },
}

/// OpenRouter's container network policy. Allowlist hosts follow the
/// provider's lowercase hostname/glob syntax and fifty-entry cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterShellNetworkPolicy {
    Disabled,
    Allowlist { allowed_domains: Vec<String> },
}

/// Local scope attached to a caller-imported OpenRouter shell container ID.
/// The reference is bound to the exact profile, endpoint, and stable account
/// identity selected by the application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerScope {
    profile_name: String,
    endpoint: String,
    account_scope: String,
}

impl OpenRouterContainerScope {
    pub fn new(
        profile_name: impl Into<String>,
        endpoint: impl Into<String>,
        account_scope: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let scope = Self {
            profile_name: profile_name.into(),
            endpoint: endpoint.into(),
            account_scope: account_scope.into(),
        };
        if scope.profile_name.trim().is_empty()
            || scope.endpoint.trim().is_empty()
            || scope.account_scope.trim().is_empty()
            || scope.profile_name.chars().any(char::is_control)
            || scope.endpoint.chars().any(char::is_control)
            || scope.account_scope.chars().any(char::is_control)
        {
            return Err(LlmError::InvalidRequest {
                message: "OpenRouter container scope requires a profile, endpoint, and stable account identity".into(),
            });
        }
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
}

/// A caller-imported container ID. The encoder sends only `id`; this local
/// scope is checked before dispatch and never appears in the provider request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerRef {
    id: String,
    scope: OpenRouterContainerScope,
}

impl OpenRouterContainerRef {
    pub fn new(id: impl Into<String>, scope: OpenRouterContainerScope) -> Result<Self, LlmError> {
        let id = id.into();
        if id.trim().is_empty() || id.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "OpenRouter container reference requires a nonempty container ID".into(),
            });
        }
        Ok(Self { id, scope })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn scope(&self) -> &OpenRouterContainerScope {
        &self.scope
    }
}

/// Raw OpenRouter Messages container metadata. Direct codec decoding has no
/// account identity, so callers explicitly supply a local scope when turning
/// its ID into a reusable shell reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterContainerMetadata {
    pub envelope: Value,
}

impl OpenRouterContainerMetadata {
    pub fn reference_for(
        &self,
        scope: OpenRouterContainerScope,
    ) -> Result<OpenRouterContainerRef, LlmError> {
        let id = self
            .envelope
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "OpenRouter container metadata has no string ID to import".into(),
            })?;
        OpenRouterContainerRef::new(id, scope)
    }
}
