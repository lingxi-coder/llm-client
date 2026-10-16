//! What a caller says about one request.

use crate::protocol::Secret;
use crate::providers::openrouter::response_cache::OpenRouterResponseCache;
use std::collections::BTreeMap;
use std::time::Duration;

/// First-party Claude request families that use Native's primary process
/// proxy route. Other providers continue through the shared default pool.
pub(crate) fn uses_native_proxy_route(profile: &crate::protocol::ProviderProfile) -> bool {
    (profile.protocol == crate::protocol::ProtocolFamily::AnthropicMessages
        && profile.provider_id.as_str() == "anthropic")
        || profile.protocol == crate::protocol::ProtocolFamily::FoundryClaude
}

#[cfg(test)]
mod native_proxy_route_tests {
    use super::uses_native_proxy_route;
    use crate::protocol::ProviderProfile;

    fn profile(provider_id: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(serde_json::json!({
            "provider_id": provider_id,
            "profile_name": "route-test",
            "base_url": "https://example.test",
            "protocol": protocol,
            "auth": "none",
            "models": [{"display_model":"m","request_model":"m","billing_model":"m"}]
        }))
        .unwrap()
    }

    #[test]
    fn only_first_party_anthropic_messages_and_foundry_use_native_proxy_pool() {
        assert!(uses_native_proxy_route(&profile(
            "anthropic",
            "anthropic_messages"
        )));
        assert!(!uses_native_proxy_route(&profile(
            "gateway",
            "anthropic_messages"
        )));
        assert!(uses_native_proxy_route(&profile("azure", "foundry_claude")));
        assert!(!uses_native_proxy_route(&profile("openai", "open_ai_chat")));
    }
}

/// Default wall-clock deadline for a completion request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Local context planning; never serialized to the provider.
    pub context: super::context::ContextRequest,
    /// The manual thinking budget was chosen automatically by the application.
    /// Re-bound it to each Claude-protocol attempt's effective output ceiling.
    /// Other protocol families use their own defaults instead of this Claude
    /// policy. Explicit thinking keeps the selected codec's normal validation.
    pub automatic_thinking_budget: bool,
    /// Explicit wire format for this call. Overrides the configured default;
    /// unsupported formats fail before encoding or network activity.
    pub protocol: Option<crate::protocol::ProtocolFamily>,
    /// Host-supplied Native Anthropic request kind, used only by SDK request
    /// serialization policies that are specific to an exact request route.
    pub anthropic_request_kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
    /// Host-admitted exact Native Anthropic endpoint for this request. This
    /// permits only request-local first-party Messages features on the exact
    /// selected profile; it is not inferred from a compatible wire protocol.
    pub native_anthropic_endpoint_admission: Option<NativeAnthropicEndpointAdmission>,
    /// Trusted Native Anthropic context observations for this request.
    pub native_anthropic_context: NativeAnthropicContextInput,
    /// Trusted current-request Native Anthropic safeguard and beta facts.
    /// This execution context is not model input or request metadata.
    pub native_anthropic_request_context: Option<NativeAnthropicRequestContext>,
    /// Typed, request-local Native model.complete gate. This is execution
    /// context and is never serialized into the provider request.
    pub model_complete_cache_gate: Option<crate::protocol::NativeModelCacheGate>,
    /// Current Native model.complete operation-specific cache-disable bit.
    pub drop_cache_control_when_caching_disabled: bool,
    /// Exact text sidecars in canonical ChatRequest message/content
    /// coordinates. The selected SDK codec remaps them to its encoded JSON
    /// body schema before exposing the prepared request.
    pub message_text_utf16_overrides: BTreeMap<String, Vec<u16>>,
    /// Request-local Fast capability observed by the host for the selected
    /// primary model. It overrides the catalog in validation and encoding,
    /// without changing the shared client or another failover connection.
    pub fast_capability: Option<crate::protocol::CapabilitySupport>,
    /// Request-local authentication supplied by an embedding host. Used only for
    /// the selected primary profile; never forwarded to fallback accounts.
    pub authenticator: Option<RequestAuthenticator>,
    /// Optional host policy applied before signing the final wire bytes.
    pub finalizer: Option<std::sync::Arc<dyn RequestFinalizer>>,
    /// The secret this request authenticates with, if the profile needs one.
    ///
    /// Passed in rather than looked up: this crate does not hold, fetch or
    /// store credentials. The caller owns them — including expiry, refresh and
    /// whatever secure storage the platform provides — and hands over one
    /// already-valid credential per request. A profile whose `auth` is `None`
    /// wants `None` here.
    pub credential: Option<Secret<String>>,
    /// Credentials for named failover profiles. The primary credential is
    /// never sent to another connection unless supplied here for that profile.
    pub fallback_credentials: BTreeMap<String, Secret<String>>,
    /// Stable, non-secret account identity for stateful response continuation.
    /// A response ID is returned as a reusable `ContinuationRef` only when this
    /// is set; the same scope is required on the next request.
    pub account_scope: Option<String>,
    /// Stable, non-secret identity of the account used for provider file
    /// caching. Supply the same value only for the same provider account; when
    /// absent, uploaded files are scoped to the current request and are not
    /// reused across calls. Qwen automatic uploads are request-scoped even
    /// when this value is stable, because Qwen does not expire stored files.
    pub file_account_scope: Option<String>,
    /// Total request deadline, including response body reads. `complete()`
    /// defaults to 120 seconds when omitted, or two hours for a request with
    /// video content; `stream()` has no default total
    /// deadline and relies on the transport's idle-read timeout. Automatic
    /// Qwen cleanup uses only the remaining budget, then retries in the background.
    pub total_timeout: Option<Duration>,
    /// Per-request OpenRouter gateway response-cache policy. This is not a
    /// provider prompt-cache breakpoint or a client-side response cache.
    pub openrouter_response_cache: Option<OpenRouterResponseCache>,
    /// Fresh authorization values for hosted remote MCP servers, keyed by
    /// OpenAI/xAI `server_label` or Anthropic server `name`. Never serialized
    /// into ChatRequest/history.
    pub mcp_authorizations: BTreeMap<String, Secret<String>>,
}

/// Request-local admission for Native Anthropic Messages behavior on an
/// explicitly selected first-party endpoint. The SDK still binds the token to
/// the exact profile name, provider identity, protocol, and base URL before
/// using it. Hosts must construct it only after their route policy has admitted
/// that endpoint for the current request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAnthropicEndpointAdmission {
    profile_name: String,
    base_url: String,
}

impl NativeAnthropicEndpointAdmission {
    /// Bind an admission to the exact selected first-party Messages profile.
    /// Host route policy remains responsible for establishing that selection.
    #[must_use]
    pub fn selected_first_party_messages_endpoint(
        profile_name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            base_url: base_url.into(),
        }
    }

    pub(crate) fn matches_profile(&self, profile: &crate::protocol::ProviderProfile) -> bool {
        profile.provider_id.as_str() == "anthropic"
            && profile.protocol == crate::protocol::ProtocolFamily::AnthropicMessages
            && profile.profile_name == self.profile_name
            && profile.base_url == self.base_url
    }
}

/// Current Native Anthropic context facts available to the SDK. `None` means
/// that source was not observed; a generic model-directory fallback is not
/// substituted for Native's own context registry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeAnthropicContextInput {
    /// Context window from the selected Native model-registry observation.
    pub model_registry_context_window: Option<u64>,
    /// Whether the selected Native model is registered for the 1M beta.
    pub model_supports_context_1m: Option<bool>,
    /// Context window from a current Native dynamic-context observation.
    pub dynamic_context_window: Option<u64>,
    /// Host-authorized suppression of the context-1m beta after a Native
    /// refusal. The SDK applies it to the final encoded request before auth.
    pub context_1m_beta_suppressed: bool,
    /// Exact membership in Native's planned `betas` list; this is not inferred
    /// from the eventual signed HTTP header.
    pub thinking_resumption_beta_requested: Option<bool>,
    /// Whether Native planned the 1M context beta for this request.
    pub context_1m_beta_requested: Option<bool>,
    /// Exact membership in Native planned thinking-binding-controls beta header.
    /// None means membership was not observed; the SDK does not infer it
    /// from thinking settings or final wire headers.
    pub native_thinking_binding_controls_beta_requested: Option<bool>,
    /// Exact Lsn-normalized manual budget for this ordinary Native main
    /// request. This provenance permits the Native 1024-token floor to exceed
    /// a smaller `max_tokens`; it does not bypass capability or range checks.
    pub native_lsn_manual_budget: Option<NativeLsnManualBudget>,
    /// Exact Native `X.model` selector before any provider wire-model mapping.
    /// This is a JSON string payload so unpaired UTF-16 units remain intact.
    /// A missing or non-string value leaves model-dependent HQ evidence unknown.
    pub native_hq_model_source: Option<crate::protocol::JsonPayload>,
    /// Host-observed result of Native's `cc(X.model)` provider mapping.
    /// `Other` means a known Native provider whose Bkt classifier is always
    /// unknown; unsupported or unrepresented Native families (including
    /// Mantle in hosts without that route) must remain absent.
    pub native_hq_model_provider_family:
        Option<crate::protocol::NativeAnthropicModelProviderFamily>,
}

/// A Native Lsn manual budget after applying its ordinary request clamp.
/// The SDK compares both values with the encoded `thinking.budget_tokens` and
/// request `max_tokens` before relaxing only the generic below-max check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeLsnManualBudget {
    /// The final Native `thinking.budget_tokens` value.
    pub budget_tokens: u32,
    /// The final request `max_tokens` value used by Native's clamp.
    pub request_max_tokens: Option<u32>,
}

impl NativeLsnManualBudget {
    /// Bind exact Lsn-normalized values to one request.
    #[must_use]
    pub const fn new(budget_tokens: u32, request_max_tokens: Option<u32>) -> Self {
        Self {
            budget_tokens,
            request_max_tokens,
        }
    }
}

/// Trusted request facts for Native Anthropic safeguards and current beta features.
/// Hosts populate this only after the corresponding Native gates admit the current
/// request; the SDK still limits use to an eligible main-generation route. It is
/// never read from or serialized as `ChatRequest` metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeAnthropicRequestContext {
    /// Source-backed Native server-classifier context, when it was produced for
    /// this request after host admission. `None` means no context was produced.
    pub server_classifier: Option<NativeAnthropicServerClassifierContext>,
    /// Whether Native admitted AFK mode for this request.
    pub afk_mode: bool,
    /// Whether Native admitted mid-conversation tool changes for this request.
    pub mid_conversation_tool_changes: bool,
    /// Host-observed intent to resume an incomplete thinking block. This is
    /// execution context only and is never serialized into the provider body.
    pub resume_incomplete_thinking: bool,
}

/// Source-backed context attached to a Native `dangerous_tool_use` safeguard.
#[derive(Clone, PartialEq, Eq)]
pub struct NativeAnthropicServerClassifierContext {
    classifier_context: crate::protocol::JsonPayload,
}

impl std::fmt::Debug for NativeAnthropicServerClassifierContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeAnthropicServerClassifierContext")
            .field("classifier_context", &"<redacted>")
            .finish()
    }
}

impl NativeAnthropicServerClassifierContext {
    /// Validate and retain the exact source JSON object supplied by the host.
    pub fn new(
        classifier_context: crate::protocol::JsonPayload,
    ) -> Result<Self, crate::protocol::LlmError> {
        if !classifier_context.is_root_object() {
            return Err(crate::protocol::LlmError::InvalidRequest {
                message: "Native server classifier context must be a JSON object".into(),
            });
        }
        Ok(Self { classifier_context })
    }

    /// The exact source-backed context object forwarded into the safeguard.
    pub fn classifier_context(&self) -> &crate::protocol::JsonPayload {
        &self.classifier_context
    }
}

/// Per-request transformation of the encoded request, before authentication.
/// Implementations must not dispatch requests or fetch credentials. The final
/// bytes are authenticated once and immutable after preparation.
pub trait RequestFinalizer: std::fmt::Debug + Send + Sync {
    fn finalize(
        &self,
        request: &mut crate::HttpRequest,
        profile: &crate::protocol::ProviderProfile,
    ) -> Result<(), crate::protocol::LlmError>;
}

/// Redacted, cloneable request-local authenticator. It is never persisted.
#[derive(Clone)]
pub struct RequestAuthenticator(pub std::sync::Arc<dyn crate::Authenticator>);
impl std::fmt::Debug for RequestAuthenticator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RequestAuthenticator(<redacted>)")
    }
}
