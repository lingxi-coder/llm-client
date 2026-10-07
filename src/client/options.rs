//! What a caller says about one request.

use crate::protocol::Secret;
use crate::providers::openrouter::response_cache::OpenRouterResponseCache;
use std::collections::BTreeMap;
use std::time::Duration;

/// Default wall-clock deadline for a completion request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Host-supplied Native Anthropic request kind, used only by SDK request
    /// serialization policies that are specific to an exact request route.
    pub anthropic_request_kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
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
