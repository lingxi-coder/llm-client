//! What a caller says about one request.

use lingxi_agent_api::protocol::Secret;
use std::collections::BTreeMap;
use std::time::Duration;

/// Default wall-clock deadline for a completion request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Streaming mode for direct codec calls. `LlmClient::complete` and
    /// `LlmClient::stream` select their own mode regardless of this field.
    pub stream: bool,
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
    /// Total request deadline, including response body reads. `complete()`
    /// defaults to 120 seconds when omitted; `stream()` has no default total
    /// deadline and relies on the transport's idle-read timeout.
    pub total_timeout: Option<Duration>,
}
