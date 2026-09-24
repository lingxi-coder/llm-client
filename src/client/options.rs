//! What a caller says about one request.

use crate::protocol::Secret;
use std::collections::BTreeMap;
use std::time::Duration;

/// Default wall-clock deadline for a completion request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
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
}
