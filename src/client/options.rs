//! What a caller says about one request.

use lingxi_agent_api::protocol::Secret;

/// Per-request options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub stream: bool,
    /// The secret this request authenticates with, if the profile needs one.
    ///
    /// Passed in rather than looked up: this crate does not hold, fetch or
    /// store credentials. The caller owns them — including expiry, refresh and
    /// whatever secure storage the platform provides — and hands over one
    /// already-valid credential per request. A profile whose `auth` is `None`
    /// wants `None` here.
    pub credential: Option<Secret<String>>,
}
