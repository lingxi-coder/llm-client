//! What a connection says it serves *right now*.
//!
//! The shipped catalog is authoritative about what a model costs and what it
//! is; it cannot be authoritative about what exists, because it is a snapshot
//! taken when someone last ran the vendor script. A directory is the other
//! half: one GET per connection, answering which model ids that endpoint will
//! accept today.
//!
//! **Why this is a second trait rather than two more methods on `WireCodec`.**
//! The two are keyed differently and fail differently:
//!
//! - *Keyed differently.* The wire an endpoint speaks does not decide the
//!   shape of its model list. Among the shipped presets one connection speaks
//!   one wire for completions while its host publishes the list in another
//!   wire's shape, and one publishes no list at all. So the shape is a route
//!   key (`ProviderProfile::model_list`), not `protocol`.
//! - *Fail differently.* A profile whose protocol has no codec cannot run a
//!   turn, so the builder refuses it (gate 33). A profile with no directory
//!   runs turns perfectly well and simply cannot refresh its list, so it
//!   builds. Folding the two into one trait would force one of those two
//!   behaviours onto the other case.
//!
//! **A wrong parser is worse than no parser.** Every shape here refuses a page
//! whose list key is absent, and refuses a row with no id, rather than
//! returning what it did manage to read. A page decoded by the wrong shape
//! yields zero rows — which downstream is indistinguishable from "this
//! provider withdrew its entire catalog", and would be merged as such.

mod anthropic;
mod gemini;
mod openai;

pub use anthropic::AnthropicMessagesDirectory;
pub use gemini::GeminiDirectory;
pub use openai::OpenAiChatDirectory;

use crate::transport::{HttpRequest, HttpResponse};
use bytes::Bytes;
use lingxi_agent_api::protocol::{LlmError, ProtocolFamily, ProviderProfile};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// One model as the provider's own list describes it.
///
/// Deliberately thin. Prices are not read from a live list even where one
/// publishes them: the catalog is what this project prices from, and a
/// provider-stated price would be a second source with no test behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveModel {
    /// The id as it goes on the wire — what a request's `model` field carries,
    /// and what a catalog row is matched against.
    pub request_model: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
}

/// One page of a directory, and how to ask for the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPage {
    pub models: Vec<LiveModel>,
    /// `None` on the last page. Opaque: it is the provider's own cursor, and
    /// the only thing a caller may do with it is hand it back.
    pub next_cursor: Option<String>,
}

/// One decoded page with any operation support the directory states explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedModelPage {
    /// Models the directory can identify as usable or legacy-compatible.
    pub page: ModelPage,
    /// Model ids explicitly known to be incompatible with this directory's
    /// protocol family.
    pub incompatible_model_ids: Vec<String>,
    /// Model ids explicitly known to support this directory's protocol family.
    /// Rows with absent operation metadata are deliberately omitted.
    pub explicitly_compatible_model_ids: Vec<String>,
}

/// How to ask one shape of endpoint what it serves, and how to read the answer.
///
/// Registered per `ProtocolFamily` like a codec, but looked up through
/// `ProviderProfile::model_list` rather than `protocol`, and optional where a
/// codec is not.
pub trait ModelDirectory: Send + Sync + 'static {
    /// The shape this reads. Also its registration key.
    fn shape(&self) -> ProtocolFamily;

    /// The GET for one page. `cursor` is `None` for the first page and
    /// otherwise a value this same implementation produced.
    ///
    /// Carries no credential: the caller attaches one, after this, the same
    /// way it does for a completion. This crate holds none (gate 64).
    fn list_request(&self, profile: &ProviderProfile, cursor: Option<&str>) -> HttpRequest;

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError>;

    /// Decode a page and identify model ids explicitly known to be incompatible
    /// with this directory's protocol family.
    ///
    /// The default preserves the original directory contract and reports no
    /// explicit operation support. Implementations that can distinguish
    /// listed-but-unusable models or known-compatible models can return those
    /// ids so provider sync updates stale copies safely.
    fn decode_page_with_exclusions(
        &self,
        resp: &HttpResponse,
    ) -> Result<DecodedModelPage, LlmError> {
        self.decode_page(resp).map(|page| DecodedModelPage {
            page,
            incompatible_model_ids: Vec::new(),
            explicitly_compatible_model_ids: Vec::new(),
        })
    }
}

// Gate 3.
const _: Option<&dyn ModelDirectory> = None;

/// A directory GET is a background refresh, not a turn: it is worth bounding
/// so a hung connection cannot hold a refresh open indefinitely.
const DIRECTORY_TIMEOUT: Duration = Duration::from_secs(30);

/// Asked for in one page where the wire allows it. Both paginating shapes here
/// cap at this; a catalog of several hundred models is otherwise a dozen round
/// trips against a default page size.
const PAGE_SIZE: &str = "1000";

/// The endpoint to list from.
///
/// How much path `base_url` carries is a per-family convention — bare origin
/// for one shape, versioned root for another — so the path a shape appends is
/// part of the shape, exactly as it is for that family's codec.
fn endpoint(profile: &ProviderProfile, default_path: &str) -> String {
    format!("{}{default_path}", profile.base_url.trim_end_matches('/'))
}

/// Percent-encode a query value.
///
/// A cursor is a provider-minted opaque string — one shape's is a model id,
/// another's is base64 with its padding intact — so nothing may be assumed
/// about the bytes in it.
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(b));
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn with_query(url: &str, params: &[(&str, &str)]) -> String {
    let mut out = url.to_owned();
    let mut sep = if url.contains('?') { '&' } else { '?' };
    for (key, value) in params {
        out.push(sep);
        out.push_str(key);
        out.push('=');
        out.push_str(&escape(value));
        sep = '&';
    }
    out
}

fn get(url: String, profile: &ProviderProfile) -> HttpRequest {
    let mut headers = vec![("accept".to_owned(), "application/json".to_owned())];
    // A profile's attribution headers belong on every request it makes, not
    // only on completions. Credentials are refused there (`wire_extras`).
    crate::codecs::extras::merge_headers(profile, &mut headers);
    HttpRequest {
        method: "GET".to_owned(),
        url,
        headers,
        body: Bytes::new(),
        timeout: Some(DIRECTORY_TIMEOUT),
    }
}

fn body_of(resp: &HttpResponse) -> Result<Value, LlmError> {
    serde_json::from_slice(&resp.body).map_err(|e| LlmError::ProviderInternal {
        message: format!("the model directory answered with something that is not JSON: {e}"),
    })
}

/// The list itself, refusing a page that does not have one.
///
/// An absent key is not an empty catalog. It means this page was read by the
/// wrong shape, or the endpoint answered something else entirely with a 200 —
/// and either one, reported as zero models, merges as a total withdrawal.
fn rows<'a>(body: &'a Value, key: &str) -> Result<&'a Vec<Value>, LlmError> {
    body.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::ProviderInternal {
            message: format!("the model directory's page has no {key:?} array"),
        })
}

fn required_id(row: &Value, key: &str) -> Result<String, LlmError> {
    row.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| LlmError::ProviderInternal {
            message: format!("a listed model has no {key:?}, so nothing could be requested of it"),
        })
}

fn text(row: &Value, key: &str) -> Option<String> {
    row.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn number(row: &Value, key: &str) -> Option<u64> {
    row.get(key).and_then(Value::as_u64)
}

/// A cursor the provider gave us, ignoring the empty string some wires send in
/// place of omitting the field.
fn cursor(body: &Value, key: &str) -> Option<String> {
    text(body, key)
}

/// A non-success status becomes the same `LlmError` a completion would have
/// produced, through the shape's own classifier: a 401 from a directory means
/// exactly what a 401 from a completion means, and a caller deciding whether to
/// prompt for a key should not have to special-case where it came from.
fn ok_or_classified(
    resp: &HttpResponse,
    classify: fn(u16, &Value, Option<Duration>) -> LlmError,
) -> Result<Value, LlmError> {
    if (200..300).contains(&resp.status) {
        return body_of(resp);
    }
    let body = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
    Err(classify(resp.status, &body, retry_after(resp)))
}

fn retry_after(resp: &HttpResponse) -> Option<Duration> {
    resp.header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Every directory compiled into this crate, one per shape that a shipped
/// route names. A family with none is not an error: that connection cannot
/// refresh its list and can still run turns.
pub(crate) fn builtin() -> Vec<Arc<dyn ModelDirectory>> {
    vec![
        Arc::new(OpenAiChatDirectory),
        Arc::new(AnthropicMessagesDirectory),
        Arc::new(GeminiDirectory),
    ]
}
