//! Pure Gemini resumable upload messages for hosts that own authentication and polling.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::LlmError;
use crate::transport::HttpRequest;

/// A Gemini File API `File` resource.
///
/// `state` is kept as a `String` (tolerant decoder convention); known wire
/// values are `PROCESSING`, `ACTIVE`, and `FAILED`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiFile {
    /// Resource name, e.g. `files/abc123`.
    pub name: String,
    /// Download/reference URI used as `file_data.file_uri` in generate calls.
    pub uri: String,
    /// MIME type recorded for the uploaded bytes.
    pub mime_type: String,
    /// Processing state: `PROCESSING`, `ACTIVE`, or `FAILED` (passed through
    /// verbatim; unknown values are preserved).
    pub state: String,
}

/// Strip one trailing `/v1beta` or `/v1` version segment from `base_url`.
///
/// See the module docs for the convention; a base without a recognized
/// trailing version segment is returned unchanged (minus a trailing slash).
fn upload_base(base_url: &str) -> &str {
    let trimmed = base_url.trim_end_matches('/');
    if let Some(root) = trimmed.strip_suffix("/v1beta") {
        return root;
    }
    if let Some(root) = trimmed.strip_suffix("/v1") {
        return root;
    }
    trimmed
}

/// Build the resumable-upload START request:
/// `POST {upload_base}/upload/v1beta/files`.
///
/// Carries only metadata (the byte count, MIME type, and display name); the
/// raw bytes go in the second leg built by [`upload_finalize_request`]. The
/// response's `x-goog-upload-url` header (see [`parse_start_response`]) is
/// the session URL for that second leg.
#[must_use]
pub fn start_upload_request(
    base_url: &str,
    num_bytes: usize,
    mime_type: &str,
    display_name: &str,
) -> HttpRequest {
    let root = upload_base(base_url);
    let mut request = HttpRequest {
        method: "POST".into(),
        url: format!("{root}/upload/v1beta/files"),
        body: serde_json::to_vec(&serde_json::json!({"file":{"display_name":display_name}}))
            .expect("file metadata JSON")
            .into(),
        headers: Vec::new(),
        timeout: None,
    };
    request.headers.push((
        "x-goog-upload-protocol".to_string(),
        "resumable".to_string(),
    ));
    request
        .headers
        .push(("x-goog-upload-command".to_string(), "start".to_string()));
    request.headers.push((
        "x-goog-upload-header-content-length".to_string(),
        num_bytes.to_string(),
    ));
    request.headers.push((
        "x-goog-upload-header-content-type".to_string(),
        mime_type.to_string(),
    ));
    request
        .headers
        .push(("content-type".to_string(), "application/json".to_string()));
    request
}

/// Extract the resumable-session URL from the START response headers.
///
/// The lookup is case-insensitive because transports are only EXPECTED to
/// lower-case header names ([`crate::StreamResponse`] convention), not
/// required to. The error message never echoes header values (no-secret
/// rule — response header maps can sit next to credential material).
pub fn parse_start_response(headers: &BTreeMap<String, String>) -> Result<String, LlmError> {
    headers
        .get("x-goog-upload-url")
        .or_else(|| {
            headers.iter().find_map(|(name, value)| {
                name.eq_ignore_ascii_case("x-goog-upload-url")
                    .then_some(value)
            })
        })
        .cloned()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Gemini upload start response missing x-goog-upload-url header".to_string(),
        })
}

/// Build the UPLOAD+FINALIZE request: `POST {upload_url}` with the raw media
/// bytes as the body.
///
/// `body_bytes` carries the payload and `body_json` stays `Value::Null` — the
/// transport bridge sends raw bytes verbatim when `body_bytes` is set.
/// `content-length` is left to the transport.
#[must_use]
pub fn upload_finalize_request(upload_url: &str, bytes: Vec<u8>) -> HttpRequest {
    let mut headers = BTreeMap::new();
    headers.insert(
        "x-goog-upload-command".to_string(),
        "upload, finalize".to_string(),
    );
    headers.insert("x-goog-upload-offset".to_string(), "0".to_string());
    HttpRequest {
        method: "POST".to_string(),
        url: upload_url.to_string(),
        headers: headers.into_iter().collect(),
        body: bytes.into(),
        timeout: None,
    }
}

/// Parse the UPLOAD+FINALIZE response body.
///
/// The upload response nests the resource under a `"file"` key:
/// `{"file": {"name": "files/abc", "uri": "...", "mimeType": "...", "state": "ACTIVE"}}`.
/// (The files.get response is NOT nested — see [`parse_file_status`].)
///
/// A `FAILED` state is data, not an error: it is passed through for the
/// caller to act on.
pub fn parse_upload_response(body: &Value) -> Result<GeminiFile, LlmError> {
    let file = body.get("file").ok_or_else(|| LlmError::InvalidRequest {
        message: "Gemini upload response missing file object".to_string(),
    })?;
    parse_file_object(file, "Gemini upload response missing")
}

/// Build the file-status poll request: `GET {upload_base}/v1beta/{file_name}`
/// where `file_name` is the resource name (`files/<id>`).
///
/// Callers poll this until `state == "ACTIVE"` for video/PDF uploads; the
/// upload driver never polls — use
/// the host polling policy for a ready-made
/// polling loop.
#[must_use]
pub fn file_status_request(base_url: &str, file_name: &str) -> HttpRequest {
    let root = upload_base(base_url);
    HttpRequest {
        method: "GET".to_string(),
        url: format!("{root}/v1beta/{file_name}"),
        headers: Vec::new(),
        body: Default::default(),
        timeout: None,
    }
}

/// Parse a files.get response body.
///
/// Asymmetry with [`parse_upload_response`]: the files.get response IS the
/// `File` resource itself at the top level — it is not wrapped in a `"file"`
/// key like the upload response.
pub fn parse_file_status(body: &Value) -> Result<GeminiFile, LlmError> {
    parse_file_object(body, "Gemini file status response missing")
}

/// Parse one camelCase `File` resource object.
///
/// `name` and `uri` are required (errors name the missing field and never
/// echo sibling values); `mimeType` and `state` default to empty strings when
/// absent (tolerant-decoder convention).
fn parse_file_object(file: &Value, error_prefix: &str) -> Result<GeminiFile, LlmError> {
    let name = file
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: format!("{error_prefix} file.name"),
        })?
        .to_string();
    let uri = file
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: format!("{error_prefix} file.uri"),
        })?
        .to_string();
    let mime_type = file
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let state = file
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(GeminiFile {
        name,
        uri,
        mime_type,
        state,
    })
}
