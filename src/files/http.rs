//! File HTTP encoding, parsing and bounded polling helpers.
use super::*;

/// Return the stable, non-secret endpoint fingerprint carried by provider
/// file references. It covers the full configured base URL without retaining
/// userinfo, query parameters, or any other URL text in the reference.
#[must_use]
pub fn provider_file_endpoint_fingerprint(base_url: &str) -> String {
    // FNV-1a 128 keeps the binding deterministic across Rust versions without
    // another dependency. This is an endpoint identifier, not an auth token.
    let hash =
        base_url
            .as_bytes()
            .iter()
            .fold(0x6c62272e07bb014262b821756295c58du128, |hash, byte| {
                (hash ^ u128::from(*byte)).wrapping_mul(0x0000000001000000000000000000013bu128)
            });
    format!("fnv1a128:{hash:032x}")
}

/// A payload resolved by the host application before provider preparation.
pub(crate) fn multipart_body(
    adapter: Adapter,
    purpose: FilePurpose,
    file: &UploadFile,
    boundary: &str,
    expires_in_seconds: Option<u64>,
) -> Result<Bytes, LlmError> {
    let (prefix, suffix) = multipart_parts(
        adapter,
        purpose,
        &file.filename,
        &file.media_type,
        boundary,
        expires_in_seconds,
    );
    let mut body = BytesMut::new();
    body.extend_from_slice(&prefix);
    body.extend_from_slice(&file.bytes);
    body.extend_from_slice(&suffix);
    Ok(body.freeze())
}

/// Construct only the small multipart framing around a caller-owned file
/// stream. Both buffered and streaming uploads use this field builder so
/// purpose and expiration wire forms cannot drift.
pub(crate) fn multipart_parts(
    adapter: Adapter,
    purpose: FilePurpose,
    filename: &str,
    media_type: &str,
    boundary: &str,
    expires_in_seconds: Option<u64>,
) -> (Bytes, Bytes) {
    let mut prefix = BytesMut::new();
    if let Some(seconds) = expires_in_seconds {
        match adapter {
            Adapter::Anthropic => {
                append_field(
                    &mut prefix,
                    boundary,
                    "expires_in_seconds",
                    &seconds.to_string(),
                );
            }
            Adapter::OpenAi => {
                append_field(&mut prefix, boundary, "expires_after[anchor]", "created_at");
                append_field(
                    &mut prefix,
                    boundary,
                    "expires_after[seconds]",
                    &seconds.to_string(),
                );
            }
            // xAI's raw multipart API takes a scalar number of seconds, and
            // requires this field to precede the file part.
            Adapter::Xai => {
                append_field(&mut prefix, boundary, "expires_after", &seconds.to_string());
            }
            _ => {}
        }
    }
    match (adapter, purpose) {
        (Adapter::OpenAi, FilePurpose::Batch) => {
            append_field(&mut prefix, boundary, "purpose", "batch")
        }
        // xAI's documented Batch file example uploads only the `file` part.
        // Its optional `purpose` metadata is an OpenAI compatibility echo.
        (Adapter::Xai, FilePurpose::Batch) => {}
        (Adapter::OpenAi, FilePurpose::ModelInput) => {
            append_field(&mut prefix, boundary, "purpose", "user_data")
        }
        (Adapter::Xai, FilePurpose::ModelInput) => {
            append_field(&mut prefix, boundary, "purpose", "assistants")
        }
        (Adapter::Moonshot, FilePurpose::Extraction) => {
            append_field(&mut prefix, boundary, "purpose", "file-extract")
        }
        (Adapter::Qwen, FilePurpose::ModelInput | FilePurpose::Extraction) => {
            append_field(&mut prefix, boundary, "purpose", "file-extract")
        }
        (Adapter::MiniMax, purpose) => {
            if let Some(value) = purpose_name(adapter, purpose) {
                append_field(&mut prefix, boundary, "purpose", value);
            }
        }
        (Adapter::Zai, FilePurpose::Auxiliary) => {
            append_field(&mut prefix, boundary, "purpose", "agent")
        }
        (Adapter::Zhipu, FilePurpose::Auxiliary) => {
            append_field(&mut prefix, boundary, "purpose", "agent")
        }
        _ => {}
    }
    let safe_filename = sanitize_filename(filename);
    prefix.extend_from_slice(format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_filename}\"\r\nContent-Type: {}\r\n\r\n",
        media_type
    ).as_bytes());
    (
        prefix.freeze(),
        Bytes::from(format!("\r\n--{boundary}--\r\n")),
    )
}

pub(crate) fn multipart_batch_parts(
    filename: &str,
    media_type: &str,
    boundary: &str,
) -> (Bytes, Bytes) {
    multipart_parts(
        Adapter::OpenAi,
        FilePurpose::Batch,
        filename,
        media_type,
        boundary,
        None,
    )
}

struct ExactUploadState {
    input: BoxStream<'static, Result<Bytes, LlmError>>,
    remaining: u64,
}

/// Validate byte count while forwarding each input chunk. The stream does not
/// finish successfully until it observes EOF at the declared size.
pub(crate) fn exact_upload_stream(
    input: BoxStream<'static, Result<Bytes, LlmError>>,
    size_bytes: u64,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::try_unfold(
        ExactUploadState {
            input,
            remaining: size_bytes,
        },
        |mut state| async move {
            match state.input.next().await {
                Some(Ok(chunk)) => {
                    let chunk_size = u64::try_from(chunk.len()).unwrap_or(u64::MAX);
                    if chunk_size > state.remaining {
                        return Err(LlmError::StreamInterrupted {
                            message: "file upload stream exceeds its declared size".into(),
                        });
                    }
                    state.remaining -= chunk_size;
                    Ok(Some((chunk, state)))
                }
                Some(Err(error)) => Err(error),
                None if state.remaining != 0 => Err(LlmError::StreamInterrupted {
                    message: "file upload stream ended before its declared size".into(),
                }),
                None => Ok(None),
            }
        },
    )
    .boxed()
}

struct MultipartUploadState {
    prefix: Option<Bytes>,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    file_finished: bool,
    suffix: Option<Bytes>,
}

/// Emit the closing boundary only after the exact-size input stream reaches
/// EOF. An overlong, short, or failed stream therefore cannot produce a
/// syntactically complete multipart upload.
pub(crate) fn multipart_upload_stream(
    prefix: Bytes,
    input: BoxStream<'static, Result<Bytes, LlmError>>,
    size_bytes: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::try_unfold(
        MultipartUploadState {
            prefix: Some(prefix),
            file: exact_upload_stream(input, size_bytes),
            file_finished: false,
            suffix: Some(suffix),
        },
        |mut state| async move {
            if let Some(prefix) = state.prefix.take() {
                return Ok(Some((prefix, state)));
            }
            if !state.file_finished {
                match state.file.next().await {
                    Some(chunk) => return Ok(Some((chunk?, state))),
                    None => state.file_finished = true,
                }
            }
            Ok(state.suffix.take().map(|suffix| (suffix, state)))
        },
    )
    .boxed()
}

pub(crate) fn validate_media_type(value: &str) -> Result<(), LlmError> {
    let Some((top_level, subtype)) = value.split_once('/') else {
        return Err(LlmError::InvalidRequest {
            message: "attachment media_type must be a MIME type".into(),
        });
    };
    let is_token = |part: &str| {
        !part.is_empty()
            && part.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#'
                            | b'$'
                            | b'%'
                            | b'&'
                            | b'\''
                            | b'*'
                            | b'+'
                            | b'-'
                            | b'.'
                            | b'^'
                            | b'_'
                            | b'`'
                            | b'|'
                            | b'~'
                    )
            })
    };
    if value.matches('/').count() != 1 || !is_token(top_level) || !is_token(subtype) {
        return Err(LlmError::InvalidRequest {
            message: "attachment media_type contains invalid MIME token characters".into(),
        });
    }
    Ok(())
}

pub(crate) fn append_field(body: &mut BytesMut, boundary: &str, name: &str, value: &str) {
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .as_bytes(),
    );
}

pub(crate) fn multipart_boundary() -> String {
    // Unique enough to avoid accidental content collision; no random dependency
    // is needed because the bytes are locally generated and not user-facing.
    format!(
        "lingxi-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |time| time.as_nanos())
    )
}

pub(crate) fn gemini_upload_url(profile: &ProviderProfile) -> String {
    let base = profile.base_url.trim_end_matches('/');
    if base.ends_with("/v1beta") {
        format!("{}/upload/v1beta/files", base.trim_end_matches("/v1beta"))
    } else {
        format!("{base}/upload/v1beta/files")
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CursorField {
    OpenAi,
    Qwen,
    Anthropic,
    Gemini,
    Xai,
    OpenRouter,
    MiniMax,
}

pub(crate) fn query_value(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(byte))
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub(crate) fn path_segment(value: &str) -> String {
    query_value(value)
}

pub(crate) fn encoded_path(value: &str) -> String {
    value
        .split('/')
        .map(path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

pub(crate) fn json_success(response: &HttpResponse, operation: &str) -> Result<Value, LlmError> {
    status_result(response, operation)?;
    serde_json::from_slice(&response.body)
        .map_err(|error| provider_shape(&format!("{operation} response was not JSON: {error}")))
}

pub(crate) fn status_result(response: &HttpResponse, operation: &str) -> Result<(), LlmError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let body = String::from_utf8_lossy(&response.body);
    Err(status_error(response.status, &body, operation))
}

pub(crate) fn status_error(status: u16, body: &str, operation: &str) -> LlmError {
    let message = format!(
        "{operation} failed with HTTP {status}: {}",
        body.chars().take(512).collect::<String>()
    );
    match status {
        401 => LlmError::Authentication { message },
        403 => LlmError::PermissionDenied { message },
        404 => LlmError::InvalidRequest { message },
        413 => LlmError::RequestTooLarge { message },
        429 => LlmError::RateLimited {
            message,
            retry_after: None,
        },
        500..=599 => LlmError::ProviderInternal { message },
        _ => LlmError::InvalidRequest { message },
    }
}

pub(crate) fn json_optional_scalar_string(value: &Value) -> Option<String> {
    (!value.is_null()).then(|| {
        value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string())
    })
}

pub(crate) fn json_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

pub(crate) use crate::runtime::delay as async_delay;

pub(crate) fn gemini_timeout_remaining(
    deadline: Instant,
    timeout: Duration,
    phase: &str,
) -> Result<Duration, LlmError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        Err(LlmError::TransportTimeout {
            message: format!("Gemini file {phase} timed out after {timeout:?}"),
        })
    } else {
        Ok(remaining)
    }
}

pub(crate) fn gemini_processing_remaining(
    deadline: Instant,
    timeout: Duration,
    pending: &ProviderFileRef,
) -> Result<Duration, LlmError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        Err(LlmError::ProviderFileProcessing {
            message: format!("Gemini file remains PROCESSING after {timeout:?}"),
            file: Box::new(pending.model_reference()),
        })
    } else {
        Ok(remaining)
    }
}

pub(crate) async fn gemini_processing_call<T>(
    operation: impl Future<Output = Result<T, LlmError>>,
    deadline: Instant,
    timeout: Duration,
    pending: &ProviderFileRef,
) -> Result<T, LlmError> {
    let remaining = gemini_processing_remaining(deadline, timeout, pending)?;
    match futures::future::select(Box::pin(operation), Box::pin(async_delay(remaining))).await {
        futures::future::Either::Left((result, _)) => {
            result.map_err(|error| gemini_processing_unresolved(pending, error))
        }
        futures::future::Either::Right((_, _)) => Err(LlmError::ProviderFileProcessing {
            message: format!("Gemini file processing timed out after {timeout:?}"),
            file: Box::new(pending.model_reference()),
        }),
    }
}

pub(crate) fn gemini_processing_unresolved(pending: &ProviderFileRef, error: LlmError) -> LlmError {
    LlmError::ProviderFileProcessing {
        message: error.to_string(),
        file: Box::new(pending.model_reference()),
    }
}

pub(crate) fn nonempty_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(crate) fn valid_download_uri(uri: &str) -> bool {
    let Ok(uri) = url::Url::parse(uri) else {
        return false;
    };
    uri.scheme() == "https"
        && uri.host_str().is_some()
        && uri.username().is_empty()
        && uri.password().is_none()
}

pub(crate) fn same_origin(left: &str, right: &str) -> bool {
    let (Ok(left), Ok(right)) = (url::Url::parse(left), url::Url::parse(right)) else {
        return false;
    };
    left.scheme() == "https"
        && left.username().is_empty()
        && left.password().is_none()
        && left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

pub(crate) fn sanitize_filename(value: &str) -> String {
    let leaf = value.rsplit(['/', '\\']).next().unwrap_or(value);
    leaf.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ' ') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches(|ch: char| ch == '.' || ch.is_ascii_whitespace())
        .chars()
        .take(180)
        .collect::<String>()
        .pipe_nonempty("attachment")
}

trait PipeNonempty {
    fn pipe_nonempty(self, fallback: &str) -> String;
}
impl PipeNonempty for String {
    fn pipe_nonempty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_owned()
        } else {
            self
        }
    }
}

pub(crate) fn unsupported(operation: &str) -> LlmError {
    LlmError::UnsupportedCapability {
        message: format!("provider file {operation} is not supported for this profile"),
    }
}

pub(crate) fn provider_shape(message: &str) -> LlmError {
    LlmError::ProviderInternal {
        message: message.to_owned(),
    }
}
