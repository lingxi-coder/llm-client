//! Server-side URL input for xAI speech-to-text.

use super::{
    invalid, multipart_boundary, transcription_option_fields, validate_transcription_request,
    XaiAudioCredentials, XaiAudioError, XaiAudioService, XaiTranscription, XaiTranscriptionRequest,
};
use crate::runtime::Deadline;
use crate::{
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest},
};
use ::url::Url;
use bytes::{Bytes, BytesMut};
use serde_json::Value;
use std::fmt;

/// A caller-supplied HTTP(S) audio URL for xAI server-side transcription.
///
/// The URL is sent to xAI in the multipart `url` field; this client does not
/// download it. Its `Debug` output is redacted because query parameters may
/// contain signed access credentials.
#[derive(Clone, PartialEq, Eq)]
pub struct XaiTranscriptionUrl {
    value: String,
    query: Option<String>,
}

impl XaiTranscriptionUrl {
    /// Validate an absolute HTTP(S) URL without fetching it.
    pub fn new(value: impl Into<String>) -> Result<Self, XaiAudioError> {
        let value = value.into();
        let Some((scheme, remainder)) = value.split_once("://") else {
            return Err(invalid("audio source must be an absolute HTTP(S) URL"));
        };
        let authority = remainder
            .split(['/', '?', '#', '\\'])
            .next()
            .unwrap_or_default();
        if (!scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https"))
            || authority.is_empty()
            || value.contains('\\')
        {
            return Err(invalid("audio source must be an absolute HTTP(S) URL"));
        }
        let parsed = Url::parse(&value)
            .map_err(|_| invalid("audio source must be an absolute HTTP(S) URL"))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none_or(str::is_empty)
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.fragment().is_some()
            || value
                .chars()
                .any(|character| character.is_whitespace() || character.is_control())
        {
            return Err(invalid(
                "audio source must be an absolute HTTP(S) URL without user info or a fragment",
            ));
        }
        // Keep the exact input bytes: signed queries can depend on their
        // original escaping and ordering. Url parsing above is validation only.
        Ok(Self {
            value,
            query: parsed.query().map(str::to_owned),
        })
    }

    fn redact_text(&self, text: &str) -> String {
        let mut redacted = text.replace(&self.value, "<redacted audio URL>");
        if let Some(query) = self.query.as_deref().filter(|query| !query.is_empty()) {
            // Redact only a complete query component. Replacing a bare short
            // query (for example `?type`) throughout an error could alter
            // unrelated JSON field names or enum discriminators.
            redacted = redacted.replace(&format!("?{query}"), "?<redacted URL query>");
        }
        redacted
    }

    fn redact_value(&self, value: Value) -> Value {
        match value {
            Value::String(text) => Value::String(self.redact_text(&text)),
            Value::Array(values) => Value::Array(
                values
                    .into_iter()
                    .map(|value| self.redact_value(value))
                    .collect(),
            ),
            Value::Object(object) => Value::Object(
                object
                    .into_iter()
                    .map(|(key, value)| (key, self.redact_value(value)))
                    .collect(),
            ),
            value => value,
        }
    }

    fn redact_error(&self, error: XaiAudioError) -> XaiAudioError {
        match error {
            XaiAudioError::Llm(source) => XaiAudioError::Llm(self.redact_llm_error(source)),
            XaiAudioError::InvalidRequest(message) => {
                XaiAudioError::InvalidRequest(self.redact_text(&message))
            }
            XaiAudioError::Provider {
                operation,
                status,
                request_id,
                body,
            } => XaiAudioError::Provider {
                operation,
                status,
                request_id,
                body: Box::new(self.redact_value(*body)),
            },
            XaiAudioError::InvalidResponse {
                operation,
                message,
                request_id,
                native,
            } => XaiAudioError::InvalidResponse {
                operation,
                message: self.redact_text(&message),
                request_id,
                native: Box::new(self.redact_value(*native)),
            },
            XaiAudioError::OutcomeUnknown {
                operation,
                source,
                response,
            } => XaiAudioError::OutcomeUnknown {
                operation,
                source: self.redact_llm_error(source),
                response: response.map(|response| {
                    let mut response = *response;
                    response.native = Box::new(self.redact_value(*response.native));
                    Box::new(response)
                }),
            },
        }
    }

    fn redact_llm_error(&self, error: LlmError) -> LlmError {
        match error {
            LlmError::Authentication { message } => LlmError::Authentication {
                message: self.redact_text(&message),
            },
            LlmError::PermissionDenied { message } => LlmError::PermissionDenied {
                message: self.redact_text(&message),
            },
            LlmError::InvalidRequest { message } => LlmError::InvalidRequest {
                message: self.redact_text(&message),
            },
            LlmError::RateLimited {
                message,
                retry_after,
            } => LlmError::RateLimited {
                message: self.redact_text(&message),
                retry_after,
            },
            LlmError::QuotaExceeded { message } => LlmError::QuotaExceeded {
                message: self.redact_text(&message),
            },
            LlmError::ContextOverflow {
                message,
                limit,
                actual,
            } => LlmError::ContextOverflow {
                message: self.redact_text(&message),
                limit,
                actual,
            },
            LlmError::RequestTooLarge { message } => LlmError::RequestTooLarge {
                message: self.redact_text(&message),
            },
            LlmError::ModelUnavailable { message } => LlmError::ModelUnavailable {
                message: self.redact_text(&message),
            },
            LlmError::ProviderInternal { message } => LlmError::ProviderInternal {
                message: self.redact_text(&message),
            },
            LlmError::Overloaded { message } => LlmError::Overloaded {
                message: self.redact_text(&message),
            },
            LlmError::Transport { message } => LlmError::Transport {
                message: self.redact_text(&message),
            },
            LlmError::TransportTimeout { message } => LlmError::TransportTimeout {
                message: self.redact_text(&message),
            },
            LlmError::FileUploadOutcomeUnknown { message } => LlmError::FileUploadOutcomeUnknown {
                message: self.redact_text(&message),
            },
            LlmError::ProviderFileProcessing { message, file } => {
                let mut file = *file;
                file.uri = file.uri.map(|uri| self.redact_text(&uri));
                LlmError::ProviderFileProcessing {
                    message: self.redact_text(&message),
                    file: Box::new(file),
                }
            }
            LlmError::TlsCert { message } => LlmError::TlsCert {
                message: self.redact_text(&message),
            },
            LlmError::StreamInterrupted { message } => LlmError::StreamInterrupted {
                message: self.redact_text(&message),
            },
            LlmError::CostUnavailable { message } => LlmError::CostUnavailable {
                message: self.redact_text(&message),
            },
            LlmError::UnsupportedCapability { message } => LlmError::UnsupportedCapability {
                message: self.redact_text(&message),
            },
        }
    }
}

impl fmt::Debug for XaiTranscriptionUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("XaiTranscriptionUrl")
            .field(&"<redacted>")
            .finish()
    }
}

impl<'a> XaiAudioService<'a> {
    /// Ask xAI to download and transcribe a caller-provided HTTP(S) audio URL.
    ///
    /// This submits one `POST /v1/stt` multipart request with a `url` field.
    /// The client does not fetch the media or retry the transcription.
    pub async fn transcribe_url(
        &self,
        audio_url: &XaiTranscriptionUrl,
        request: &XaiTranscriptionRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiTranscription, XaiAudioError> {
        let pinned_service = self.pin()?;
        let result = pinned_service
            .transcribe_url_inner(audio_url, request, credentials)
            .await;
        result.map_err(|error| audio_url.redact_error(error))
    }

    async fn transcribe_url_inner(
        &self,
        audio_url: &XaiTranscriptionUrl,
        request: &XaiTranscriptionRequest,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiTranscription, XaiAudioError> {
        validate_transcription_request(request)?;
        super::validate_credential(&credentials.api_key)?;

        let boundary = multipart_boundary();
        let (prefix, suffix) = transcription_url_multipart_parts(&boundary, audio_url, request);
        let body_len = prefix
            .len()
            .checked_add(suffix.len())
            .ok_or_else(|| invalid("multipart request size overflows"))?;
        let mut body = BytesMut::with_capacity(body_len);
        body.extend_from_slice(&prefix);
        body.extend_from_slice(&suffix);
        let deadline = Deadline::after(Some(self.config.request_timeout));
        let http = HttpRequest {
            method: "POST".into(),
            url: self.route_url("stt")?,
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                ),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            body: body.freeze(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .execute_bounded(http, super::MAX_JSON_RESPONSE_BYTES)
            .await
            .map_err(|source| super::map_outcome("speech-to-text", source))?;
        self.decode_transcription_response(response)
    }
}

fn transcription_url_multipart_parts(
    boundary: &str,
    audio_url: &XaiTranscriptionUrl,
    request: &XaiTranscriptionRequest,
) -> (Bytes, Bytes) {
    let mut prefix = transcription_option_fields(boundary, request);
    super::append_field(&mut prefix, boundary, "url", &audio_url.value);
    // append_field already ends the last form value with CRLF, so start the
    // closing delimiter directly to keep the URL value byte-for-byte intact.
    let suffix = Bytes::from(format!("--{boundary}--\r\n").into_bytes());
    (prefix.freeze(), suffix)
}
