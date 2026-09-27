//! Scoped transcription through Zhipu's official GLM-ASR model server example.
//!
//! The currently published model repository documents GLM-ASR-Nano through a
//! self-hosted SGLang server that exposes an OpenAI-compatible chat endpoint.
//! This module implements only that documented chat request. It does not claim
//! that Zhipu hosts a cloud transcription endpoint, and it does not guess a TTS
//! route from third-party examples.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, Secret},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use thiserror::Error;
use url::Url;

const MODEL: &str = "glm-asr";
const TRANSCRIPTION_PROMPT: &str = "Please transcribe this audio into text";
const MAX_AUDIO_REFERENCE_CHARS: usize = 4096;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Caller-selected OpenAI-compatible `/v1` root for the self-hosted GLM-ASR
/// SGLang server. The official repository example uses `http://127.0.0.1:8000/v1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAsrRoute {
    endpoint: String,
}

impl GlmAsrRoute {
    pub fn new(endpoint: impl Into<String>) -> Result<Self, GlmAsrError> {
        let route = Self {
            endpoint: endpoint.into(),
        };
        route.validate()?;
        Ok(route)
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn validate(&self) -> Result<(), GlmAsrError> {
        let url = Url::parse(&self.endpoint).map_err(|_| invalid("invalid GLM-ASR base URL"))?;
        let local_http = url.scheme() == "http"
            && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
        if (url.scheme() != "https" && !local_http)
            || url.host_str().is_none()
            || url.username() != ""
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !url.path().trim_end_matches('/').ends_with("/v1")
        {
            return Err(invalid(
                "GLM-ASR base URL must be an HTTPS /v1 endpoint or a loopback HTTP /v1 endpoint",
            ));
        }
        Ok(())
    }

    fn fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(self.endpoint.trim_end_matches('/'))
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.endpoint.trim_end_matches('/'))
    }
}

/// Stable caller-owned account and endpoint identity for one transcription
/// service. The endpoint fingerprint prevents reusing this scope with another
/// local server or proxy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAsrScope {
    profile_name: String,
    account_scope: String,
    endpoint_fingerprint: String,
}

impl GlmAsrScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        route: &GlmAsrRoute,
    ) -> Result<Self, GlmAsrError> {
        route.validate()?;
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            endpoint_fingerprint: route.fingerprint(),
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), GlmAsrError> {
        if self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || self.endpoint_fingerprint.trim().is_empty()
        {
            return Err(invalid(
                "GLM-ASR scope requires a profile, non-secret account scope, and endpoint",
            ));
        }
        Ok(())
    }
}

/// One URL or server-readable path for the SGLang server's documented
/// `audio_url.url` content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAsrRequest {
    pub audio_url: String,
}

impl GlmAsrRequest {
    pub fn new(audio_url: impl Into<String>) -> Result<Self, GlmAsrError> {
        let request = Self {
            audio_url: audio_url.into(),
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), GlmAsrError> {
        if self.audio_url.trim().is_empty()
            || self.audio_url.chars().count() > MAX_AUDIO_REFERENCE_CHARS
            || self.audio_url.chars().any(char::is_control)
        {
            return Err(invalid(
                "GLM-ASR audio reference must contain 1 to 4096 non-control characters",
            ));
        }
        Ok(())
    }
}

/// Decoded text returned by the GLM-ASR chat completion endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmAsrTranscription {
    pub text: String,
    pub model: Option<String>,
    pub response_id: Option<String>,
    pub request_id: Option<String>,
    pub scope: GlmAsrScope,
    /// Complete provider response for forward-compatible access to usage and
    /// fields not represented by this small transcription contract.
    pub native: Value,
}

/// How far one GLM-ASR request may have progressed. This service never retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmAsrDispatch {
    /// Local validation prevented the HTTP request.
    NotSent,
    /// The server returned an explicit client error.
    Rejected,
    /// The server may have processed the request, but no successful result was confirmed.
    Unknown,
    /// The server returned success but its response was not a usable transcript.
    Accepted,
}

#[derive(Debug, Error)]
pub enum GlmAsrError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid GLM-ASR input: {0}")]
    InvalidInput(String),
    #[error("GLM-ASR returned HTTP {status}: {message}")]
    Provider {
        status: u16,
        message: String,
        request_id: Option<String>,
        dispatch: GlmAsrDispatch,
        native: Box<Value>,
    },
    #[error("GLM-ASR accepted the request but returned an invalid response: {message}")]
    InvalidResponse {
        message: String,
        request_id: Option<String>,
        native: Box<Value>,
    },
    #[error("GLM-ASR request outcome is unknown: {source}")]
    OutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

impl GlmAsrError {
    pub fn dispatch(&self) -> GlmAsrDispatch {
        match self {
            Self::Llm(_) | Self::InvalidInput(_) => GlmAsrDispatch::NotSent,
            Self::Provider { dispatch, .. } => *dispatch,
            Self::InvalidResponse { .. } => GlmAsrDispatch::Accepted,
            Self::OutcomeUnknown { .. } => GlmAsrDispatch::Unknown,
        }
    }
}

/// Typed adapter for the self-hosted GLM-ASR SGLang example.
pub struct GlmAsrService<'a> {
    http: &'a dyn Transport,
    credential: Secret<String>,
    route: GlmAsrRoute,
    scope: GlmAsrScope,
}

impl<'a> GlmAsrService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        credential: Secret<String>,
        route: GlmAsrRoute,
        scope: GlmAsrScope,
    ) -> Result<Self, GlmAsrError> {
        route.validate()?;
        scope.validate()?;
        if credential.expose_secret().trim().is_empty() {
            return Err(invalid("GLM-ASR bearer credential must be non-empty"));
        }
        if scope.endpoint_fingerprint != route.fingerprint() {
            return Err(LlmError::PermissionDenied {
                message: "GLM-ASR scope belongs to another endpoint".into(),
            }
            .into());
        }
        Ok(Self {
            http,
            credential,
            route,
            scope,
        })
    }

    pub fn scope(&self) -> &GlmAsrScope {
        &self.scope
    }

    /// Transcribe one server-readable audio reference through a single chat
    /// completion request. The operation is not retried automatically.
    pub async fn transcribe(
        &self,
        request: &GlmAsrRequest,
    ) -> Result<GlmAsrTranscription, GlmAsrError> {
        request.validate()?;
        let body = json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": request.audio_url}
                    },
                    {"type": "text", "text": TRANSCRIPTION_PROMPT}
                ]
            }],
            "max_tokens": 1024
        });
        let request = HttpRequest {
            method: "POST".into(),
            url: self.route.chat_completions_url(),
            headers: vec![
                (
                    "authorization".into(),
                    format!("Bearer {}", self.credential.expose_secret()),
                ),
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Bytes::from(
                serde_json::to_vec(&body)
                    .map_err(|_| invalid("GLM-ASR request cannot be encoded as JSON"))?,
            ),
            timeout: Some(DEFAULT_TIMEOUT),
        };
        let response = HttpExecutor::new(self.http)
            .execute_bounded(request, MAX_RESPONSE_BYTES)
            .await
            .map_err(map_transport_error)?;
        decode_response(response, &self.scope)
    }
}

fn decode_response(
    response: HttpResponse,
    scope: &GlmAsrScope,
) -> Result<GlmAsrTranscription, GlmAsrError> {
    let header_request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let parsed = serde_json::from_slice::<Value>(&response.body);
    let native = parsed
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    let request_id = header_request_id.or_else(|| {
        native
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    if !(200..300).contains(&response.status) {
        let dispatch = if (400..500).contains(&response.status) {
            GlmAsrDispatch::Rejected
        } else {
            GlmAsrDispatch::Unknown
        };
        let message = native
            .pointer("/error/message")
            .and_then(Value::as_str)
            .or_else(|| native.get("message").and_then(Value::as_str))
            .unwrap_or("GLM-ASR request failed")
            .to_owned();
        return Err(GlmAsrError::Provider {
            status: response.status,
            message,
            request_id,
            dispatch,
            native: Box::new(native),
        });
    }
    let text = native
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| GlmAsrError::InvalidResponse {
            message: "response omitted choices[0].message.content".into(),
            request_id: request_id.clone(),
            native: Box::new(native.clone()),
        })?
        .to_owned();
    Ok(GlmAsrTranscription {
        text,
        model: native
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        response_id: native.get("id").and_then(Value::as_str).map(str::to_owned),
        request_id,
        scope: scope.clone(),
        native,
    })
}

fn map_transport_error(source: LlmError) -> GlmAsrError {
    match source {
        LlmError::Transport { .. }
        | LlmError::TransportTimeout { .. }
        | LlmError::StreamInterrupted { .. } => GlmAsrError::OutcomeUnknown { source },
        other => GlmAsrError::Llm(other),
    }
}

fn invalid(message: impl Into<String>) -> GlmAsrError {
    GlmAsrError::InvalidInput(message.into())
}
