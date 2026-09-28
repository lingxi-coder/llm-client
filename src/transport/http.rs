use super::{HttpRequest, HttpStreamRequest, StreamResponse, Transport};
use crate::protocol::LlmError;
use async_trait::async_trait;
use futures::StreamExt;
use std::time::Duration;

/// Pooled HTTP/HTTPS transport backed by reqwest and rustls.
///
/// Execute requests inside a Tokio runtime with I/O and time enabled. Clones
/// share the connection pool. Redirects and automatic retries are disabled so
/// a provider cannot forward credentials or silently replay a billed request.
/// Connections have a 30-second timeout and reads have a 60-second idle
/// timeout by default. [`HttpRequest::timeout`] controls the total deadline,
/// including streaming body reads.
/// Responses WebSocket sessions are available with `responses-websocket`.
#[derive(Clone)]
pub struct HttpTransport {
    pub(super) client: reqwest::Client,
    pub(super) read_timeout: Duration,
}

impl HttpTransport {
    pub fn new() -> Result<Self, LlmError> {
        Self::with_read_timeout(Duration::from_secs(60))
    }

    /// Build with a custom timeout for each idle response-body read.
    pub fn with_read_timeout(read_timeout: Duration) -> Result<Self, LlmError> {
        let mut transport =
            Self::with_client_configurator(|builder| builder.read_timeout(read_timeout))?;
        transport.read_timeout = read_timeout;
        Ok(transport)
    }

    /// Customize connection settings, such as a private CA, client TLS identity
    /// or proxy, while retaining this transport's request execution policy.
    ///
    /// The configurator receives the default connection and idle-read timeouts.
    /// Redirects and automatic retries are disabled after it returns, so these
    /// policies cannot accidentally replay a request or forward credentials.
    pub fn with_client_configurator(
        configure: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
    ) -> Result<Self, LlmError> {
        let builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60));
        let client = configure(builder)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| LlmError::Transport {
                message: "could not initialize HTTP client".into(),
            })?;
        Ok(Self {
            client,
            read_timeout: Duration::from_secs(60),
        })
    }

    async fn send_request(&self, req: HttpRequest) -> Result<reqwest::Response, LlmError> {
        let invalid = || LlmError::InvalidRequest {
            message: "invalid HTTP method, URL or headers".into(),
        };
        let method = reqwest::Method::from_bytes(req.method.as_bytes()).map_err(|_| invalid())?;
        let url = url::Url::parse(&req.url).map_err(|_| invalid())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid());
        }
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in req.headers {
            let name =
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid())?;
            let mut value =
                reqwest::header::HeaderValue::from_str(&value).map_err(|_| invalid())?;
            // Custom authenticators may use any header for secrets.
            value.set_sensitive(true);
            headers.append(name, value);
        }
        let mut request = self
            .client
            .request(method, url)
            .headers(headers)
            .body(req.body);
        if let Some(timeout) = req.timeout {
            request = request.timeout(timeout);
        }
        request
            .send()
            .await
            .map_err(|err| network_error(err, false))
    }

    async fn send_stream_request(
        &self,
        req: HttpStreamRequest,
    ) -> Result<reqwest::Response, LlmError> {
        let invalid = || LlmError::InvalidRequest {
            message: "invalid streaming HTTP method, URL or headers".into(),
        };
        let method = reqwest::Method::from_bytes(req.method.as_bytes()).map_err(|_| invalid())?;
        let url = url::Url::parse(&req.url).map_err(|_| invalid())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid());
        }
        let mut headers = reqwest::header::HeaderMap::new();
        let mut saw_content_length = false;
        for (name, value) in req.headers {
            let name =
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid())?;
            if name == reqwest::header::TRANSFER_ENCODING {
                return Err(invalid());
            }
            if name == reqwest::header::CONTENT_LENGTH {
                // Resumable provider uploads include their exact byte count in
                // both places. Normalize an agreeing header; reject ambiguous
                // framing before polling the single-use body.
                if saw_content_length || value != req.content_length.to_string() {
                    return Err(invalid());
                }
                saw_content_length = true;
                continue;
            }
            let mut value =
                reqwest::header::HeaderValue::from_str(&value).map_err(|_| invalid())?;
            value.set_sensitive(true);
            headers.append(name, value);
        }
        headers.insert(
            reqwest::header::CONTENT_LENGTH,
            reqwest::header::HeaderValue::from_str(&req.content_length.to_string())
                .map_err(|_| invalid())?,
        );
        let mut request = self
            .client
            .request(method, url)
            .headers(headers)
            .body(reqwest::Body::wrap_stream(req.body));
        if let Some(timeout) = req.timeout {
            request = request.timeout(timeout);
        }
        request
            .send()
            .await
            .map_err(|err| network_error(err, false))
    }
}

fn response_headers(response: &reqwest::Response) -> Vec<(String, String)> {
    response
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect()
}

fn certificate_error(error: &dyn std::error::Error) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        let message = error.to_string().to_ascii_lowercase();
        if message.contains("certificate") || message.contains("unknownissuer") {
            return true;
        }
        current = error.source();
    }
    false
}

// Do not expose reqwest error strings: URLs, query credentials and underlying
// proxy errors may carry secrets. Preserve actionable categories instead.
pub(super) fn network_error(err: reqwest::Error, streaming_body: bool) -> LlmError {
    if certificate_error(&err) {
        LlmError::TlsCert {
            message: "TLS certificate validation failed".into(),
        }
    } else if err.is_timeout() {
        LlmError::TransportTimeout {
            message: "HTTP request timed out".into(),
        }
    } else if err.is_builder() {
        LlmError::InvalidRequest {
            message: "could not build HTTP request".into(),
        }
    } else if streaming_body {
        LlmError::StreamInterrupted {
            message: "HTTP response stream interrupted".into(),
        }
    } else {
        LlmError::Transport {
            message: "HTTP connection or response read failed".into(),
        }
    }
}

#[async_trait]
impl Transport for HttpTransport {
    #[cfg(feature = "responses-websocket")]
    async fn connect_websocket(
        &self,
        request: HttpRequest,
    ) -> Result<Box<dyn super::WebSocketConnection>, LlmError> {
        super::websocket::responses(self, request).await
    }

    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        let response = self.send_request(req).await?;
        Ok(StreamResponse {
            status: response.status().as_u16(),
            headers: response_headers(&response),
            body: response
                .bytes_stream()
                .map(|chunk| chunk.map_err(|err| network_error(err, true)))
                .boxed(),
        })
    }

    async fn send_stream(&self, req: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let response = self.send_stream_request(req).await?;
        Ok(StreamResponse {
            status: response.status().as_u16(),
            headers: response_headers(&response),
            body: response
                .bytes_stream()
                .map(|chunk| chunk.map_err(|err| network_error(err, true)))
                .boxed(),
        })
    }
}
