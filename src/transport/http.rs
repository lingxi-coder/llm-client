use super::{HttpRequest, HttpResponse, StreamResponse, Transport, WebSocketSession};
use async_trait::async_trait;
use futures::StreamExt;
use lingxi_agent_api::protocol::LlmError;
use std::time::Duration;

/// Pooled HTTP/HTTPS transport backed by reqwest and rustls.
///
/// Execute requests inside a Tokio runtime with I/O and time enabled. Clones
/// share the connection pool. Redirects and automatic retries are disabled so
/// a provider cannot forward credentials or silently replay a billed request.
/// Connections have a 30-second timeout; [`HttpRequest::timeout`] controls the
/// total deadline, including streaming body reads. `None` means no total limit.
/// WebSocket sessions are not supported by this implementation.
#[derive(Clone)]
pub struct HttpTransport {
    client: reqwest::Client,
}

impl HttpTransport {
    pub fn new() -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|_| LlmError::Transport {
                message: "could not initialize HTTP client".into(),
            })?;
        Ok(Self { client })
    }

    async fn send(&self, req: HttpRequest) -> Result<reqwest::Response, LlmError> {
        let invalid = || LlmError::InvalidRequest {
            message: "invalid HTTP method, URL or headers".into(),
        };
        let method = reqwest::Method::from_bytes(req.method.as_bytes()).map_err(|_| invalid())?;
        let url = reqwest::Url::parse(&req.url).map_err(|_| invalid())?;
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

// Do not expose reqwest error strings: URLs, query credentials and underlying
// proxy errors may carry secrets. Preserve actionable categories instead.
fn network_error(err: reqwest::Error, streaming_body: bool) -> LlmError {
    if err.is_timeout() {
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
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        let response = self.send(req).await?;
        let status = response.status().as_u16();
        let headers = response_headers(&response);
        let body = response
            .bytes()
            .await
            .map_err(|err| network_error(err, false))?;
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn execute_no_follow(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.execute(req).await
    }

    async fn open_stream(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        let response = self.send(req).await?;
        Ok(StreamResponse {
            status: response.status().as_u16(),
            headers: response_headers(&response),
            body: response
                .bytes_stream()
                .map(|chunk| chunk.map_err(|err| network_error(err, true)))
                .boxed(),
        })
    }

    async fn open_responses_websocket_session(
        &self,
        _req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "built-in HTTP transport does not support WebSocket sessions".into(),
        })
    }
}
