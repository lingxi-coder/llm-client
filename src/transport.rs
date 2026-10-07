//! Built-in HTTP transport, injectable network interfaces and clocks.

mod http;
#[cfg(feature = "responses-websocket")]
pub(crate) mod websocket;
pub use http::HttpTransport;

use crate::protocol::LlmError;
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream, StreamExt};
use std::time::{Duration, SystemTime};

pub(crate) const MAX_ERROR_BODY_SIZE: usize = 64 * 1024;

/// Collect the useful prefix of an HTTP error body without letting a broken
/// connection erase an already-received status and headers.
pub(crate) async fn collect_error_body<S, E>(stream: S) -> Bytes
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    let mut stream = stream;
    let mut body = BytesMut::with_capacity(MAX_ERROR_BODY_SIZE);
    while body.len() < MAX_ERROR_BODY_SIZE {
        match stream.next().await {
            Some(Ok(chunk)) => {
                let take = chunk.len().min(MAX_ERROR_BODY_SIZE - body.len());
                body.extend_from_slice(&chunk[..take]);
            }
            Some(Err(_)) => break,
            None => break,
        }
    }
    body.freeze()
}

/// An outgoing request. `Debug` omits the URL, header values and body because
/// authenticators and callers may put credentials or private content in them.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
    pub timeout: Option<Duration>,
}

/// An outgoing request with a streamed body and an exact declared byte count.
/// The body is consumed once; transports must not replay it or follow redirects.
/// An optional Content-Length header must be unique and match `content_length`;
/// the built-in transport normalizes it and rejects Transfer-Encoding.
pub struct HttpStreamRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: BoxStream<'static, Result<Bytes, LlmError>>,
    pub content_length: u64,
    pub timeout: Option<Duration>,
}

impl std::fmt::Debug for HttpStreamRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpStreamRequest")
            .field("method", &self.method)
            .field("url", &"<redacted>")
            .field(
                "header_names",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("content_length", &self.content_length)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl std::fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &"<redacted>")
            .field(
                "header_names",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("body_bytes", &self.body.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    /// Preserve JSON provider payloads and non-JSON HTTP error text for policy
    /// decisions. Successful binary/SSE bodies do not become an error payload.
    pub fn payload_value(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|_| {
            let text = String::from_utf8_lossy(&self.body);
            let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
            serde_json::from_str(text).unwrap_or_else(|_| {
                if self.status >= 400 {
                    serde_json::Value::String(text.to_owned())
                } else {
                    serde_json::Value::Null
                }
            })
        })
    }
}

/// A streamed response. The status and headers arrive whole, before the body,
/// and the body arrives in arbitrary byte chunks.
///
/// Returning the frames alone would be simpler and was what this trait did, but
/// it put every response header out of reach on the one path an agent turn
/// actually takes: `retry-after` on a 429, and the rate-limit and quota headers
/// a provider reports its plan state through. A streamed 429 could not even say
/// how long to wait.
pub struct StreamResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl StreamResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }
}

/// Header names are case-insensitive, and providers do not agree on the case.
fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// A raw HTTP transport. Implementations must not follow redirects or retry
/// requests automatically. The body contains arbitrary byte chunks; dropping
/// it must cancel the response read. Use [`HttpExecutor`] for collection and
/// runtime-enforced deadlines, including with custom transports.
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError>;

    /// Override to support large streaming uploads. The default fails before
    /// consuming input, so custom transports never silently buffer it.
    async fn send_stream(&self, _req: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "transport does not support streaming request bodies".into(),
        })
    }

    /// Open a reusable Responses connection without generating a response.
    /// Hosts may inject their native WebSocket stack alongside HTTP.
    async fn connect_websocket(
        &self,
        _handshake: HttpRequest,
    ) -> Result<Box<dyn WebSocketConnection>, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "this transport does not support WebSocket connections".into(),
        })
    }
}

/// One reusable connection. Each body item is one complete JSON text message.
/// Implementations must not reconnect or resend a generation automatically.
#[async_trait]
pub trait WebSocketConnection: Send {
    async fn send(&mut self, payload: Bytes) -> Result<StreamResponse, LlmError>;
    async fn close(&mut self) -> Result<(), LlmError>;
}

/// Shared request execution independent of the concrete HTTP backend.
/// Limits are enforced while reading, rather than after buffering a response.
#[derive(Clone, Copy)]
pub struct HttpExecutor<'a> {
    transport: &'a dyn Transport,
    deadline: crate::runtime::Deadline,
}

impl<'a> HttpExecutor<'a> {
    pub fn new(transport: &'a dyn Transport) -> Self {
        Self {
            transport,
            deadline: Default::default(),
        }
    }

    pub(crate) fn with_deadline(mut self, deadline: crate::runtime::Deadline) -> Self {
        self.deadline = deadline;
        self
    }

    pub async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.send_with_dispatch_report(request)
            .await
            .map_err(|(error, _)| error)
    }

    /// The flag is false when the deadline expired before the transport
    /// future was polled, so callers can distinguish an unsent attempt.
    pub(crate) async fn send_with_dispatch_report(
        &self,
        mut request: HttpRequest,
    ) -> Result<StreamResponse, (LlmError, bool)> {
        let deadline = self.deadline.cap(request.timeout);
        request.timeout = deadline.remaining().map_err(|error| (error, false))?;
        let mut transport_attempted = false;
        let outcome = deadline
            .run(async {
                transport_attempted = true;
                self.transport.send(request).await
            })
            .await;
        match outcome {
            Ok(Ok(response)) => Ok(Self::bound_response(response, deadline)),
            Ok(Err(error)) | Err(error) => Err((error, transport_attempted)),
        }
    }

    pub async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let deadline = self.deadline.cap(request.timeout);
        request.timeout = deadline.remaining()?;
        let response = deadline.run(self.transport.send_stream(request)).await??;
        Ok(Self::bound_response(response, deadline))
    }

    pub(crate) fn bound_response(
        response: StreamResponse,
        deadline: crate::runtime::Deadline,
    ) -> StreamResponse {
        let body = futures::stream::unfold(Some(response.body), move |state| async move {
            let mut body = state?;
            match deadline.run(body.next()).await {
                Ok(Some(Ok(chunk))) => Some((Ok(chunk), Some(body))),
                Ok(Some(Err(error))) | Err(error) => Some((Err(error), None)),
                Ok(None) => None,
            }
        })
        .boxed();
        StreamResponse {
            status: response.status,
            headers: response.headers,
            body,
        }
    }

    pub async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.collect(request, None).await
    }

    pub async fn execute_bounded(
        &self,
        request: HttpRequest,
        limit: usize,
    ) -> Result<HttpResponse, LlmError> {
        self.collect(request, Some(limit)).await
    }

    pub async fn execute_stream_bounded(
        &self,
        request: HttpStreamRequest,
        limit: usize,
    ) -> Result<HttpResponse, LlmError> {
        let response = self.send_stream(request).await?;
        Self::collect_response(response, Some(limit)).await
    }

    async fn collect(
        &self,
        request: HttpRequest,
        limit: Option<usize>,
    ) -> Result<HttpResponse, LlmError> {
        let response = self.send(request).await?;
        Self::collect_response(response, limit).await
    }

    pub(crate) async fn collect_response(
        response: StreamResponse,
        limit: Option<usize>,
    ) -> Result<HttpResponse, LlmError> {
        Self::collect_response_with_partial(response, limit)
            .await
            .map_err(|(error, _)| error)
    }

    /// Keep the status, headers, and bytes already received when a successful
    /// response body stops early. Decision calls use this to avoid replaying a
    /// generation whose outcome may already have been produced.
    pub(crate) async fn collect_response_with_partial(
        response: StreamResponse,
        limit: Option<usize>,
    ) -> Result<HttpResponse, (LlmError, HttpResponse)> {
        let status = response.status;
        let headers = response.headers;
        let body = if (200..300).contains(&response.status) {
            let mut chunks = response.body;
            let mut body = BytesMut::new();
            while let Some(chunk) = chunks.next().await {
                let chunk = chunk.map_err(|error| {
                    let error = match error {
                        LlmError::StreamInterrupted { message } => LlmError::Transport { message },
                        other => other,
                    };
                    (
                        error,
                        HttpResponse {
                            status,
                            headers: headers.clone(),
                            body: body.clone().freeze(),
                        },
                    )
                })?;
                if limit.is_some_and(|limit| chunk.len() > limit.saturating_sub(body.len())) {
                    return Err((
                        LlmError::Transport {
                            message: "HTTP response body exceeds operation limit".into(),
                        },
                        HttpResponse {
                            status,
                            headers,
                            body: body.freeze(),
                        },
                    ));
                }
                body.extend_from_slice(&chunk);
            }
            body.freeze()
        } else {
            collect_error_body(response.body).await
        };
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

impl From<HttpResponse> for StreamResponse {
    fn from(response: HttpResponse) -> Self {
        Self {
            status: response.status,
            headers: response.headers,
            body: futures::stream::once(async move { Ok(response.body) }).boxed(),
        }
    }
}

pub trait Clock: Send + Sync + 'static {
    fn now(&self) -> SystemTime;
}

/// Wall-clock time for price schedules in normal applications.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

const _: Option<&dyn Transport> = None;
const _: Option<&dyn Clock> = None;

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTransport(AtomicUsize);

    #[async_trait::async_trait]
    impl Transport for CountingTransport {
        async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Err(LlmError::Transport {
                message: "unexpected dispatch".into(),
            })
        }
    }

    #[tokio::test]
    async fn expired_deadline_does_not_mark_transport_attempted() {
        let transport = CountingTransport(AtomicUsize::new(0));
        let expired = std::time::Instant::now() - Duration::from_secs(1);
        let request = HttpRequest {
            method: "POST".into(),
            url: "https://example.test/decision".into(),
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        };
        let outcome = HttpExecutor::new(&transport)
            .with_deadline(crate::runtime::Deadline::at(Some(expired)))
            .send_with_dispatch_report(request)
            .await;
        assert!(matches!(
            outcome,
            Err((LlmError::TransportTimeout { .. }, false))
        ));
        assert_eq!(transport.0.load(Ordering::Relaxed), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provider_payload_preserves_error_text_json_scalars_and_success_body_ownership() {
        for (status, bytes, expected) in [
            (
                400,
                b"could not process image".as_slice(),
                serde_json::json!("could not process image"),
            ),
            (
                422,
                b"\xff field".as_slice(),
                serde_json::json!("\u{fffd} field"),
            ),
            (500, b"".as_slice(), serde_json::json!("")),
            (
                400,
                b"{\"message\":\"known\"}".as_slice(),
                serde_json::json!({"message":"known"}),
            ),
            (400, b"false".as_slice(), serde_json::json!(false)),
            (
                400,
                b"\xef\xbb\xbfcould not process image".as_slice(),
                serde_json::json!("could not process image"),
            ),
            (
                400,
                b"{\"message\":\"\xff\"}".as_slice(),
                serde_json::json!({"message":"\u{fffd}"}),
            ),
            (200, b"data: event\n\n".as_slice(), serde_json::Value::Null),
            (
                200,
                b"{\"content\":[]}".as_slice(),
                serde_json::json!({"content":[]}),
            ),
        ] {
            let response = HttpResponse {
                status,
                headers: vec![],
                body: Bytes::copy_from_slice(bytes),
            };
            assert_eq!(response.payload_value(), expected);
            assert_eq!(response.body.as_ref(), bytes);
        }
    }
}
