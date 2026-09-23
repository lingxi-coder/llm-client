//! Built-in HTTP transport, injectable network interfaces and clocks.

mod http;
pub use http::HttpTransport;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream, StreamExt};
use lingxi_agent_api::protocol::LlmError;
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
}

/// A streamed response. The status and headers arrive whole, before the body,
/// and the body arrives in frames.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    Text(String),
    Binary(Bytes),
}

#[async_trait]
pub trait WebSocketSession: Send {
    async fn send(&mut self, msg: WsMessage) -> Result<(), LlmError>;
    async fn recv(&mut self) -> Result<Option<WsMessage>, LlmError>;
    async fn close(&mut self);
}

/// The shared transport. Use [`HttpTransport`] for built-in networking or
/// inject a platform-specific implementation through [`crate::LlmClientBuilder::with_transport`].
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError>;

    /// Return redirects as 3xx responses without following them. Custom
    /// transports must opt in explicitly before receiving account credentials.
    async fn execute_no_follow(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "transport does not implement no-redirect requests".into(),
        })
    }

    /// No-redirect request with a maximum successful response-body size.
    /// The built-in transport enforces this while reading; custom transports
    /// should do the same rather than relying on this post-read fallback.
    async fn execute_no_follow_bounded(
        &self,
        req: HttpRequest,
        max_body_size: usize,
    ) -> Result<HttpResponse, LlmError> {
        let response = self.execute_no_follow(req).await?;
        if response.body.len() > max_body_size {
            return Err(LlmError::Transport {
                message: "HTTP response body exceeds account limit".into(),
            });
        }
        Ok(response)
    }

    async fn open_stream(&self, req: HttpRequest) -> Result<StreamResponse, LlmError>;

    /// Open a streaming request without following redirects. Custom transports
    /// must opt in explicitly before sending credentialed requests.
    async fn open_stream_no_follow(&self, _req: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "transport does not implement no-redirect streaming requests".into(),
        })
    }

    async fn open_responses_websocket_session(
        &self,
        req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError>;
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

pub trait UrlOpener: Send + Sync + 'static {
    fn open(&self, url: &str) -> Result<(), String>;
}

// Gate 3.
const _: Option<&dyn Transport> = None;
const _: Option<&dyn WebSocketSession> = None;
const _: Option<&dyn Clock> = None;
const _: Option<&dyn UrlOpener> = None;
