//! `Transport` — the platform-provided HTTP/WebSocket client — and `LlmServices`.

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use lingxi_agent_api::protocol::LlmError;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
    pub timeout: Option<Duration>,
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

/// The shared transport. One implementation per platform (`LlmServices.http`);
/// provider crates never pick an HTTP client (gate 15).
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError>;

    /// `execute`, but a redirect comes back as the 3xx it is, `location`
    /// header and all, instead of being followed. A caller that must judge
    /// a redirect before going there (a fetch that stays on the host it was
    /// given) needs this; the default is for transports that cannot switch
    /// it off, and follows.
    async fn execute_no_follow(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.execute(req).await
    }

    async fn open_stream(&self, req: HttpRequest) -> Result<StreamResponse, LlmError>;

    async fn open_responses_websocket_session(
        &self,
        req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError>;
}

pub trait Clock: Send + Sync + 'static {
    fn now(&self) -> SystemTime;
}

pub trait UrlOpener: Send + Sync + 'static {
    fn open(&self, url: &str) -> Result<(), String>;
}

/// What the composition root hands the LLM layer: the two things every provider
/// needs, and nothing else. Not a service locator — a capability that needs
/// more takes it as its own constructor argument (review C6).
///
/// There is deliberately no credential store here. This crate does not hold or
/// fetch secrets; a credential arrives per request on `RequestOptions`, from
/// whoever owns it.
#[derive(Clone)]
pub struct LlmServices {
    pub http: Arc<dyn Transport>,
    pub clock: Arc<dyn Clock>,
}

// Gate 3.
const _: Option<&dyn Transport> = None;
const _: Option<&dyn WebSocketSession> = None;
const _: Option<&dyn Clock> = None;
const _: Option<&dyn UrlOpener> = None;
