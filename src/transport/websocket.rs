//! Shared HTTP upgrade path: HTTP, Responses and realtime use the same TLS/proxy policy.
use super::{HttpRequest, HttpTransport, StreamResponse, WebSocketConnection};
use crate::protocol::LlmError;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_tungstenite::{
    tungstenite::{
        client::IntoClientRequest,
        handshake::derive_accept_key,
        protocol::{Role, WebSocketConfig},
        Message,
    },
    WebSocketStream,
};

pub(crate) type Socket = WebSocketStream<reqwest::Upgraded>;
fn invalid() -> LlmError {
    LlmError::InvalidRequest {
        message: "invalid WebSocket handshake".into(),
    }
}
fn interrupted() -> LlmError {
    LlmError::StreamInterrupted {
        message: "WebSocket stream interrupted".into(),
    }
}
fn timeout() -> LlmError {
    LlmError::TransportTimeout {
        message: "WebSocket operation timed out".into(),
    }
}

pub(crate) async fn connect(
    transport: &HttpTransport,
    request: HttpRequest,
    max_frame_bytes: usize,
) -> Result<(Socket, Vec<(String, String)>), LlmError> {
    let mut url = url::Url::parse(&request.url).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "ws" | "wss")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || max_frame_bytes == 0
    {
        return Err(invalid());
    }
    let handshake = url.as_str().into_client_request().map_err(|_| invalid())?;
    let key = handshake
        .headers()
        .get("sec-websocket-key")
        .ok_or_else(invalid)?;
    let expected = derive_accept_key(key.as_bytes());
    let scheme = if url.scheme() == "wss" {
        "https"
    } else {
        "http"
    };
    url.set_scheme(scheme).map_err(|_| invalid())?;
    let mut headers = handshake.headers().clone();
    for (name, value) in request.headers {
        let name =
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid())?;
        // Generated handshake fields must not be overwritten by credentials or callers.
        if matches!(
            name.as_str(),
            "host"
                | "connection"
                | "upgrade"
                | "sec-websocket-key"
                | "sec-websocket-version"
                | "sec-websocket-extensions"
        ) {
            return Err(invalid());
        }
        let mut value = reqwest::header::HeaderValue::from_str(&value).map_err(|_| invalid())?;
        value.set_sensitive(true);
        headers.append(name, value);
    }
    let offered_protocols = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let deadline = request.timeout.unwrap_or(Duration::from_secs(15));
    tokio::time::timeout(deadline, async {
        let response = transport
            .client
            .get(url)
            .version(reqwest::Version::HTTP_11)
            .headers(headers)
            .send()
            .await
            .map_err(|e| super::http::network_error(e, false))?;
        if response.status() != reqwest::StatusCode::SWITCHING_PROTOCOLS {
            let status = response.status().as_u16();
            let message = format!("WebSocket handshake rejected with HTTP {status}");
            return Err(match status {
                401 => LlmError::Authentication { message },
                403 => LlmError::PermissionDenied { message },
                429 => LlmError::RateLimited {
                    message,
                    retry_after: response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(Duration::from_secs),
                },
                529 => LlmError::Overloaded { message },
                500..=599 => LlmError::ProviderInternal { message },
                // Preserve 426 for the session's explicit pre-dispatch HTTP fallback.
                _ => LlmError::Transport { message },
            });
        }
        let h = response.headers();
        let contains = |name: &str, token: &str| {
            h.get(name)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.split(',').any(|v| v.trim().eq_ignore_ascii_case(token)))
        };
        if !contains("connection", "upgrade")
            || !contains("upgrade", "websocket")
            || h.get("sec-websocket-accept").and_then(|v| v.to_str().ok())
                != Some(expected.as_str())
            || h.contains_key("sec-websocket-extensions")
        {
            return Err(invalid());
        }
        if let Some(protocol) = h.get("sec-websocket-protocol") {
            let selected = protocol.to_str().map_err(|_| invalid())?;
            if !offered_protocols
                .as_ref()
                .is_some_and(|p| p.split(',').any(|p| p.trim() == selected))
            {
                return Err(invalid());
            }
        }
        let headers = h
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let upgraded = response
            .upgrade()
            .await
            .map_err(|e| super::http::network_error(e, false))?;
        let socket = WebSocketStream::from_raw_socket(
            upgraded,
            Role::Client,
            Some(WebSocketConfig {
                max_message_size: Some(max_frame_bytes),
                max_frame_size: Some(max_frame_bytes),
                ..Default::default()
            }),
        )
        .await;
        Ok((socket, headers))
    })
    .await
    .map_err(|_| timeout())?
}

/// The server ends a connection after 60 minutes. Like Codex's pooled
/// sockets, an idle connection is retired at 55 so no send meets the limit.
const MAX_CONNECTION_AGE: Duration = Duration::from_secs(55 * 60);

struct ResponsesConnection {
    // Taking the socket exclusively prevents overlapping generations. Early drop
    // discards it; only a terminal provider event makes it reusable.
    socket: Arc<Mutex<Slot>>,
    headers: Vec<(String, String)>,
    cancel: tokio::sync::watch::Sender<bool>,
    idle_timeout: Option<Duration>,
    opened: tokio::time::Instant,
}

enum Slot {
    /// Between generations a reader owns the socket so Pings are answered
    /// and a peer close is noticed before the next send, as in Codex.
    Idle(IdleReader),
    /// A generation's response stream owns the socket.
    Busy,
    Closed,
}

struct IdleReader {
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Option<Socket>>,
}

impl IdleReader {
    fn spawn(mut socket: Socket, pong_limit: Duration) -> Self {
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = &mut stopped => return Some(socket),
                    frame = socket.next() => match frame {
                        // Reading a Ping queues its Pong; flushing sends it.
                        Some(Ok(Message::Ping(_))) => {
                            if !matches!(
                                tokio::time::timeout(pong_limit, socket.flush()).await,
                                Ok(Ok(()))
                            ) {
                                return None;
                            }
                        }
                        // Nothing is in flight, so only a connection error matters.
                        Some(Ok(Message::Text(text)))
                            if event_type(&text).as_deref() != Some("error") => {}
                        Some(Ok(Message::Pong(_) | Message::Binary(_) | Message::Frame(_))) => {}
                        _ => return None,
                    },
                }
            }
        });
        Self { stop, task }
    }

    /// Stop reading and take the socket back unless the peer closed it.
    async fn resume(self) -> Option<Socket> {
        let _ = self.stop.send(());
        self.task.await.ok().flatten()
    }
}

/// One generation's hold on the socket. A terminal event hands the socket
/// back to an idle reader; any other end leaves the connection closed.
struct Generation {
    socket: Option<Socket>,
    slot: std::sync::Weak<Mutex<Slot>>,
    pong_limit: Duration,
    returned: bool,
}

impl Generation {
    fn finish(mut self) {
        self.returned = true;
        if let (Some(socket), Some(slot)) = (self.socket.take(), self.slot.upgrade()) {
            *slot.lock().expect("socket") = Slot::Idle(IdleReader::spawn(socket, self.pong_limit));
        }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        if self.returned {
            return;
        }
        if let Some(slot) = self.slot.upgrade() {
            if let Ok(mut slot) = slot.lock() {
                *slot = Slot::Closed;
            }
        }
    }
}

fn event_type(text: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()?
        .get("type")?
        .as_str()
        .map(str::to_owned)
}
pub(crate) async fn responses(
    transport: &HttpTransport,
    mut request: HttpRequest,
) -> Result<Box<dyn WebSocketConnection>, LlmError> {
    let mut url = url::Url::parse(&request.url).map_err(|_| invalid())?;
    let scheme = match url.scheme() {
        "https" => "wss",
        "http" => "ws",
        "wss" => "wss",
        "ws" => "ws",
        _ => return Err(invalid()),
    };
    url.set_scheme(scheme).map_err(|_| invalid())?;
    request.url = url.to_string();
    let beta = "responses_websockets=2026-02-06";
    if let Some((_, value)) = request
        .headers
        .iter_mut()
        .find(|(k, _)| k.eq_ignore_ascii_case("openai-beta"))
    {
        if !value.split(',').any(|v| v.trim() == beta) {
            value.push(',');
            value.push_str(beta);
        }
    } else {
        request.headers.push(("OpenAI-Beta".into(), beta.into()));
    }
    let (socket, headers) = connect(transport, request, 16 * 1024 * 1024).await?;
    let pong_limit = transport.read_timeout.unwrap_or(Duration::from_secs(60));
    Ok(Box::new(ResponsesConnection {
        socket: Arc::new(Mutex::new(Slot::Idle(IdleReader::spawn(socket, pong_limit)))),
        headers,
        cancel: tokio::sync::watch::channel(false).0,
        idle_timeout: transport.read_timeout,
        opened: tokio::time::Instant::now(),
    }))
}
#[async_trait]
impl WebSocketConnection for ResponsesConnection {
    async fn send(&mut self, payload: Bytes) -> Result<StreamResponse, LlmError> {
        let text = String::from_utf8(payload.to_vec()).map_err(|_| invalid())?;
        let reader = {
            let mut slot = self.socket.lock().expect("socket");
            match std::mem::replace(&mut *slot, Slot::Busy) {
                Slot::Idle(reader) => reader,
                other => {
                    *slot = other;
                    return Err(LlmError::InvalidRequest {
                        message: "WebSocket connection is busy or no longer usable".into(),
                    });
                }
            }
        };
        let pong_limit = self.idle_timeout.unwrap_or(Duration::from_secs(60));
        // From here every early return, including a dropped future, closes the slot.
        let mut generation = Generation {
            socket: None,
            slot: Arc::downgrade(&self.socket),
            pong_limit,
            returned: false,
        };
        let socket = generation
            .socket
            .insert(reader.resume().await.ok_or_else(interrupted)?);
        tokio::time::timeout(pong_limit, socket.send(Message::Text(text)))
            .await
            .map_err(|_| timeout())?
            .map_err(|_| interrupted())?;
        let cancel = self.cancel.subscribe();
        let idle_timeout = self.idle_timeout;
        let body = futures::stream::unfold(
            (Some(generation), cancel),
            move |(generation, mut cancel)| async move {
                let mut generation = generation?;
                loop {
                    let socket = generation.socket.as_mut()?;
                    let frame = match guarded_io(idle_timeout, &mut cancel, async {
                        socket
                            .next()
                            .await
                            .ok_or_else(interrupted)?
                            .map_err(|_| interrupted())
                    })
                    .await
                    {
                        Ok(frame) => frame,
                        Err(error) => return Some((Err(error), (None, cancel))),
                    };
                    match frame {
                        Message::Text(text) => {
                            let terminal = matches!(
                                event_type(&text).as_deref(),
                                Some(
                                    "response.completed"
                                        | "response.incomplete"
                                        | "response.failed"
                                        | "error"
                                )
                            );
                            let next = if terminal {
                                generation.finish();
                                None
                            } else {
                                Some(generation)
                            };
                            return Some((Ok(Bytes::from(text)), (next, cancel)));
                        }
                        Message::Ping(_) => {
                            if let Err(error) = guarded_io(Some(pong_limit), &mut cancel, async {
                                socket.flush().await.map_err(|_| interrupted())
                            })
                            .await
                            {
                                return Some((Err(error), (None, cancel)));
                            }
                        }
                        Message::Pong(_) => {}
                        _ => return Some((Err(interrupted()), (None, cancel))),
                    }
                }
            },
        )
        .boxed();
        Ok(StreamResponse {
            status: 200,
            headers: self.headers.clone(),
            body,
        })
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        let _ = self.cancel.send(true);
        let slot = std::mem::replace(&mut *self.socket.lock().expect("socket"), Slot::Closed);
        // Detach in-flight readers so they cannot return their socket to a closed session.
        self.socket = Arc::new(Mutex::new(Slot::Closed));
        if let Slot::Idle(reader) = slot {
            if let Some(mut socket) = reader.resume().await {
                tokio::time::timeout(Duration::from_secs(5), socket.close(None))
                    .await
                    .map_err(|_| timeout())?
                    .map_err(|_| interrupted())?;
            }
        }
        Ok(())
    }
    fn is_closed(&self) -> bool {
        match &*self.socket.lock().expect("socket") {
            Slot::Idle(reader) => {
                reader.task.is_finished() || self.opened.elapsed() >= MAX_CONNECTION_AGE
            }
            Slot::Busy => false,
            Slot::Closed => true,
        }
    }
}

impl Drop for ResponsesConnection {
    fn drop(&mut self) {
        if let Ok(slot) = self.socket.lock() {
            if let Slot::Idle(reader) = &*slot {
                reader.task.abort();
            }
        }
    }
}

// Both socket reads and Pong writes must observe close(), even under backpressure.
async fn guarded_io<T>(
    limit: Option<Duration>,
    cancel: &mut tokio::sync::watch::Receiver<bool>,
    operation: impl std::future::Future<Output = Result<T, LlmError>>,
) -> Result<T, LlmError> {
    if *cancel.borrow() {
        return Err(interrupted());
    }
    let deadline = async {
        match limit {
            Some(limit) => tokio::time::sleep(limit).await,
            None => std::future::pending().await,
        }
    };
    // Only close() cancels. Dropping the connection lets an in-flight
    // generation finish on the socket it already owns.
    let closed = async {
        if cancel.wait_for(|closed| *closed).await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        biased;
        () = closed => Err(interrupted()),
        _ = deadline => Err(timeout()),
        result = operation => result,
    }
}

#[cfg(test)]
mod guarded_io_tests {
    use super::*;
    #[tokio::test(start_paused = true)]
    async fn blocked_pong_write_observes_deadline_and_cancellation() {
        let (close, mut cancel) = tokio::sync::watch::channel(false);
        let error = guarded_io::<()>(
            Some(Duration::from_secs(5)),
            &mut cancel,
            std::future::pending(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, LlmError::TransportTimeout { .. }));
        let task = tokio::spawn(async move {
            guarded_io::<()>(None, &mut cancel, std::future::pending()).await
        });
        tokio::task::yield_now().await;
        close.send(true).unwrap();
        assert!(matches!(
            task.await.unwrap(),
            Err(LlmError::StreamInterrupted { .. })
        ));
    }
}
