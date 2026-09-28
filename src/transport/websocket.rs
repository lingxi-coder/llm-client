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

struct ResponsesConnection {
    // Taking the socket exclusively prevents overlapping generations. Early drop
    // discards it; only a terminal provider event makes it reusable.
    socket: Arc<Mutex<Option<Socket>>>,
    headers: Vec<(String, String)>,
    cancel: tokio::sync::watch::Sender<bool>,
    idle_timeout: Duration,
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
    Ok(Box::new(ResponsesConnection {
        socket: Arc::new(Mutex::new(Some(socket))),
        headers,
        cancel: tokio::sync::watch::channel(false).0,
        idle_timeout: transport.read_timeout,
    }))
}
#[async_trait]
impl WebSocketConnection for ResponsesConnection {
    async fn send(&mut self, payload: Bytes) -> Result<StreamResponse, LlmError> {
        let mut socket =
            self.socket
                .lock()
                .expect("socket")
                .take()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "WebSocket connection is busy or no longer usable".into(),
                })?;
        let text = String::from_utf8(payload.to_vec()).map_err(|_| invalid())?;
        tokio::time::timeout(self.idle_timeout, socket.send(Message::Text(text)))
            .await
            .map_err(|_| timeout())?
            .map_err(|_| interrupted())?;
        let slot = Arc::downgrade(&self.socket);
        let cancel = self.cancel.subscribe();
        let idle_timeout = self.idle_timeout;
        let body = futures::stream::unfold((Some(socket), cancel), move |(state, mut cancel)| {
            let slot = slot.clone();
            async move {
                let mut socket = state?;
                loop {
                    let read = tokio::select! {
                        biased;
                        _ = cancel.changed() => return Some((Err(interrupted()), (None, cancel))),
                        value = tokio::time::timeout(idle_timeout, socket.next()) => value,
                    };
                    let frame = match read {
                        Ok(Some(Ok(frame))) => frame,
                        Err(_) => return Some((Err(timeout()), (None, cancel))),
                        _ => return Some((Err(interrupted()), (None, cancel))),
                    };
                    match frame {
                        Message::Text(text) => {
                            let terminal = serde_json::from_str::<serde_json::Value>(&text)
                                .ok()
                                .and_then(|v| {
                                    v.get("type").and_then(|v| v.as_str()).map(str::to_owned)
                                })
                                .is_some_and(|kind| {
                                    matches!(
                                        kind.as_str(),
                                        "response.completed"
                                            | "response.incomplete"
                                            | "response.failed"
                                            | "error"
                                    )
                                });
                            let next = if terminal {
                                if let Some(slot) = slot.upgrade() {
                                    *slot.lock().expect("socket") = Some(socket);
                                }
                                None
                            } else {
                                Some(socket)
                            };
                            return Some((Ok(Bytes::from(text)), (next, cancel)));
                        }
                        Message::Ping(_) => {
                            if socket.flush().await.is_err() {
                                return Some((Err(interrupted()), (None, cancel)));
                            }
                        }
                        Message::Pong(_) => {}
                        _ => return Some((Err(interrupted()), (None, cancel))),
                    }
                }
            }
        })
        .boxed();
        Ok(StreamResponse {
            status: 200,
            headers: self.headers.clone(),
            body,
        })
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        let _ = self.cancel.send(true);
        let socket = self.socket.lock().expect("socket").take();
        // Detach in-flight readers so they cannot return their socket to a closed session.
        self.socket = Arc::new(Mutex::new(None));
        if let Some(mut socket) = socket {
            tokio::time::timeout(Duration::from_secs(5), socket.close(None))
                .await
                .map_err(|_| timeout())?
                .map_err(|_| interrupted())?;
        }
        Ok(())
    }
}
