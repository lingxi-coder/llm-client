//! Optional Tokio WebSocket transport backed by Rustls.

use super::{
    RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
    RealtimeSink, RealtimeTransport,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{Sink, SinkExt, Stream, StreamExt};
use std::borrow::Cow;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    handshake::client::Request,
    http::{header::HeaderName, header::HeaderValue},
    protocol::{frame::coding::CloseCode, frame::CloseFrame},
    Error as WebSocketError, Message,
};

/// Realtime uses the same configured network client as HTTP and Responses.
#[async_trait]
impl RealtimeTransport for crate::HttpTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(RealtimeError::InvalidConfig {
                message: "HttpTransport realtime requires a Tokio runtime".into(),
            });
        }
        if request.max_frame_bytes == 0 {
            return Err(RealtimeError::InvalidConfig {
                message: "max_frame_bytes must be non-zero".into(),
            });
        }

        let handshake = build_handshake_request(&request)?;
        let wire = crate::HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url: request.endpoint.clone(),
            headers: request.headers,
            body: Default::default(),
            timeout: None,
        };
        drop(handshake);
        let (socket, _) = crate::transport::websocket::connect(self, wire, request.max_frame_bytes)
            .await
            .map_err(|error| match error {
                crate::protocol::LlmError::InvalidRequest { message } => {
                    RealtimeError::InvalidConfig { message }
                }
                crate::protocol::LlmError::TlsCert { message } => {
                    RealtimeError::TlsCert { message }
                }
                other => RealtimeError::Transport {
                    message: other.to_string(),
                },
            })?;
        let (outbound, inbound) = socket.split();
        let inbound = inbound
            .map(|result| result.map_err(safe_websocket_error))
            .filter_map(|result| async move {
                match result {
                    Ok(message) => match incoming_frame(message) {
                        Ok(Some(frame)) => Some(Ok(frame)),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    },
                    Err(error) => Some(Err(error)),
                }
            })
            .boxed();

        Ok(RealtimeConnection {
            outbound: Box::new(WebSocketSink { sink: outbound }),
            inbound,
        })
    }
}

fn build_handshake_request(request: &RealtimeConnectRequest) -> Result<Request, RealtimeError> {
    validate_endpoint(&request.endpoint)?;
    let mut handshake = request
        .endpoint
        .as_str()
        .into_client_request()
        .map_err(safe_websocket_error)?;
    for (name, value) in &request.headers {
        let name =
            HeaderName::from_bytes(name.as_bytes()).map_err(|_| RealtimeError::InvalidConfig {
                message: "realtime request contains an invalid HTTP header name".into(),
            })?;
        let value = HeaderValue::from_str(value).map_err(|_| RealtimeError::InvalidConfig {
            message: "realtime request contains an invalid HTTP header value".into(),
        })?;
        handshake.headers_mut().append(name, value);
    }
    Ok(handshake)
}

struct WebSocketSink<S> {
    sink: futures::stream::SplitSink<S, Message>,
}

#[async_trait]
impl<S> RealtimeSink for WebSocketSink<S>
where
    S: Stream<Item = Result<Message, WebSocketError>>
        + Sink<Message, Error = WebSocketError>
        + Unpin
        + Send
        + 'static,
{
    async fn send(&mut self, frame: RealtimeFrame) -> Result<(), RealtimeError> {
        self.sink
            .send(outgoing_message(frame)?)
            .await
            .map_err(safe_websocket_error)
    }

    async fn ping(&mut self, payload: Bytes) -> Result<(), RealtimeError> {
        self.sink
            .send(outgoing_ping(payload)?)
            .await
            .map_err(safe_websocket_error)
    }

    fn abort(&mut self) {
        // Dropping both split halves releases the socket; no flush is needed.
    }

    async fn close(&mut self, close: RealtimeClose) -> Result<(), RealtimeError> {
        self.sink
            .send(outgoing_close(close))
            .await
            .map_err(safe_websocket_error)
    }
}

fn validate_endpoint(endpoint: &str) -> Result<(), RealtimeError> {
    let parsed = url::Url::parse(endpoint).map_err(|_| RealtimeError::InvalidConfig {
        message: "realtime endpoint must be an absolute wss:// URL".into(),
    })?;
    if parsed.scheme() != "wss"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(RealtimeError::InvalidConfig {
            message: "realtime endpoint must be an absolute wss:// URL without user-info".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn websocket_config(
    max_frame_bytes: usize,
) -> tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
    tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
        max_message_size: Some(max_frame_bytes),
        max_frame_size: Some(max_frame_bytes),
        ..Default::default()
    }
}

fn outgoing_message(frame: RealtimeFrame) -> Result<Message, RealtimeError> {
    match frame {
        RealtimeFrame::Text(bytes) => String::from_utf8(bytes.to_vec())
            .map(Message::Text)
            .map_err(|_| RealtimeError::InvalidInput {
                message: "text WebSocket frames must contain UTF-8".into(),
            }),
        RealtimeFrame::Binary(bytes) => Ok(Message::Binary(bytes.to_vec())),
    }
}

fn outgoing_ping(payload: Bytes) -> Result<Message, RealtimeError> {
    const MAX_CONTROL_PAYLOAD_BYTES: usize = 125;

    if payload.len() > MAX_CONTROL_PAYLOAD_BYTES {
        return Err(RealtimeError::InvalidInput {
            message: format!(
                "WebSocket Ping payload is {} bytes; maximum is {MAX_CONTROL_PAYLOAD_BYTES}",
                payload.len()
            ),
        });
    }

    Ok(Message::Ping(payload.to_vec()))
}

fn outgoing_close(close: RealtimeClose) -> Message {
    Message::Close(Some(CloseFrame {
        code: CloseCode::from(close.code),
        reason: Cow::Owned(close.reason),
    }))
}

fn incoming_frame(message: Message) -> Result<Option<RealtimeFrame>, RealtimeError> {
    match message {
        Message::Text(text) => Ok(Some(RealtimeFrame::Text(Bytes::from(text.into_bytes())))),
        Message::Binary(bytes) => Ok(Some(RealtimeFrame::Binary(Bytes::from(bytes)))),
        // Tungstenite queues Pong replies for Ping frames and flushes them as
        // it continues reading or writing. These are transport control frames.
        Message::Ping(_) | Message::Pong(_) => Ok(None),
        Message::Close(None) => Ok(None),
        Message::Close(Some(close)) if close.code == CloseCode::Normal => Ok(None),
        Message::Close(Some(close)) => Err(RealtimeError::Transport {
            // Keep peer-provided close reasons out of diagnostics. A non-normal
            // close interrupts an in-progress realtime exchange.
            message: format!(
                "remote WebSocket closed with status code {}",
                u16::from(close.code)
            ),
        }),
        Message::Frame(_) => Err(RealtimeError::Transport {
            message: "WebSocket transport received an unsupported raw frame".into(),
        }),
    }
}

fn safe_websocket_error(error: WebSocketError) -> RealtimeError {
    use tokio_tungstenite::tungstenite::error::CapacityError;

    if let WebSocketError::Capacity(CapacityError::MessageTooLong { size, max_size }) = &error {
        return RealtimeError::FrameTooLarge {
            actual: *size,
            max: *max_size,
        };
    }

    let message = match error {
        WebSocketError::ConnectionClosed => "WebSocket connection closed".into(),
        WebSocketError::AlreadyClosed => "WebSocket connection is already closed".into(),
        WebSocketError::Io(error) => format!("WebSocket I/O error ({:?})", error.kind()),
        WebSocketError::Tls(_) => "WebSocket TLS handshake failed".into(),
        WebSocketError::Capacity(_) => "WebSocket capacity limit exceeded".into(),
        WebSocketError::Protocol(_) => "WebSocket protocol error".into(),
        WebSocketError::WriteBufferFull(_) => "WebSocket write buffer is full".into(),
        WebSocketError::Utf8 => "WebSocket text message is not UTF-8".into(),
        WebSocketError::AttackAttempt => "WebSocket protocol attack detected".into(),
        WebSocketError::Url(_) => "invalid WebSocket URL".into(),
        WebSocketError::Http(response) => {
            format!(
                "WebSocket handshake rejected with HTTP {}",
                response.status()
            )
        }
        WebSocketError::HttpFormat(_) => "invalid WebSocket handshake request".into(),
    };
    RealtimeError::Transport { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::protocol::Role;

    #[test]
    fn websocket_message_conversion_preserves_data_and_filters_control_frames() {
        assert_eq!(
            outgoing_message(RealtimeFrame::text("hello")).unwrap(),
            Message::Text("hello".into())
        );
        assert_eq!(
            outgoing_message(RealtimeFrame::binary(Bytes::from_static(b"\x01\x02"))).unwrap(),
            Message::Binary(vec![1, 2])
        );
        assert!(outgoing_message(RealtimeFrame::text(Bytes::from_static(b"\xff"))).is_err());

        assert_eq!(
            incoming_frame(Message::Text("hello".into())).unwrap(),
            Some(RealtimeFrame::text("hello"))
        );
        assert_eq!(
            incoming_frame(Message::Binary(vec![1, 2])).unwrap(),
            Some(RealtimeFrame::binary(Bytes::from_static(b"\x01\x02")))
        );
        assert_eq!(incoming_frame(Message::Ping(vec![])).unwrap(), None);
        assert_eq!(incoming_frame(Message::Pong(vec![])).unwrap(), None);
        assert_eq!(incoming_frame(Message::Close(None)).unwrap(), None);
        assert_eq!(
            incoming_frame(Message::Close(Some(CloseFrame {
                code: CloseCode::Normal,
                reason: Cow::Borrowed("complete"),
            })))
            .unwrap(),
            None
        );
    }

    #[test]
    fn non_normal_remote_close_codes_interrupt_without_exposing_reasons() {
        for code in [
            CloseCode::Away,
            CloseCode::Protocol,
            CloseCode::Policy,
            CloseCode::Abnormal,
            CloseCode::Error,
            CloseCode::Iana(4321),
        ] {
            let code_number = u16::from(code);
            let error = incoming_frame(Message::Close(Some(CloseFrame {
                code,
                reason: Cow::Borrowed("private peer reason"),
            })))
            .expect_err("non-normal close codes must not be treated as clean EOF");

            assert!(matches!(error, RealtimeError::Transport { .. }));
            assert!(error.to_string().contains(&code_number.to_string()));
            assert!(!error.to_string().contains("private peer reason"));
        }
    }

    #[tokio::test]
    async fn ping_is_encoded_as_a_websocket_control_frame() {
        let (client_io, server_io) = tokio::io::duplex(1024);
        let client =
            tokio_tungstenite::WebSocketStream::from_raw_socket(client_io, Role::Client, None)
                .await;
        let mut server =
            tokio_tungstenite::WebSocketStream::from_raw_socket(server_io, Role::Server, None)
                .await;
        let (outbound, _inbound) = client.split();
        let mut sink = WebSocketSink { sink: outbound };
        let payload = Bytes::from(vec![0x5a; 125]);

        sink.ping(payload.clone()).await.unwrap();

        let message = tokio::time::timeout(Duration::from_secs(1), server.next())
            .await
            .expect("encoded Ping should arrive")
            .expect("WebSocket stream should remain open")
            .unwrap();
        assert_eq!(message, Message::Ping(payload.to_vec()));
    }

    #[tokio::test]
    async fn oversized_ping_is_rejected_before_any_frame_is_sent() {
        let (client_io, server_io) = tokio::io::duplex(1024);
        let client =
            tokio_tungstenite::WebSocketStream::from_raw_socket(client_io, Role::Client, None)
                .await;
        let mut server =
            tokio_tungstenite::WebSocketStream::from_raw_socket(server_io, Role::Server, None)
                .await;
        let (outbound, _inbound) = client.split();
        let mut sink = WebSocketSink { sink: outbound };

        let error = sink
            .ping(Bytes::from(vec![0x5a; 126]))
            .await
            .expect_err("RFC 6455 control payloads cannot exceed 125 bytes");
        assert!(matches!(error, RealtimeError::InvalidInput { .. }));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), server.next())
                .await
                .is_err()
        );
    }

    #[test]
    fn websocket_limits_apply_to_messages_and_individual_frames() {
        let config = websocket_config(8192);
        assert_eq!(config.max_message_size, Some(8192));
        assert_eq!(config.max_frame_size, Some(8192));
    }

    #[test]
    fn handshake_preserves_caller_endpoint_and_headers() {
        let request = RealtimeConnectRequest {
            endpoint: "wss://example.invalid/realtime?model=test-model".into(),
            headers: vec![("Authorization".into(), "Bearer test-secret".into())],
            max_frame_bytes: 4096,
        };
        let handshake = build_handshake_request(&request).unwrap();

        assert_eq!(handshake.uri().path(), "/realtime");
        assert_eq!(handshake.uri().query(), Some("model=test-model"));
        assert_eq!(
            handshake.headers().get("authorization").unwrap(),
            "Bearer test-secret"
        );
    }

    #[test]
    fn websocket_close_preserves_the_validated_code_and_reason() {
        assert_eq!(
            outgoing_close(RealtimeClose::normal("normal")),
            Message::Close(Some(CloseFrame {
                code: CloseCode::Normal,
                reason: Cow::Owned("normal".into()),
            }))
        );
    }

    #[test]
    fn oversized_transport_messages_map_to_the_core_frame_limit_error() {
        let error = safe_websocket_error(WebSocketError::Capacity(
            tokio_tungstenite::tungstenite::error::CapacityError::MessageTooLong {
                size: 9,
                max_size: 8,
            },
        ));
        assert_eq!(error, RealtimeError::FrameTooLarge { actual: 9, max: 8 });
    }
}
