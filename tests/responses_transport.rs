#![cfg(feature = "responses-websocket")]
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use lingxi_llm_client::{HttpRequest, HttpTransport, Transport};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

async fn read_headers(stream: &mut tokio::net::TcpStream) {
    use tokio::io::AsyncReadExt;
    let mut received = Vec::new();
    let mut chunk = [0; 512];
    while !received.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
        let count = stream.read(&mut chunk).await.unwrap();
        assert!(count > 0, "handshake ended before complete request headers");
        received.extend_from_slice(&chunk[..count]);
        assert!(
            received.len() <= 16_384,
            "fixture request headers exceeded bound"
        );
    }
}

fn request(url: String) -> HttpRequest {
    HttpRequest {
        http1_header_layout: None,
        method: "GET".into(),
        url,
        headers: vec![("Authorization".into(), "Bearer private-token".into())],
        body: Bytes::new(),
        timeout: Some(std::time::Duration::from_secs(2)),
    }
}
#[tokio::test]
// Tungstenite fixes the callback error type; this trait cannot return a box.
#[allow(clippy::result_large_err)]
async fn responses_reuse_terminal_connection_without_replaying() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_hdr_async(
            stream,
            |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
                assert_eq!(request.headers()["authorization"], "Bearer private-token");
                assert!(request.headers()["openai-beta"]
                    .to_str()
                    .unwrap()
                    .contains("responses_websockets="));
                Ok(response)
            },
        )
        .await
        .unwrap();
        for terminal in [
            "response.completed",
            "response.incomplete",
            "response.failed",
            "error",
        ] {
            assert_eq!(
                socket.next().await.unwrap().unwrap().into_text().unwrap(),
                "{}"
            );
            socket
                .send(Message::Text(format!("{{\"type\":\"{terminal}\"}}")))
                .await
                .unwrap();
        }
    });
    let transport = HttpTransport::new().unwrap();
    let mut connection = transport.connect_websocket(request(url)).await.unwrap();
    for _ in 0..4 {
        let response = connection.send(Bytes::from_static(b"{}")).await.unwrap();
        let chunks = response.body.collect::<Vec<_>>().await;
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_ok());
    }
    server.await.unwrap();
}
#[tokio::test]
async fn dropping_incomplete_body_discards_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.next().await.unwrap().unwrap();
        socket
            .send(Message::Text("{\"type\":\"response.created\"}".into()))
            .await
            .unwrap();
        // Client cancellation must release the connection.
        tokio::time::timeout(std::time::Duration::from_secs(2), socket.next())
            .await
            .unwrap();
    });
    let transport = HttpTransport::new().unwrap();
    let mut connection = transport.connect_websocket(request(url)).await.unwrap();
    let mut response = connection.send(Bytes::from_static(b"{}")).await.unwrap();
    response.body.next().await.unwrap().unwrap();
    drop(response);
    assert!(connection.send(Bytes::from_static(b"{}")).await.is_err());
    server.await.unwrap();
}
#[tokio::test]
async fn rejects_bad_accept_and_redacts_query() {
    use tokio::io::AsyncWriteExt;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/responses?secret=private-query",
        listener.local_addr().unwrap()
    );
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_headers(&mut stream).await;
        stream.write_all(b"HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: wrong\r\n\r\n").await.unwrap();
    });
    let error = HttpTransport::new()
        .unwrap()
        .connect_websocket(request(url))
        .await
        .err()
        .unwrap();
    assert!(!error.to_string().contains("private-query"));
    assert!(!error.to_string().contains("private-token"));
    server.await.unwrap();
}
#[tokio::test]
// Tungstenite fixes the callback error type; this trait cannot return a box.
#[allow(clippy::result_large_err)]
async fn configured_http_proxy_is_used_for_websocket() {
    // The endpoint intentionally cannot resolve. A successful handshake proves
    // WebSockets use the configured HTTP proxy instead of an independent socket.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        tokio_tungstenite::accept_hdr_async(
            stream,
            |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
                assert!(request.uri().to_string().contains("provider.invalid"));
                Ok(response)
            },
        )
        .await
        .unwrap()
    });
    let transport = HttpTransport::with_client_configurator(|builder| {
        builder.proxy(reqwest::Proxy::http(proxy).unwrap())
    })
    .unwrap();
    let _connection = transport
        .connect_websocket(request("http://provider.invalid/responses".into()))
        .await
        .unwrap();
    let _socket = server.await.unwrap();
}

#[tokio::test]
async fn closing_a_connection_interrupts_an_owned_response_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.next().await.unwrap().unwrap();
        let _ = socket.next().await;
    });
    let transport = HttpTransport::new().unwrap();
    let mut connection = transport.connect_websocket(request(url)).await.unwrap();
    let mut response = connection.send(Bytes::from_static(b"{}")).await.unwrap();
    connection.close().await.unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), response.body.next())
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    server.await.unwrap();
}
#[tokio::test]
async fn configured_idle_timeout_bounds_silent_responses() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.next().await.unwrap().unwrap();
        let _ = socket.next().await;
    });
    let transport = HttpTransport::with_read_timeout_and_client_configurator(
        Some(std::time::Duration::from_millis(100)),
        |b| b.connect_timeout(std::time::Duration::from_secs(2)),
    )
    .unwrap();
    let mut connection = transport.connect_websocket(request(url)).await.unwrap();
    let mut response = connection.send(Bytes::from_static(b"{}")).await.unwrap();
    let error = tokio::time::timeout(std::time::Duration::from_secs(2), response.body.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        error,
        lingxi_llm_client::protocol::LlmError::TransportTimeout { .. }
    ));
    assert!(connection.send(Bytes::from_static(b"{}")).await.is_err());
    server.await.unwrap();
}
#[tokio::test]
async fn handshake_rejections_are_classified_without_echoing_provider_secrets() {
    use tokio::io::AsyncWriteExt;
    for status in [401, 403, 426, 429, 503] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/responses", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_headers(&mut stream).await;
            stream.write_all(format!("HTTP/1.1 {status} Rejected\r\nRetry-After: 3\r\nContent-Length: 7\r\n\r\nprivate").as_bytes()).await.unwrap();
        });
        let error = HttpTransport::new()
            .unwrap()
            .connect_websocket(request(url))
            .await
            .err()
            .unwrap();
        assert!(!error.to_string().contains("private"));
        use lingxi_llm_client::protocol::LlmError as E;
        match status {
            401 => assert!(matches!(error, E::Authentication { .. })),
            403 => assert!(matches!(error, E::PermissionDenied { .. })),
            429 => assert!(
                matches!(error,E::RateLimited{retry_after:Some(value),..} if value.as_secs()==3)
            ),
            503 => assert!(matches!(error, E::ProviderInternal { .. })),
            426 => assert!(error.to_string().contains("426")),
            _ => unreachable!(),
        }
        server.await.unwrap();
    }
}

#[tokio::test]
async fn host_owned_idle_policy_keeps_responses_alive_past_five_minutes() {
    use futures::poll;
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let (finish, waiting) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.next().await.unwrap().unwrap();
        waiting.await.unwrap();
        socket
            .send(Message::Text(r#"{"type":"response.completed"}"#.into()))
            .await
            .unwrap();
    });
    let transport = HttpTransport::with_read_timeout_and_client_configurator(None, |b| b).unwrap();
    let mut connection = transport.connect_websocket(request(url)).await.unwrap();
    let mut response = connection.send(Bytes::from_static(b"{}")).await.unwrap();
    tokio::time::pause();
    let next = response.body.next();
    tokio::pin!(next);
    assert!(poll!(&mut next).is_pending());
    tokio::time::advance(Duration::from_secs(301)).await;
    assert!(poll!(&mut next).is_pending());
    tokio::time::resume();
    finish.send(()).unwrap();
    assert!(next.await.unwrap().unwrap().ends_with(b"completed\"}"));
    server.await.unwrap();
}
