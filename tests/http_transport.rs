//! Real loopback HTTP tests for the built-in transport (no provider credentials).
use bytes::Bytes;
use futures::StreamExt;
use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, LlmError, MessageRole, ProviderProfile,
    Secret, StreamEvent, ToolChoice,
};
use lingxi_llm_client::{HttpRequest, HttpTransport, LlmClientBuilder, RequestOptions, Transport};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

async fn read_request(socket: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 1024];
        let n = socket.read(&mut buf).await.unwrap();
        assert_ne!(n, 0, "request ended before its body");
        bytes.extend_from_slice(&buf[..n]);
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let len: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + len {
                return String::from_utf8(bytes).unwrap();
            }
        }
    }
}

async fn server(response: String) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        socket.write_all(response.as_bytes()).await.unwrap();
        request
    });
    (url, task)
}

fn response(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
        body.len()
    )
}

fn request(url: String) -> HttpRequest {
    HttpRequest {
        method: "POST".into(),
        url,
        headers: vec![("x-test".into(), "value".into())],
        body: Bytes::from_static(b"payload"),
        timeout: Some(Duration::from_secs(3)),
    }
}

#[tokio::test]
async fn execute_preserves_request_and_non_success_response() {
    let (url, task) = server(response(
        "429 Too Many Requests",
        "Retry-After: 7\r\n",
        "limited",
    ))
    .await;
    let http = HttpTransport::new().unwrap();
    let reply = http
        .execute(request(format!("{url}/path?q=1")))
        .await
        .unwrap();
    assert_eq!(reply.status, 429);
    assert_eq!(reply.header("RETRY-AFTER"), Some("7"));
    assert_eq!(reply.body, "limited");
    let seen = task.await.unwrap();
    assert!(seen.starts_with("POST /path?q=1 HTTP/1.1\r\n"));
    assert!(seen.to_ascii_lowercase().contains("x-test: value\r\n"));
    assert!(seen.ends_with("\r\n\r\npayload"));
}

#[tokio::test]
async fn every_http_entry_point_returns_redirect_without_following() {
    let http = HttpTransport::new().unwrap();
    for mode in 0..3 {
        let (url, task) = server(response(
            "307 Temporary Redirect",
            "Location: http://127.0.0.1:1/secret\r\n",
            "redirect",
        ))
        .await;
        if mode == 2 {
            let mut reply = http.open_stream(request(url)).await.unwrap();
            assert_eq!(reply.status, 307);
            assert_eq!(reply.header("location"), Some("http://127.0.0.1:1/secret"));
            let mut body = Vec::new();
            while let Some(chunk) = reply.body.next().await {
                body.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(body, b"redirect");
        } else {
            let reply = if mode == 0 {
                http.execute(request(url)).await
            } else {
                http.execute_no_follow(request(url)).await
            }
            .unwrap();
            assert_eq!(reply.status, 307);
            assert_eq!(reply.header("location"), Some("http://127.0.0.1:1/secret"));
            assert_eq!(reply.body, "redirect");
        }
        task.await.unwrap();
    }
}

#[tokio::test]
async fn streaming_delivers_first_chunk_before_server_finishes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (release, wait) = oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nfirst\r\n")
            .await
            .unwrap();
        wait.await.unwrap();
        socket.write_all(b"4\r\nlast\r\n0\r\n\r\n").await.unwrap();
    });
    let mut reply = HttpTransport::new()
        .unwrap()
        .open_stream(request(url))
        .await
        .unwrap();
    let chunk = tokio::time::timeout(Duration::from_secs(2), reply.body.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(chunk, "first");
    release.send(()).unwrap();
    let mut rest = Vec::new();
    while let Some(chunk) = reply.body.next().await {
        rest.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(rest, b"last");
    task.await.unwrap();
}

#[tokio::test]
async fn total_timeout_covers_streaming_and_buffered_body_reads() {
    for streaming in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (release, wait) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nx")
                .await
                .unwrap();
            let _ = wait.await;
        });
        let http = HttpTransport::new().unwrap();
        let mut req = request(url);
        req.timeout = Some(Duration::from_millis(150));
        let error = tokio::time::timeout(Duration::from_secs(3), async {
            if streaming {
                let mut reply = http.open_stream(req).await.unwrap();
                loop {
                    match reply.body.next().await.expect("body must time out") {
                        Ok(_) => {}
                        Err(error) => break error,
                    }
                }
            } else {
                http.execute(req).await.unwrap_err()
            }
        })
        .await
        .unwrap();
        assert!(
            matches!(error, LlmError::TransportTimeout { .. }),
            "{error:?}"
        );
        drop(release);
        task.await.unwrap();
    }
}

#[tokio::test]
async fn custom_read_idle_timeout_ends_a_stalled_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nx")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
    });
    let http = HttpTransport::with_read_timeout(Duration::from_millis(100)).unwrap();
    let mut req = request(url);
    req.timeout = None;
    let mut response = http.open_stream(req).await.unwrap();
    assert_eq!(response.body.next().await.unwrap().unwrap(), "x");
    let error = response.body.next().await.unwrap().unwrap_err();
    assert!(
        matches!(error, LlmError::TransportTimeout { .. }),
        "{error:?}"
    );
    task.await.unwrap();
}

#[tokio::test]
async fn truncated_stream_is_reported_as_interrupted() {
    let (url, task) = server("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort".into()).await;
    let mut reply = HttpTransport::new()
        .unwrap()
        .open_stream(request(url))
        .await
        .unwrap();
    let mut error = None;
    while let Some(chunk) = reply.body.next().await {
        if let Err(err) = chunk {
            error = Some(err);
            break;
        }
    }
    assert!(matches!(error, Some(LlmError::StreamInterrupted { .. })));
    task.await.unwrap();
}

#[tokio::test]
async fn invalid_requests_do_not_disclose_secrets_and_websocket_is_unsupported() {
    let http = HttpTransport::new().unwrap();
    let mut invalid_header = request("http://127.0.0.1:1/?key=private-secret".into());
    invalid_header.headers = vec![("Authorization".into(), "private-secret\ninvalid".into())];
    for req in [request("private-secret invalid URL".into()), invalid_header] {
        let err = http.execute(req).await.unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest { .. }));
        assert!(!format!("{err:?} {err}").contains("private-secret"));
    }
    let err = http
        .open_responses_websocket_session(request("ws://127.0.0.1:1".into()))
        .await
        .err()
        .unwrap();
    assert!(matches!(err, LlmError::UnsupportedCapability { .. }));
}

fn completion() -> CompletionRequest {
    CompletionRequest {
        model: "test-model".into(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hello".into(),
                thought_signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::auto(),
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

#[tokio::test]
async fn builtin_builder_completes_and_streams_with_api_key_and_bearer_authentication() {
    for auth in ["api_key", "bearer"] {
        for streaming in [false, true] {
            let body = if streaming {
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
            } else {
                "{\"id\":\"test-id\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}"
            };
            let content_type = if streaming {
                "Content-Type: text/event-stream\r\n"
            } else {
                "Content-Type: application/json\r\n"
            };
            let (url, task) = server(response("200 OK", content_type, body)).await;
            let profile: ProviderProfile = serde_json::from_value(json!({ "provider_id": "local", "profile_name": "local", "base_url": url, "protocol": "open_ai_chat", "auth": auth, "models": [{"display_model":"test-model", "request_model":"wire-model", "billing_model":"wire-model"}] })).unwrap();
            let client = LlmClientBuilder::new(&[profile]).unwrap().build().unwrap();
            let opts = RequestOptions {
                credential: Some(Secret::new("test-key".to_owned())),
                ..Default::default()
            };
            if streaming {
                let mut stream = client.stream(&completion(), &opts).await.unwrap();
                let mut text = String::new();
                while let Some(event) = stream.next().await {
                    if let StreamEvent::TextDelta { text: delta, .. } = event.unwrap() {
                        text.push_str(&delta);
                    }
                }
                assert_eq!(text, "hello");
            } else {
                let reply = client.complete(&completion(), &opts).await.unwrap();
                assert!(reply.message.content.iter().any(
                    |block| matches!(block, ContentBlock::Text { text, .. } if text == "hello")
                ));
            }
            let seen = task.await.unwrap();
            assert!(seen
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n"));
            assert!(seen.starts_with("POST /chat/completions HTTP/1.1\r\n"));
            let body: Value = serde_json::from_str(seen.split_once("\r\n\r\n").unwrap().1).unwrap();
            assert_eq!(body["model"], "wire-model");
            assert_eq!(body["stream"].as_bool().unwrap_or(false), streaming);
        }
    }
}
