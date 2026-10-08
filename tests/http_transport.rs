//! Real loopback HTTP tests for the built-in transport (no provider credentials).
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::protocol::{
    ChatRequest, ContentBlock, ConversationMessage, LlmError, MessageRole, ProviderProfile, Secret,
    StreamEvent, ToolChoice,
};
use lingxi_llm_client::{
    HttpExecutor, HttpRequest, HttpStreamRequest, HttpTransport, LlmClientBuilder, RequestOptions,
    Transport,
};
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

fn interrupted_response(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
        body.len() + 32
    )
}

fn rate_limit_failover_client(primary: &str, backup: &str) -> lingxi_llm_client::LlmClient {
    let profile = |name: &str, base_url: &str, order: u32| {
        serde_json::from_value(json!({
            "provider_id": "loopback",
            "profile_name": name,
            "base_url": base_url,
            "protocol": "open_ai_chat",
            "auth": "none",
            "models": [{
                "display_model": "test-model",
                "request_model": "test-model",
                "billing_model": "test-model"
            }],
            "connection": {
                "group": "loopback",
                "order": order,
                "failover": {
                    "rateLimit": true,
                    "overloaded": false,
                    "serverError": false,
                    "network": false,
                    "auth": false
                }
            }
        }))
        .expect("loopback profile fixture parses")
    };
    let profiles: Vec<ProviderProfile> =
        vec![profile("primary", primary, 0), profile("backup", backup, 1)];
    LlmClientBuilder::new(&profiles)
        .unwrap()
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap()
}

fn request(url: String) -> HttpRequest {
    HttpRequest {
        http1_header_layout: None,
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
    let reply = HttpExecutor::new(&http)
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
async fn execute_keeps_status_headers_and_partial_body_for_interrupted_error_responses() {
    let body = r#"{"error":{"message":"quota unavailable"}}"#;
    let (url, task) = server(interrupted_response(
        "429 Too Many Requests",
        "Retry-After: 11\r\n",
        body,
    ))
    .await;

    let reply = HttpExecutor::new(&HttpTransport::new().unwrap())
        .execute(request(url))
        .await
        .expect("a response status already arrived before its body was truncated");

    assert_eq!(reply.status, 429);
    assert_eq!(reply.header("retry-after"), Some("11"));
    assert_eq!(reply.body, body);
    task.await.unwrap();
}

#[tokio::test]
async fn complete_fails_over_on_interrupted_429_when_only_rate_limits_are_enabled() {
    let error_body = r#"{"error":{"message":"quota unavailable"}}"#;
    let (primary, primary_task) = server(interrupted_response(
        "429 Too Many Requests",
        "Retry-After: 11\r\n",
        error_body,
    ))
    .await;
    let success_body = r#"{"id":"reply","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
    let (backup, backup_task) = server(response(
        "200 OK",
        "Content-Type: application/json\r\n",
        success_body,
    ))
    .await;
    let client = rate_limit_failover_client(&primary, &backup);

    let reply = client
        .chat()
        .complete(&completion(), &RequestOptions::default())
        .await
        .expect("truncated 429 still qualifies for configured rate-limit failover");

    assert_eq!(reply.executed_profile.as_deref(), Some("backup"));
    assert!(reply
        .message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { text, .. } if text == "ok")));
    assert!(primary_task
        .await
        .unwrap()
        .starts_with("POST /chat/completions HTTP/1.1"));
    assert!(backup_task
        .await
        .unwrap()
        .starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn exhausted_interrupted_429_keeps_retry_after() {
    let error_body = r#"{"error":{"message":"quota unavailable"}}"#;
    let (primary, primary_task) = server(interrupted_response(
        "429 Too Many Requests",
        "Retry-After: 11\r\n",
        error_body,
    ))
    .await;
    let (backup, backup_task) = server(interrupted_response(
        "429 Too Many Requests",
        "Retry-After: 11\r\n",
        error_body,
    ))
    .await;
    let client = rate_limit_failover_client(&primary, &backup);

    let error = client
        .chat()
        .complete(&completion(), &RequestOptions::default())
        .await
        .expect_err("both interrupted 429 responses should remain rate-limit errors");

    assert!(
        matches!(error, LlmError::RateLimited { ref message, retry_after: Some(delay) }
        if message.contains("quota unavailable") && delay == Duration::from_secs(11)),
        "{error:?}"
    );
    assert!(primary_task
        .await
        .unwrap()
        .starts_with("POST /chat/completions HTTP/1.1"));
    assert!(backup_task
        .await
        .unwrap()
        .starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn interrupted_429_body_does_not_extend_the_client_deadline_for_failover() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let primary = format!("http://{}", listener.local_addr().unwrap());
    let (release, wait) = oneshot::channel::<()>();
    let primary_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 11\r\nContent-Length: 128\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"quota\"}}",
            )
            .await
            .unwrap();
        let _ = wait.await;
        request
    });
    let (backup, backup_task) = server(response(
        "200 OK",
        "Content-Type: application/json\r\n",
        r#"{"id":"reply","choices":[{"message":{"role":"assistant","content":"unexpected fallback"},"finish_reason":"stop"}]}"#,
    ))
    .await;
    let client = rate_limit_failover_client(&primary, &backup);

    let result = tokio::time::timeout(
        Duration::from_secs(3),
        client.chat().complete(
            &completion(),
            &RequestOptions {
                total_timeout: Some(Duration::from_millis(250)),
                ..RequestOptions::default()
            },
        ),
    )
    .await
    .expect("the configured request deadline should bound the stalled error body");
    assert!(
        matches!(result, Err(LlmError::TransportTimeout { .. })),
        "{result:?}"
    );
    assert!(
        !backup_task.is_finished(),
        "the expired deadline must prevent another attempt"
    );
    drop(release);
    assert!(primary_task
        .await
        .unwrap()
        .starts_with("POST /chat/completions HTTP/1.1"));
    backup_task.abort();
    let _ = backup_task.await;
}

#[tokio::test]
async fn interrupted_401_and_5xx_keep_their_categories_without_rate_limit_failover() {
    let body = r#"{"error":{"message":"provider unavailable"}}"#;
    for (status, expected_category) in [
        ("401 Unauthorized", "authentication"),
        ("500 Internal Server Error", "provider_internal"),
        ("529 Overloaded", "overloaded"),
    ] {
        let (primary, primary_task) = server(interrupted_response(status, "", body)).await;
        let (backup, backup_task) = server(response(
            "200 OK",
            "Content-Type: application/json\r\n",
            r#"{"id":"reply","choices":[{"message":{"role":"assistant","content":"unexpected fallback"},"finish_reason":"stop"}]}"#,
        ))
        .await;
        let client = rate_limit_failover_client(&primary, &backup);
        let error = client
            .chat()
            .complete(&completion(), &RequestOptions::default())
            .await
            .expect_err("only rate-limit failures should trigger the backup");

        match expected_category {
            "authentication" => assert!(
                matches!(error, LlmError::Authentication { .. }),
                "{error:?}"
            ),
            "provider_internal" => assert!(
                matches!(error, LlmError::ProviderInternal { .. }),
                "{error:?}"
            ),
            "overloaded" => assert!(matches!(error, LlmError::Overloaded { .. }), "{error:?}"),
            _ => unreachable!(),
        }
        assert!(primary_task
            .await
            .unwrap()
            .starts_with("POST /chat/completions HTTP/1.1"));
        assert!(
            !backup_task.is_finished(),
            "{status} must not match rate-limit-only failover"
        );
        backup_task.abort();
        let _ = backup_task.await;
    }
}

#[tokio::test]
async fn buffered_error_body_is_bounded_and_successful_body_interruptions_still_error() {
    let large = "x".repeat(96 * 1024);
    let (url, task) = server(response("500 Internal Server Error", "", &large)).await;
    let reply = HttpExecutor::new(&HttpTransport::new().unwrap())
        .execute(request(url))
        .await
        .unwrap();
    assert_eq!(reply.body.len(), 64 * 1024);
    task.await.unwrap();

    let (url, task) = server(interrupted_response("200 OK", "", "partial success")).await;
    let error = HttpExecutor::new(&HttpTransport::new().unwrap())
        .execute(request(url))
        .await
        .expect_err("a successful response with an interrupted body must still fail");
    assert!(matches!(error, LlmError::Transport { .. }));
    task.await.unwrap();
}

#[tokio::test]
async fn every_http_entry_point_returns_redirect_without_following() {
    let http = HttpTransport::new().unwrap();
    for mode in 0..2 {
        let (url, task) = server(response(
            "307 Temporary Redirect",
            "Location: http://127.0.0.1:1/secret\r\n",
            "redirect",
        ))
        .await;
        if mode == 1 {
            let mut reply = http.send(request(url)).await.unwrap();
            assert_eq!(reply.status, 307);
            assert_eq!(reply.header("location"), Some("http://127.0.0.1:1/secret"));
            let mut body = Vec::new();
            while let Some(chunk) = reply.body.next().await {
                body.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(body, b"redirect");
        } else {
            let reply = HttpExecutor::new(&http)
                .execute(request(url))
                .await
                .unwrap();
            assert_eq!(reply.status, 307);
            assert_eq!(reply.header("location"), Some("http://127.0.0.1:1/secret"));
            assert_eq!(reply.body, "redirect");
        }
        task.await.unwrap();
    }
}

#[tokio::test]
async fn configurator_cannot_enable_redirects_or_automatic_retries() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let classifications = Arc::new(AtomicUsize::new(0));
    let observed = classifications.clone();
    let http = HttpTransport::with_client_configurator(|builder| {
        builder
            .redirect(reqwest::redirect::Policy::limited(4))
            .retry(
                reqwest::retry::for_host("127.0.0.1").classify_fn(move |result| {
                    observed.fetch_add(1, Ordering::SeqCst);
                    result.retryable()
                }),
            )
    })
    .unwrap();
    for (status, expected) in [("307 Temporary Redirect", 307), ("503 Unavailable", 503)] {
        for streaming in [false, true] {
            let (url, task) = server(response(
                status,
                "Location: http://127.0.0.1:1/secret\r\n",
                "body",
            ))
            .await;
            let reply = if streaming {
                http.send_stream(HttpStreamRequest {
                    http1_header_layout: None,
                    method: "POST".into(),
                    url,
                    headers: vec![],
                    body: futures::stream::once(async { Ok(Bytes::from_static(b"payload")) })
                        .boxed(),
                    content_length: 7,
                    timeout: Some(Duration::from_secs(3)),
                })
                .await
                .unwrap()
            } else {
                http.send(request(url)).await.unwrap()
            };
            assert_eq!(reply.status, expected);
            task.await.unwrap();
        }
    }
    assert_eq!(classifications.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn streamed_upload_normalizes_matching_provider_content_length() {
    for name in ["content-length", "Content-Length"] {
        let (url, task) = server(response("200 OK", "", "uploaded")).await;
        let reply = HttpTransport::new()
            .unwrap()
            .send_stream(HttpStreamRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url,
                headers: vec![(name.into(), "7".into())],
                body: futures::stream::once(async { Ok(Bytes::from_static(b"payload")) }).boxed(),
                content_length: 7,
                timeout: Some(Duration::from_secs(3)),
            })
            .await
            .unwrap();
        assert_eq!(reply.status, 200);
        let wire = task.await.unwrap().to_ascii_lowercase();
        assert_eq!(wire.matches("content-length: 7\r\n").count(), 1);
        assert!(!wire.contains("transfer-encoding"));
        assert!(wire.ends_with("\r\n\r\npayload"));
    }
}

#[tokio::test]
async fn streamed_upload_rejects_ambiguous_framing_before_consuming_input() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    for headers in [
        vec![("Content-Length".into(), "8".into())],
        vec![("Content-Length".into(), "7, 7".into())],
        vec![
            ("Content-Length".into(), "7".into()),
            ("content-length".into(), "7".into()),
        ],
        vec![("Transfer-Encoding".into(), "chunked".into())],
    ] {
        let polled = Arc::new(AtomicBool::new(false));
        let observed = polled.clone();
        let result = HttpTransport::new()
            .unwrap()
            .send_stream(HttpStreamRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url: "http://127.0.0.1:1/upload".into(),
                headers,
                body: futures::stream::once(async move {
                    observed.store(true, Ordering::SeqCst);
                    Ok(Bytes::from_static(b"payload"))
                })
                .boxed(),
                content_length: 7,
                timeout: Some(Duration::from_secs(3)),
            })
            .await;
        assert!(matches!(result, Err(LlmError::InvalidRequest { .. })));
        assert!(!polled.load(Ordering::SeqCst));
    }
}

#[tokio::test]
async fn bounded_account_read_rejects_an_oversized_success_body() {
    let (url, task) = server(response("200 OK", "", &"x".repeat(8_192))).await;
    let error = HttpExecutor::new(&HttpTransport::new().unwrap())
        .execute_bounded(request(url), 1_024)
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::Transport { .. }));
    task.await.unwrap();
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
        .send(request(url))
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
                let mut reply = http.send(req).await.unwrap();
                loop {
                    match reply.body.next().await.expect("body must time out") {
                        Ok(_) => {}
                        Err(error) => break error,
                    }
                }
            } else {
                HttpExecutor::new(&http).execute(req).await.unwrap_err()
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
    let mut response = http.send(req).await.unwrap();
    assert_eq!(response.body.next().await.unwrap().unwrap(), "x");
    let error = response.body.next().await.unwrap().unwrap_err();
    assert!(
        matches!(error, LlmError::TransportTimeout { .. }),
        "{error:?}"
    );
    task.await.unwrap();
}

#[tokio::test]
async fn truncated_stream_preserves_its_network_cause() {
    let (url, task) = server("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort".into()).await;
    let mut reply = HttpTransport::new()
        .unwrap()
        .send(request(url))
        .await
        .unwrap();
    let mut error = None;
    while let Some(chunk) = reply.body.next().await {
        if let Err(err) = chunk {
            error = Some(err);
            break;
        }
    }
    assert!(matches!(error, Some(LlmError::Transport { .. })));
    task.await.unwrap();
}

fn completion() -> ChatRequest {
    ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        controls: Default::default(),
        service_tier: None,
        model: "test-model".into(),
        native_options: Vec::new(),
        hosted_tools: vec![],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hello".into(),
                thought_signature: None,
                citations: None,
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
            let client = LlmClientBuilder::new(&[profile])
                .unwrap()
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build()
                .unwrap();
            let opts = RequestOptions {
                credential: Some(Secret::new("test-key".to_owned())),
                ..Default::default()
            };
            if streaming {
                let mut stream = client.chat().stream(&completion(), &opts).await.unwrap();
                let mut text = String::new();
                while let Some(event) = stream.next().await {
                    if let StreamEvent::TextDelta { text: delta, .. } = event.unwrap() {
                        text.push_str(&delta);
                    }
                }
                assert_eq!(text, "hello");
            } else {
                let reply = client.chat().complete(&completion(), &opts).await.unwrap();
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

#[tokio::test]
async fn certificate_in_url_does_not_change_connection_or_timeout_category() {
    let transport = HttpTransport::with_client_configurator(|b| b.no_proxy()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/certificate/v1/messages",
        listener.local_addr().unwrap()
    );
    let request = || HttpRequest {
        http1_header_layout: None,
        method: "POST".into(),
        url: url.clone(),
        headers: vec![],
        body: Bytes::new(),
        timeout: Some(Duration::from_millis(50)),
    };
    // A listening TCP socket that never answers HTTP causes a request deadline.
    assert!(matches!(
        transport.send(request()).await,
        Err(LlmError::TransportTimeout { .. })
    ));
    drop(listener);
    // The same URL without a listener is an ordinary connection failure.
    assert!(matches!(
        transport.send(request()).await,
        Err(LlmError::Transport { .. })
    ));
}
