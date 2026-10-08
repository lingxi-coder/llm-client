use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{files::FileService, protocol::*, *};
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
struct Capture {
    calls: Mutex<Vec<Call>>,
}

struct Call {
    declared: u64,
    actual: u64,
    first: String,
    last: String,
    headers: Vec<(String, String)>,
}

#[async_trait]
impl Transport for Capture {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("buffered send must not be used for streaming upload")
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let mut total = 0u64;
        let mut first = None;
        let mut last = String::new();
        while let Some(chunk) = request.body.next().await {
            let chunk = chunk?;
            total += chunk.len() as u64;
            if first.is_none() {
                first = Some(String::from_utf8_lossy(&chunk).into_owned());
            }
            if chunk.len() < 1024 {
                last = String::from_utf8_lossy(&chunk).into_owned();
            }
        }
        self.calls.lock().unwrap().push(Call {
            declared: request.content_length,
            actual: total,
            first: first.unwrap_or_default(),
            last,
            headers: request.headers,
        });
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&json!({"id":"file_batch","purpose":"batch"}))
                .unwrap()
                .into(),
        }
        .into())
    }
}

fn profile(base_url: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"openai", "base_url":base_url,
        "protocol":"open_ai_responses", "auth":"none"
    }))
    .unwrap()
}

#[tokio::test]
async fn large_batch_upload_consumes_chunks_without_buffering() {
    let http = Capture::default();
    let profile = profile("https://api.openai.com/v1");
    let file = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let chunk = Bytes::from(vec![b'x'; 1024 * 1024]);
    let body = stream::iter((0..65).map(move |_| Ok(chunk.clone()))).boxed();
    let uploaded = file
        .upload_batch_stream(
            "input.jsonl",
            "application/x-ndjson",
            65 * 1024 * 1024,
            body,
            None,
        )
        .await
        .unwrap();
    assert_eq!(uploaded.file_id, "file_batch");
    assert_eq!(uploaded.purpose.as_deref(), Some("batch"));
    assert_eq!(uploaded.account_scope.as_deref(), Some("acct-a"));
    assert_eq!(uploaded.size_bytes, Some(65 * 1024 * 1024));
    let calls = http.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].declared, calls[0].actual);
    assert!(calls[0].first.contains("name=\"purpose\"\r\n\r\nbatch"));
    assert!(calls[0].first.contains("filename=\"input.jsonl\""));
    assert!(calls[0].last.ends_with("--\r\n"));
}

#[tokio::test]
async fn batch_stream_uses_the_profile_authenticator_and_scoped_reference() {
    let http = Capture::default();
    let mut profile = profile("https://api.openai.com/v1");
    profile.auth = AuthStrategy::ApiKey;
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".into());
    let file = FileService::new(&http, &profile, Some(&auth), Some(&key), Some("acct-a"));
    let uploaded = file
        .upload_batch_stream(
            "input.jsonl",
            "application/x-ndjson",
            1,
            stream::once(async { Ok(Bytes::from_static(b"x")) }).boxed(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(uploaded.profile_name, "openai");
    assert_eq!(uploaded.account_scope.as_deref(), Some("acct-a"));
    let calls = http.calls.lock().unwrap();
    assert!(calls[0]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
            && value == "Bearer test-key"));
}

#[tokio::test]
async fn batch_stream_timeout_covers_authentication_before_upload() {
    struct PendingAuth;
    #[async_trait]
    impl Authenticator for PendingAuth {
        async fn apply(
            &self,
            _: &mut HttpRequest,
            _: &ProviderProfile,
            _: Option<&Secret<String>>,
        ) -> Result<(), LlmError> {
            std::future::pending().await
        }
    }
    let http = Capture::default();
    let mut profile = profile("https://api.openai.com/v1");
    profile.auth = AuthStrategy::ApiKey;
    let auth = PendingAuth;
    let file = FileService::new(&http, &profile, Some(&auth), None, Some("acct-a"));
    assert!(matches!(
        file.upload_batch_stream(
            "input.jsonl",
            "application/x-ndjson",
            1,
            stream::once(async { Ok(Bytes::from_static(b"x")) }).boxed(),
            Some(std::time::Duration::from_millis(5)),
        )
        .await,
        Err(LlmError::TransportTimeout { .. })
    ));
    assert!(http.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn batch_upload_rejects_bad_scope_and_stream_length() {
    let http = Capture::default();
    let profile = profile("https://api.openai.com/v1");
    let missing_scope = FileService::new(&http, &profile, None, None, None);
    assert!(missing_scope
        .upload_batch_stream(
            "input.jsonl",
            "application/x-ndjson",
            1,
            stream::once(async { Ok(Bytes::from_static(b"x")) }).boxed(),
            None
        )
        .await
        .is_err());
    assert!(http.calls.lock().unwrap().is_empty());

    let file = FileService::new(&http, &profile, None, None, Some("acct-a"));
    for (declared, actual) in [(2, b"x".as_slice()), (1, b"xx".as_slice())] {
        let bytes = Bytes::copy_from_slice(actual);
        let error = file
            .upload_batch_stream(
                "input.jsonl",
                "application/x-ndjson",
                declared,
                stream::once(async move { Ok(bytes) }).boxed(),
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, LlmError::InvalidRequest { .. }));
    }
    assert!(http.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn transport_without_stream_support_does_not_consume_or_buffer_input() {
    struct BufferedOnly;
    #[async_trait]
    impl Transport for BufferedOnly {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            panic!("stream upload must not fall back to buffered send")
        }
    }
    let http = BufferedOnly;
    let profile = profile("https://api.openai.com/v1");
    let file = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let polled = Arc::new(AtomicBool::new(false));
    let flag = polled.clone();
    let body = stream::once(async move {
        flag.store(true, Ordering::SeqCst);
        Ok(Bytes::from_static(b"x"))
    })
    .boxed();
    assert!(matches!(
        file.upload_batch_stream("input.jsonl", "application/x-ndjson", 1, body, None)
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(!polled.load(Ordering::SeqCst));
}

#[tokio::test]
async fn builtin_transport_sends_exact_multipart_content_length() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut received = Vec::new();
        let header_end = loop {
            if let Some(index) = received.windows(4).position(|part| part == b"\r\n\r\n") {
                break index + 4;
            }
            let mut chunk = [0u8; 4096];
            let count = socket.read(&mut chunk).await.unwrap();
            assert!(count > 0);
            received.extend_from_slice(&chunk[..count]);
        };
        let headers = String::from_utf8_lossy(&received[..header_end]);
        let content_length: usize = headers
            .lines()
            .find_map(|line| {
                line.split_once(':')
                    .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.trim().parse().ok())
            })
            .unwrap();
        while received.len() - header_end < content_length {
            let mut chunk = [0u8; 4096];
            let count = socket.read(&mut chunk).await.unwrap();
            assert!(count > 0);
            received.extend_from_slice(&chunk[..count]);
        }
        let body = &received[header_end..header_end + content_length];
        assert_eq!(body, b"streamed-body");
        let response = b"{\"id\":\"file_batch\",\"purpose\":\"batch\"}";
        let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len());
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(response).await.unwrap();
    });
    let http = HttpTransport::new().unwrap();
    let response = HttpExecutor::new(&http)
        .execute_stream_bounded(
            HttpStreamRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url: format!("http://{addr}/upload"),
                headers: vec![("content-type".into(), "application/octet-stream".into())],
                body: stream::iter(vec![
                    Ok(Bytes::from_static(b"streamed-")),
                    Ok(Bytes::from_static(b"body")),
                ])
                .boxed(),
                content_length: 13,
                timeout: Some(std::time::Duration::from_secs(5)),
            },
            1024,
        )
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    server.await.unwrap();
}
