use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    qwen_tts::{
        QwenTtsDispatchOutcome, QwenTtsError, QwenTtsLanguage, QwenTtsRegion, QwenTtsRequest,
        QwenTtsScope, QwenTtsService, QwenTtsStreamEventKind, QWEN_TTS_BEIJING_HTTP_ENDPOINT,
        QWEN_TTS_SINGAPORE_HTTP_ENDPOINT,
    },
    HttpRequest, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug)]
struct Sent {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    sent: Mutex<Vec<Sent>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn sent(&self) -> Vec<Sent> {
        self.sent.lock().unwrap().clone()
    }
}

impl Clone for Sent {
    fn clone(&self) -> Self {
        Self {
            method: self.method.clone(),
            url: self.url.clone(),
            headers: self.headers.clone(),
            body: self.body.clone(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(Sent {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Qwen TTS request")?;
        Ok(StreamResponse {
            status: reply.status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn scope(region: QwenTtsRegion) -> QwenTtsScope {
    QwenTtsScope::new("qwen-main", "account-42", region, "workspace-abc").unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("region-specific-key".to_owned())),
        account_scope: Some("account-42".to_owned()),
        ..Default::default()
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
    })
}

fn success_response() -> Value {
    json!({
        "request_id": "tts-request-1",
        "output": {
            "finish_reason": "stop",
            "audio": {
                "data": "",
                "url": "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/out.wav?Expires=1&Signature=private",
                "id": "audio-1",
                "expires_at": 1_800_000_000_i64
            }
        },
        "usage": { "characters": 12 }
    })
}

#[tokio::test]
async fn synthesizes_once_with_official_qwen3_tts_wire_and_returns_signed_url() {
    let transport = MockTransport::new([reply(200, success_response())]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Singapore)).unwrap();
    let request =
        QwenTtsRequest::new("你好，世界。", "Cherry").with_language_type(QwenTtsLanguage::Chinese);

    let result = service.synthesize(&request, &options()).await.unwrap();
    assert_eq!(result.request_id.as_deref(), Some("tts-request-1"));
    assert_eq!(result.audio_id.as_deref(), Some("audio-1"));
    assert_eq!(result.expires_at_unix_seconds, Some(1_800_000_000));
    assert_eq!(result.characters, Some(12));
    assert_eq!(result.scope().region(), QwenTtsRegion::Singapore);
    assert!(!format!("{result:?}").contains("Signature=private"));

    let sent = transport.sent();
    assert_eq!(
        sent.len(),
        1,
        "one synthesis call must not fetch its audio URL"
    );
    assert_eq!(sent[0].method, "POST");
    assert_eq!(sent[0].url, QWEN_TTS_SINGAPORE_HTTP_ENDPOINT);
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer region-specific-key"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "qwen3-tts-flash");
    assert_eq!(body["input"]["text"], "你好，世界。");
    assert_eq!(body["input"]["voice"], "Cherry");
    assert_eq!(body["input"]["language_type"], "Chinese");
    assert!(body["input"].get("instructions").is_none());
}

#[tokio::test]
async fn request_validation_and_account_mismatch_fail_before_send() {
    let transport = MockTransport::new([]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();

    let long_text = "x".repeat(601);
    let error = service
        .synthesize(&QwenTtsRequest::new(long_text, "Cherry"), &options())
        .await
        .unwrap_err();
    assert!(matches!(error, QwenTtsError::InvalidInput(_)));
    assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::NotSent);

    let wrong_account = RequestOptions {
        account_scope: Some("different-account".into()),
        ..options()
    };
    assert!(matches!(
        service
            .synthesize(&QwenTtsRequest::new("hello", "Cherry"), &wrong_account)
            .await,
        Err(QwenTtsError::ScopeMismatch)
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn region_selects_the_matching_dashscope_host() {
    let transport = MockTransport::new([
        reply(200, success_response()),
        reply(200, success_response()),
    ]);
    let request = QwenTtsRequest::new("hello", "Cherry");
    let options = options();

    let beijing = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    beijing.synthesize(&request, &options).await.unwrap();
    let singapore = QwenTtsService::new(&transport, scope(QwenTtsRegion::Singapore)).unwrap();
    singapore.synthesize(&request, &options).await.unwrap();

    let sent = transport.sent();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].url, QWEN_TTS_BEIJING_HTTP_ENDPOINT);
    assert_eq!(sent[1].url, QWEN_TTS_SINGAPORE_HTTP_ENDPOINT);
}

#[tokio::test]
async fn missing_key_is_not_sent() {
    let transport = MockTransport::new([]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let error = service
        .synthesize(
            &QwenTtsRequest::new("hello", "Cherry"),
            &RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::NotSent);
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn network_failure_is_unknown_and_not_retried() {
    let transport = MockTransport::new([Err(LlmError::Transport {
        message: "connection lost after request send".into(),
    })]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let error = service
        .synthesize(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::Unknown);
    assert_eq!(transport.sent().len(), 1);
}

#[tokio::test]
async fn rejection_and_accepted_malformed_response_have_distinct_outcomes() {
    let rejected_transport = MockTransport::new([reply(
        400,
        json!({ "code": "InvalidParameter", "message": "unsupported voice" }),
    )]);
    let service = QwenTtsService::new(&rejected_transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let rejected = service
        .synthesize(&QwenTtsRequest::new("hello", "unknown"), &options())
        .await
        .unwrap_err();
    assert_eq!(
        rejected.dispatch_outcome(),
        QwenTtsDispatchOutcome::Rejected
    );

    let malformed_transport = MockTransport::new([reply(
        200,
        json!({
            "request_id": "accepted-id",
            "output": { "finish_reason": "stop", "audio": { "data": "" } }
        }),
    )]);
    let service = QwenTtsService::new(&malformed_transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let malformed = service
        .synthesize(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap_err();
    assert_eq!(
        malformed.dispatch_outcome(),
        QwenTtsDispatchOutcome::Accepted
    );
    assert!(matches!(
        malformed,
        QwenTtsError::AcceptedInvalidResponse {
            request_id: Some(ref id),
            ..
        } if id == "accepted-id"
    ));
}

struct SseTransport {
    response: Mutex<Option<StreamResponse>>,
    sent: Mutex<Vec<HttpRequest>>,
}

impl SseTransport {
    fn chunks(chunks: Vec<Result<Bytes, LlmError>>) -> Self {
        Self::with_body(stream::iter(chunks).boxed())
    }

    fn with_body(body: futures::stream::BoxStream<'static, Result<Bytes, LlmError>>) -> Self {
        Self {
            response: Mutex::new(Some(StreamResponse {
                status: 200,
                headers: vec![(
                    "content-type".into(),
                    "text/event-stream; charset=utf-8".into(),
                )],
                body,
            })),
            sent: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for SseTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        Ok(self
            .response
            .lock()
            .unwrap()
            .take()
            .expect("unexpected retry"))
    }
}

fn sse(value: Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn delta_response() -> Value {
    json!({
        "request_id": "tts-request-1",
        "output": {
            "finish_reason": null,
            "audio": { "data": "AQACAAMA", "id": "audio-1" }
        },
        "usage": { "characters": 3 }
    })
}

#[tokio::test]
async fn sse_decodes_pcm_and_returns_final_url_and_native_usage_after_clean_eof() {
    for region in [QwenTtsRegion::Beijing, QwenTtsRegion::Singapore] {
        let mut final_response = success_response();
        final_response["usage"]["provider_extension"] = json!(7);
        let transport = SseTransport::chunks(vec![
            Ok(sse(delta_response())),
            Ok(sse(final_response)),
            Ok(Bytes::from_static(b": final keep-alive\n\n")),
        ]);
        let service = QwenTtsService::new(&transport, scope(region)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        assert_eq!(output.scope().region(), region);
        let delta = output.next_event().await.unwrap().unwrap();
        assert!(
            matches!(&delta.kind, QwenTtsStreamEventKind::AudioDelta { data }
            if data.as_ref() == [1, 0, 2, 0, 3, 0])
        );
        assert_eq!(delta.native["output"]["audio"]["id"], "audio-1");
        assert_eq!(delta.native["usage"]["characters"], 3);
        assert!(!format!("{delta:?}").contains("AQACAAMA"));
        assert_eq!(output.request_id(), Some("tts-request-1"));

        let completed = output.next_event().await.unwrap().unwrap();
        let QwenTtsStreamEventKind::Completed { synthesis } = &completed.kind else {
            panic!("expected completion after clean EOF")
        };
        assert_eq!(synthesis.characters, Some(12));
        assert_eq!(synthesis.finish_reason.as_deref(), Some("stop"));
        assert_eq!(synthesis.audio_id.as_deref(), Some("audio-1"));
        assert_eq!(synthesis.expires_at_unix_seconds, Some(1_800_000_000));
        assert!(synthesis.audio_url().contains("Signature=private"));
        assert_eq!(completed.native["usage"]["provider_extension"], 7);
        assert!(!format!("{completed:?}").contains("Signature=private"));
        assert!(output.next_event().await.unwrap().is_none());

        let sent = transport.sent.lock().unwrap();
        assert_eq!(sent.len(), 1, "stream does not download or retry");
        let endpoint = match region {
            QwenTtsRegion::Beijing => QWEN_TTS_BEIJING_HTTP_ENDPOINT,
            QwenTtsRegion::Singapore => QWEN_TTS_SINGAPORE_HTTP_ENDPOINT,
        };
        assert_eq!(sent[0].url, endpoint);
        assert!(
            sent[0]
                .headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("X-DashScope-SSE")
                    && value == "enable")
        );
        assert!(sent[0].headers.iter().any(
            |(name, value)| name.eq_ignore_ascii_case("accept") && value == "text/event-stream"
        ));
        let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
        assert_eq!(body["model"], "qwen3-tts-flash");
        assert_eq!(body["input"]["text"], "hello");
        assert!(
            body.get("stream").is_none(),
            "streaming is a native HTTP header"
        );
    }
}

#[tokio::test]
async fn sse_accepts_every_byte_boundary_and_unterminated_final_event() {
    let delta = sse(delta_response());
    let final_event = format!("data: {}", success_response());
    let wire = [delta.as_ref(), final_event.as_bytes()].concat();
    let transport = SseTransport::chunks(
        wire.into_iter()
            .map(|byte| Ok(Bytes::from(vec![byte])))
            .collect(),
    );
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let mut output = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap();
    assert!(matches!(
        output.next_event().await.unwrap().unwrap().kind,
        QwenTtsStreamEventKind::AudioDelta { .. }
    ));
    assert!(matches!(
        output.next_event().await.unwrap().unwrap().kind,
        QwenTtsStreamEventKind::Completed { .. }
    ));
    assert!(output.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_url_without_stop_is_not_completion() {
    let mut value = success_response();
    value["output"]["finish_reason"] = Value::Null;
    let transport = SseTransport::chunks(vec![Ok(sse(value))]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let mut output = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap();
    assert!(matches!(
        output.next_event().await.unwrap().unwrap().kind,
        QwenTtsStreamEventKind::Metadata
    ));
    let error = output.next_event().await.unwrap_err();
    assert!(matches!(error, QwenTtsError::StreamInterrupted { .. }));
    assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::Accepted);
    assert!(output.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_empty_or_audio_only_eof_is_interrupted() {
    for chunks in [vec![], vec![Ok(sse(delta_response()))]] {
        let transport = SseTransport::chunks(chunks);
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        let first = output.next_event().await;
        let error = match first {
            Ok(_) => output.next_event().await.unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(error, QwenTtsError::StreamInterrupted { .. }));
        assert!(output.next_event().await.unwrap().is_none());
    }
}

#[tokio::test]
async fn sse_never_reports_completion_before_a_late_native_error() {
    for same_chunk in [true, false] {
        let final_event = sse(success_response());
        let error_event =
            sse(json!({"code":"InternalError", "message":"late failure", "request_id":"late-id"}));
        let chunks = if same_chunk {
            vec![Ok(Bytes::from(
                [final_event.as_ref(), error_event.as_ref()].concat(),
            ))]
        } else {
            vec![Ok(final_event), Ok(error_event)]
        };
        let transport = SseTransport::chunks(chunks);
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        assert!(
            matches!(output.next_event().await, Err(QwenTtsError::StreamProvider { error })
            if error.code.as_deref() == Some("InternalError") && error.request_id.as_deref() == Some("late-id"))
        );
        assert!(output.next_event().await.unwrap().is_none());
    }
}

#[tokio::test]
async fn sse_preserves_audio_before_error_but_does_not_hide_transport_error_after_stop() {
    let transport = SseTransport::chunks(vec![
        Ok(sse(delta_response())),
        Ok(sse(success_response())),
        Err(LlmError::StreamInterrupted {
            message: "lost final response bytes".into(),
        }),
    ]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let mut output = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap();
    assert!(matches!(
        output.next_event().await.unwrap().unwrap().kind,
        QwenTtsStreamEventKind::AudioDelta { .. }
    ));
    assert!(matches!(
        output.next_event().await,
        Err(QwenTtsError::StreamInterrupted { .. })
    ));
    assert!(output.next_event().await.unwrap().is_none());
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn sse_rejects_invalid_audio_final_payloads_and_late_data() {
    let mut invalid_base64 = delta_response();
    invalid_base64["output"]["audio"]["data"] = json!("!not-base64");
    let mut missing_url = success_response();
    missing_url["output"]["audio"]
        .as_object_mut()
        .unwrap()
        .remove("url");
    let mut nonempty_final_audio = success_response();
    nonempty_final_audio["output"]["audio"]["data"] = json!("AQACAA==");
    let mut unknown_finish = delta_response();
    unknown_finish["output"]["finish_reason"] = json!("length");
    for chunks in [
        vec![Ok(sse(invalid_base64))],
        vec![Ok(sse(missing_url))],
        vec![Ok(sse(nonempty_final_audio))],
        vec![Ok(sse(unknown_finish))],
        vec![Ok(Bytes::from_static(b"data: not json\n\n"))],
        vec![Ok(sse(success_response())), Ok(sse(delta_response()))],
        vec![Ok(sse(success_response())), Ok(sse(success_response()))],
    ] {
        let transport = SseTransport::chunks(chunks);
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        assert!(matches!(
            output.next_event().await,
            Err(QwenTtsError::AcceptedInvalidResponse { .. })
        ));
        assert!(output.next_event().await.unwrap().is_none());
    }
}

#[tokio::test]
async fn sse_preflight_uses_existing_validation_and_account_scope() {
    let transport = SseTransport::chunks(vec![]);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    for (request, opts) in [
        (QwenTtsRequest::new("x".repeat(601), "Cherry"), options()),
        (QwenTtsRequest::new("hello", ""), options()),
        (
            QwenTtsRequest::new("hello", "Cherry"),
            RequestOptions::default(),
        ),
        (
            QwenTtsRequest::new("hello", "Cherry"),
            RequestOptions {
                account_scope: Some("wrong-account".into()),
                ..options()
            },
        ),
    ] {
        let error = service
            .synthesize_stream(&request, &opts)
            .await
            .err()
            .unwrap();
        assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::NotSent);
    }
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sse_http_errors_keep_dispatch_classification() {
    for status in [400, 408, 429, 500] {
        let transport =
            MockTransport::new([reply(status, json!({"code":"failed", "message":"failure"}))]);
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let error = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .err()
            .unwrap();
        assert_eq!(
            error.dispatch_outcome(),
            if status < 500 && status != 408 {
                QwenTtsDispatchOutcome::Rejected
            } else {
                QwenTtsDispatchOutcome::Unknown
            }
        );
        assert_eq!(transport.sent().len(), 1);
    }
}

#[tokio::test]
async fn sse_provider_errors_retain_native_diagnostics_usage_and_request_id() {
    for native in [
        json!({"code":"InternalError", "message":"failed", "usage":{"characters":4}, "detail":{"trace":"secret"}}),
        json!({"status_code":500, "message":"failed", "usage":{"characters":4}, "detail":{"trace":"secret"}}),
    ] {
        let transport = SseTransport::chunks(vec![Ok(sse(native.clone()))]);
        transport
            .response
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .headers
            .push(("x-request-id".into(), "header-id".into()));
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        let error = output.next_event().await.unwrap_err();
        assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::Accepted);
        let QwenTtsError::StreamProvider { error } = error else {
            panic!("expected native provider error");
        };
        assert_eq!(error.native, native);
        assert_eq!(error.request_id.as_deref(), Some("header-id"));
        assert!(!format!("{error:?}").contains("secret"));
        assert!(output.next_event().await.unwrap().is_none());
    }
}

#[tokio::test]
async fn http_408_retains_header_request_id_and_is_unknown_in_both_modes() {
    for streaming in [false, true] {
        let transport =
            SseTransport::chunks(vec![Ok(Bytes::from_static(br#"{"message":"timed out"}"#))]);
        {
            let mut response = transport.response.lock().unwrap();
            let response = response.as_mut().unwrap();
            response.status = 408;
            response
                .headers
                .push(("x-request-id".into(), "header-id".into()));
        }
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let request = QwenTtsRequest::new("hello", "Cherry");
        let error = if streaming {
            service
                .synthesize_stream(&request, &options())
                .await
                .err()
                .unwrap()
        } else {
            service.synthesize(&request, &options()).await.unwrap_err()
        };
        assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::Unknown);
        assert!(
            matches!(error, QwenTtsError::OutcomeUnknown { request_id, .. } if request_id.as_deref() == Some("header-id"))
        );
        assert_eq!(transport.sent.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn sse_rejects_explicit_non_sse_content_type() {
    let transport = SseTransport::chunks(vec![Ok(sse(success_response()))]);
    transport.response.lock().unwrap().as_mut().unwrap().headers =
        vec![("content-type".into(), "application/json".into())];
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let error = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .err()
        .unwrap();
    assert_eq!(error.dispatch_outcome(), QwenTtsDispatchOutcome::Accepted);
}

#[tokio::test]
async fn sse_bounds_unterminated_events_and_total_wire_bytes() {
    let oversize_event = Bytes::from(vec![b'x'; 8 * 1024 * 1024 + 1]);
    let comment = Bytes::from([b": ".as_slice(), &vec![b'x'; 1024 * 1024 - 4], b"\n\n"].concat());
    let mut oversized_wire = vec![Ok(comment); 64];
    oversized_wire.push(Ok(Bytes::from_static(b"x")));
    for chunks in [vec![Ok(oversize_event)], oversized_wire] {
        let transport = SseTransport::chunks(chunks);
        let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
        let mut output = service
            .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
            .await
            .unwrap();
        assert!(
            matches!(output.next_event().await, Err(QwenTtsError::StreamInterrupted { source: LlmError::StreamInterrupted { message }, .. }) if message.contains("limit"))
        );
        assert!(output.next_event().await.unwrap().is_none());
    }
}

#[tokio::test(start_paused = true)]
async fn sse_deadline_applies_while_waiting_for_clean_eof() {
    let body = stream::once(async { Ok(sse(success_response())) })
        .chain(stream::pending())
        .boxed();
    let transport = SseTransport::with_body(body);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let opts = RequestOptions {
        total_timeout: Some(std::time::Duration::from_secs(1)),
        ..options()
    };
    let mut output = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &opts)
        .await
        .unwrap();
    assert!(matches!(
        output.next_event().await,
        Err(QwenTtsError::StreamInterrupted {
            source: LlmError::TransportTimeout { .. },
            ..
        })
    ));
    assert!(output.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn sse_drop_releases_body_without_waiting_or_retrying() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct DropProbe(Arc<AtomicBool>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = DropProbe(dropped.clone());
    let body = stream::unfold(probe, |probe| async move {
        futures::future::pending::<()>().await;
        Some((Ok(Bytes::new()), probe))
    })
    .boxed();
    let transport = SseTransport::with_body(body);
    let service = QwenTtsService::new(&transport, scope(QwenTtsRegion::Beijing)).unwrap();
    let output = service
        .synthesize_stream(&QwenTtsRequest::new("hello", "Cherry"), &options())
        .await
        .unwrap();
    assert!(!dropped.load(Ordering::SeqCst));
    drop(output);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}
