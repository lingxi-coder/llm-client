use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    files::UploadFileStream,
    protocol::{LlmError, Secret},
    qwen_knowledge::{
        QwenKnowledgeCategoryType, QwenKnowledgeError, QwenKnowledgeFileUploadRequest,
        QwenKnowledgeImportRequest, QwenKnowledgeParser, QwenKnowledgeParserConfig,
        QwenKnowledgeRegion, QwenKnowledgeRegisterFileRequest, QwenKnowledgeScope,
        QwenKnowledgeService,
    },
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

enum Reply {
    Json { status: u16, body: Value },
    Stream { status: u16 },
}

#[derive(Debug, Clone)]
struct RecordedStream {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    content_length: u64,
    body: Bytes,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
    streams: Mutex<Vec<RecordedStream>>,
    stream_attempts: AtomicUsize,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            streams: Mutex::new(Vec::new()),
            stream_attempts: AtomicUsize::new(0),
        }
    }

    fn next_reply(&self) -> Reply {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request")
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let Reply::Json { status, body } = self.next_reply() else {
            panic!("expected JSON request reply")
        };
        let bytes = serde_json::to_vec(&body).unwrap();
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "req-qwen-files-1".into())],
            body: stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        self.stream_attempts.fetch_add(1, Ordering::SeqCst);
        let mut body = request.body;
        let mut collected = Vec::new();
        while let Some(chunk) = body.next().await {
            collected.extend_from_slice(&chunk?);
        }
        self.streams.lock().unwrap().push(RecordedStream {
            method: request.method,
            url: request.url,
            headers: request.headers,
            content_length: request.content_length,
            body: Bytes::from(collected),
        });
        let Reply::Stream { status } = self.next_reply() else {
            panic!("expected streamed request reply")
        };
        Ok(StreamResponse {
            status,
            headers: vec![("x-oss-request-id".into(), "oss-req-1".into())],
            body: stream::empty().boxed(),
        })
    }
}

fn json_reply(body: Value) -> Reply {
    Reply::Json { status: 200, body }
}

fn scope(account: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new(
        "qwen-main",
        account,
        QwenKnowledgeRegion::Beijing,
        "llm-workspace-1",
    )
    .unwrap()
}

fn service<'a>(mock: &'a Mock, account: &str) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(
        mock,
        Secret::new("qwen-key-secret".to_owned()),
        scope(account),
    )
    .unwrap()
}

fn lease_body(url: &str, headers: Value) -> Value {
    json!({
        "code":"Success",
        "message":"",
        "requestId":"lease-request-1",
        "data":{
            "type":"OSS.PreSignedUrl",
            "param":{"url":url,"method":"PUT","headers":headers},
            "leaseId":"lease.opaque-123"
        },
        "status":200
    })
}

async fn requested_lease(
    service: &QwenKnowledgeService<'_>,
) -> Result<lingxi_llm_client::qwen_knowledge::QwenKnowledgeFileUploadLease, QwenKnowledgeError> {
    let request = QwenKnowledgeFileUploadRequest::new(
        "default",
        "guide.md",
        11,
        "d41d8cd98f00b204e9800998ecf8427e",
    );
    service.request_file_upload_lease(&request).await
}

fn upload_input<I>(chunks: I, polls: Arc<AtomicUsize>) -> UploadFileStream
where
    I: IntoIterator<Item = Result<Bytes, LlmError>> + Send + 'static,
    I::IntoIter: Send + 'static,
{
    let body = stream::iter(chunks).inspect(move |_| {
        polls.fetch_add(1, Ordering::SeqCst);
    });
    UploadFileStream::new("guide.md", "text/plain", 11, body)
}

#[tokio::test]
async fn lease_put_register_describe_and_typed_import_are_separate_scoped_calls() {
    let mock = Mock::new([
        json_reply(lease_body(
            "https://oss-example.aliyuncs.com/bucket/guide.md?signature=lease-secret",
            json!({"x-bailian-extra":"opaque-upload-token","Content-Type":"application/octet-stream"}),
        )),
        Reply::Stream { status: 200 },
        json_reply(json!({
            "code":"Success","status_code":200,
            "requestId":"register-request-1",
            "data":{"fileId":"file-123","parser":"AUTO_SELECT","status":"PARSING"}
        })),
        json_reply(json!({
            "code":"Success","status":200,
            "data":{"fileId":"file-123","status":"PARSING","fileName":"guide.md","fileType":"md","parser":"AUTO_SELECT","sizeBytes":11,"uploadTime":"2026-09-26 12:00:00","category":"default"}
        })),
        json_reply(json!({
            "code":"Success","success":true,"status_code":200,
            "data":{"ingestionId":"job-9","status":"PENDING"}
        })),
    ]);
    let service = service(&mock, "acct-1");
    let lease = requested_lease(&service).await.unwrap();
    assert_eq!(lease.file_name(), "guide.md");
    assert_eq!(lease.size_bytes(), 11);
    assert_eq!(
        lease.category_type(),
        QwenKnowledgeCategoryType::Unstructured
    );
    let lease_debug = format!("{lease:?}");
    assert!(!lease_debug.contains("lease-secret"));
    assert!(!lease_debug.contains("opaque-123"));

    let polls = Arc::new(AtomicUsize::new(0));
    service
        .upload_file_content(
            &lease,
            upload_input(
                [
                    Ok(Bytes::from_static(b"hello ")),
                    Ok(Bytes::from_static(b"world")),
                ],
                polls.clone(),
            ),
        )
        .await
        .unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), 2);

    let registered = service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::AutoSelect)
                .with_tags(["guide", "policy"]),
        )
        .await
        .unwrap();
    assert_eq!(registered.reference.file_id(), "file-123");
    assert_eq!(
        registered.reference.category_type(),
        Some(QwenKnowledgeCategoryType::Unstructured)
    );
    assert_eq!(registered.status.as_deref(), Some("PARSING"));

    let details = service.describe_file(&registered.reference).await.unwrap();
    assert_eq!(details.status.as_deref(), Some("PARSING"));
    assert_eq!(details.size_bytes, Some(11));

    let knowledge = service.knowledge_ref("index-1").unwrap();
    let import = QwenKnowledgeImportRequest::from_files([registered.reference.file_id()]);
    let result = service
        .submit_import_job_with_files(
            &knowledge,
            &import,
            std::slice::from_ref(&registered.reference),
        )
        .await
        .unwrap();
    assert_eq!(result.reference.job_id(), "job-9");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].method, "POST");
    assert!(requests[0].url.ends_with("/applyFileUploadLease"));
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "category":"default",
            "fileName":"guide.md",
            "sizeBytes":"11",
            "contentMd5":"d41d8cd98f00b204e9800998ecf8427e",
            "categoryType":"UNSTRUCTURED"
        })
    );
    assert_eq!(
        requests[1]
            .headers
            .iter()
            .find(|(key, _)| key == "authorization")
            .unwrap()
            .1,
        "Bearer qwen-key-secret"
    );
    assert!(requests[1].url.ends_with("/addFile"));
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
        json!({
            "leaseId":"lease.opaque-123",
            "category":"default",
            "categoryType":"UNSTRUCTURED",
            "parser":"AUTO_SELECT",
            "tags":["guide","policy"]
        })
    );
    assert!(requests[2].url.ends_with("/describeFile"));
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[2].body).unwrap(),
        json!({"fileId":"file-123"})
    );
    assert!(requests[3].url.ends_with("/job/create"));
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[3].body).unwrap()["docIds"],
        json!(["file-123"])
    );
    drop(requests);

    let streams = mock.streams.lock().unwrap();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].method, "PUT");
    assert!(streams[0]
        .url
        .starts_with("https://oss-example.aliyuncs.com/"));
    assert_eq!(streams[0].content_length, 11);
    assert_eq!(streams[0].body, Bytes::from_static(b"hello world"));
    assert!(streams[0]
        .headers
        .iter()
        .all(|(key, _)| !key.eq_ignore_ascii_case("authorization")));
    assert!(streams[0].headers.iter().any(|(key, value)| {
        key.eq_ignore_ascii_case("content-type") && value == "application/octet-stream"
    }));
    assert!(streams[0]
        .headers
        .iter()
        .any(|(key, val)| key == "x-bailian-extra" && val == "opaque-upload-token"));
}

#[tokio::test]
async fn lease_preflight_rejects_wrong_origin_and_credentials_before_stream_polling() {
    for (url, headers) in [
        (
            "https://console.aliyuncs.com/upload?signature=secret",
            json!({"Content-Type":"text/plain"}),
        ),
        (
            "https://bucket.oss-cn-beijing.aliyuncs.com/upload?signature=secret",
            json!({"Content-Type":"text/plain","authorization":"borrowed-key"}),
        ),
        (
            "https://bucket.oss-cn-beijing.aliyuncs.com/upload?signature=secret",
            json!({"Content-Type":"text/plain","content-type":"application/json"}),
        ),
    ] {
        let mock = Mock::new([json_reply(lease_body(url, headers))]);
        let service = service(&mock, "acct-1");
        let error = requested_lease(&service).await.unwrap_err();
        assert_eq!(
            error.dispatch(),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown
        );
        assert!(
            format!("{error:?}").contains("redacted")
                || format!("{error:?}").contains("ResponseOutcomeUnknown")
        );
        assert!(mock.streams.lock().unwrap().is_empty());
        assert_eq!(mock.stream_attempts.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn malformed_success_status_is_unknown_and_sensitive_lease_fields_are_redacted() {
    let missing_status = json!({
        "code":"Success",
        "data":{
            "type":"OSS.PreSignedUrl",
            "param":{"url":"https://oss-example.aliyuncs.com/path?secret=raw","method":"PUT","headers":{"Content-Type":"text/plain"}},
            "leaseId":"secret-lease-id"
        }
    });
    let conflict = json!({
        "code":"Success","status":200,"status_code":400,
        "data":{
            "type":"OSS.PreSignedUrl",
            "param":{"url":"https://oss-example.aliyuncs.com/path?secret=raw","method":"PUT","headers":{"Content-Type":"text/plain"}},
            "leaseId":"secret-lease-id"
        }
    });
    for body in [missing_status, conflict] {
        let mock = Mock::new([json_reply(body)]);
        let error = requested_lease(&service(&mock, "acct-1"))
            .await
            .unwrap_err();
        assert_eq!(
            error.dispatch(),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown
        );
        let debug = format!("{error:?}");
        assert!(!debug.contains("secret-lease-id"));
        assert!(!debug.contains("secret=raw"));
    }
}

#[tokio::test]
async fn file_envelope_rejects_false_or_malformed_success_and_marks_server_status_unknown() {
    let cases = [
        (
            json!({
                "code":"Success","success":false,"status":200,"status_code":200,
                "data":{"type":"OSS.PreSignedUrl"}
            }),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Rejected,
        ),
        (
            json!({
                "code":"Success","success":"true","status":200,
                "data":{"type":"OSS.PreSignedUrl"}
            }),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown,
        ),
        (
            json!({
                "code":"Success","status":503,
                "data":{"type":"OSS.PreSignedUrl"}
            }),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown,
        ),
    ];

    for (body, expected_dispatch) in cases {
        let mock = Mock::new([json_reply(body)]);
        let error = requested_lease(&service(&mock, "acct-1"))
            .await
            .unwrap_err();
        assert_eq!(error.dispatch(), expected_dispatch);
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn explicit_business_rejection_is_rejected_but_malformed_registration_is_unknown() {
    let rejected = Mock::new([json_reply(json!({
        "code":"InvalidParameter",
        "message":"invalid category",
        "status":400
    }))]);
    let error = requested_lease(&service(&rejected, "acct-1"))
        .await
        .unwrap_err();
    assert_eq!(
        error.dispatch(),
        lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Rejected
    );

    let malformed_registration = Mock::new([
        json_reply(lease_body(
            "https://oss-example.aliyuncs.com/path?sig=secret",
            json!({"Content-Type":"text/plain"}),
        )),
        json_reply(json!({
            "code":"Success","status_code":200,
            "data":{"parser":"AUTO_SELECT","status":"PARSING"}
        })),
    ]);
    let service = service(&malformed_registration, "acct-1");
    let lease = requested_lease(&service).await.unwrap();
    let error = service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::AutoSelect),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.dispatch(),
        lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown
    );
}

#[tokio::test]
async fn exact_size_stream_errors_are_unknown_mutations_and_never_close_successfully() {
    for chunks in [
        vec![Ok(Bytes::from_static(b"short"))],
        vec![Ok(Bytes::from_static(b"hello world!"))],
    ] {
        let mock = Mock::new([
            json_reply(lease_body(
                "https://bucket.oss-cn-beijing.aliyuncs.com/upload?signature=secret",
                json!({"Content-Type":"text/plain"}),
            )),
            Reply::Stream { status: 200 },
        ]);
        let service = service(&mock, "acct-1");
        let lease = requested_lease(&service).await.unwrap();
        let file = UploadFileStream::new("guide.md", "text/plain", 11, stream::iter(chunks));
        let error = service.upload_file_content(&lease, file).await.unwrap_err();
        assert_eq!(
            error.dispatch(),
            lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::Unknown
        );
        assert!(matches!(error, QwenKnowledgeError::OutcomeUnknown { .. }));
        assert!(mock.streams.lock().unwrap().is_empty());
        assert_eq!(mock.stream_attempts.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn upload_scope_and_metadata_mismatches_do_not_poll_or_dispatch_the_body() {
    let mock = Mock::new([json_reply(lease_body(
        "https://bucket.oss-cn-beijing.aliyuncs.com/upload?signature=secret",
        json!({"Content-Type":"text/plain"}),
    ))]);
    let service_a = service(&mock, "acct-1");
    let lease = requested_lease(&service_a).await.unwrap();
    let service_b = service(&mock, "acct-2");

    let polls = Arc::new(AtomicUsize::new(0));
    let error = service_b
        .upload_file_content(
            &lease,
            upload_input([Ok(Bytes::from_static(b"hello world"))], polls.clone()),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.dispatch(),
        lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::NotSent
    );
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(mock.stream_attempts.load(Ordering::SeqCst), 0);

    for (name, declared_size) in [("other.md", 11), ("guide.md", 10)] {
        let body = stream::iter([Ok::<_, LlmError>(Bytes::from_static(b"hello world"))]).inspect({
            let polls = polls.clone();
            move |_| {
                polls.fetch_add(1, Ordering::SeqCst);
            }
        });
        let error = service_a
            .upload_file_content(
                &lease,
                UploadFileStream::new(name, "text/plain", declared_size, body),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    }
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(mock.stream_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn missing_required_parser_configuration_fails_before_registration_dispatch() {
    let mock = Mock::new([json_reply(lease_body(
        "https://oss-example.aliyuncs.com/path?sig=secret",
        json!({"Content-Type":"text/plain"}),
    ))]);
    let service = service(&mock, "acct-1");
    let lease = requested_lease(&service).await.unwrap();
    let error = service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::DashQwenVlParser),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn typed_import_rejects_cross_scope_and_session_files_before_dispatch() {
    let mock = Mock::new([]);
    let first = service(&mock, "acct-1");
    let other_scope_file = scope("acct-2").file_ref("file-123").unwrap();
    let knowledge = first.knowledge_ref("index-1").unwrap();
    let request = QwenKnowledgeImportRequest::from_files(["file-123"]);
    let error = first
        .submit_import_job_with_files(&knowledge, &request, &[other_scope_file])
        .await
        .unwrap_err();
    assert_eq!(
        error.dispatch(),
        lingxi_llm_client::qwen_knowledge::QwenKnowledgeDispatch::NotSent
    );

    let session_file = scope("acct-1")
        .file_ref_with_category_type("file-session", QwenKnowledgeCategoryType::SessionFile)
        .unwrap();
    let session_request = QwenKnowledgeImportRequest::from_files(["file-session"]);
    let error = first
        .submit_import_job_with_files(&knowledge, &session_request, &[session_file])
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn qwen_vl_parser_accepts_multiline_prompt_and_emits_documented_config() {
    let mock = Mock::new([
        json_reply(lease_body(
            "https://oss-example.aliyuncs.com/path?sig=secret",
            json!({"Content-Type":"text/plain"}),
        )),
        json_reply(json!({
            "code":"Success","status_code":200,
            "data":{"fileId":"file-vl","parser":"DASH_QWEN_VL_PARSER","status":"PARSING"}
        })),
    ]);
    let service = service(&mock, "acct-1");
    let lease = requested_lease(&service).await.unwrap();
    service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::DashQwenVlParser)
                .with_parser_config(QwenKnowledgeParserConfig::new(
                    "Read the table.\nKeep row order.",
                )),
        )
        .await
        .unwrap();
    let requests = mock.requests.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap()["parserConfig"],
        json!({"modelName":"qwen3-vl-plus","modelPrompt":"Read the table.\nKeep row order."})
    );
}
