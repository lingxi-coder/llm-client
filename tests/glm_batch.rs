use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use lingxi_llm_client::{
    glm_batch::{
        GlmBatchChatRequest, GlmBatchError, GlmBatchInput, GlmBatchLine, GlmBatchListOptions,
        GlmBatchMessage, GlmBatchMetadata, GlmBatchRegion, GlmBatchScope, GlmBatchService,
        GlmBatchStatus,
    },
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<Bytes>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn next_response(&self) -> Result<StreamResponse, LlmError> {
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: futures::stream::iter(reply.body.into_iter().map(Ok)).boxed(),
        })
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        self.next_response()
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let HttpStreamRequest {
            method,
            url,
            headers,
            mut body,
            timeout,
            ..
        } = request;
        let mut collected = BytesMut::new();
        while let Some(chunk) = body.next().await {
            collected.extend_from_slice(&chunk?);
        }
        self.requests.lock().unwrap().push(HttpRequest {
            method,
            url,
            headers,
            body: collected.freeze(),
            timeout,
        });
        self.next_response()
    }
}

fn response(status: u16, body: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: vec![Bytes::from(serde_json::to_vec(&body).unwrap())],
    })
}

fn raw_response(
    status: u16,
    chunks: impl IntoIterator<Item = &'static str>,
) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: chunks
            .into_iter()
            .map(|chunk| Bytes::from_static(chunk.as_bytes()))
            .collect(),
    })
}

fn scope(account: &str) -> GlmBatchScope {
    GlmBatchScope::new("zhipu-main", account, GlmBatchRegion::ChinaMainland).unwrap()
}

fn service<'a>(transport: &'a MockTransport, account: &str) -> GlmBatchService<'a> {
    GlmBatchService::new(
        transport,
        Secret::new("glm-test-key".into()),
        scope(account),
    )
    .unwrap()
}

fn input() -> GlmBatchInput {
    let body = GlmBatchChatRequest::new(
        "glm-5.3",
        vec![GlmBatchMessage {
            role: "user".into(),
            content: json!("hello"),
        }],
    )
    .unwrap()
    .with_parameter("temperature", json!(0.2))
    .unwrap();
    GlmBatchInput::new(vec![GlmBatchLine::new("row-1", body).unwrap()]).unwrap()
}

fn batch(id: &str, status: &str, output_file_id: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "batch",
        "endpoint": "/v4/chat/completions",
        "input_file_id": "file-input-1",
        "completion_window": "24h",
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": null,
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "future_field": "retained"
    })
}

#[test]
fn region_and_typed_input_validation_are_explicit() {
    assert!(matches!(
        GlmBatchScope::new("zhipu", "acct", GlmBatchRegion::International),
        Err(GlmBatchError::UnsupportedRegion { .. })
    ));
    assert_eq!(input().len(), 1);
    assert!(input().encode_jsonl().unwrap().ends_with(b"\n"));

    let first = GlmBatchLine::new(
        "same",
        GlmBatchChatRequest::new(
            "glm-5.3",
            vec![GlmBatchMessage {
                role: "user".into(),
                content: json!("x"),
            }],
        )
        .unwrap(),
    )
    .unwrap();
    let duplicate = first.clone();
    assert!(GlmBatchInput::new(vec![first, duplicate]).is_err());
    assert!(GlmBatchChatRequest::new("glm-5.3", vec![]).is_err());
    assert!(GlmBatchChatRequest::new(
        "glm-5.3",
        vec![GlmBatchMessage {
            role: "user".into(),
            content: json!("x")
        }],
    )
    .unwrap()
    .with_parameter("stream", json!(true))
    .is_err());
}

#[tokio::test]
async fn upload_submit_get_cancel_and_stream_output_use_scoped_routes() {
    let mock = MockTransport::new([
        response(200, json!({"id":"file-input-1", "purpose":"batch"})),
        response(200, batch("batch_1", "validating", None)),
        response(200, batch("batch_1", "completed", Some("file-output-1"))),
        response(200, batch("batch_1", "cancelling", Some("file-output-1"))),
        raw_response(200, ["{\"custom_id\":\"row-1\",", "\"response\":{}}\n"]),
    ]);
    let glm = service(&mock, "account-a");
    let uploaded = glm.upload_input(&input()).await.unwrap();
    assert_eq!(uploaded.file_id(), "file-input-1");
    let metadata = GlmBatchMetadata {
        values: [("model".into(), "glm-5.3".into())].into(),
    };
    let created = glm.submit(&uploaded, &metadata).await.unwrap();
    assert_eq!(created.reference.batch_id(), "batch_1");
    assert_eq!(created.request_counts.unwrap().total, 1);
    assert_eq!(created.native["future_field"], "retained");

    let fetched = glm.get(&created.reference).await.unwrap();
    assert_eq!(fetched.status, GlmBatchStatus::Completed);
    assert!(fetched.status.is_terminal());
    let output = fetched.output_ref().unwrap();
    let cancelled = glm.cancel(&created.reference).await.unwrap();
    assert_eq!(cancelled.status, GlmBatchStatus::Cancelling);
    let stream = glm.stream_result(&output).await.unwrap();
    let bytes = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let actual = bytes
        .into_iter()
        .flat_map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(actual, b"{\"custom_id\":\"row-1\",\"response\":{}}\n");

    let requests = mock.requests();
    assert_eq!(requests.len(), 5);
    assert_eq!(
        requests[0].url,
        "https://open.bigmodel.cn/api/paas/v4/files"
    );
    assert!(String::from_utf8_lossy(&requests[0].body).contains("name=\"purpose\"\r\n\r\nbatch"));
    assert!(String::from_utf8_lossy(&requests[0].body).contains("\"url\":\"/v4/chat/completions\""));
    assert_eq!(
        requests[1].url,
        "https://open.bigmodel.cn/api/paas/v4/batches"
    );
    let submit: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(submit["input_file_id"], "file-input-1");
    assert_eq!(submit["endpoint"], "/v4/chat/completions");
    assert_eq!(submit["completion_window"], "24h");
    assert_eq!(submit["metadata"]["model"], "glm-5.3");
    assert_eq!(
        requests[2].url,
        "https://open.bigmodel.cn/api/paas/v4/batches/batch_1"
    );
    assert_eq!(
        requests[3].url,
        "https://open.bigmodel.cn/api/paas/v4/batches/batch_1/cancel"
    );
    assert_eq!(
        requests[4].url,
        "https://open.bigmodel.cn/api/paas/v4/files/file-output-1/content"
    );
}

#[tokio::test]
async fn list_fetches_one_scoped_page_and_validates_options_before_io() {
    let mock = MockTransport::new([response(
        200,
        json!({
            "object":"list",
            "data":[batch("batch_2", "completed", Some("file-output-2"))],
            "first_id":"batch_2",
            "last_id":"batch_2",
            "has_more":true
        }),
    )]);
    let glm = service(&mock, "account-a");
    let page = glm
        .list(&GlmBatchListOptions::new().after("batch_1").limit(7))
        .await
        .unwrap();

    assert_eq!(page.batches.len(), 1);
    assert_eq!(page.batches[0].reference.batch_id(), "batch_2");
    assert_eq!(
        page.batches[0].reference.scope().account_scope(),
        "account-a"
    );
    assert_eq!(page.first_id.as_deref(), Some("batch_2"));
    assert_eq!(page.last_id.as_deref(), Some("batch_2"));
    assert!(page.has_more);

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert!(requests[0]
        .url
        .starts_with("https://open.bigmodel.cn/api/paas/v4/batches?"));
    assert!(requests[0].url.contains("after=batch_1"));
    assert!(requests[0].url.contains("limit=7"));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "Authorization" && value == "Bearer glm-test-key"));

    let no_io = MockTransport::new([]);
    let invalid = service(&no_io, "account-a")
        .list(&GlmBatchListOptions::new().limit(0))
        .await
        .unwrap_err();
    assert!(matches!(invalid, GlmBatchError::InvalidInput(_)));
    assert!(no_io.requests().is_empty());
    let invalid = service(&no_io, "account-a")
        .list(&GlmBatchListOptions::new().after("bad/id"))
        .await
        .unwrap_err();
    assert!(matches!(invalid, GlmBatchError::InvalidInput(_)));
    assert!(no_io.requests().is_empty());
}

#[tokio::test]
async fn list_rejects_duplicate_or_non_advancing_pages() {
    let empty_page = MockTransport::new([response(
        200,
        json!({
            "object":"list",
            "data":[],
            "first_id":null,
            "last_id":null,
            "has_more":true
        }),
    )]);
    assert!(matches!(
        service(&empty_page, "account-a")
            .list(&GlmBatchListOptions::new())
            .await,
        Err(GlmBatchError::InvalidResponse { .. })
    ));

    let duplicate_page = MockTransport::new([response(
        200,
        json!({
            "object":"list",
            "data":[
                batch("batch_2", "completed", None),
                batch("batch_2", "completed", None)
            ],
            "first_id":"batch_2",
            "last_id":"batch_2",
            "has_more":false
        }),
    )]);
    assert!(matches!(
        service(&duplicate_page, "account-a")
            .list(&GlmBatchListOptions::new())
            .await,
        Err(GlmBatchError::InvalidResponse { .. })
    ));

    let stuck_cursor = MockTransport::new([response(
        200,
        json!({
            "object":"list",
            "data":[batch("batch_2", "completed", None)],
            "first_id":"batch_2",
            "last_id":"batch_2",
            "has_more":true
        }),
    )]);
    assert!(matches!(
        service(&stuck_cursor, "account-a")
            .list(&GlmBatchListOptions::new().after("batch_2"))
            .await,
        Err(GlmBatchError::InvalidResponse { .. })
    ));
}

#[tokio::test]
async fn mutation_unknown_is_not_retried_and_references_cannot_cross_accounts() {
    let mock = MockTransport::new([
        response(200, json!({"id":"file-input-1", "purpose":"batch"})),
        Err(LlmError::Transport {
            message: "socket closed".into(),
        }),
    ]);
    let glm = service(&mock, "account-a");
    let file = glm.upload_input(&input()).await.unwrap();
    assert!(matches!(
        glm.submit(&file, &GlmBatchMetadata::default()).await,
        Err(GlmBatchError::OutcomeUnknown {
            operation: "submit",
            ..
        })
    ));
    assert_eq!(mock.requests().len(), 2);

    let other_mock = MockTransport::new([]);
    let other = service(&other_mock, "account-b");
    let reference = GlmBatchJobRefForTest::reference();
    assert!(matches!(
        other.get(&reference).await,
        Err(GlmBatchError::ScopeMismatch)
    ));
    assert!(other_mock.requests().is_empty());
}

struct GlmBatchJobRefForTest;

impl GlmBatchJobRefForTest {
    fn reference() -> lingxi_llm_client::glm_batch::GlmBatchJobRef {
        serde_json::from_value(json!({
            "scope": {
                "provider_id":"zhipu",
                "profile_name":"zhipu-main",
                "account_scope":"account-a",
                "region":"china_mainland",
                "endpoint_fingerprint": scope("account-a").endpoint_fingerprint()
            },
            "endpoint_fingerprint": scope("account-a").endpoint_fingerprint(),
            "batch_id":"batch_1"
        }))
        .unwrap()
    }
}
