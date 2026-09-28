use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::batch::*,
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug)]
struct RecordedRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    content_length: Option<u64>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<RecordedRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn reply(&self) -> Reply {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request")
    }

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: futures::stream::iter([Ok(Bytes::from(reply.body))]).boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self.reply();
        self.requests.lock().unwrap().push(RecordedRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
            content_length: None,
        });
        Ok(Self::response(reply))
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let reply = self.reply();
        let mut body = request.body;
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        assert_eq!(bytes.len() as u64, request.content_length);
        self.requests.lock().unwrap().push(RecordedRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: bytes,
            content_length: Some(request.content_length),
        });
        Ok(Self::response(reply))
    }
}

fn scope(account: &str, region: QwenBatchRegion, workspace: Option<&str>) -> QwenBatchScope {
    QwenBatchScope::new("qwen-prod", account, region, workspace.map(str::to_owned)).unwrap()
}

fn service<'a>(mock: &'a MockTransport, account: &str) -> QwenBatchService<'a> {
    QwenBatchService::new(
        mock,
        scope(account, QwenBatchRegion::Beijing, Some("workspace-1")),
    )
    .unwrap()
}

fn chat_line(custom_id: &str, model: &str, thinking: Option<bool>) -> QwenBatchLine {
    QwenBatchLine {
        custom_id: custom_id.into(),
        body: QwenBatchRequestBody::Chat(QwenBatchChatRequest {
            model: model.into(),
            messages: vec![QwenBatchChatMessage {
                role: QwenBatchChatRole::User,
                content: "你好".into(),
            }],
            temperature: None,
            top_p: None,
            max_tokens: Some(64),
            enable_thinking: thinking,
        }),
    }
}

fn task(status: &str) -> Value {
    json!({
        "id":"batch_task_1",
        "object":"batch",
        "endpoint":"/v1/chat/completions",
        "input_file_id":"file-batch-input_1",
        "completion_window":"24h",
        "status":status,
        "output_file_id": if status == "completed" { Some("file-batch_output-result_1") } else { None::<&str> },
        "error_file_id": if status == "completed" { Some("file-batch_error-errors_1") } else { None::<&str> },
        "request_counts":{"total":1,"completed":1,"failed":0},
        "metadata":{"ds_name":"nightly"},
        "created_at":1711402400
    })
}

fn reply(status: u16, body: Value) -> Reply {
    Reply {
        status,
        body: serde_json::to_vec(&body).unwrap(),
    }
}

fn request_body(request: &RecordedRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn encodes_typed_chat_and_embedding_lines_as_supported_jsonl() {
    let chat = QwenBatchInput::new(vec![chat_line("row-1", "qwen-plus", Some(false))]).unwrap();
    let row: Value = serde_json::from_slice(chat.encode_jsonl().unwrap().trim_ascii_end()).unwrap();
    assert_eq!(row["custom_id"], "row-1");
    assert_eq!(row["method"], "POST");
    assert_eq!(row["url"], "/v1/chat/completions");
    assert_eq!(row["body"]["model"], "qwen-plus");
    assert_eq!(row["body"]["enable_thinking"], false);

    let embeddings = QwenBatchInput::new(vec![QwenBatchLine {
        custom_id: "emb-1".into(),
        body: QwenBatchRequestBody::Embeddings(QwenBatchEmbeddingRequest {
            model: "text-embedding-v3".into(),
            input: QwenBatchEmbeddingInput::Texts(vec!["one".into(), "two".into()]),
            encoding_format: Some(QwenBatchEncodingFormat::Float),
        }),
    }])
    .unwrap();
    let row: Value =
        serde_json::from_slice(embeddings.encode_jsonl().unwrap().trim_ascii_end()).unwrap();
    assert_eq!(row["url"], "/v1/embeddings");
    assert_eq!(row["body"]["input"], json!(["one", "two"]));
}

#[test]
fn rejects_mixed_models_thinking_modes_duplicates_and_out_of_range_windows() {
    assert!(QwenBatchInput::new(vec![
        chat_line("a", "qwen-plus", None),
        chat_line("b", "qwen-max", None),
    ])
    .is_err());
    assert!(QwenBatchInput::new(vec![
        chat_line("a", "qwen-plus", Some(true)),
        chat_line("b", "qwen-plus", Some(false)),
    ])
    .is_err());
    assert!(QwenBatchInput::new(vec![
        chat_line("same", "qwen-plus", None),
        chat_line("same", "qwen-plus", None),
    ])
    .is_err());
    assert!(QwenBatchCompletionWindow::from_hours(23).is_err());
    assert!(QwenBatchCompletionWindow::from_hours(337).is_err());
    assert_eq!(
        QwenBatchCompletionWindow::from_hours(336).unwrap().hours(),
        336
    );
}

#[tokio::test]
async fn upload_submit_query_cancel_and_stream_output_use_documented_routes() {
    let mock = Arc::new(MockTransport::new([
        reply(
            200,
            json!({"id":"file-batch-input_1","purpose":"batch","status":"processed"}),
        ),
        reply(200, task("validating")),
        reply(200, task("completed")),
        reply(200, task("cancelling")),
        Reply {
            status: 200,
            body: b"{\"custom_id\":\"row-1\",\"response\":{}}\n".to_vec(),
        },
        Reply {
            status: 200,
            body: b"{\"custom_id\":\"row-2\",\"error\":{}}\n".to_vec(),
        },
    ]));
    let service = service(&mock, "account-a");
    let input = QwenBatchInput::new(vec![chat_line("row-1", "qwen-plus", None)]).unwrap();
    let file = service
        .upload_input(&input, &request_options())
        .await
        .unwrap();
    assert_eq!(file.endpoint(), QwenBatchEndpoint::ChatCompletions);
    assert_eq!(file.scope().workspace_id(), Some("workspace-1"));

    let options = QwenBatchSubmitOptions {
        completion_window: QwenBatchCompletionWindow::from_hours(24).unwrap(),
        metadata: QwenBatchMetadata {
            name: Some("nightly".into()),
            description: Some("nightly qwen run".into()),
        },
    };
    let created = service
        .submit(&file, &options, &request_options())
        .await
        .unwrap();
    assert_eq!(created.status, QwenBatchStatus::Validating);
    let queried = service
        .query(&created.reference, &request_options())
        .await
        .unwrap();
    assert_eq!(queried.status, QwenBatchStatus::Completed);
    let cancelled = service
        .cancel(&queried.reference, &request_options())
        .await
        .unwrap();
    assert_eq!(cancelled.status, QwenBatchStatus::Cancelling);
    let output = queried.output_ref().unwrap();
    let mut content = service
        .stream_result(&output, &request_options())
        .await
        .unwrap();
    let first = content.next().await.unwrap().unwrap();
    assert!(String::from_utf8(first.to_vec()).unwrap().contains("row-1"));
    assert!(content.next().await.is_none());
    let error_file = queried.error_ref().unwrap();
    assert_eq!(error_file.file_id(), "file-batch_error-errors_1");
    let mut errors = service
        .stream_result(&error_file, &request_options())
        .await
        .unwrap();
    let first = errors.next().await.unwrap().unwrap();
    assert!(String::from_utf8(first.to_vec()).unwrap().contains("row-2"));
    assert!(errors.next().await.is_none());

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 6);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1/files"
    );
    assert_eq!(
        requests[0].content_length.unwrap() as usize,
        requests[0].body.len()
    );
    let upload_content_type = requests[0]
        .headers
        .iter()
        .find(|(name, _)| name == "Content-Type")
        .unwrap()
        .1
        .clone();
    assert!(upload_content_type.starts_with("multipart/form-data; boundary="));
    let upload_body = String::from_utf8(requests[0].body.clone()).unwrap();
    assert!(upload_body.contains("name=\"purpose\"\r\n\r\nbatch"));
    assert!(upload_body.contains("name=\"file\"; filename=\"batch-input.jsonl\""));
    assert!(upload_body.contains("\"url\":\"/v1/chat/completions\""));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "Authorization" && value == "Bearer sk-test-key"));

    assert_eq!(
        requests[1].url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1/batches"
    );
    let submitted = request_body(&requests[1]);
    assert_eq!(submitted["endpoint"], "/v1/chat/completions");
    assert_eq!(submitted["completion_window"], "24h");
    assert_eq!(submitted["metadata"]["ds_name"], "nightly");
    assert_eq!(requests[2].method, "GET");
    assert!(requests[2].url.ends_with("/batches/batch_task_1"));
    assert_eq!(requests[3].method, "POST");
    assert!(requests[3].url.ends_with("/batches/batch_task_1/cancel"));
    assert!(requests[4]
        .url
        .ends_with("/files/file-batch_output-result_1/content"));
    assert!(requests[5]
        .url
        .ends_with("/files/file-batch_error-errors_1/content"));
}

#[tokio::test]
async fn query_and_cancel_reject_response_for_a_different_task_id() {
    let input = QwenBatchInput::new(vec![chat_line("row-1", "qwen-plus", None)]).unwrap();
    let options = QwenBatchSubmitOptions::new(QwenBatchCompletionWindow::from_hours(24).unwrap());

    let query_mock = MockTransport::new([
        reply(
            200,
            json!({"id":"file-batch-input_1","purpose":"batch","status":"processed"}),
        ),
        reply(200, task("validating")),
        {
            let mut response = task("completed");
            response["id"] = json!("batch_task_other");
            reply(200, response)
        },
    ]);
    let query_service = service(&query_mock, "account-a");
    let file = query_service
        .upload_input(&input, &request_options())
        .await
        .unwrap();
    let submitted = query_service
        .submit(&file, &options, &request_options())
        .await
        .unwrap();
    assert!(matches!(
        query_service.query(&submitted.reference, &request_options()).await,
        Err(QwenBatchError::InvalidResponse(message))
            if message.contains("query response task ID")
    ));
    {
        let query_requests = query_mock.requests.lock().unwrap();
        assert_eq!(query_requests.len(), 3);
        assert!(query_requests[2].url.ends_with("/batches/batch_task_1"));
    }

    let cancel_mock = MockTransport::new([
        reply(
            200,
            json!({"id":"file-batch-input_1","purpose":"batch","status":"processed"}),
        ),
        reply(200, task("validating")),
        {
            let mut response = task("cancelling");
            response["id"] = json!("batch_task_other");
            reply(200, response)
        },
    ]);
    let cancel_service = service(&cancel_mock, "account-a");
    let file = cancel_service
        .upload_input(&input, &request_options())
        .await
        .unwrap();
    let submitted = cancel_service
        .submit(&file, &options, &request_options())
        .await
        .unwrap();
    assert!(matches!(
        cancel_service.cancel(&submitted.reference, &request_options()).await,
        Err(QwenBatchError::OutcomeUnknownResponse {
            operation: "cancel",
            expected_id,
            actual_id,
        }) if expected_id == "batch_task_1" && actual_id == "batch_task_other"
    ));
    let cancel_requests = cancel_mock.requests.lock().unwrap();
    assert_eq!(cancel_requests.len(), 3);
    assert!(cancel_requests[2]
        .url
        .ends_with("/batches/batch_task_1/cancel"));
}

#[tokio::test]
async fn list_uses_documented_filters_and_returns_scoped_task_references() {
    let mock = MockTransport::new([reply(
        200,
        json!({
            "object":"list",
            "data":[task("completed")],
            "first_id":"batch_task_1",
            "last_id":"batch_task_1",
            "has_more":true
        }),
    )]);
    let page = service(&mock, "account-a")
        .list(
            &QwenBatchListOptions::new()
                .after("batch_task_0")
                .limit(2)
                .task_name("nightly")
                .input_file_ids(vec!["file-batch-input_1".into()])
                .statuses(vec!["completed".into(), "expired".into()])
                .create_after("20250304000000")
                .create_before("20250306123000"),
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(page.batches.len(), 1);
    assert_eq!(page.batches[0].reference.job_id(), "batch_task_1");
    assert_eq!(
        page.batches[0].reference.scope().account_scope(),
        "account-a"
    );
    assert_eq!(page.first_id.as_deref(), Some("batch_task_1"));
    assert_eq!(page.last_id.as_deref(), Some("batch_task_1"));
    assert!(page.has_more);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
    assert!(requests[0]
        .url
        .starts_with("https://dashscope.aliyuncs.com/compatible-mode/v1/batches?"));
    for query in [
        "after=batch_task_0",
        "limit=2",
        "ds_name=nightly",
        "input_file_ids=file-batch-input_1",
        "status=completed%2Cexpired",
        "create_after=20250304000000",
        "create_before=20250306123000",
    ] {
        assert!(
            requests[0].url.contains(query),
            "missing {query} in {}",
            requests[0].url
        );
    }
}

#[tokio::test]
async fn list_preflights_options_and_rejects_non_advancing_pages() {
    let no_io = MockTransport::new([]);
    for options in [
        QwenBatchListOptions::new().limit(0),
        QwenBatchListOptions::new().limit(101),
        QwenBatchListOptions::new().after("not-a-batch-id"),
        QwenBatchListOptions::new().statuses(vec!["unknown".into()]),
        QwenBatchListOptions::new().create_after("2025-03-04"),
    ] {
        assert!(matches!(
            service(&no_io, "account-a")
                .list(&options, &request_options())
                .await,
            Err(QwenBatchError::InvalidInput(_))
        ));
    }
    assert!(no_io.requests.lock().unwrap().is_empty());

    let stuck_cursor = MockTransport::new([reply(
        200,
        json!({
            "object":"list",
            "data":[task("completed")],
            "first_id":"batch_task_1",
            "last_id":"batch_task_1",
            "has_more":true
        }),
    )]);
    assert!(matches!(
        service(&stuck_cursor, "account-a")
            .list(
                &QwenBatchListOptions::new().after("batch_task_1"),
                &request_options()
            )
            .await,
        Err(QwenBatchError::InvalidResponse(_))
    ));

    let duplicate_page = MockTransport::new([reply(
        200,
        json!({
            "object":"list",
            "data":[task("completed"), task("completed")],
            "first_id":"batch_task_1",
            "last_id":"batch_task_1",
            "has_more":false
        }),
    )]);
    assert!(matches!(
        service(&duplicate_page, "account-a")
            .list(&QwenBatchListOptions::new(), &request_options())
            .await,
        Err(QwenBatchError::InvalidResponse(_))
    ));
}

#[tokio::test]
async fn rejects_foreign_scope_before_network_and_maps_provider_errors() {
    let mock = MockTransport::new([
        reply(
            200,
            json!({"id":"file-batch-input_1","purpose":"batch","status":"processed"}),
        ),
        reply(200, task("validating")),
    ]);
    let source = service(&mock, "account-a");
    let input = QwenBatchInput::new(vec![chat_line("row-1", "qwen-plus", None)]).unwrap();
    let file = source
        .upload_input(&input, &request_options())
        .await
        .unwrap();
    let snapshot = source
        .submit(
            &file,
            &QwenBatchSubmitOptions::new(QwenBatchCompletionWindow::from_hours(24).unwrap()),
            &request_options(),
        )
        .await
        .unwrap();
    let count = mock.requests.lock().unwrap().len();

    let serialized = serde_json::to_value(&snapshot.reference).unwrap();
    assert_eq!(serialized["scope"]["provider_id"], "qwen");
    assert_eq!(serialized["scope"]["profile_name"], "qwen-prod");
    assert_eq!(serialized["scope"]["account_scope"], "account-a");
    assert_eq!(serialized["scope"]["region"], "beijing");
    assert_eq!(serialized["scope"]["workspace_id"], "workspace-1");

    let mut foreign_refs = Vec::new();
    for (path, value) in [
        (&["scope", "profile_name"][..], json!("qwen-other")),
        (&["scope", "account_scope"][..], json!("account-b")),
        (&["scope", "region"][..], json!("singapore")),
        (&["scope", "workspace_id"][..], json!("workspace-2")),
        (&["endpoint_fingerprint"][..], json!("different-endpoint")),
    ] {
        let mut value_json = serialized.clone();
        let mut target = &mut value_json;
        for part in &path[..path.len() - 1] {
            target = &mut target[*part];
        }
        target[path[path.len() - 1]] = value;
        foreign_refs.push(serde_json::from_value::<QwenBatchJobRef>(value_json).unwrap());
    }
    for foreign_ref in foreign_refs {
        assert!(matches!(
            source.query(&foreign_ref, &request_options()).await,
            Err(QwenBatchError::Llm(LlmError::PermissionDenied { .. }))
        ));
    }
    assert_eq!(mock.requests.lock().unwrap().len(), count);

    let wrong_account = QwenBatchService::new(
        &mock,
        scope("account-b", QwenBatchRegion::Beijing, Some("workspace-1")),
    )
    .unwrap();
    assert!(matches!(
        wrong_account
            .query(&snapshot.reference, &request_options())
            .await,
        Err(QwenBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), count);

    let denied = MockTransport::new([reply(401, json!({"message":"bad key"}))]);
    let denied_service = QwenBatchService::new(
        &denied,
        scope("account-a", QwenBatchRegion::Beijing, Some("workspace-1")),
    )
    .unwrap();
    assert!(matches!(
        denied_service
            .query(&snapshot.reference, &request_options())
            .await,
        Err(QwenBatchError::Llm(LlmError::Authentication { .. }))
    ));
}

#[tokio::test]
async fn singapore_scope_uses_singapore_batch_host_and_separate_scope() {
    let mock = MockTransport::new([
        reply(
            200,
            json!({"id":"file-batch-input_1","purpose":"batch","status":"processed"}),
        ),
        reply(200, task("validating")),
    ]);
    let service =
        QwenBatchService::new(&mock, scope("account-a", QwenBatchRegion::Singapore, None)).unwrap();
    let input = QwenBatchInput::new(vec![chat_line("row-1", "qwen-plus", None)]).unwrap();
    let file = service
        .upload_input(&input, &request_options())
        .await
        .unwrap();
    service
        .submit(
            &file,
            &QwenBatchSubmitOptions::new(QwenBatchCompletionWindow::from_hours(24).unwrap()),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(
        mock.requests.lock().unwrap()[0].url,
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/files"
    );
    assert_eq!(
        mock.requests.lock().unwrap()[1].url,
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/batches"
    );
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("sk-test-key".into())),
        ..Default::default()
    }
}

#[tokio::test]
async fn credentials_are_required_per_operation_and_can_rotate() {
    let mock = MockTransport::new([
        reply(401, json!({"error": "rejected"})),
        reply(401, json!({"error": "rejected"})),
    ]);
    let service = service(&mock, "account-a");
    let query = QwenBatchListOptions::default();
    for credential in [None, Some(Secret::new(" ".into()))] {
        let options = lingxi_llm_client::RequestOptions {
            credential,
            ..Default::default()
        };
        assert!(service.list(&query, &options).await.is_err());
    }
    assert!(mock.requests.lock().unwrap().is_empty());
    for key in ["first-key", "rotated-key"] {
        let options = lingxi_llm_client::RequestOptions {
            credential: Some(Secret::new(key.into())),
            ..Default::default()
        };
        assert!(service.list(&query, &options).await.is_err());
    }
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for (request, key) in requests.iter().zip(["first-key", "rotated-key"]) {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("Authorization")
                && value == &format!("Bearer {key}")));
    }
}
