use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::knowledge::{
        QwenKnowledgeAddChunkRequest, QwenKnowledgeChunkFields, QwenKnowledgeDispatch,
        QwenKnowledgeError, QwenKnowledgeListChunksRequest, QwenKnowledgeRegion,
        QwenKnowledgeScope, QwenKnowledgeService, QwenKnowledgeUpdateChunkRequest,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

enum MockReply {
    Response { status: u16, body: Value },
    TransportFailure,
}

struct Mock {
    replies: Mutex<VecDeque<MockReply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = MockReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request")
        {
            MockReply::Response { status, body } => {
                let bytes = serde_json::to_vec(&body).unwrap();
                Ok(StreamResponse {
                    status,
                    headers: vec![("x-request-id".into(), "req-qwen-chunk-1".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            MockReply::TransportFailure => Err(LlmError::TransportTimeout {
                message: "mock timeout after request dispatch".into(),
            }),
        }
    }
}

fn make_service<'a>(mock: &'a Mock, account: &str) -> QwenKnowledgeService<'a> {
    let scope = QwenKnowledgeScope::new(
        "qwen-main",
        account,
        QwenKnowledgeRegion::Beijing,
        "llm-workspace-1",
    )
    .unwrap();
    QwenKnowledgeService::new(mock, scope).unwrap()
}

fn success(data: Value) -> Value {
    json!({
        "code":"Success",
        "status_code":200,
        "success":true,
        "message":"success",
        "request_id":"body-request-1",
        "data":data,
        "status":"SUCCESS"
    })
}

#[tokio::test]
async fn lists_typed_chunks_with_provider_page_fields_and_document_scope() {
    let mock = Mock::new([MockReply::Response {
        status: 200,
        body: success(json!({
            "total":37,
            "nodes":[{
                "text":"Original chunk body",
                "score":0.0,
                "metadata":{
                    "_id":"llm-kb-file-0-0",
                    "doc_id":"file-7",
                    "doc_name":"guide.md",
                    "title":"Guide",
                    "pipeline_id":"index-1",
                    "workspace_id":"llm-workspace-1",
                    "future_field":{"preserved":true}
                },
                "provider_extension":"kept"
            }],
            "isDowngrade":false
        })),
    }]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let document = knowledge.document_ref("file-7").unwrap();
    let result = service
        .list_chunks(
            &knowledge,
            &QwenKnowledgeListChunksRequest::new(2, 10).for_document(&document),
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(result.total_count, 37);
    assert_eq!(result.page_num, 2);
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].reference.chunk_id(), "llm-kb-file-0-0");
    assert_eq!(
        result.chunks[0].reference.document().unwrap().document_id(),
        "file-7"
    );
    assert_eq!(result.chunks[0].native["provider_extension"], "kept");
    assert_eq!(result.chunks[0].metadata["future_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/chunklist"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"indexId":"index-1","pageNum":2,"pageSize":10,"docId":"file-7"})
    );
}

#[tokio::test]
async fn filtered_list_rejects_a_chunk_from_another_document_and_keeps_request_id() {
    let mock = Mock::new([MockReply::Response {
        status: 200,
        body: success(json!({
            "total":1,
            "nodes":[{
                "text":"Unexpected document",
                "metadata":{
                    "_id":"llm-kb-file-0-0",
                    "doc_id":"file-other",
                    "pipeline_id":"index-1",
                    "workspace_id":"llm-workspace-1"
                }
            }]
        })),
    }]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let document = knowledge.document_ref("file-7").unwrap();
    let error = service
        .list_chunks(
            &knowledge,
            &QwenKnowledgeListChunksRequest::default().for_document(&document),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::InvalidResponse {
            operation: "list_chunks",
            request_id: Some(ref id),
            ..
        } if id == "req-qwen-chunk-1"
    ));
}

#[tokio::test]
async fn adds_updates_and_deletes_with_document_and_chunk_scope() {
    let mock = Mock::new([
        MockReply::Response {
            status: 200,
            body: success(json!({})),
        },
        MockReply::Response {
            status: 200,
            body: json!({
                "code":"Success",
                "status_code":200,
                "request_id":"update-sample-request"
            }),
        },
        MockReply::Response {
            status: 200,
            body: json!({
                "code":"Success",
                "status_code":200,
                "request_id":"delete-sample-request"
            }),
        },
    ]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let document = knowledge.document_ref("file-7").unwrap();

    let fields = QwenKnowledgeChunkFields::document("Added indexed paragraph.")
        .with_title("Added title")
        .unwrap()
        .with_image_urls(["https://images.example/a.png"])
        .unwrap();
    let add = QwenKnowledgeAddChunkRequest::for_document(&document, fields);
    let add_result = service
        .add_chunk(&knowledge, &add, &request_options())
        .await
        .unwrap();
    assert_eq!(add_result.request_id.as_deref(), Some("req-qwen-chunk-1"));

    let chunk = lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChunkRef::new(
        &knowledge,
        &document,
        "llm-kb-file-0-0",
    )
    .unwrap();
    let update_result = service
        .update_chunk(
            &chunk,
            &QwenKnowledgeUpdateChunkRequest::new(
                "Updated chunk content is at least ten characters.",
                false,
            )
            .with_title(""),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(
        update_result.native,
        json!({
            "code":"Success",
            "status_code":200,
            "request_id":"update-sample-request"
        })
    );
    let delete_result = service
        .delete_chunks(&knowledge, std::slice::from_ref(&chunk), &request_options())
        .await
        .unwrap();
    assert_eq!(
        delete_result.native,
        json!({
            "code":"Success",
            "status_code":200,
            "request_id":"delete-sample-request"
        })
    );

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/chunk/create"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "pipelineId":"index-1",
            "dataId":"file-7",
            "field":{
                "content":"Added indexed paragraph.",
                "title":"Added title",
                "image_urls":["https://images.example/a.png"]
            }
        })
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
        json!({
            "pipelineId":"index-1",
            "chunkId":"llm-kb-file-0-0",
            "dataId":"file-7",
            "content":"Updated chunk content is at least ten characters.",
            "title":"",
            "isDisplayedChunkContent":false
        })
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[2].body).unwrap(),
        json!({"pipelineId":"index-1","chunkIds":["llm-kb-file-0-0"]})
    );
}

#[tokio::test]
async fn rejects_invalid_limits_and_cross_scope_refs_before_http() {
    let mock = Mock::new([]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let document = knowledge.document_ref("file-7").unwrap();

    let invalid_page = service
        .list_chunks(
            &knowledge,
            &QwenKnowledgeListChunksRequest::new(0, 101),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(invalid_page, QwenKnowledgeError::InvalidInput(_)));

    let invalid_add =
        QwenKnowledgeAddChunkRequest::new(QwenKnowledgeChunkFields::document("x".repeat(6001)));
    assert!(matches!(
        service
            .add_chunk(&knowledge, &invalid_add, &request_options())
            .await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));

    let chunk = lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChunkRef::new(
        &knowledge, &document, "chunk-1",
    )
    .unwrap();
    let other_service = make_service(&mock, "account-b");
    assert!(matches!(
        other_service
            .delete_chunks(&knowledge, std::slice::from_ref(&chunk), &request_options())
            .await,
        Err(QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let too_many = (0..11)
        .map(|index| {
            lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChunkRef::without_document(
                &knowledge,
                format!("chunk-{index}"),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        service
            .delete_chunks(&knowledge, &too_many, &request_options())
            .await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn update_requires_a_chunk_reference_bound_to_its_document() {
    let mock = Mock::new([]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let chunk =
        lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChunkRef::without_document(
            &knowledge, "chunk-1",
        )
        .unwrap();
    let error = service
        .update_chunk(
            &chunk,
            &QwenKnowledgeUpdateChunkRequest::new("This content exceeds ten characters.", true),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn interrupted_mutation_is_unknown_and_is_not_retried() {
    let mock = Mock::new([MockReply::TransportFailure]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let error = service
        .add_chunk(
            &knowledge,
            &QwenKnowledgeAddChunkRequest::new(QwenKnowledgeChunkFields::document("text")),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        QwenKnowledgeError::OutcomeUnknown {
            operation: "add_chunk",
            ..
        }
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn chunk_mutations_reject_explicit_failure_and_contradictory_business_status() {
    let mock = Mock::new([
        MockReply::Response {
            status: 200,
            body: json!({
                "code":"Success",
                "status_code":200,
                "success":false,
                "request_id":"explicit-failure"
            }),
        },
        MockReply::Response {
            status: 200,
            body: json!({
                "code":"Success",
                "status_code":500,
                "request_id":"contradictory-failure"
            }),
        },
    ]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let document = knowledge.document_ref("file-7").unwrap();
    let chunk = lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChunkRef::new(
        &knowledge, &document, "chunk-1",
    )
    .unwrap();

    let explicit_failure = service
        .update_chunk(
            &chunk,
            &QwenKnowledgeUpdateChunkRequest::new(
                "Updated content that is longer than ten characters.",
                true,
            ),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        explicit_failure,
        QwenKnowledgeError::Provider {
            status: 200,
            dispatch: QwenKnowledgeDispatch::Rejected,
            ..
        }
    ));

    let contradictory_status = service
        .delete_chunks(&knowledge, std::slice::from_ref(&chunk), &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        contradictory_status,
        QwenKnowledgeError::Provider {
            status: 200,
            dispatch: QwenKnowledgeDispatch::Unknown,
            ..
        }
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn status_only_success_is_not_accepted_for_add_chunk() {
    let mock = Mock::new([MockReply::Response {
        status: 200,
        body: json!({
            "code":"Success",
            "status_code":200,
            "request_id":"unexpected-status-only-add"
        }),
    }]);
    let service = make_service(&mock, "account-a");
    let knowledge = service.knowledge_ref("index-1").unwrap();
    let error = service
        .add_chunk(
            &knowledge,
            &QwenKnowledgeAddChunkRequest::new(QwenKnowledgeChunkFields::document("text")),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::ResponseOutcomeUnknown {
            operation: "add_chunk",
            ..
        }
    ));
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("qwen-key-test".into())),
        ..Default::default()
    }
}
