use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    qwen_knowledge::{
        QwenKnowledgeChunkMode, QwenKnowledgeCreateRequest, QwenKnowledgeDispatch,
        QwenKnowledgeDocumentPageRequest, QwenKnowledgeError, QwenKnowledgeImportRequest,
        QwenKnowledgeRegion, QwenKnowledgeRetrieveRequest, QwenKnowledgeScope,
        QwenKnowledgeService, QwenKnowledgeUpdateRequest,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::collections::VecDeque;

enum MockReply {
    Response { status: u16, body: Value },
    TransportFailure,
}

struct Mock {
    replies: std::sync::Mutex<VecDeque<MockReply>>,
    requests: std::sync::Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = MockReply>) -> Self {
        Self {
            replies: std::sync::Mutex::new(replies.into_iter().collect()),
            requests: std::sync::Mutex::new(Vec::new()),
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
                    headers: vec![("x-request-id".into(), "req-qwen-kb-1".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            MockReply::TransportFailure => Err(LlmError::TransportTimeout {
                message: "mock timeout after request dispatch".into(),
            }),
        }
    }
}

fn response(status: u16, body: Value) -> MockReply {
    MockReply::Response { status, body }
}

fn scope(region: QwenKnowledgeRegion, account: &str, workspace: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new("qwen-main", account, region, workspace).unwrap()
}

fn service<'a>(
    mock: &'a Mock,
    region: QwenKnowledgeRegion,
    account: &str,
    workspace: &str,
) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(
        mock,
        Secret::new("qwen-key-test".to_owned()),
        scope(region, account, workspace),
    )
    .unwrap()
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

fn document_row(id: &str) -> Value {
    json!({
        "doc_id":id,
        "doc_name":"returns.md",
        "doc_type":"md",
        "size":512,
        "status":"FINISH",
        "code":"FINISH",
        "message":"index built",
        "ingestion_id":"job-9",
        "future_field":{"kept":true}
    })
}

#[tokio::test]
async fn retrieves_chunks_through_the_documented_beijing_route() {
    let mock = Mock::new([response(
        200,
        success(json!({"nodes":[{
            "text":"Returns are accepted within 30 days.",
            "score":0.92,
            "metadata":{"doc_id":"file-a","title":"Returns"},
            "future_field":{"preserved":true}
        }]})),
    )]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let result = service
        .retrieve(
            &knowledge,
            &QwenKnowledgeRetrieveRequest::new("What is the return window?").with_top_k(5),
        )
        .await
        .unwrap();

    assert_eq!(result.request_id.as_deref(), Some("req-qwen-kb-1"));
    assert_eq!(result.matches.len(), 1);
    assert_eq!(
        result.matches[0].text,
        "Returns are accepted within 30 days."
    );
    assert_eq!(result.matches[0].score, Some(0.92));
    assert_eq!(result.matches[0].metadata["doc_id"], "file-a");
    assert_eq!(result.matches[0].native["future_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/retrieve"
    );
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1,
        "Bearer qwen-key-test"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "index_id":"index-123",
            "query":"What is the return window?",
            "top_k":5
        })
    );
}

#[tokio::test]
async fn creates_knowledge_base_and_initial_import_with_exact_camel_case_contract() {
    let mock = Mock::new([response(
        200,
        success(json!({
            "pipelineId":"index-123",
            "ingestionId":"job-9",
            "status":"PENDING"
        })),
    )]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let mut request =
        QwenKnowledgeCreateRequest::new("Returns", "Return policy documents", ["file-1"]);
    request.category_ids.push("category-2".into());
    request.knowledge_type = Some("document".into());
    request.knowledge_scene = Some("basic_document_qa".into());
    request.embedding_model_name = Some("text-embedding-v4".into());

    let result = service.create_knowledge_base(&request).await.unwrap();
    assert_eq!(result.knowledge.index_id(), "index-123");
    assert_eq!(result.job.job_id(), "job-9");
    assert_eq!(result.status.as_deref(), Some("PENDING"));
    assert_eq!(result.knowledge.scope().account_scope(), "acct-1");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/create_v2"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "name":"Returns",
            "description":"Return policy documents",
            "structureType":"unstructured",
            "sinkType":"DEFAULT",
            "sourceType":"DATA_CENTER_FILE",
            "docIds":["file-1"],
            "categoryIds":["category-2"],
            "dataSources":[{"sourceType":"DATA_CENTER_FILE"}],
            "knowledgeType":"document",
            "knowledgeScene":"basic_document_qa",
            "embeddingModelName":"text-embedding-v4"
        })
    );
}

#[tokio::test]
async fn lists_and_finds_knowledge_bases_using_only_the_documented_list_route() {
    let mock = Mock::new([
        response(
            200,
            success(json!({
                "page_number":1,"page_size":1,"total_count":2,
                "rows":[{"id":"other-index","name":"Other","structureType":"unstructured"}]
            })),
        ),
        response(
            200,
            success(json!({
                "page_number":2,"page_size":1,"total_count":2,
                "rows":[{"id":"index-123","name":"Returns","description":"Policy","structureType":"unstructured"}]
            })),
        ),
    ]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let target = service.knowledge_ref("index-123").unwrap();
    let found = service.find_knowledge_base(&target).await.unwrap().unwrap();
    assert_eq!(found.name.as_deref(), Some("Returns"));
    assert_eq!(found.description.as_deref(), Some("Policy"));

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].url, "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/list?page_number=1&page_size=100");
    assert_eq!(requests[1].url, "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/list?page_number=2&page_size=100");
}

#[tokio::test]
async fn update_and_delete_knowledge_base_use_documented_fields_and_routes() {
    let mock = Mock::new([
        response(
            200,
            success(json!({"id":"index-123","created_at":1700000000,"updated_at":1700000001})),
        ),
        response(200, success(Value::Null)),
    ]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    service
        .update_knowledge_base(
            &knowledge,
            &QwenKnowledgeUpdateRequest::new()
                .with_name("Returns 2026")
                .with_description("Updated return information")
                .with_rerank_min_score(0.12),
        )
        .await
        .unwrap();
    service.delete_knowledge_base(&knowledge).await.unwrap();

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/update"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"id":"index-123","name":"Returns 2026","description":"Updated return information","rerankMinScore":0.12})
    );
    assert_eq!(
        requests[1].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/delete"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
        json!({"index_id":"index-123"})
    );
}

#[tokio::test]
async fn lists_details_and_deletes_only_same_knowledge_base_documents() {
    let mock = Mock::new([
        response(
            200,
            success(
                json!({"page_number":1,"page_size":1,"total_count":1,"rows":[document_row("file-doc-1")]}),
            ),
        ),
        response(
            200,
            success(
                json!({"page_number":1,"page_size":1,"total_count":1,"rows":[document_row("file-doc-1")]}),
            ),
        ),
        response(200, success(json!({"deleted":["file-doc-1"]}))),
    ]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let page = service
        .list_documents(&knowledge, QwenKnowledgeDocumentPageRequest::new(1, 1))
        .await
        .unwrap();
    let detail = service
        .list_document_details(&knowledge, QwenKnowledgeDocumentPageRequest::new(1, 1))
        .await
        .unwrap();
    assert_eq!(page.documents[0].reference.document_id(), "file-doc-1");
    assert_eq!(detail.documents[0].native["future_field"]["kept"], true);
    assert_eq!(detail.documents[0].status.as_deref(), Some("FINISH"));
    let deleted = service
        .delete_documents(&knowledge, &[page.documents[0].reference.clone()])
        .await
        .unwrap();
    assert_eq!(
        deleted.deleted_document_ids.unwrap(),
        vec!["file-doc-1".to_owned()]
    );

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].url, "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/files?index_id=index-123&page_number=1&page_size=1");
    assert_eq!(requests[1].url, "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/list/index/file/details");
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
        json!({"indexId":"index-123","pageNumber":1,"pageSize":1})
    );
    assert_eq!(
        requests[2].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/delete_file"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[2].body).unwrap(),
        json!({"index_id":"index-123","doc_ids":["file-doc-1"]})
    );
}

#[tokio::test]
async fn appends_files_and_queries_import_status_once() {
    let mock = Mock::new([
        response(
            200,
            success(json!({"pipelineId":"index-123","ingestionId":"job-9","status":"PENDING"})),
        ),
        response(
            200,
            success(json!({
                "id":"job-9","ingestion_status":"COMPLETED","ingestion_message":"COMPLETED",
                "total_count":1,"page_number":1,"page_size":10,"rows":[document_row("file-doc-1")]
            })),
        ),
    ]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let imported = service
        .submit_import_job(
            &knowledge,
            &QwenKnowledgeImportRequest {
                chunk_mode: Some(QwenKnowledgeChunkMode::Length),
                chunk_size: Some(800),
                overlap_size: Some(100),
                ..QwenKnowledgeImportRequest::from_files(["file-2"])
            },
        )
        .await
        .unwrap();
    let status = service
        .get_import_job_status(
            &imported.reference,
            QwenKnowledgeDocumentPageRequest::default(),
        )
        .await
        .unwrap();
    assert_eq!(status.ingestion_status.as_deref(), Some("COMPLETED"));
    assert_eq!(status.documents.len(), 1);
    assert_eq!(status.documents[0].reference.document_id(), "file-doc-1");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/job/create"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"indexId":"index-123","sourceType":"DATA_CENTER_FILE","docIds":["file-2"],"chunkMode":"length","chunkSize":800,"overlapSize":100})
    );
    assert_eq!(requests[1].url, "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index_job/status?index_id=index-123&job_id=job-9&page_number=1&page_size=10");
}

#[tokio::test]
async fn cross_scope_knowledge_document_and_job_refs_are_rejected_before_http() {
    let mock = Mock::new([]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let foreign_knowledge = scope(QwenKnowledgeRegion::Beijing, "acct-2", "llm-workspace-2")
        .knowledge_ref("index-123")
        .unwrap();
    let foreign_document = foreign_knowledge.document_ref("file-doc-1").unwrap();
    let error = service
        .delete_documents(&knowledge, &[foreign_document])
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));

    let foreign_job = foreign_knowledge.job_ref("job-9").unwrap();
    let error = service
        .get_import_job_status(&foreign_job, QwenKnowledgeDocumentPageRequest::default())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn singapore_mutations_are_preflighted_without_guessing_a_route() {
    let mock = Mock::new([]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Singapore,
        "acct-sg",
        "llm-workspace-sg",
    );
    let error = service
        .create_knowledge_base(&QwenKnowledgeCreateRequest::new(
            "Returns",
            "Description",
            ["file-1"],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        QwenKnowledgeError::Llm(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn uncertain_create_is_not_retried_and_reports_unknown_dispatch() {
    let mock = Mock::new([MockReply::TransportFailure]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let error = service
        .create_knowledge_base(&QwenKnowledgeCreateRequest::new(
            "Returns",
            "Description",
            ["file-1"],
        ))
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert!(matches!(error, QwenKnowledgeError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn server_failure_is_unknown_but_explicit_client_rejection_is_definite() {
    let server = Mock::new([response(
        503,
        json!({"code":"SystemError","success":false,"message":"busy"}),
    )]);
    let server_service = service(
        &server,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let error = server_service
        .delete_knowledge_base(&server_service.knowledge_ref("index-123").unwrap())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(server.requests.lock().unwrap().len(), 1);

    let client = Mock::new([response(
        400,
        json!({"code":"Index.InvalidParameter","success":false,"message":"bad input"}),
    )]);
    let client_service = service(
        &client,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let error = client_service
        .delete_knowledge_base(&client_service.knowledge_ref("index-123").unwrap())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Rejected);
}

#[tokio::test]
async fn business_errors_on_http_200_are_not_reported_as_empty_success() {
    let mock = Mock::new([response(
        200,
        json!({
            "code":"Index.NotFound",
            "success":false,
            "message":"knowledge base not found",
            "request_id":"provider-request-7"
        }),
    )]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let error = service
        .retrieve(&knowledge, &QwenKnowledgeRetrieveRequest::new("query"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::Provider {
            status: 200,
            code: Some(ref code),
            request_id: Some(ref request_id),
            ..
        } if code == "Index.NotFound" && request_id == "req-qwen-kb-1"
    ));
}

#[tokio::test]
async fn invalid_requests_are_rejected_before_http() {
    let mock = Mock::new([]);
    let service = service(
        &mock,
        QwenKnowledgeRegion::Beijing,
        "acct-1",
        "llm-workspace-1",
    );
    let knowledge = service.knowledge_ref("index-123").unwrap();
    assert!(service
        .retrieve(&knowledge, &QwenKnowledgeRetrieveRequest::new("  "))
        .await
        .is_err());
    assert!(service
        .submit_import_job(
            &knowledge,
            &QwenKnowledgeImportRequest::from_files(std::iter::empty::<&str>()),
        )
        .await
        .is_err());
    assert!(service
        .update_knowledge_base(&knowledge, &QwenKnowledgeUpdateRequest::new())
        .await
        .is_err());
    assert!(mock.requests.lock().unwrap().is_empty());
}
