use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    qwen_knowledge::{
        QwenKnowledgeDispatch, QwenKnowledgeError, QwenKnowledgeRef, QwenKnowledgeRegion,
        QwenKnowledgeScope, QwenKnowledgeSearchKbConfig, QwenKnowledgeSearchRef,
        QwenKnowledgeSearchRequest, QwenKnowledgeService,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

enum Reply {
    Json { status: u16, body: Value },
    Failure,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
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
            Reply::Json { status, body } => {
                let bytes = serde_json::to_vec(&body).unwrap();
                Ok(StreamResponse {
                    status,
                    headers: vec![("x-request-id".into(), "header-search-request".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            Reply::Failure => Err(LlmError::TransportTimeout {
                message: "mock search timeout".into(),
            }),
        }
    }
}

fn reply(body: Value) -> Reply {
    Reply::Json { status: 200, body }
}

fn scope(account: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new(
        "qwen-main",
        account,
        QwenKnowledgeRegion::Beijing,
        "llm-search-workspace",
    )
    .unwrap()
}

fn service<'a>(mock: &'a Mock, account: &str) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(
        mock,
        Secret::new("qwen-search-api-key".to_owned()),
        scope(account),
    )
    .unwrap()
}

fn search_ref(scope: &QwenKnowledgeScope) -> QwenKnowledgeSearchRef {
    scope
        .knowledge_search_ref("aid-published-search-1")
        .unwrap()
}

fn kb_ref(scope: &QwenKnowledgeScope, id: &str) -> QwenKnowledgeRef {
    scope.knowledge_ref(id).unwrap()
}

fn search_response(nodes: Value) -> Value {
    json!({
        "code":"Success",
        "success":true,
        "status_code":200,
        "status":"SUCCESS",
        "request_id":"body-search-request",
        "data":{
            "total":nodes.as_array().map_or(0, |items| items.len()),
            "nodes":nodes,
            "cost_time":12,
            "future_data_field":{"retained":true}
        },
        "future_envelope_field":"retained"
    })
}

fn result_node(workspace_id: &str, pipeline_id: &str) -> Value {
    json!({
        "score":0.97,
        "text":"title: Guide\ncontent: A useful result",
        "metadata":{
            "workspace_id":workspace_id,
            "pipeline_id":pipeline_id,
            "doc_id":"doc_1",
            "_id":"chunk_1",
            "custom_product_code":"P-42",
            "future_metadata_field":{"kept":true}
        },
        "future_node_field":[1,2,3]
    })
}

#[tokio::test]
async fn native_search_uses_published_agent_and_returns_raw_nodes_and_metadata() {
    let mock = Mock::new([reply(search_response(json!([result_node(
        "llm-search-workspace",
        "kb-product-1"
    )])))]);
    let service = service(&mock, "acct-1");
    let knowledge = kb_ref(service.scope(), "kb-filter-1");
    let request =
        QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "Find the product warranty.")
            .with_agent_version("published-v3")
            .with_images(["https://cdn.example.test/warranty.jpg"])
            .with_kb_search_config(
                QwenKnowledgeSearchKbConfig::new(knowledge).with_search_filters([
                    json!({"doc_id":["doc-a","doc-b"]}),
                    json!({"tags":["warranty"]}),
                    json!({"price":{"gte":100,"lte":500}}),
                ]),
            );
    let result = service.knowledge_search(&request).await.unwrap();

    assert_eq!(result.total, 1);
    assert_eq!(result.cost_time_ms, 12);
    assert_eq!(result.request_id.as_deref(), Some("header-search-request"));
    assert_eq!(result.native["request_id"], "body-search-request");
    assert_eq!(result.native["future_envelope_field"], "retained");
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0].score, Some(0.97));
    assert_eq!(
        result.nodes[0].knowledge.as_ref().unwrap().index_id(),
        "kb-product-1"
    );
    assert_eq!(
        result.nodes[0].metadata["future_metadata_field"]["kept"],
        true
    );
    assert_eq!(
        result.nodes[0].native["future_node_field"],
        json!([1, 2, 3])
    );

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-search-workspace.cn-beijing.maas.aliyuncs.com/api/v1/indices/knowledge/search"
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer qwen-search-api-key"));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["agent_id"], "aid-published-search-1");
    assert_eq!(body["agent_version"], "published-v3");
    assert_eq!(body["query"], "Find the product warranty.");
    assert_eq!(
        body["images"],
        json!(["https://cdn.example.test/warranty.jpg"])
    );
    assert_eq!(body["kb_search_configs"][0]["id"], "kb-filter-1");
    assert_eq!(
        body["kb_search_configs"][0]["search_filters"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(body.get("rerank").is_none());
    assert!(body.get("strategy").is_none());
}

#[tokio::test]
async fn image_only_search_sends_the_required_empty_query() {
    let mock = Mock::new([reply(search_response(json!([])))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeSearchRequest::images_only(
        search_ref(service.scope()),
        ["https://images.example.test/product.jpg"],
    );
    let result = service.knowledge_search(&request).await.unwrap();
    assert!(result.nodes.is_empty());
    let body: Value = serde_json::from_slice(&mock.requests.lock().unwrap()[0].body).unwrap();
    assert_eq!(body["query"], "");
    assert_eq!(
        body["images"],
        json!(["https://images.example.test/product.jpg"])
    );
}

#[tokio::test]
async fn explicit_search_filters_do_not_restrict_agent_bound_result_knowledge_bases() {
    let mock = Mock::new([reply(search_response(json!([result_node(
        "llm-search-workspace",
        "another-agent-bound-kb"
    )])))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "find something")
        .with_kb_search_config(QwenKnowledgeSearchKbConfig::new(kb_ref(
            service.scope(),
            "kb-with-extra-filter",
        )));
    let result = service.knowledge_search(&request).await.unwrap();
    assert_eq!(
        result.nodes[0].knowledge.as_ref().unwrap().index_id(),
        "another-agent-bound-kb"
    );
}

#[tokio::test]
async fn search_and_knowledge_refs_from_another_scope_are_rejected_before_http() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1");
    let foreign_request =
        QwenKnowledgeSearchRequest::new(search_ref(&scope("acct-2")), "find something");
    let error = service
        .knowledge_search(&foreign_request)
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::Llm(_)));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);

    let foreign_kb_request =
        QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "find something")
            .with_kb_search_config(QwenKnowledgeSearchKbConfig::new(kb_ref(
                &scope("acct-2"),
                "kb-foreign",
            )));
    let error = service
        .knowledge_search(&foreign_kb_request)
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::Llm(_)));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn duplicate_knowledge_ids_and_documented_filter_limits_fail_before_http() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1");
    let knowledge = kb_ref(service.scope(), "kb-1");
    let duplicate = QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "query")
        .with_kb_search_configs([
            QwenKnowledgeSearchKbConfig::new(knowledge.clone()),
            QwenKnowledgeSearchKbConfig::new(knowledge.clone()),
        ]);
    assert!(matches!(
        service.knowledge_search(&duplicate).await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));

    let too_many_tags = QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "query")
        .with_kb_search_config(
            QwenKnowledgeSearchKbConfig::for_unstructured(knowledge.clone())
                .with_search_filters([json!({"tags":vec!["tag"; 1001]})]),
        );
    assert!(matches!(
        service.knowledge_search(&too_many_tags).await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));

    let large_filter = json!({"metadata_field":"x".repeat(80_100)});
    let too_many_bytes = QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "query")
        .with_kb_search_configs([
            QwenKnowledgeSearchKbConfig::new(knowledge.clone())
                .with_search_filters([large_filter.clone()]),
            QwenKnowledgeSearchKbConfig::new(kb_ref(service.scope(), "kb-2"))
                .with_search_filters([large_filter]),
        ]);
    assert!(matches!(
        service.knowledge_search(&too_many_bytes).await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn generic_filter_configs_leave_knowledge_type_specific_limits_to_the_provider() {
    let mock = Mock::new([reply(search_response(json!([])))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeSearchRequest::new(search_ref(service.scope()), "query")
        .with_kb_search_config(
            QwenKnowledgeSearchKbConfig::new(kb_ref(service.scope(), "kb-type-not-declared"))
                .with_search_filters([json!({"tags":vec!["tag"; 1001]})]),
        );
    service.knowledge_search(&request).await.unwrap();

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["kb_search_configs"][0]["search_filters"][0]["tags"]
            .as_array()
            .unwrap()
            .len(),
        1001
    );
}

#[tokio::test]
async fn empty_search_intent_and_mismatched_result_workspace_fail_without_retry() {
    let empty_mock = Mock::new([]);
    let empty_service = service(&empty_mock, "acct-1");
    let empty = QwenKnowledgeSearchRequest::new(search_ref(empty_service.scope()), "");
    assert!(matches!(
        empty_service.knowledge_search(&empty).await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));
    assert!(empty_mock.requests.lock().unwrap().is_empty());

    let invalid_image = QwenKnowledgeSearchRequest::images_only(
        search_ref(empty_service.scope()),
        ["file:///tmp/private.png"],
    );
    assert!(matches!(
        empty_service.knowledge_search(&invalid_image).await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));
    assert!(empty_mock.requests.lock().unwrap().is_empty());

    let mismatch = Mock::new([reply(search_response(json!([result_node(
        "other-workspace",
        "kb-1"
    )])))]);
    let error = service(&mismatch, "acct-1")
        .knowledge_search(&QwenKnowledgeSearchRequest::new(
            search_ref(&scope("acct-1")),
            "query",
        ))
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::InvalidResponse { .. }));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);
    assert_eq!(mismatch.requests.lock().unwrap().len(), 1);

    let timeout = Mock::new([Reply::Failure]);
    let error = service(&timeout, "acct-1")
        .knowledge_search(&QwenKnowledgeSearchRequest::new(
            search_ref(&scope("acct-1")),
            "query",
        ))
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::Llm(_)));
    assert_eq!(timeout.requests.lock().unwrap().len(), 1);
}
