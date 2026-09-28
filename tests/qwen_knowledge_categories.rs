use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::knowledge::{
        QwenKnowledgeCategoryCreateRequest, QwenKnowledgeCategoryListRequest,
        QwenKnowledgeDispatch, QwenKnowledgeError, QwenKnowledgeRegion, QwenKnowledgeScope,
        QwenKnowledgeService,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

enum MockReply {
    Response { status: u16, body: Value },
    Failure,
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
                    headers: vec![("x-request-id".into(), "category-header-id".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            MockReply::Failure => Err(LlmError::TransportTimeout {
                message: "mock timeout after category request dispatch".into(),
            }),
        }
    }
}

fn response(status: u16, body: Value) -> MockReply {
    MockReply::Response { status, body }
}

fn success(data: Value) -> Value {
    json!({
        "code":"Success",
        "status_code":200,
        "status":200,
        "data":data,
        "requestId":"category-body-id",
        "future_envelope_field":{"preserved":true}
    })
}

fn category_row(id: &str, name: &str, is_default: bool) -> Value {
    json!({
        "categoryId":id,
        "categoryName":name,
        "type":"UNSTRUCTURED",
        "isDefault":is_default,
        "future_category_field":{"preserved":true}
    })
}

fn scope(region: QwenKnowledgeRegion, account: &str, workspace: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new("qwen-profile", account, region, workspace).unwrap()
}

fn make_service<'a>(
    mock: &'a Mock,
    region: QwenKnowledgeRegion,
    account: &str,
) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(mock, scope(region, account, "llm-category-workspace")).unwrap()
}

#[tokio::test]
async fn list_categories_sends_current_filter_fields_and_preserves_native_rows() {
    let mock = Mock::new([response(
        200,
        success(json!({
            "hasNext":false,
            "maxResult":20,
            "totalCount":1,
            "maxId":100602,
            "categoryList":[category_row("cate_child_1", "Product docs", false)],
            "future_page_field":"preserved"
        })),
    )]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let parent = service.scope().category_ref("cate_parent_1").unwrap();
    let connector = service.scope().connector_ref("file_conn_1").unwrap();
    let request = QwenKnowledgeCategoryListRequest::new()
        .with_parent_category(parent)
        .with_category_name("Product docs")
        .with_connector_ref(connector)
        .with_next_token("cursor-before")
        .with_max_result(40);

    let page = service
        .list_categories(&request, &request_options())
        .await
        .unwrap();

    assert_eq!(page.categories.len(), 1);
    assert_eq!(page.categories[0].reference.category_id(), "cate_child_1");
    assert_eq!(page.categories[0].category_name, "Product docs");
    assert!(!page.categories[0].is_default);
    assert_eq!(
        page.categories[0].category_type,
        lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeCategoryType::Unstructured
    );
    assert_eq!(
        page.categories[0].native["future_category_field"]["preserved"],
        true
    );
    assert_eq!(page.total_count, Some(1));
    assert_eq!(page.request_id.as_deref(), Some("category-header-id"));
    assert_eq!(page.native["requestId"], "category-body-id");
    assert_eq!(page.native["future_envelope_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-category-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/listCategory"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "type":"UNSTRUCTURED",
            "parentId":"cate_parent_1",
            "categoryName":"Product docs",
            "connectorId":"file_conn_1",
            "nextToken":"cursor-before",
            "maxResult":40
        })
    );
}

#[tokio::test]
async fn list_categories_returns_cursor_and_rejects_nonprogressing_or_duplicate_pages() {
    let mock = Mock::new([
        response(
            200,
            success(json!({
                "hasNext":true,
                "nextToken":"cursor-after",
                "categoryList":[category_row("cate_page_1", "First", false)]
            })),
        ),
        response(
            200,
            success(json!({
                "hasNext":false,
                "categoryList":[category_row("cate_page_2", "Second", false)]
            })),
        ),
    ]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let first = service
        .list_categories(&QwenKnowledgeCategoryListRequest::new(), &request_options())
        .await
        .unwrap();
    let cursor = first.next_token.clone().unwrap();
    let second = service
        .list_categories(
            &QwenKnowledgeCategoryListRequest::new().with_next_token(cursor.clone()),
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(first.next_token.as_deref(), Some("cursor-after"));
    assert!(!second.has_next);
    {
        let requests = mock.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
            json!({"type":"UNSTRUCTURED", "nextToken":cursor})
        );
    }

    for (data, request) in [
        (
            json!({"hasNext":true,"categoryList":[]}),
            QwenKnowledgeCategoryListRequest::new(),
        ),
        (
            json!({"hasNext":true,"nextToken":"","categoryList":[]}),
            QwenKnowledgeCategoryListRequest::new(),
        ),
        (
            json!({"hasNext":true,"nextToken":"same","categoryList":[]}),
            QwenKnowledgeCategoryListRequest::new().with_next_token("same"),
        ),
        (
            json!({
                "hasNext":false,
                "categoryList":[
                    category_row("cate_duplicate", "One", false),
                    category_row("cate_duplicate", "Two", false)
                ]
            }),
            QwenKnowledgeCategoryListRequest::new(),
        ),
    ] {
        let mock = Mock::new([response(200, success(data))]);
        let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
        let error = service
            .list_categories(&request, &request_options())
            .await
            .unwrap_err();
        assert!(matches!(error, QwenKnowledgeError::InvalidResponse { .. }));
    }
}

#[tokio::test]
async fn create_category_encodes_optional_parent_and_connector_and_returns_scoped_ref() {
    let mock = Mock::new([response(
        200,
        success(json!({
            "categoryId":"cate_created_1",
            "categoryName":"created from server"
        })),
    )]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let parent = service.scope().category_ref("cate_parent_1").unwrap();
    let connector = service.scope().connector_ref("file_conn_1").unwrap();
    let request = QwenKnowledgeCategoryCreateRequest::new("新しい分類")
        .with_parent_category(parent)
        .with_connector_ref(connector);

    let result = service
        .create_category(&request, &request_options())
        .await
        .unwrap();
    assert_eq!(result.reference.category_id(), "cate_created_1");
    assert_eq!(result.reference.scope().account_scope(), "account-1");
    assert_eq!(result.category_name.as_deref(), Some("created from server"));
    assert_eq!(result.request_id.as_deref(), Some("category-header-id"));
    assert_eq!(result.native["data"]["categoryId"], "cate_created_1");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-category-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/addCategory"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "categoryName":"新しい分類",
            "categoryType":"UNSTRUCTURED",
            "parentCategoryId":"cate_parent_1",
            "connectorId":"file_conn_1"
        })
    );
}

#[tokio::test]
async fn create_category_validates_name_and_parent_scope_before_dispatch() {
    let mock = Mock::new([]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let other_scope = scope(
        QwenKnowledgeRegion::Beijing,
        "other-account",
        "llm-category-workspace",
    );
    let other_parent = other_scope.category_ref("cate_parent_1").unwrap();
    let other_connector = other_scope.connector_ref("file_conn_1").unwrap();
    for request in [
        QwenKnowledgeCategoryCreateRequest::new(""),
        QwenKnowledgeCategoryCreateRequest::new("x".repeat(21)),
        QwenKnowledgeCategoryCreateRequest::new("child").with_parent_category(other_parent),
        QwenKnowledgeCategoryCreateRequest::new("child").with_connector_ref(other_connector),
    ] {
        let error = service
            .create_category(&request, &request_options())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            QwenKnowledgeError::InvalidInput(_)
                | QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
        ));
    }
    let foreign_connector = scope(
        QwenKnowledgeRegion::Beijing,
        "other-account",
        "llm-category-workspace",
    )
    .connector_ref("file_conn_1")
    .unwrap();
    let list_error = service
        .list_categories(
            &QwenKnowledgeCategoryListRequest::new().with_connector_ref(foreign_connector),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        list_error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    let mut forged_ref =
        serde_json::to_value(service.scope().connector_ref("file_conn_1").unwrap()).unwrap();
    forged_ref["endpoint_fingerprint"] = json!("wrong-endpoint");
    let forged_ref = serde_json::from_value(forged_ref).unwrap();
    let forged_error = service
        .create_category(
            &QwenKnowledgeCategoryCreateRequest::new("child").with_connector_ref(forged_ref),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        forged_error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    let mut forged_category =
        serde_json::to_value(service.scope().category_ref("cate_category_1").unwrap()).unwrap();
    forged_category["category_type"] = json!("SESSION_FILE");
    let forged_category = serde_json::from_value(forged_category).unwrap();
    let namespace_error = service
        .delete_category(&forged_category, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        namespace_error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn delete_category_uses_scoped_reference_and_retains_native_response() {
    let mock = Mock::new([response(
        200,
        success(json!({"categoryId":"cate_delete_1"})),
    )]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let category = service.scope().category_ref("cate_delete_1").unwrap();
    let result = service
        .delete_category(&category, &request_options())
        .await
        .unwrap();
    assert_eq!(result.reference, category);
    assert_eq!(result.request_id.as_deref(), Some("category-header-id"));
    assert_eq!(result.native["data"]["categoryId"], "cate_delete_1");

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-category-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/deleteCategory"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"categoryId":"cate_delete_1"})
    );
}

#[tokio::test]
async fn category_scope_errors_and_unconfirmed_mutations_never_retry() {
    let foreign = scope(
        QwenKnowledgeRegion::Beijing,
        "foreign-account",
        "llm-category-workspace",
    )
    .category_ref("cate_foreign_1")
    .unwrap();
    let mock = Mock::new([]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let error = service
        .delete_category(&foreign, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    for reply in [
        response(200, success(json!({"categoryId":"cate_other_1"}))),
        response(200, success(json!({}))),
        MockReply::Failure,
    ] {
        let mock = Mock::new([reply]);
        let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
        let category = service.scope().category_ref("cate_delete_1").unwrap();
        let error = service
            .delete_category(&category, &request_options())
            .await
            .unwrap_err();
        assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }

    let mock = Mock::new([MockReply::Failure]);
    let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let error = service
        .create_category(
            &QwenKnowledgeCategoryCreateRequest::new("one"),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    for data in [
        json!({}),
        json!({"categoryId":"not valid"}),
        json!({"categoryId":"cate_created_1","categoryName":false}),
    ] {
        let mock = Mock::new([response(200, success(data))]);
        let service = make_service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
        let error = service
            .create_category(
                &QwenKnowledgeCategoryCreateRequest::new("one"),
                &request_options(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            QwenKnowledgeError::ResponseOutcomeUnknown { .. }
        ));
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn unsupported_region_is_rejected_before_category_http_dispatch() {
    let mock = Mock::new([]);
    let service = make_service(&mock, QwenKnowledgeRegion::Singapore, "account-1");
    let error = service
        .list_categories(&QwenKnowledgeCategoryListRequest::new(), &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::Llm(LlmError::UnsupportedCapability { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("qwen-category-key".to_owned())),
        ..Default::default()
    }
}
