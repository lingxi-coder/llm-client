use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::knowledge::{
        QwenKnowledgeDispatch, QwenKnowledgeError, QwenKnowledgeFileListRequest,
        QwenKnowledgeFileTagUpdate, QwenKnowledgeFileTagUpdateMode,
        QwenKnowledgeFileTagUpdateRequest, QwenKnowledgeRegion, QwenKnowledgeScope,
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
                    headers: vec![("x-request-id".into(), "header-qwen-file-1".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            MockReply::Failure => Err(LlmError::TransportTimeout {
                message: "mock timeout after request dispatch".into(),
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
        "requestId":"body-qwen-file-1",
        "future_envelope_field":{"preserved":true}
    })
}

fn scope(region: QwenKnowledgeRegion, account: &str, workspace: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new("qwen-profile", account, region, workspace).unwrap()
}

fn service<'a>(
    mock: &'a Mock,
    region: QwenKnowledgeRegion,
    account: &str,
) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(mock, scope(region, account, "llm-files-workspace")).unwrap()
}

fn file_row(file_id: &str, file_name: &str) -> Value {
    json!({
        "fileId":file_id,
        "status":"PARSE_SUCCESS",
        "fileName":file_name,
        "fileType":"md",
        "parser":"DASHSCOPE_DOCMIND",
        "sizeBytes":21,
        "uploadTime":"2026-04-01 16:03:45",
        "category":"cate-1",
        "autoId":338031,
        "parseErrorMessage":"",
        "future_row_field":{"kept":true}
    })
}

#[tokio::test]
async fn list_files_uses_the_documented_post_body_and_preserves_native_rows() {
    let mock = Mock::new([response(
        200,
        success(json!({
            "hasNext":false,
            "maxResult":10,
            "totalCount":1,
            "maxId":338031,
            "fileList":[file_row("file_abc123_1", "policy.md")],
            "future_page_field":"kept"
        })),
    )]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let request = QwenKnowledgeFileListRequest::new("cate-1")
        .with_file_name("policy")
        .with_file_ids(["file_abc123_1"])
        .with_next_token("cursor-previous")
        .with_max_result(10);
    let result = service
        .list_files(&request, &request_options())
        .await
        .unwrap();

    assert_eq!(result.files.len(), 1);
    assert_eq!(result.files[0].reference.file_id(), "file_abc123_1");
    assert_eq!(
        result.files[0].reference.scope().account_scope(),
        "account-1"
    );
    assert_eq!(result.files[0].file_name.as_deref(), Some("policy.md"));
    assert_eq!(result.files[0].status.as_deref(), Some("PARSE_SUCCESS"));
    assert_eq!(result.files[0].native["future_row_field"]["kept"], true);
    assert!(!result.has_next);
    assert_eq!(result.total_count, Some(1));
    assert_eq!(result.request_id.as_deref(), Some("header-qwen-file-1"));
    assert_eq!(result.native["requestId"], "body-qwen-file-1");
    assert_eq!(result.native["future_envelope_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-files-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/listFile"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "categoryId":"cate-1",
            "fileName":"policy",
            "fileIds":["file_abc123_1"],
            "nextToken":"cursor-previous",
            "maxResult":10
        })
    );
}

#[tokio::test]
async fn list_files_advances_cursor_and_rejects_nonprogressing_pages() {
    let mock = Mock::new([
        response(
            200,
            success(json!({
                "hasNext":true,
                "nextToken":"cursor-2",
                "maxResult":20,
                "totalCount":2,
                "fileList":[file_row("file_first_1", "one.md")]
            })),
        ),
        response(
            200,
            success(json!({
                "hasNext":false,
                "maxResult":20,
                "totalCount":2,
                "fileList":[file_row("file_second_2", "two.md")]
            })),
        ),
        response(
            200,
            success(json!({
                "hasNext":true,
                "nextToken":"cursor-2",
                "fileList":[]
            })),
        ),
    ]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let first = service
        .list_files(
            &QwenKnowledgeFileListRequest::new("cate-1"),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(first.next_token.as_deref(), Some("cursor-2"));

    let second = service
        .list_files(
            &QwenKnowledgeFileListRequest::new("cate-1").with_next_token(first.next_token.unwrap()),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(second.files[0].reference.file_id(), "file_second_2");

    let error = service
        .list_files(
            &QwenKnowledgeFileListRequest::new("cate-1").with_next_token("cursor-2"),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidResponse { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 3);
    assert_eq!(
        serde_json::from_slice::<Value>(&mock.requests.lock().unwrap()[1].body).unwrap()
            ["nextToken"],
        "cursor-2"
    );
}

#[tokio::test]
async fn list_files_requires_progress_when_has_next_and_rejects_duplicate_page_ids() {
    for data in [
        json!({"hasNext":true,"fileList":[]}),
        json!({"hasNext":true,"nextToken":"","fileList":[]}),
        json!({
            "hasNext":false,
            "fileList":[
                file_row("file_same_1", "one.md"),
                file_row("file_same_1", "one.md")
            ]
        }),
    ] {
        let mock = Mock::new([response(200, success(data))]);
        let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
        assert!(matches!(
            service
                .list_files(
                    &QwenKnowledgeFileListRequest::new("cate-1"),
                    &request_options()
                )
                .await,
            Err(QwenKnowledgeError::InvalidResponse { .. })
        ));
    }

    let mock = Mock::new([response(
        200,
        success(json!({"hasNext":true,"nextToken":"cursor-1","fileList":[]})),
    )]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    assert!(matches!(
        service
            .list_files(
                &QwenKnowledgeFileListRequest::new("cate-1").with_next_token("cursor-1"),
                &request_options()
            )
            .await,
        Err(QwenKnowledgeError::InvalidResponse { .. })
    ));
}

#[tokio::test]
async fn delete_file_uses_scoped_file_id_and_accepts_documented_empty_data() {
    let mock = Mock::new([response(200, success(json!({})))]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let result = service
        .delete_file(&file, &request_options())
        .await
        .unwrap();
    assert_eq!(result.reference, file);
    assert_eq!(result.deleted_file_id, None);
    assert_eq!(result.status, None);
    assert_eq!(result.native["future_envelope_field"]["preserved"], true);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-files-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/deleteFile"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"fileId":"file_abc123_1"})
    );
}

#[tokio::test]
async fn delete_file_preserves_confirmed_status_and_marks_mismatched_success_unknown() {
    let mock = Mock::new([
        response(
            200,
            success(json!({"fileId":"file_abc123_1","status":"DELETED"})),
        ),
        response(
            200,
            success(json!({"fileId":"file_other_2","status":"DELETED"})),
        ),
    ]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let result = service
        .delete_file(&file, &request_options())
        .await
        .unwrap();
    assert_eq!(result.deleted_file_id.as_deref(), Some("file_abc123_1"));
    assert_eq!(result.status.as_deref(), Some("DELETED"));

    let error = service
        .delete_file(&file, &request_options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert!(matches!(
        error,
        QwenKnowledgeError::ResponseOutcomeUnknown { .. }
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn batch_update_tags_encodes_mode_and_preserves_per_file_results() {
    let mock = Mock::new([response(
        200,
        success(json!({
            "results":[
                {"fileId":"file_second_2","success":false,"future_error":"permission"},
                {"fileId":"file_first_1","success":true,"future_result":{"kept":true}}
            ],
            "future_data_field":7
        })),
    )]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let first = service.scope().file_ref("file_first_1").unwrap();
    let second = service.scope().file_ref("file_second_2").unwrap();
    let request = QwenKnowledgeFileTagUpdateRequest::new([
        QwenKnowledgeFileTagUpdate::new(first.clone(), ["产品", "V2"]),
        QwenKnowledgeFileTagUpdate::new(second.clone(), ["FAQ"]),
    ])
    .with_update_mode(QwenKnowledgeFileTagUpdateMode::Overwrite);
    let result = service
        .update_file_tags(&request, &request_options())
        .await
        .unwrap();
    let rows = result.per_file_results.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].reference, second);
    assert!(!rows[0].success);
    assert_eq!(rows[0].native["future_error"], "permission");
    assert_eq!(rows[1].reference, first);
    assert!(rows[1].success);
    assert_eq!(rows[1].native["future_result"]["kept"], true);
    assert_eq!(result.native["data"]["future_data_field"], 7);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://llm-files-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/batchUpdateFileTag");
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "fileInfos":[
                {"fileId":"file_first_1","tags":["产品","V2"]},
                {"fileId":"file_second_2","tags":["FAQ"]}
            ],
            "updateMode":"OVERWRITE"
        })
    );
}

#[tokio::test]
async fn batch_update_tags_accepts_documented_empty_data_and_omits_optional_mode() {
    let mock = Mock::new([response(200, success(json!({})))]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let request = QwenKnowledgeFileTagUpdateRequest::new([QwenKnowledgeFileTagUpdate::new(
        file,
        std::iter::empty::<String>(),
    )]);
    let result = service
        .update_file_tags(&request, &request_options())
        .await
        .unwrap();
    assert_eq!(result.per_file_results, None);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"fileInfos":[{"fileId":"file_abc123_1","tags":[]}]})
    );
}

#[tokio::test]
async fn mutations_preflight_scopes_limits_and_uncertain_dispatch_without_retry() {
    let mock = Mock::new([MockReply::Failure]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let foreign = scope(
        QwenKnowledgeRegion::Beijing,
        "account-2",
        "llm-other-workspace",
    )
    .file_ref("file_foreign_1")
    .unwrap();
    assert!(matches!(
        service.delete_file(&foreign, &request_options()).await,
        Err(QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let foreign_update = QwenKnowledgeFileTagUpdateRequest::new([QwenKnowledgeFileTagUpdate::new(
        foreign.clone(),
        ["tag"],
    )]);
    assert!(matches!(
        service
            .update_file_tags(&foreign_update, &request_options())
            .await,
        Err(QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let too_many_tags = QwenKnowledgeFileTagUpdateRequest::new([QwenKnowledgeFileTagUpdate::new(
        file.clone(),
        std::iter::repeat_n("t".to_owned(), 101),
    )]);
    assert!(matches!(
        service
            .update_file_tags(&too_many_tags, &request_options())
            .await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));

    assert!(matches!(
        service
            .list_files(
                &QwenKnowledgeFileListRequest::new("cate-1")
                    .with_file_ids((0..21).map(|index| format!("file_id_{index}"))),
                &request_options()
            )
            .await,
        Err(QwenKnowledgeError::InvalidInput(_))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    let error = service
        .delete_file(&file, &request_options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert!(matches!(error, QwenKnowledgeError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_tag_preflight_enforces_each_documented_limit_without_io() {
    let mock = Mock::new([]);
    let service = service(&mock, QwenKnowledgeRegion::Beijing, "account-1");
    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let requests = [
        QwenKnowledgeFileTagUpdateRequest::new((0..21).map(|index| {
            QwenKnowledgeFileTagUpdate::new(
                service.scope().file_ref(format!("file_{index}_1")).unwrap(),
                ["tag"],
            )
        })),
        QwenKnowledgeFileTagUpdateRequest::new([QwenKnowledgeFileTagUpdate::new(
            file.clone(),
            ["x".repeat(33)],
        )]),
        QwenKnowledgeFileTagUpdateRequest::new([QwenKnowledgeFileTagUpdate::new(
            file.clone(),
            std::iter::repeat_n("1234567890".to_owned(), 71),
        )]),
    ];
    for request in requests {
        assert!(matches!(
            service.update_file_tags(&request, &request_options()).await,
            Err(QwenKnowledgeError::InvalidInput(_))
        ));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn unsupported_region_fails_before_file_management_io() {
    let mock = Mock::new([]);
    let service = service(&mock, QwenKnowledgeRegion::Singapore, "account-sg");
    let file = service.scope().file_ref("file_abc123_1").unwrap();
    let error = service
        .delete_file(&file, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        QwenKnowledgeError::Llm(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("qwen-file-key".to_owned())),
        ..Default::default()
    }
}
