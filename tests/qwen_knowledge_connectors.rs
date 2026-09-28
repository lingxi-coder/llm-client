use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::knowledge::{
        QwenKnowledgeCategoryRef, QwenKnowledgeCategoryType, QwenKnowledgeConnectorCreateRequest,
        QwenKnowledgeConnectorLookup, QwenKnowledgeDispatch, QwenKnowledgeError,
        QwenKnowledgeOssImportFile, QwenKnowledgeOssImportRequest, QwenKnowledgeParser,
        QwenKnowledgeParserConfig, QwenKnowledgeRegion, QwenKnowledgeScope, QwenKnowledgeService,
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
                    headers: vec![("x-request-id".into(), "header-qwen-connector-1".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
                })
            }
            Reply::Failure => Err(LlmError::TransportTimeout {
                message: "mock timeout after request dispatch".into(),
            }),
        }
    }
}

fn reply(body: Value) -> Reply {
    Reply::Json { status: 200, body }
}

fn scope(account: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new(
        "qwen-profile",
        account,
        QwenKnowledgeRegion::Beijing,
        "llm-connector-workspace",
    )
    .unwrap()
}

fn service<'a>(mock: &'a Mock, account: &str) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(mock, scope(account)).unwrap()
}

fn create_response(connector_id: &str) -> Value {
    json!({
        "code":"Success",
        "status_code":200,
        "data":{"connectorId":connector_id,"future_create_field":"kept"},
        "requestId":"request-create-1"
    })
}

#[tokio::test]
async fn create_connector_uses_documented_file_shape_and_binds_result_scope() {
    let mock = Mock::new([reply(create_response("conn_abc123"))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeConnectorCreateRequest::new(
        "Product docs",
        "Connector for product documentation",
    );
    let result = service
        .create_connector(&request, &request_options())
        .await
        .unwrap();

    assert_eq!(result.reference.connector_id(), "conn_abc123");
    assert_eq!(result.reference.scope(), service.scope());
    assert_eq!(
        result.request_id.as_deref(),
        Some("header-qwen-connector-1")
    );
    assert_eq!(result.native["requestId"], "request-create-1");
    assert_eq!(result.native["data"]["future_create_field"], "kept");
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-connector-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/addConnector"
    );
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["connectorType"], "FILE");
    assert_eq!(body["connectorName"], "Product docs");
    assert_eq!(body["description"], "Connector for product documentation");
    assert_eq!(body["fileConnectorConfig"], json!({"storeType":"PLATFORM"}));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer qwen-connector-api-key"));
}

#[tokio::test]
async fn custom_connector_and_get_by_name_use_documented_fields_and_status_only_sample() {
    let mock = Mock::new([
        reply(create_response("conn_custom_1")),
        reply(json!({
            "code":"Success",
            "message":"",
            "requestId":"request-get-1",
            "data":{
                "connectorId":"conn_custom_1",
                "connectorName":"My custom data",
                "connectorType":"FILE",
                "connectorSubType":"UNSTRUCTURED",
                "description":"My bucket connector",
                "gmtCreate":"2026-09-26 10:00:00",
                "gmtModified":"2026-09-26 10:00:00"
            },
            "status":200
        })),
    ]);
    let service = service(&mock, "acct-1");
    let created = service
        .create_connector(
            &QwenKnowledgeConnectorCreateRequest::new("My custom data", "My bucket connector")
                .with_custom_oss("cn-beijing", "my-docs-bucket"),
            &request_options(),
        )
        .await
        .unwrap();
    let details = service
        .get_connector(
            &QwenKnowledgeConnectorLookup::by_name("My custom data"),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(details.reference, created.reference);
    assert_eq!(details.connector_sub_type.as_deref(), Some("UNSTRUCTURED"));
    assert_eq!(details.created_at.as_deref(), Some("2026-09-26 10:00:00"));

    let requests = mock.requests.lock().unwrap();
    let create_body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        create_body["fileConnectorConfig"],
        json!({"storeType":"CUSTOM","regionId":"cn-beijing","bucketName":"my-docs-bucket"})
    );
    let get_body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(get_body, json!({"connectorName":"My custom data"}));
    assert_eq!(
        requests[1].url,
        requests[0].url.replace("addConnector", "getConnector")
    );
}

#[tokio::test]
async fn connector_id_lookup_is_typed_and_cross_scope_is_rejected_before_http() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1");
    let foreign = scope("acct-2").connector_ref("conn_foreign").unwrap();
    let error = service
        .get_connector(
            &QwenKnowledgeConnectorLookup::by_id(foreign),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::Llm(_)));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::NotSent);
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn oss_import_encodes_parser_tags_and_file_ids_as_scoped_refs() {
    let mock = Mock::new([reply(json!({
        "code":"Success",
        "status_code":200,
        "requestId":"request-import-1",
        "data":{"fileIds":["file_1","file_2"],"future":"kept"}
    }))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeOssImportRequest::new(
        "cate_abc123",
        "my-docs-bucket",
        "cn-beijing",
        [
            QwenKnowledgeOssImportFile::new("guide.pdf", "docs/guide.pdf")
                .with_parser(QwenKnowledgeParser::DashQwenVlParser)
                .with_parser_config(QwenKnowledgeParserConfig::new("Read text and tables.")),
        ],
    )
    .with_tags(["manual", "product"])
    .with_overwrite_file_by_oss_key(true);
    let result = service
        .import_files_from_oss(&request, &request_options())
        .await
        .unwrap();

    let imported = result.imported_files.unwrap();
    assert_eq!(imported.len(), 2);
    assert_eq!(imported[0].file_id(), "file_1");
    assert_eq!(imported[0].scope(), service.scope());
    assert_eq!(
        result.request_id.as_deref(),
        Some("header-qwen-connector-1")
    );
    assert_eq!(result.native["requestId"], "request-import-1");
    assert_eq!(result.native["data"]["future"], "kept");
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url,
        "https://llm-connector-workspace.cn-beijing.maas.aliyuncs.com/api/v1/connector/dash/addFilesFromAuthorizedOss"
    );
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["categoryId"], "cate_abc123");
    assert_eq!(body["categoryType"], "UNSTRUCTURED");
    assert_eq!(body["ossBucket"], "my-docs-bucket");
    assert_eq!(body["ossRegionId"], "cn-beijing");
    assert_eq!(body["fileDetails"][0]["parser"], "DASH_QWEN_VL_PARSER");
    assert_eq!(
        body["fileDetails"][0]["parserConfig"],
        json!({"modelName":"qwen3-vl-plus","modelPrompt":"Read text and tables."})
    );
    assert_eq!(body["tags"], json!(["manual", "product"]));
    assert_eq!(body["overWriteFileByOssKey"], true);
}

#[tokio::test]
async fn empty_import_data_is_preserved_without_inventing_a_job_or_file_ids() {
    let mock = Mock::new([reply(json!({
        "code":"Success","status_code":200,"requestId":"request-empty-1","data":{}
    }))]);
    let service = service(&mock, "acct-1");
    let request = QwenKnowledgeOssImportRequest::new(
        "cate_1",
        "docs-bucket",
        "cn-beijing",
        [QwenKnowledgeOssImportFile::new(
            "guide.md",
            "manuals/guide.md",
        )],
    );
    let result = service
        .import_files_from_oss(&request, &request_options())
        .await
        .unwrap();
    assert!(result.imported_files.is_none());
    assert_eq!(result.native["data"], json!({}));
}

#[tokio::test]
async fn documented_input_limits_and_session_parser_rules_fail_before_dispatch() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1");
    let too_many_files = (0..11)
        .map(|index| {
            QwenKnowledgeOssImportFile::new(format!("{index}.txt"), format!("{index}.txt"))
        })
        .collect::<Vec<_>>();
    let error = service
        .import_files_from_oss(
            &QwenKnowledgeOssImportRequest::new(
                "cate_1",
                "docs-bucket",
                "cn-beijing",
                too_many_files,
            ),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));

    let session_override = QwenKnowledgeOssImportRequest::new(
        "cate_1",
        "docs-bucket",
        "cn-beijing",
        [QwenKnowledgeOssImportFile::new("guide.md", "guide.md")
            .with_parser(QwenKnowledgeParser::AutoSelect)],
    )
    .with_category_type(QwenKnowledgeCategoryType::SessionFile);
    let error = service
        .import_files_from_oss(&session_override, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));

    let long_name = "x".repeat(21);
    let error = service
        .create_connector(
            &QwenKnowledgeConnectorCreateRequest::new(long_name, "Description"),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn typed_oss_category_refs_retain_scope_and_reject_forged_fingerprints() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1");
    let foreign = scope("acct-2").category_ref("cate_1").unwrap();
    let foreign_request = QwenKnowledgeOssImportRequest::for_category(
        foreign,
        "docs-bucket",
        "cn-beijing",
        [QwenKnowledgeOssImportFile::new("guide.md", "guide.md")],
    );
    let error = service
        .import_files_from_oss(&foreign_request, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::Llm(_)));

    let category = service.scope().category_ref("cate_1").unwrap();
    let mut serialized = serde_json::to_value(&category).unwrap();
    serialized["endpoint_fingerprint"] = json!("forged-endpoint");
    let forged: QwenKnowledgeCategoryRef = serde_json::from_value(serialized).unwrap();
    let forged_request = QwenKnowledgeOssImportRequest::for_category(
        forged,
        "docs-bucket",
        "cn-beijing",
        [QwenKnowledgeOssImportFile::new("guide.md", "guide.md")],
    );
    let error = service
        .import_files_from_oss(&forged_request, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(error, QwenKnowledgeError::Llm(_)));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn mutation_transport_and_malformed_response_errors_never_retry() {
    let timeout = Mock::new([Reply::Failure]);
    let error = service(&timeout, "acct-1")
        .import_files_from_oss(
            &QwenKnowledgeOssImportRequest::new(
                "cate_1",
                "docs-bucket",
                "cn-beijing",
                [QwenKnowledgeOssImportFile::new("guide.md", "guide.md")],
            ),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(&error, QwenKnowledgeError::OutcomeUnknown { .. }));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(timeout.requests.lock().unwrap().len(), 1);

    let malformed = Mock::new([reply(json!({
        "code":"Success","status_code":200,"requestId":"request-missing-id","data":{}
    }))]);
    let error = service(&malformed, "acct-1")
        .create_connector(
            &QwenKnowledgeConnectorCreateRequest::new("A connector", "A description"),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        &error,
        QwenKnowledgeError::ResponseOutcomeUnknown { .. }
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(malformed.requests.lock().unwrap().len(), 1);
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("qwen-connector-api-key".to_owned())),
        ..Default::default()
    }
}
