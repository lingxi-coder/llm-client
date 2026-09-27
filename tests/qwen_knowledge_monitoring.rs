use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    qwen_knowledge::{
        monitoring::{QwenKnowledgeMonitoringRequest, QwenKnowledgeMonitoringResult},
        QwenKnowledgeError, QwenKnowledgeRegion, QwenKnowledgeScope, QwenKnowledgeService,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Mock {
    replies: Mutex<VecDeque<(u16, Value)>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = (u16, Value)>) -> Self {
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
        let (status, body) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request");
        let body = serde_json::to_vec(&body).unwrap();
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "monitor-header-id".into())],
            body: futures::stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
        })
    }
}

fn success(data: Value) -> Value {
    json!({
        "code": "Success",
        "status_code": 200,
        "success": true,
        "message": "success",
        "request_id": "monitor-body-id",
        "data": data,
        "status": "SUCCESS"
    })
}

fn scope(account: &str, workspace: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new(
        "qwen-main",
        account,
        QwenKnowledgeRegion::Beijing,
        workspace,
    )
    .unwrap()
}

fn service<'a>(mock: &'a Mock, account: &str, workspace: &str) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(
        mock,
        Secret::new("qwen-key-test".to_owned()),
        scope(account, workspace),
    )
    .unwrap()
}

#[tokio::test]
async fn monitoring_uses_the_documented_post_contract_and_preserves_raw_data() {
    let monitoring_data = json!({
        "pipelineCommercialType": "standard",
        // The current endpoint page's example shows an array here even though
        // its field table describes an object. Keep both wire shape and fields.
        "storageMonitorData": [],
        "qpsMonitorData": {
            "peakQps": 3,
            "totalRequests": 8,
            "avgQpsOfActiveSeconds": 0.75,
            "monitorData": [{
                "windowRange": 1780900000,
                "windowRangeEnd": 1780900060,
                "peakQpsInWindowRange": 3,
                "totalRequests": 8,
                "avgQpsOfActiveSeconds": 0.75,
                "successData": {"totalRequests": 7},
                "limitData": {"totalRequests": 1},
                "failData": {"totalRequests": 0},
                "provider_future_field": "preserved"
            }]
        },
        "provider_future_field": {"preserved": true}
    });
    let mock = Mock::new([(200, success(monitoring_data.clone()))]);
    let service = service(&mock, "acct-1", "llm-workspace-1");
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let request = QwenKnowledgeMonitoringRequest::new(1_780_900_000, 1_783_492_000);

    assert_eq!(request.start_timestamp_secs(), 1_780_900_000);
    assert_eq!(request.end_timestamp_secs(), 1_783_492_000);
    let result: QwenKnowledgeMonitoringResult = service
        .get_knowledge_base_monitoring(&knowledge, &request)
        .await
        .unwrap();

    assert_eq!(result.knowledge, knowledge);
    assert_eq!(result.request_id.as_deref(), Some("monitor-header-id"));
    assert_eq!(result.data, monitoring_data);
    assert_eq!(result.native["request_id"], "monitor-body-id");
    assert_eq!(result.native["data"], result.data);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-workspace-1.cn-beijing.maas.aliyuncs.com/api/v1/indices/rag/index/monitor"
    );
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .find(|(name, _)| name == "content-type")
            .map(|(_, value)| value.as_str()),
        Some("application/json")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "indexId": "index-123",
            "startTimestamp": "1780900000",
            "endTimestamp": "1783492000"
        })
    );
}

#[tokio::test]
async fn accepts_the_official_monitoring_success_sample_without_success_boolean() {
    let mock = Mock::new([(
        200,
        json!({
            "code":"Success",
            "status_code":200,
            "data":{"storageMonitorData":[],"qpsMonitorData":[]},
            "request_id":"monitor-sample-request"
        }),
    )]);
    let service = service(&mock, "acct-1", "llm-workspace-1");
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let result = service
        .get_knowledge_base_monitoring(
            &knowledge,
            &QwenKnowledgeMonitoringRequest::new(1_780_900_000, 1_780_900_060),
        )
        .await
        .unwrap();

    assert_eq!(
        result.data,
        json!({"storageMonitorData":[],"qpsMonitorData":[]})
    );
    assert_eq!(result.native["request_id"], "monitor-sample-request");
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn monitoring_rejects_cross_scope_references_before_sending() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1", "llm-workspace-1");
    let foreign_knowledge = scope("acct-2", "llm-workspace-2")
        .knowledge_ref("index-foreign")
        .unwrap();
    let request = QwenKnowledgeMonitoringRequest::new(1_780_900_000, 1_780_900_060);

    let error = service
        .get_knowledge_base_monitoring(&foreign_knowledge, &request)
        .await
        .expect_err("monitoring must remain bound to the service scope");

    assert!(matches!(
        error,
        QwenKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn monitoring_rejects_invalid_windows_before_sending() {
    let mock = Mock::new([]);
    let service = service(&mock, "acct-1", "llm-workspace-1");
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let too_long = QwenKnowledgeMonitoringRequest::new(0, 30 * 24 * 60 * 60 + 1);
    let reversed = QwenKnowledgeMonitoringRequest::new(2, 1);

    for request in [&too_long, &reversed] {
        let error = service
            .get_knowledge_base_monitoring(&knowledge, request)
            .await
            .expect_err("invalid monitoring windows must not be sent");
        assert!(matches!(error, QwenKnowledgeError::InvalidInput(_)));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn monitoring_requires_the_documented_data_object_and_keeps_response_context() {
    let mut body = success(Value::Null);
    body.as_object_mut().unwrap().remove("data");
    let mock = Mock::new([(200, body.clone())]);
    let service = service(&mock, "acct-1", "llm-workspace-1");
    let knowledge = service.knowledge_ref("index-123").unwrap();
    let request = QwenKnowledgeMonitoringRequest::new(1_780_900_000, 1_780_900_060);

    let error = service
        .get_knowledge_base_monitoring(&knowledge, &request)
        .await
        .expect_err("the documented successful envelope requires data");

    match error {
        QwenKnowledgeError::InvalidResponse {
            request_id, native, ..
        } => {
            assert_eq!(request_id.as_deref(), Some("monitor-header-id"));
            assert_eq!(*native, body);
        }
        other => panic!("expected invalid response, got {other}"),
    }
}
