use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::kimi::batch::*,
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    chunks: Vec<Vec<u8>>,
}

impl Reply {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            chunks: vec![serde_json::to_vec(&value).unwrap()],
        }
    }

    fn bytes(status: u16, chunks: Vec<Vec<u8>>) -> Self {
        Self { status, chunks }
    }
}

struct RecordedRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    requests: Mutex<Vec<RecordedRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: futures::stream::iter(
                reply.chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))),
            )
            .boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(RecordedRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request")?;
        Ok(Self::response(reply))
    }
}

fn scope(account: &str) -> KimiBatchScope {
    KimiBatchScope::new(
        "kimi-intl-production",
        account,
        "https://api.moonshot.ai/v1",
    )
    .unwrap()
}

fn service<'a>(http: &'a MockTransport, account: &str) -> KimiBatchService<'a> {
    KimiBatchService::new(http, scope(account)).unwrap()
}

fn task(status: &str) -> Value {
    json!({
        "id":"batch_test_1",
        "object":"batch",
        "endpoint":"/v1/chat/completions",
        "input_file_id":"file_input_1",
        "completion_window":"24h",
        "status":status,
        "request_counts":{"total":2,"completed":1,"failed":0},
        "output_file_id": if status == "completed" { Some("file_output_1") } else { None::<&str> },
        "error_file_id":null,
        "created_at":1711402400
    })
}

fn input_file(account: &str) -> KimiBatchInputFileRef {
    KimiBatchInputFileRef::new(scope(account), KimiBatchModel::KimiK2_6, "file_input_1").unwrap()
}

#[tokio::test]
async fn submit_get_cancel_and_stream_result_use_documented_routes() {
    let http = MockTransport::new([
        Ok(Reply::json(200, task("validating"))),
        Ok(Reply::json(200, task("completed"))),
        Ok(Reply::json(200, task("cancelling"))),
        Ok(Reply::bytes(
            200,
            vec![
                b"{\"custom_id\":\"first\"}\n".to_vec(),
                b"{\"custom_id\":\"second\"}\n".to_vec(),
            ],
        )),
    ]);
    let service = service(&http, "account-a");

    let created = service
        .submit(
            &input_file("account-a"),
            KimiBatchSubmitOptions::default(),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(created.status, KimiBatchStatus::Validating);
    assert_eq!(created.reference.model(), KimiBatchModel::KimiK2_6);

    let completed = service
        .get(&created.reference, &request_options())
        .await
        .unwrap();
    assert_eq!(completed.status, KimiBatchStatus::Completed);
    assert!(completed.status.is_terminal());
    let output = completed.output_ref().unwrap();

    let cancelled = service
        .cancel(&created.reference, &request_options())
        .await
        .unwrap();
    assert_eq!(cancelled.status, KimiBatchStatus::Cancelling);

    let mut stream = service
        .stream_result(&output, &request_options())
        .await
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        Bytes::from_static(b"{\"custom_id\":\"first\"}\n")
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        Bytes::from_static(b"{\"custom_id\":\"second\"}\n")
    );
    assert!(stream.next().await.is_none());

    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://api.moonshot.ai/v1/batches");
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "input_file_id":"file_input_1",
            "endpoint":"/v1/chat/completions",
            "completion_window":"24h"
        })
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "Authorization" && value == "Bearer sk-kimi-test"));
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://api.moonshot.ai/v1/batches/batch_test_1"
    );
    assert_eq!(requests[2].method, "POST");
    assert_eq!(
        requests[2].url,
        "https://api.moonshot.ai/v1/batches/batch_test_1/cancel"
    );
    assert_eq!(
        requests[3].url,
        "https://api.moonshot.ai/v1/files/file_output_1/content"
    );
}

#[tokio::test]
async fn list_returns_a_scoped_page_without_inventing_a_missing_model() {
    let list = json!({
        "object":"list",
        "data":[task("completed")],
        "has_more":true
    });
    let http = MockTransport::new([
        Ok(Reply::json(200, list)),
        Ok(Reply::json(200, task("completed"))),
    ]);
    let service = service(&http, "account-a");
    let page = service
        .list(
            &KimiBatchListOptions::new().after("batch_test_0").limit(3),
            &request_options(),
        )
        .await
        .unwrap();

    assert!(page.has_more);
    assert_eq!(page.batches.len(), 1);
    let listed = &page.batches[0];
    assert_eq!(listed.batch_id(), "batch_test_1");
    assert_eq!(listed.status, KimiBatchStatus::Completed);
    assert_eq!(listed.scope().account_scope(), "account-a");
    assert_eq!(listed.native["id"], "batch_test_1");

    let reference = listed
        .reference_with_model(KimiBatchModel::KimiK2_6)
        .unwrap();
    let fetched = service.get(&reference, &request_options()).await.unwrap();
    assert_eq!(fetched.reference.model(), KimiBatchModel::KimiK2_6);

    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert!(requests[0]
        .url
        .starts_with("https://api.moonshot.ai/v1/batches?"));
    assert!(requests[0].url.contains("after=batch_test_0"));
    assert!(requests[0].url.contains("limit=3"));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "Authorization" && value == "Bearer sk-kimi-test"));
    assert!(requests[1].url.ends_with("/batches/batch_test_1"));
}

#[tokio::test]
async fn list_rejects_invalid_options_before_network_access() {
    let http = MockTransport::new([]);
    assert!(matches!(
        service(&http, "account-a")
            .list(&KimiBatchListOptions::new().limit(0), &request_options())
            .await,
        Err(KimiBatchError::InvalidInput(_))
    ));
    assert!(matches!(
        service(&http, "account-a")
            .list(
                &KimiBatchListOptions::new().after("bad/id"),
                &request_options()
            )
            .await,
        Err(KimiBatchError::InvalidInput(_))
    ));
    assert!(http.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn list_rejects_duplicate_or_non_advancing_pages() {
    let empty_page = MockTransport::new([Ok(Reply::json(
        200,
        json!({"object":"list", "data":[], "has_more":true}),
    ))]);
    assert!(matches!(
        service(&empty_page, "account-a")
            .list(&KimiBatchListOptions::new(), &request_options())
            .await,
        Err(KimiBatchError::InvalidResponse(_))
    ));

    let repeated_page = MockTransport::new([Ok(Reply::json(
        200,
        json!({
            "object":"list",
            "data":[task("validating"), task("completed")],
            "has_more":false
        }),
    ))]);
    assert!(matches!(
        service(&repeated_page, "account-a")
            .list(&KimiBatchListOptions::new(), &request_options())
            .await,
        Err(KimiBatchError::InvalidResponse(_))
    ));

    let stuck_cursor = MockTransport::new([Ok(Reply::json(
        200,
        json!({
            "object":"list",
            "data":[task("validating")],
            "has_more":true
        }),
    ))]);
    assert!(matches!(
        service(&stuck_cursor, "account-a")
            .list(
                &KimiBatchListOptions::new().after("batch_test_1"),
                &request_options()
            )
            .await,
        Err(KimiBatchError::InvalidResponse(_))
    ));
}

#[tokio::test]
async fn references_are_bound_to_profile_account_and_endpoint_before_network_access() {
    let http = MockTransport::new([Ok(Reply::json(200, task("validating")))]);
    let source = service(&http, "account-a");
    let created = source
        .submit(
            &input_file("account-a"),
            KimiBatchSubmitOptions::default(),
            &request_options(),
        )
        .await
        .unwrap();
    let before = http.requests.lock().unwrap().len();

    let foreign_account = service(&http, "account-b");
    assert!(matches!(
        foreign_account
            .get(&created.reference, &request_options())
            .await,
        Err(KimiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let mut encoded = serde_json::to_value(&created.reference).unwrap();
    encoded["scope"]["profile_name"] = json!("other-profile");
    let foreign_profile: KimiBatchJobRef = serde_json::from_value(encoded).unwrap();
    assert!(matches!(
        source.get(&foreign_profile, &request_options()).await,
        Err(KimiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));

    let mut encoded = serde_json::to_value(&created.reference).unwrap();
    encoded["endpoint_fingerprint"] = json!("different-endpoint");
    let foreign_endpoint: KimiBatchJobRef = serde_json::from_value(encoded).unwrap();
    assert!(matches!(
        source.get(&foreign_endpoint, &request_options()).await,
        Err(KimiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(http.requests.lock().unwrap().len(), before);
}

#[tokio::test]
async fn submit_reports_unknown_outcome_after_transport_failure() {
    let http = MockTransport::new([Err(LlmError::TransportTimeout {
        message: "HTTP request timed out".into(),
    })]);
    let result = service(&http, "account-a")
        .submit(
            &input_file("account-a"),
            KimiBatchSubmitOptions::default(),
            &request_options(),
        )
        .await;
    assert!(matches!(
        result,
        Err(KimiBatchError::OutcomeUnknown {
            operation: "submission",
            source: LlmError::TransportTimeout { .. }
        })
    ));
}

#[test]
fn scope_models_and_identifiers_are_validated() {
    assert!(KimiBatchScope::new("kimi-cn", "account-a", "https://api.moonshot.cn/v1").is_err());
    assert!(KimiBatchScope::new("", "account-a", "https://api.moonshot.ai/v1").is_err());
    assert!(KimiBatchInputFileRef::new(
        scope("account-a"),
        KimiBatchModel::KimiK2_7Code,
        "file/other-account"
    )
    .is_err());
    assert!(serde_json::from_str::<KimiBatchModel>("\"kimi-k3\"").is_err());
    assert!(serde_json::from_str::<KimiBatchEndpoint>("\"/v1/responses\"").is_err());
}

#[tokio::test]
async fn successful_but_mismatched_submit_reply_reports_unknown_outcome() {
    let mut body = task("validating");
    body["input_file_id"] = json!("file_unexpected");
    let http = MockTransport::new([Ok(Reply::json(200, body))]);
    let result = service(&http, "account-a")
        .submit(
            &input_file("account-a"),
            KimiBatchSubmitOptions::default(),
            &request_options(),
        )
        .await;
    assert!(matches!(
        result,
        Err(KimiBatchError::ResponseOutcomeUnknown {
            operation: "submission",
            ..
        })
    ));
}

#[tokio::test]
async fn output_reference_is_only_available_after_completion() {
    let http = MockTransport::new([
        Ok(Reply::json(200, task("validating"))),
        Ok(Reply::json(200, task("completed"))),
    ]);
    let service = service(&http, "account-a");
    let created = service
        .submit(
            &input_file("account-a"),
            KimiBatchSubmitOptions::default(),
            &request_options(),
        )
        .await
        .unwrap();
    assert!(created.output_ref().is_none());
    let complete = service
        .get(&created.reference, &request_options())
        .await
        .unwrap();
    assert_eq!(complete.output_ref().unwrap().file_id(), "file_output_1");
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("sk-kimi-test".into())),
        ..Default::default()
    }
}

#[tokio::test]
async fn credentials_are_required_per_operation_and_can_rotate() {
    let mock = MockTransport::new([
        Ok(Reply::json(401, json!({"error": "rejected"}))),
        Ok(Reply::json(401, json!({"error": "rejected"}))),
    ]);
    let service = service(&mock, "account-a");
    let query = KimiBatchListOptions::default();
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
