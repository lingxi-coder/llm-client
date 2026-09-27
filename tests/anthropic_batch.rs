use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    anthropic_batch::*,
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<Bytes>,
    headers: Vec<(String, String)>,
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
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: futures::stream::iter(reply.body.into_iter().map(Ok)).boxed(),
        })
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        panic!("Anthropic Message Batches use inline JSON requests")
    }
}

fn response(status: u16, body: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: vec![Bytes::from(serde_json::to_vec(&body).unwrap())],
        headers: vec![("content-type".into(), "application/json".into())],
    })
}

fn result_stream(
    status: u16,
    chunks: impl IntoIterator<Item = &'static str>,
) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: chunks
            .into_iter()
            .map(|chunk| Bytes::from_static(chunk.as_bytes()))
            .collect(),
        headers: vec![("content-type".into(), "application/x-jsonlines".into())],
    })
}

fn batch(id: &str, status: &str) -> Value {
    json!({
        "id": id,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 1,
            "succeeded": 2,
            "errored": 3,
            "canceled": 4,
            "expired": 5
        },
        "created_at": "2026-09-25T10:00:00Z",
        "expires_at": "2026-09-26T10:00:00Z",
        "ended_at": null,
        "cancel_initiated_at": null,
        "archived_at": null,
        "results_url": format!("https://api.anthropic.com/v1/messages/batches/{id}/results"),
        "future_field": "preserved"
    })
}

fn scope(account: &str) -> AnthropicBatchScope {
    AnthropicBatchScope::new(
        "anthropic-prod",
        account,
        "https://api.anthropic.com",
        Some("wrkspc_test".into()),
    )
    .unwrap()
}

fn service<'a>(transport: &'a MockTransport, account: &str) -> AnthropicBatchService<'a> {
    AnthropicBatchService::new(transport, Secret::new("sk-ant-test".into()), scope(account))
        .unwrap()
}

fn request(custom_id: &str) -> AnthropicBatchRequest {
    AnthropicBatchRequest::new(
        custom_id,
        AnthropicBatchParams::new(
            "claude-sonnet-5",
            512,
            vec![AnthropicBatchMessage {
                role: AnthropicBatchRole::User,
                content: json!([{"type":"text", "text":"hello"}]),
            }],
        )
        .unwrap()
        .with_parameter("system", json!("Be concise"))
        .unwrap(),
    )
    .unwrap()
}

fn input() -> AnthropicBatchInput {
    AnthropicBatchInput::new(vec![request("item-1")]).unwrap()
}

fn json_body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn typed_inline_input_validates_ids_core_fields_and_batch_only_restrictions() {
    assert_eq!(input().len(), 1);
    assert!(matches!(
        AnthropicBatchParams::new(
            "claude-sonnet-5",
            0,
            vec![AnthropicBatchMessage {
                role: AnthropicBatchRole::User,
                content: json!("hello"),
            }]
        ),
        Err(AnthropicBatchError::InvalidInput(_))
    ));
    assert!(matches!(
        AnthropicBatchParams::new("claude-sonnet-5", 10, vec![]),
        Err(AnthropicBatchError::InvalidInput(_))
    ));
    assert!(matches!(
        AnthropicBatchParams::new(
            "claude-sonnet-5",
            10,
            vec![AnthropicBatchMessage {
                role: AnthropicBatchRole::User,
                content: json!("hello"),
            }]
        )
        .unwrap()
        .with_parameter("stream", json!(true)),
        Err(AnthropicBatchError::InvalidInput(_))
    ));
    assert!(matches!(
        AnthropicBatchParams::new(
            "claude-sonnet-5",
            10,
            vec![AnthropicBatchMessage {
                role: AnthropicBatchRole::User,
                content: json!("hello"),
            }]
        )
        .unwrap()
        .with_parameter("speed", json!("fast")),
        Err(AnthropicBatchError::InvalidInput(_))
    ));
    assert!(matches!(
        AnthropicBatchInput::new(vec![request("duplicate"), request("duplicate")]),
        Err(AnthropicBatchError::InvalidInput(_))
    ));
    assert!(AnthropicBatchRequest::new("not valid", request("valid").params).is_err());
}

#[tokio::test]
async fn create_get_list_and_cancel_use_scoped_routes_and_documented_headers() {
    let replies = [
        response(200, batch("msgbatch_1", "in_progress")),
        response(200, batch("msgbatch_1", "in_progress")),
        response(
            200,
            json!({
                "data": [batch("msgbatch_1", "in_progress")],
                "first_id": "first",
                "last_id": "last",
                "has_more": true,
            }),
        ),
        response(200, batch("msgbatch_1", "canceling")),
    ];
    let mock = MockTransport::new(replies);
    let service = service(&mock, "account-a");

    let created = service.create(&input()).await.unwrap();
    assert_eq!(created.reference.batch_id(), "msgbatch_1");
    assert_eq!(created.request_counts.succeeded, 2);
    assert_eq!(created.native["future_field"], "preserved");

    let fetched = service.get(&created.reference).await.unwrap();
    assert_eq!(fetched.status, AnthropicBatchStatus::InProgress);
    let page = service
        .list(
            &AnthropicBatchListOptions::new()
                .limit(9)
                .after_id("cursor/one"),
        )
        .await
        .unwrap();
    assert!(page.has_more);
    assert_eq!(page.batches[0].reference.batch_id(), "msgbatch_1");
    let canceled = service.cancel(&created.reference).await.unwrap();
    assert_eq!(canceled.status, AnthropicBatchStatus::Canceling);

    let requests = mock.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://api.anthropic.com/v1/messages/batches"
    );
    assert_eq!(
        json_body(&requests[0])["requests"][0]["custom_id"],
        "item-1"
    );
    assert_eq!(
        json_body(&requests[0])["requests"][0]["params"]["max_tokens"],
        512
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "X-Api-Key" && value == "sk-ant-test"));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| { name == "anthropic-version" && value == "2023-06-01" }));
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name == "anthropic-beta" && value == "message-batches-2024-09-24"
    }));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| { name == "anthropic-workspace-id" && value == "wrkspc_test" }));
    assert!(requests[1].url.ends_with("/v1/messages/batches/msgbatch_1"));
    assert!(requests[2].url.contains("limit=9"));
    assert!(requests[2].url.contains("after_id=cursor%2Fone"));
    assert!(requests[3]
        .url
        .ends_with("/v1/messages/batches/msgbatch_1/cancel"));
}

#[tokio::test]
async fn delete_uses_documented_route_and_validates_the_confirmation() {
    let mock = MockTransport::new([
        response(200, batch("msgbatch_delete", "ended")),
        response(
            200,
            json!({"id":"msgbatch_delete", "type":"message_batch_deleted"}),
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let created = batch_service.create(&input()).await.unwrap();

    let deleted = batch_service.delete(&created.reference).await.unwrap();
    assert_eq!(deleted.id, "msgbatch_delete");
    assert_eq!(deleted.object_type, "message_batch_deleted");

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].method, "DELETE");
    assert!(requests[1]
        .url
        .ends_with("/v1/messages/batches/msgbatch_delete"));
    assert!(requests[1].body.is_empty());
    assert!(requests[1]
        .headers
        .iter()
        .any(|(name, value)| name == "anthropic-beta" && value == "message-batches-2024-09-24"));

    let malformed = MockTransport::new([
        response(200, batch("msgbatch_delete", "ended")),
        response(
            200,
            json!({"id":"another_batch", "type":"message_batch_deleted"}),
        ),
    ]);
    let service = service(&malformed, "account-a");
    let created = service.create(&input()).await.unwrap();
    assert!(matches!(
        service.delete(&created.reference).await,
        Err(AnthropicBatchError::OutcomeUnknownResponse {
            operation: "delete",
            ..
        })
    ));
    assert_eq!(malformed.requests().len(), 2);
}

#[tokio::test]
async fn delete_checks_scope_before_io_and_preserves_known_provider_rejections() {
    let source = MockTransport::new([response(200, batch("msgbatch_delete", "ended"))]);
    let reference = service(&source, "account-a")
        .create(&input())
        .await
        .unwrap()
        .reference;

    let foreign = MockTransport::new(std::iter::empty::<Result<Reply, LlmError>>());
    assert!(matches!(
        service(&foreign, "account-b").delete(&reference).await,
        Err(AnthropicBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(foreign.requests().is_empty());

    let rejected = MockTransport::new([response(409, json!({"error":"batch still processing"}))]);
    assert!(matches!(
        service(&rejected, "account-a").delete(&reference).await,
        Err(AnthropicBatchError::Provider { status: 409, .. })
    ));
    assert_eq!(rejected.requests().len(), 1);

    let ambiguous = MockTransport::new([Err(LlmError::Transport {
        message: "connection closed after delete dispatch".into(),
    })]);
    assert!(matches!(
        service(&ambiguous, "account-a").delete(&reference).await,
        Err(AnthropicBatchError::OutcomeUnknown {
            operation: "delete",
            ..
        })
    ));
    assert_eq!(ambiguous.requests().len(), 1);
}

#[tokio::test]
async fn create_marks_transport_and_unreadable_success_as_unknown_without_retry() {
    let mock = MockTransport::new([Err(LlmError::Transport {
        message: "connection closed after dispatch".into(),
    })]);
    let error = service(&mock, "account-a")
        .create(&input())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        AnthropicBatchError::OutcomeUnknown {
            operation: "create",
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);

    let mock = MockTransport::new([
        response(200, batch("msgbatch_1", "in_progress")),
        Err(LlmError::Transport {
            message: "connection closed after dispatch".into(),
        }),
    ]);
    let batches = service(&mock, "account-a");
    let job = batches.create(&input()).await.unwrap();
    let error = batches.cancel(&job.reference).await.unwrap_err();
    assert!(matches!(
        error,
        AnthropicBatchError::OutcomeUnknown {
            operation: "cancel",
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 2);

    let mock = MockTransport::new([Ok(Reply {
        status: 200,
        body: vec![Bytes::from_static(b"not-json")],
        headers: vec![],
    })]);
    let error = service(&mock, "account-a")
        .create(&input())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        AnthropicBatchError::OutcomeUnknownResponse {
            operation: "create",
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn unknown_batch_statuses_serialize_as_their_provider_string() {
    let status = AnthropicBatchStatus::Other("paused".into());
    assert_eq!(serde_json::to_value(status).unwrap(), json!("paused"));
}

#[tokio::test]
async fn references_cannot_cross_account_or_endpoint_scope() {
    let mock_a = MockTransport::new([response(200, batch("msgbatch_1", "ended"))]);
    let job = service(&mock_a, "account-a")
        .create(&input())
        .await
        .unwrap();
    let mock_b = MockTransport::new(std::iter::empty::<Result<Reply, LlmError>>());
    let error = service(&mock_b, "account-b")
        .get(&job.reference)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        AnthropicBatchError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock_b.requests().is_empty());

    let other_endpoint = AnthropicBatchService::new(
        &mock_b,
        Secret::new("sk-ant-test".into()),
        AnthropicBatchScope::new(
            "anthropic-prod",
            "account-a",
            "https://proxy.example.test/anthropic",
            Some("wrkspc_test".into()),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        other_endpoint.get(&job.reference).await,
        Err(AnthropicBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock_b.requests().is_empty());
}

#[tokio::test]
async fn results_are_parsed_incrementally_and_keep_per_item_failures_as_values() {
    let mock = MockTransport::new([
        response(200, batch("msgbatch_1", "ended")),
        result_stream(
            200,
            [
                "{\"custom_id\":\"ok\",\"result\":{\"type\":\"succeeded\",\"message\":{\"id\":\"msg_1\"}}}\n{\"custom_id\":\"bad\",\"result\":{\"type\":\"errored\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad row\"}}}\n",
                "{\"custom_id\":\"stop\",\"result\":{\"type\":\"canceled\"}}\n{\"custom_id\":\"later\",\"result\":{\"type\":\"expired\"}}\n{\"custom_id\":\"future\",\"result\":{\"type\":\"new_status\",\"x\":1}}",
            ],
        ),
    ]);
    let service = service(&mock, "account-a");
    let job = service.create(&input()).await.unwrap();
    let items: Vec<_> = service
        .stream_results(&job.reference)
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(items.len(), 5);
    assert!(matches!(
        &items[0],
        Ok(AnthropicBatchItemResult {
            custom_id,
            outcome: AnthropicBatchItemOutcome::Succeeded { .. },
            ..
        }) if custom_id == "ok"
    ));
    assert!(matches!(
        &items[1],
        Ok(AnthropicBatchItemResult {
            custom_id,
            outcome: AnthropicBatchItemOutcome::Errored { error },
            ..
        }) if custom_id == "bad" && error["type"] == "invalid_request_error"
    ));
    assert!(matches!(
        &items[2],
        Ok(AnthropicBatchItemResult {
            outcome: AnthropicBatchItemOutcome::Canceled,
            ..
        })
    ));
    assert!(matches!(
        &items[3],
        Ok(AnthropicBatchItemResult {
            outcome: AnthropicBatchItemOutcome::Expired,
            ..
        })
    ));
    assert!(matches!(
        &items[4],
        Ok(AnthropicBatchItemResult {
            outcome: AnthropicBatchItemOutcome::Other { type_name, .. },
            ..
        }) if type_name == "new_status"
    ));
    assert_eq!(
        mock.requests()[1].url,
        "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"
    );
}

#[tokio::test]
async fn provider_http_errors_are_known_and_malformed_result_lines_end_only_the_stream() {
    let mock = MockTransport::new([
        response(200, batch("msgbatch_1", "ended")),
        Ok(Reply {
            status: 500,
            body: vec![Bytes::from_static(b"{\"error\":\"unavailable\"}")],
            headers: vec![("request-id".into(), "req_1".into())],
        }),
    ]);
    let batches = service(&mock, "account-a");
    let job = batches.create(&input()).await.unwrap();
    let error = batches.stream_results(&job.reference).await.err().unwrap();
    assert!(matches!(
        error,
        AnthropicBatchError::Provider {
            status: 500,
            request_id: Some(ref request_id),
            ..
        } if request_id == "req_1"
    ));

    let mock = MockTransport::new([
        response(200, batch("msgbatch_1", "ended")),
        result_stream(
            200,
            ["{\"custom_id\":\"ok\",\"result\":{\"type\":\"errored\"}}\n"],
        ),
    ]);
    let batches = service(&mock, "account-a");
    let job = batches.create(&input()).await.unwrap();
    let mut stream = batches.stream_results(&job.reference).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Err(AnthropicBatchError::InvalidResult(_)))
    ));
    assert!(stream.next().await.is_none());
}
