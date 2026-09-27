use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    embeddings::{GeminiEmbeddingMedia, GeminiEmbeddingSource},
    gemini_batch::*,
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    requests: Mutex<Vec<HttpRequest>>,
    stream_requests: Mutex<Vec<StreamRequestRecord>>,
}

struct StreamRequestRecord {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    content_length: u64,
    body: Vec<u8>,
}

type StreamRequestSnapshot = (String, String, Vec<(String, String)>, u64, Vec<u8>);

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            stream_requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn stream_requests(&self) -> Vec<StreamRequestSnapshot> {
        self.stream_requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| {
                (
                    request.method.clone(),
                    request.url.clone(),
                    request.headers.clone(),
                    request.content_length,
                    request.body.clone(),
                )
            })
            .collect()
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
            body: futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let HttpStreamRequest {
            method,
            url,
            headers,
            mut body,
            content_length,
            ..
        } = request;
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        self.stream_requests
            .lock()
            .unwrap()
            .push(StreamRequestRecord {
                method,
                url,
                headers,
                content_length,
                body: bytes,
            });
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
        headers: vec![("content-type".into(), "application/json".into())],
    })
}

fn raw_reply(
    status: u16,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body,
        headers,
    })
}

fn operation(id: &str, state: &str, response: Option<Value>) -> Value {
    let mut operation = json!({
        "name": format!("batches/{id}"),
        "done": true,
        "metadata": {"state": state}
    });
    if let Some(response) = response {
        operation["response"] = response;
    }
    operation
}

fn updated_batch_resource(id: &str, input_config: Value, priority: Option<&str>) -> Value {
    let mut batch = json!({
        "name": format!("batches/{id}"),
        "model": "models/gemini-3.8-flash",
        "displayName": "updated-batch",
        "inputConfig": input_config
    });
    if let Some(priority) = priority {
        batch["priority"] = json!(priority);
    }
    batch
}

fn update_request(input: GeminiBatchInput) -> GeminiBatchUpdateRequest {
    GeminiBatchUpdateRequest::new("gemini-3.8-flash", "updated-batch", input).unwrap()
}

fn service<'a>(http: &'a MockTransport, account: &str) -> GeminiBatchService<'a> {
    GeminiBatchService::new(
        http,
        GeminiBatchScope::new(
            "google-production",
            account,
            "https://generativelanguage.googleapis.com/v1beta/",
        )
        .unwrap(),
    )
    .unwrap()
}

fn credential() -> Secret<String> {
    Secret::new("google-test-key".to_owned())
}

fn create_request() -> GeminiBatchCreateRequest {
    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "hello"}]
    })])
    .unwrap()
    .with_parameter("generationConfig", json!({"temperature": 0.2}))
    .unwrap();
    let row = GeminiBatchRequest::new(request).with_key("row-1").unwrap();
    GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "translation-eval",
        GeminiBatchInput::new(vec![row]).unwrap(),
    )
    .unwrap()
}

fn embedding_operation(id: &str, state: &str, response: Option<Value>) -> Value {
    let mut operation = json!({
        "name": format!("batches/{id}"),
        "done": state.ends_with("SUCCEEDED"),
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.EmbedContentBatch",
            "model": "models/gemini-embedding-2",
            "state": state
        }
    });
    if let Some(response) = response {
        operation["response"] = response;
    }
    operation
}

fn embedding_create_request(count: usize) -> GeminiEmbeddingBatchCreateRequest {
    let dimension = 768;
    let config = GeminiBatchEmbeddingConfig::new()
        .with_output_dimensionality(dimension)
        .unwrap();
    let requests = (0..count)
        .map(|index| {
            GeminiBatchEmbedContentItem::new(
                GeminiBatchEmbedContentRequest::text(format!("document {index}"))
                    .unwrap()
                    .with_config(config.clone()),
            )
            .with_key(format!("doc-{index}"))
            .unwrap()
        })
        .collect();
    GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "embedding-corpus",
        GeminiEmbeddingBatchInput::new(requests).unwrap(),
    )
    .unwrap()
}

fn body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn typed_inline_input_enforces_required_contents_and_nonempty_input() {
    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "hello"}]
    })])
    .unwrap()
    .with_parameter("generationConfig", json!({"temperature": 0.2}))
    .unwrap();
    assert_eq!(request.as_value()["contents"].as_array().unwrap().len(), 1);
    assert_eq!(request.as_value()["generationConfig"]["temperature"], 0.2);
    assert!(GeminiBatchGenerateContentRequest::from_value(json!({})).is_err());
    assert!(GeminiBatchGenerateContentRequest::new(Vec::new()).is_err());
    assert!(GeminiBatchInput::new(Vec::new()).is_err());
    assert!(GeminiBatchRequest::new(request.clone())
        .with_metadata(json!("not an object"))
        .is_err());
    assert!(GeminiBatchRequest::new(request).with_key("  ").is_err());
    assert!(GeminiBatchScope::new("p", "a", "http://example.test/v1beta").is_err());
}

#[tokio::test]
async fn jsonl_file_input_uploads_once_and_creates_with_file_name() {
    let mock = MockTransport::new([
        raw_reply(
            200,
            b"{}".to_vec(),
            vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session/token".into(),
            )],
        ),
        reply(200, json!({"file": {"name": "files/input-123"}})),
        reply(200, operation("file-input", "BATCH_STATE_PENDING", None)),
    ]);
    let service = service(&mock, "account-a");
    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "summarize"}]
    })])
    .unwrap()
    .with_parameter("generationConfig", json!({"temperature": 0.2}))
    .unwrap();
    let input =
        GeminiBatchJsonlInput::new(vec![
            GeminiBatchJsonlRequest::new("summary-1", request).unwrap()
        ])
        .unwrap();

    let file = service
        .upload_input_jsonl("summaries.jsonl", input, &credential())
        .await
        .unwrap();
    assert_eq!(file.resource_name(), "files/input-123");
    let stream = mock.stream_requests();
    assert_eq!(stream.len(), 1);
    assert_eq!(stream[0].0, "POST");
    assert_eq!(
        stream[0].1,
        "https://generativelanguage.googleapis.com/upload/session/token"
    );
    assert_eq!(stream[0].3 as usize, stream[0].4.len());
    assert!(!stream[0]
        .2
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key")));
    let uploaded_line: Value =
        serde_json::from_slice(&stream[0].4[..stream[0].4.len() - 1]).unwrap();
    assert_eq!(uploaded_line["key"], "summary-1");
    assert_eq!(
        uploaded_line["request"]["generationConfig"]["temperature"],
        0.2
    );

    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "summaries",
        GeminiBatchInput::from_file(file),
    )
    .unwrap();
    service.create(&create, &credential()).await.unwrap();
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        body(&requests[1])["batch"]["input_config"]["file_name"],
        "files/input-123"
    );
    assert!(body(&requests[1])["batch"]["input_config"]["requests"].is_null());
}

#[tokio::test]
async fn uncertain_jsonl_finalize_is_reported_and_never_replayed() {
    let mock = MockTransport::new([
        raw_reply(
            200,
            b"{}".to_vec(),
            vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session/token".into(),
            )],
        ),
        Err(LlmError::TransportTimeout {
            message: "upload response timed out after finalize".into(),
        }),
    ]);
    let body =
        futures::stream::iter(vec![Ok::<Bytes, LlmError>(Bytes::from_static(b"{}\n"))]).boxed();
    let error = service(&mock, "account-a")
        .upload_input_stream("input.jsonl", 3, body, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GeminiBatchError::OutcomeUnknown {
            operation: "upload_input",
            reference: None,
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);
    assert_eq!(mock.stream_requests().len(), 1);
}

#[tokio::test]
async fn create_uses_documented_route_header_and_decodes_operation() {
    let mock = MockTransport::new([reply(
        200,
        operation("batch-123", "BATCH_STATE_PENDING", None),
    )]);
    let service = service(&mock, "account-a");
    let snapshot = service
        .create(&create_request(), &credential())
        .await
        .unwrap();

    assert_eq!(snapshot.reference.resource_name(), "batches/batch-123");
    assert_eq!(snapshot.state, Some(GeminiBatchState::Pending));
    assert_eq!(snapshot.done, Some(true));
    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:batchGenerateContent"
    );
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key"))
            .map(|(_, value)| value.as_str()),
        Some("google-test-key")
    );
    let request_body = body(&requests[0]);
    assert_eq!(
        request_body["batch"]["input_config"]["requests"]["requests"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(request_body["batch"]["display_name"], "translation-eval");
    assert_eq!(
        request_body["batch"]["input_config"]["requests"]["requests"][0]["metadata"]["key"],
        "row-1"
    );
    assert_eq!(
        request_body["batch"]["input_config"]["requests"]["requests"][0]["request"]
            ["generationConfig"]["temperature"],
        0.2
    );
}

#[tokio::test]
async fn update_generate_content_batch_uses_documented_patch_resource_and_mask() {
    let update_input = create_request().input().clone();
    let input_config = json!({
        "requests": {"requests": [{
            "request": {
                "contents": [{"role":"user", "parts":[{"text":"hello"}]}],
                "generationConfig": {"temperature": 0.2}
            },
            "metadata": {"key":"row-1"}
        }]}
    });
    let mock = MockTransport::new([
        reply(200, operation("batch-update", "BATCH_STATE_PENDING", None)),
        reply(
            200,
            updated_batch_resource("batch-update", input_config.clone(), Some("-17")),
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    let update = update_request(update_input)
        .with_priority(-17)
        .with_update_mask([
            GeminiBatchUpdateField::DisplayName,
            GeminiBatchUpdateField::Priority,
        ])
        .unwrap();
    let result = batch_service
        .update_generate_content_batch(&reference, &update, &credential())
        .await
        .unwrap();

    assert_eq!(result.reference, reference);
    assert_eq!(result.model, "models/gemini-3.8-flash");
    assert_eq!(result.display_name, "updated-batch");
    assert_eq!(result.input_config, input_config);
    assert_eq!(result.priority, Some(-17));
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    let request = &requests[1];
    assert_eq!(request.method, "PATCH");
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/batches/batch-update:updateGenerateContentBatch?updateMask=displayName%2Cpriority"
    );
    assert!(request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type") && value == "application/json"
    }));
    let encoded = body(request);
    assert_eq!(encoded["model"], "models/gemini-3.8-flash");
    assert_eq!(encoded["displayName"], "updated-batch");
    assert_eq!(encoded["inputConfig"], input_config);
    assert_eq!(encoded["priority"], "-17");
    assert!(encoded.get("name").is_none());
    assert!(encoded.get("display_name").is_none());
    assert!(encoded.get("input_config").is_none());
}

#[tokio::test]
async fn update_generate_content_batch_encodes_file_resource_fields_in_camel_case() {
    let mock = MockTransport::new([
        reply(
            200,
            operation("batch-update-file", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            updated_batch_resource(
                "batch-update-file",
                json!({"fileName":"files/input-123"}),
                None,
            ),
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    let file =
        GeminiBatchFileRef::from_resource_name(batch_service.scope(), "files/input-123").unwrap();
    let update = update_request(GeminiBatchInput::from_file(file));
    let result = batch_service
        .update_generate_content_batch(&reference, &update, &credential())
        .await
        .unwrap();
    assert_eq!(result.input_config, json!({"fileName":"files/input-123"}));
    assert_eq!(
        body(&mock.requests()[1])["inputConfig"]["fileName"],
        "files/input-123"
    );
}

#[tokio::test]
async fn update_preflights_scope_and_reports_unknown_mutation_outcomes() {
    let setup = MockTransport::new([reply(
        200,
        operation("batch-update-scope", "BATCH_STATE_PENDING", None),
    )]);
    let reference = service(&setup, "account-a")
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    let update = update_request(create_request().input().clone());

    let wrong_scope = MockTransport::new([]);
    assert!(matches!(
        service(&wrong_scope, "account-b")
            .update_generate_content_batch(&reference, &update, &credential())
            .await,
        Err(GeminiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(wrong_scope.requests().is_empty());

    let other_scope = GeminiBatchScope::new(
        "google-production",
        "account-b",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    .unwrap();
    let foreign_file =
        GeminiBatchFileRef::from_resource_name(&other_scope, "files/input-123").unwrap();
    let foreign_input = update_request(GeminiBatchInput::from_file(foreign_file));
    let local_mock = MockTransport::new([]);
    assert!(matches!(
        service(&local_mock, "account-a")
            .update_generate_content_batch(&reference, &foreign_input, &credential())
            .await,
        Err(GeminiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(local_mock.requests().is_empty());

    let transport_failure = MockTransport::new([
        reply(
            200,
            operation("batch-update-uncertain", "BATCH_STATE_PENDING", None),
        ),
        Err(LlmError::Transport {
            message: "connection closed after dispatch".into(),
        }),
    ]);
    let update_service = service(&transport_failure, "account-a");
    let reference = update_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    assert!(matches!(
        update_service
            .update_generate_content_batch(&reference, &update, &credential())
            .await,
        Err(GeminiBatchError::OutcomeUnknown {
            operation: "update",
            reference: Some(error_reference),
            ..
        }) if *error_reference == reference
    ));
    assert_eq!(transport_failure.requests().len(), 2);

    let malformed_success = MockTransport::new([
        reply(
            200,
            operation("batch-update-malformed", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            updated_batch_resource(
                "different-batch",
                json!({"fileName":"files/input-123"}),
                None,
            ),
        ),
    ]);
    let update_service = service(&malformed_success, "account-a");
    let reference = update_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    assert!(matches!(
        update_service
            .update_generate_content_batch(&reference, &update, &credential())
            .await,
        Err(GeminiBatchError::OutcomeUnknownResponse {
            operation: "update",
            reference: Some(error_reference),
            reason,
        }) if *error_reference == reference && reason.contains("different batch resource name")
    ));

    let rejected = MockTransport::new([
        reply(
            200,
            operation("batch-update-rejected", "BATCH_STATE_PENDING", None),
        ),
        reply(400, json!({"error":{"message":"invalid update"}})),
    ]);
    let update_service = service(&rejected, "account-a");
    let reference = update_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    assert!(matches!(
        update_service
            .update_generate_content_batch(&reference, &update, &credential())
            .await,
        Err(GeminiBatchError::Provider { status: 400, .. })
    ));
}

#[test]
fn update_request_validates_masks_and_preserves_signed_int64_priority() {
    let base = || update_request(create_request().input().clone());
    assert!(base()
        .with_update_mask(std::iter::empty::<GeminiBatchUpdateField>())
        .is_err());
    assert!(base()
        .with_update_mask([
            GeminiBatchUpdateField::DisplayName,
            GeminiBatchUpdateField::DisplayName,
        ])
        .is_err());
    assert!(base()
        .with_update_mask([GeminiBatchUpdateField::Priority])
        .is_err());
    assert!(base()
        .with_priority(i64::MIN)
        .with_update_mask([GeminiBatchUpdateField::DisplayName])
        .is_err());
    assert_eq!(base().with_priority(i64::MIN).priority(), Some(i64::MIN));
}

#[tokio::test]
async fn update_revalidates_final_mask_and_inline_size_before_http() {
    let mock = MockTransport::new([reply(
        200,
        operation("batch-update-preflight", "BATCH_STATE_PENDING", None),
    )]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;

    let priority_after_mask = update_request(create_request().input().clone())
        .with_update_mask([GeminiBatchUpdateField::DisplayName])
        .unwrap()
        .with_priority(-3);
    assert!(matches!(
        batch_service
            .update_generate_content_batch(&reference, &priority_after_mask, &credential())
            .await,
        Err(GeminiBatchError::InvalidInput(message)) if message.contains("must include priority")
    ));
    assert_eq!(mock.requests().len(), 1);

    let large_request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "x".repeat(20_000_000)}]
    })])
    .unwrap();
    let large_input = GeminiBatchInput::new(vec![GeminiBatchRequest::new(large_request)]).unwrap();
    let large_update = update_request(large_input);
    assert!(matches!(
        batch_service
            .update_generate_content_batch(&reference, &large_update, &credential())
            .await,
        Err(GeminiBatchError::InvalidInput(message)) if message.contains("smaller than 20 MB")
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn create_transport_uncertainty_is_explicit_and_never_retried() {
    let mock = MockTransport::new([Err(LlmError::Transport {
        message: "connection closed after dispatch".into(),
    })]);
    let error = service(&mock, "account-a")
        .create(&create_request(), &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GeminiBatchError::OutcomeUnknown {
            operation: "create",
            reference: None,
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn list_encodes_page_token_and_scopes_returned_operations() {
    let mock = MockTransport::new([reply(
        200,
        json!({
            "operations": [
                operation("batch-a", "BATCH_STATE_RUNNING", None),
                operation("batch-b", "BATCH_STATE_SUCCEEDED", None)
            ],
            "nextPageToken": "next/page"
        }),
    )]);
    let service = service(&mock, "account-a");
    let options = GeminiBatchListOptions::new()
        .with_page_size(5)
        .unwrap()
        .with_page_token("page/one")
        .unwrap();
    let refreshed_credential = Secret::new("google-refreshed-key".to_owned());
    let page = service.list(&options, &refreshed_credential).await.unwrap();

    assert_eq!(page.batches.len(), 2);
    assert_eq!(
        page.batches[0].reference.scope().account_scope(),
        "account-a"
    );
    assert_eq!(page.next_page_token.as_deref(), Some("next/page"));
    let request = &mock.requests()[0];
    assert_eq!(request.method, "GET");
    assert!(request
        .url
        .starts_with("https://generativelanguage.googleapis.com/v1beta/batches?pageSize=5&"));
    assert!(request.url.contains("pageToken=page%2Fone"));
    assert_eq!(
        request
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key"))
            .map(|(_, value)| value.as_str()),
        Some("google-refreshed-key")
    );
    assert!(GeminiBatchListOptions::new().with_page_size(0).is_err());
}

#[tokio::test]
async fn reference_scope_is_checked_before_network_access() {
    let create_mock = MockTransport::new([reply(
        200,
        operation("batch-scope", "BATCH_STATE_PENDING", None),
    )]);
    let reference = service(&create_mock, "account-a")
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;

    let other_mock = MockTransport::new([]);
    let error = service(&other_mock, "account-b")
        .get(&reference, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GeminiBatchError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(other_mock.requests().is_empty());

    let file =
        GeminiBatchFileRef::from_resource_name(reference.scope(), "files/input-123").unwrap();
    let file_create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "foreign-file",
        GeminiBatchInput::from_file(file),
    )
    .unwrap();
    assert!(matches!(
        service(&other_mock, "account-b")
            .create(&file_create, &credential())
            .await
            .unwrap_err(),
        GeminiBatchError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(other_mock.requests().is_empty());
}

#[tokio::test]
async fn cancel_is_best_effort_empty_post_and_transport_failure_is_unknown() {
    let mock = MockTransport::new([
        reply(200, operation("batch-cancel", "BATCH_STATE_RUNNING", None)),
        reply(200, json!({})),
    ]);
    let current_service = service(&mock, "account-a");
    let reference = current_service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    current_service
        .cancel(&reference, &credential())
        .await
        .unwrap();
    let requests = mock.requests();
    assert_eq!(requests[1].method, "POST");
    assert!(requests[1].url.ends_with("/batches/batch-cancel:cancel"));
    assert!(requests[1].body.is_empty());

    let malformed = MockTransport::new([reply(200, json!({"accepted": true}))]);
    let error = service(&malformed, "account-a")
        .cancel(&reference, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GeminiBatchError::OutcomeUnknownResponse {
            operation: "cancel",
            reference: Some(_),
            ..
        }
    ));

    let failing = MockTransport::new([Err(LlmError::TransportTimeout {
        message: "cancel response timed out".into(),
    })]);
    let error = service(&failing, "account-a")
        .cancel(&reference, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GeminiBatchError::OutcomeUnknown {
            operation: "cancel",
            reference: Some(_),
            ..
        }
    ));
    assert_eq!(failing.requests().len(), 1);
}

#[tokio::test]
async fn delete_uses_documented_route_and_does_not_cancel_the_batch() {
    let mock = MockTransport::new([
        reply(200, operation("batch-delete", "BATCH_STATE_RUNNING", None)),
        reply(200, json!({})),
    ]);
    let service = service(&mock, "account-a");
    let reference = service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;

    service.delete(&reference, &credential()).await.unwrap();
    let requests = mock.requests();
    assert_eq!(requests[1].method, "DELETE");
    assert!(requests[1].url.ends_with("/batches/batch-delete"));
    assert!(requests[1].body.is_empty());
    assert!(requests[1]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("x-goog-api-key")
            && value == "google-test-key"));
}

#[tokio::test]
async fn delete_preflights_scope_and_marks_unreadable_success_or_transport_unknown() {
    let source = MockTransport::new([reply(
        200,
        operation("batch-delete", "BATCH_STATE_RUNNING", None),
    )]);
    let reference = service(&source, "account-a")
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;

    let foreign = MockTransport::new([]);
    assert!(matches!(
        service(&foreign, "account-b")
            .delete(&reference, &credential())
            .await,
        Err(GeminiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(foreign.requests().is_empty());

    let malformed = MockTransport::new([reply(200, json!({"stillRunning": true}))]);
    assert!(matches!(
        service(&malformed, "account-a")
            .delete(&reference, &credential())
            .await,
        Err(GeminiBatchError::OutcomeUnknownResponse {
            operation: "delete",
            reference: Some(_),
            ..
        })
    ));
    assert_eq!(malformed.requests().len(), 1);

    let transport_error = MockTransport::new([Err(LlmError::TransportTimeout {
        message: "delete response timed out".into(),
    })]);
    assert!(matches!(
        service(&transport_error, "account-a")
            .delete(&reference, &credential())
            .await,
        Err(GeminiBatchError::OutcomeUnknown {
            operation: "delete",
            reference: Some(_),
            ..
        })
    ));
    assert_eq!(transport_error.requests().len(), 1);
}

#[tokio::test]
async fn results_decode_each_inline_response_or_item_error() {
    let final_response = json!({
        "output": {
            "inlinedResponses": {
                "inlinedResponses": [
                    {
                        "metadata": {"key": "row-1"},
                        "response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
                    },
                    {
                        "metadata": {"key": "row-2"},
                        "error": {"code": 400, "message": "invalid request"}
                    }
                ]
            }
        },
        "state": "BATCH_STATE_SUCCEEDED"
    });
    let mock = MockTransport::new([
        reply(
            200,
            operation("batch-results", "BATCH_STATE_SUCCEEDED", None),
        ),
        reply(
            200,
            operation(
                "batch-results",
                "BATCH_STATE_SUCCEEDED",
                Some(final_response),
            ),
        ),
    ]);
    let service = service(&mock, "account-a");
    let reference = service
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    let results = service.results(&reference, &credential()).await.unwrap();
    let items = results.items.unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].metadata.as_ref().unwrap()["key"], "row-1");
    assert_eq!(
        items[0].response.as_ref().unwrap()["candidates"][0]["content"]["parts"][0]["text"],
        "ok"
    );
    assert_eq!(items[0].error, None);
    assert_eq!(items[1].error.as_ref().unwrap()["code"], 400);
    assert_eq!(items[1].response, None);
    assert_eq!(mock.requests().len(), 2);
    assert_eq!(mock.requests()[1].method, "GET");
}

#[tokio::test]
async fn file_backed_result_name_is_scoped_and_downloaded_as_a_stream() {
    let output = json!({
        "output": {"responsesFile": "files/output-789"},
        "state": "BATCH_STATE_SUCCEEDED"
    });
    let jsonl = b"{\"response\":{\"candidates\":[]}}\n".to_vec();
    let mock = MockTransport::new([
        reply(
            200,
            operation("batch-file-output", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            operation("batch-file-output", "BATCH_STATE_SUCCEEDED", Some(output)),
        ),
        raw_reply(
            200,
            jsonl.clone(),
            vec![("content-type".into(), "application/jsonl".into())],
        ),
    ]);
    let service = service(&mock, "account-a");
    let input_file =
        GeminiBatchFileRef::from_resource_name(service.scope(), "files/input-456").unwrap();
    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "file-output",
        GeminiBatchInput::from_file(input_file),
    )
    .unwrap();
    let created = service.create(&create, &credential()).await.unwrap();
    let results = service
        .results(&created.reference, &credential())
        .await
        .unwrap();
    let output_file = results.snapshot.output_file.unwrap();
    assert_eq!(output_file.resource_name(), "files/output-789");

    let downloaded = service
        .download_results(&output_file, &credential())
        .await
        .unwrap()
        .fold(Vec::new(), |mut output, chunk| async move {
            output.extend_from_slice(&chunk.unwrap());
            output
        })
        .await;
    assert_eq!(downloaded, jsonl);
    let request = &mock.requests()[2];
    assert_eq!(request.method, "GET");
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/files/output-789:download?alt=media"
    );
}

#[tokio::test]
async fn malformed_successful_create_is_unknown_and_http_errors_are_provider_errors() {
    let malformed = MockTransport::new([reply(200, json!({"unexpected": true}))]);
    assert!(matches!(
        service(&malformed, "account-a")
            .create(&create_request(), &credential())
            .await
            .unwrap_err(),
        GeminiBatchError::OutcomeUnknownResponse {
            operation: "create",
            ..
        }
    ));

    let rejected = MockTransport::new([reply(429, json!({"error": {"message": "quota"}}))]);
    assert!(matches!(
        service(&rejected, "account-a")
            .create(&create_request(), &credential())
            .await
            .unwrap_err(),
        GeminiBatchError::Provider { status: 429, .. }
    ));
}

#[tokio::test]
async fn embedding_create_uses_async_route_current_config_and_typed_reference() {
    let mock = MockTransport::new([reply(
        200,
        embedding_operation("embed-create", "BATCH_STATE_PENDING", None),
    )]);
    let service = service(&mock, "account-a");
    let created = service
        .create_embedding_batch(&embedding_create_request(2), &credential())
        .await
        .unwrap();

    assert_eq!(created.reference.model(), Some("gemini-embedding-2"));
    assert_eq!(
        created.reference.expected_output_dimensions(),
        Some([Some(768), Some(768)].as_slice())
    );
    let request = &mock.requests()[0];
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:asyncBatchEmbedContent"
    );
    assert!(request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-api-key") && value == "google-test-key"
    }));
    let encoded = body(request);
    assert_eq!(encoded["batch"]["display_name"], "embedding-corpus");
    let rows = encoded["batch"]["input_config"]["requests"]["requests"]
        .as_array()
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["metadata"]["key"], "doc-0");
    assert_eq!(rows[0]["request"]["model"], "models/gemini-embedding-2");
    assert_eq!(
        rows[0]["request"]["content"]["parts"][0]["text"],
        "document 0"
    );
    assert_eq!(
        rows[0]["request"]["embedContentConfig"],
        json!({
            "autoTruncate": false,
            "outputDimensionality": 768
        })
    );
    assert!(rows[0]["request"].get("taskType").is_none());
}

#[tokio::test]
async fn embedding_batch_encodes_typed_multimodal_content_and_preflights_unsupported_media() {
    let mock = MockTransport::new([reply(
        200,
        embedding_operation("embed-media", "BATCH_STATE_PENDING", None),
    )]);
    let batch_service = service(&mock, "account-a");
    let content = GeminiBatchEmbeddingContent::new(vec![
        GeminiBatchEmbeddingPart::Text("Describe this image and audio.".into()),
        GeminiBatchEmbeddingPart::Media(GeminiEmbeddingMedia {
            mime_type: "image/png".into(),
            source: GeminiEmbeddingSource::Inline(vec![137, 80, 78, 71]),
            duration_seconds: None,
            page_count: None,
        }),
        GeminiBatchEmbeddingPart::Media(GeminiEmbeddingMedia {
            mime_type: "audio/mpeg".into(),
            source: GeminiEmbeddingSource::FileUri(
                "https://generativelanguage.googleapis.com/v1beta/files/audio-1".into(),
            ),
            duration_seconds: Some(120),
            page_count: None,
        }),
    ])
    .unwrap();
    let request = GeminiBatchEmbedContentRequest::new(content).with_config(
        GeminiBatchEmbeddingConfig::new()
            .with_output_dimensionality(768)
            .unwrap(),
    );
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "multimodal-embeddings",
        GeminiEmbeddingBatchInput::new(vec![GeminiBatchEmbedContentItem::new(request)]).unwrap(),
    )
    .unwrap();
    batch_service
        .create_embedding_batch(&create, &credential())
        .await
        .unwrap();
    let encoded = body(&mock.requests()[0]);
    let parts = encoded["batch"]["input_config"]["requests"]["requests"][0]["request"]["content"]
        ["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts[0]["text"], "Describe this image and audio.");
    assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
    assert_eq!(parts[1]["inlineData"]["data"], "iVBORw==");
    assert_eq!(parts[2]["fileData"]["mimeType"], "audio/mpeg");
    assert_eq!(
        parts[2]["fileData"]["fileUri"],
        "https://generativelanguage.googleapis.com/v1beta/files/audio-1"
    );

    let invalid = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "unsupported-media",
        GeminiEmbeddingBatchInput::new(vec![GeminiBatchEmbedContentItem::new(
            GeminiBatchEmbedContentRequest::new(
                GeminiBatchEmbeddingContent::new(vec![GeminiBatchEmbeddingPart::Media(
                    GeminiEmbeddingMedia {
                        mime_type: "image/gif".into(),
                        source: GeminiEmbeddingSource::Inline(vec![1, 2, 3]),
                        duration_seconds: None,
                        page_count: None,
                    },
                )])
                .unwrap(),
            ),
        )])
        .unwrap(),
    )
    .unwrap();
    let rejected = MockTransport::new([]);
    assert!(matches!(
        service(&rejected, "account-a")
            .create_embedding_batch(&invalid, &credential())
            .await,
        Err(GeminiBatchError::InvalidInput(message)) if message.contains("supports PNG/JPEG")
    ));
    assert!(rejected.requests().is_empty());
}

#[test]
fn embedding_jsonl_encoder_builds_unique_keyed_native_embed_requests() {
    let config_768 = GeminiBatchEmbeddingConfig::new()
        .with_task_type(GeminiBatchEmbeddingTaskType::RetrievalDocument)
        .with_title("alpha collection")
        .unwrap()
        .with_output_dimensionality(768)
        .unwrap();
    let config_512 = GeminiBatchEmbeddingConfig::new()
        .with_task_type(GeminiBatchEmbeddingTaskType::RetrievalQuery)
        .with_output_dimensionality(512)
        .unwrap();
    let rows = vec![
        GeminiEmbeddingBatchJsonlRequest::new(
            "doc-a",
            GeminiBatchEmbedContentRequest::text("alpha")
                .unwrap()
                .with_config(config_768),
        )
        .unwrap(),
        GeminiEmbeddingBatchJsonlRequest::new(
            "doc-b",
            GeminiBatchEmbedContentRequest::text("beta")
                .unwrap()
                .with_config(config_512),
        )
        .unwrap(),
    ];
    let input = GeminiEmbeddingBatchJsonlInput::new("gemini-embedding-001", rows).unwrap();
    let encoded = String::from_utf8(input.to_bytes().unwrap().to_vec()).unwrap();
    let lines = encoded
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["key"], "doc-a");
    assert_eq!(lines[0]["request"]["model"], "models/gemini-embedding-001");
    assert_eq!(lines[0]["request"]["content"]["parts"][0]["text"], "alpha");
    assert_eq!(
        lines[0]["request"]["embedContentConfig"]["outputDimensionality"],
        768
    );
    assert_eq!(
        lines[0]["request"]["embedContentConfig"]["taskType"],
        "RETRIEVAL_DOCUMENT"
    );
    assert_eq!(
        lines[0]["request"]["embedContentConfig"]["title"],
        "alpha collection"
    );
    assert_eq!(lines[1]["key"], "doc-b");
    assert_eq!(
        lines[1]["request"]["embedContentConfig"]["outputDimensionality"],
        512
    );
    assert_eq!(
        lines[1]["request"]["embedContentConfig"]["taskType"],
        "RETRIEVAL_QUERY"
    );
    assert!(lines[0].get("metadata").is_none());

    let duplicate = GeminiEmbeddingBatchJsonlInput::new(
        "gemini-embedding-2",
        vec![
            GeminiEmbeddingBatchJsonlRequest::new(
                "same",
                GeminiBatchEmbedContentRequest::text("one").unwrap(),
            )
            .unwrap(),
            GeminiEmbeddingBatchJsonlRequest::new(
                "same",
                GeminiBatchEmbedContentRequest::text("two").unwrap(),
            )
            .unwrap(),
        ],
    );
    assert!(
        matches!(duplicate, Err(GeminiBatchError::InvalidInput(message)) if message.contains("unique"))
    );
}

fn keyed_embedding_jsonl_input() -> GeminiEmbeddingBatchJsonlInput {
    GeminiEmbeddingBatchJsonlInput::new(
        "gemini-embedding-2",
        vec![
            GeminiEmbeddingBatchJsonlRequest::new(
                "doc-a",
                GeminiBatchEmbedContentRequest::text("alpha")
                    .unwrap()
                    .with_config(
                        GeminiBatchEmbeddingConfig::new()
                            .with_output_dimensionality(768)
                            .unwrap(),
                    ),
            )
            .unwrap(),
            GeminiEmbeddingBatchJsonlRequest::new(
                "doc-b",
                GeminiBatchEmbedContentRequest::text("beta")
                    .unwrap()
                    .with_config(
                        GeminiBatchEmbeddingConfig::new()
                            .with_output_dimensionality(512)
                            .unwrap(),
                    ),
            )
            .unwrap(),
            GeminiEmbeddingBatchJsonlRequest::new(
                "doc-c",
                GeminiBatchEmbedContentRequest::text("gamma")
                    .unwrap()
                    .with_config(
                        GeminiBatchEmbeddingConfig::new()
                            .with_output_dimensionality(256)
                            .unwrap(),
                    ),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

async fn read_keyed_embedding_file_results(
    rows: Vec<Value>,
) -> (
    Vec<Result<GeminiEmbeddingBatchItemResult, GeminiBatchError>>,
    Vec<StreamRequestSnapshot>,
) {
    let mut output = rows
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .join("\n")
        .into_bytes();
    if !output.is_empty() {
        output.push(b'\n');
    }
    let keyed_rows = keyed_embedding_jsonl_input();
    let mock = MockTransport::new([
        raw_reply(
            200,
            b"{}".to_vec(),
            vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session/embed-jsonl".into(),
            )],
        ),
        reply(200, json!({"file": {"name": "files/embed-input"}})),
        reply(
            200,
            embedding_operation("embed-jsonl", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            embedding_operation(
                "embed-jsonl",
                "BATCH_STATE_SUCCEEDED",
                Some(json!({"output": {"responsesFile": "files/embed-output"}})),
            ),
        ),
        raw_reply(
            200,
            output,
            vec![("content-type".into(), "application/jsonl".into())],
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let input_file = batch_service
        .upload_embedding_input_jsonl("embeddings.jsonl", keyed_rows, &credential())
        .await
        .unwrap();
    assert_eq!(input_file.model(), Some("gemini-embedding-2"));
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "embedding-file",
        GeminiEmbeddingBatchInput::from_file(input_file),
    )
    .unwrap();
    let created = batch_service
        .create_embedding_batch(&create, &credential())
        .await
        .unwrap();
    let current = batch_service
        .get_embedding_batch(&created.reference, &credential())
        .await
        .unwrap();
    let output_file = current.output_file.unwrap();
    let results = batch_service
        .download_embedding_results(&output_file, &credential())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let uploaded = mock.stream_requests();
    (results, uploaded)
}

#[tokio::test]
async fn embedding_jsonl_file_outputs_follow_input_order_and_correlate_keyless_rows() {
    let (results, uploaded) = read_keyed_embedding_file_results(vec![
        json!({"key":"doc-a", "embedding":{"values":vec![0.25; 768]}}),
        json!({"embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.75; 256]}}),
    ])
    .await;

    assert_eq!(results.len(), 3);
    for (index, key) in ["doc-a", "doc-b", "doc-c"].into_iter().enumerate() {
        assert_eq!(
            results[index].as_ref().unwrap().metadata.as_ref().unwrap()["key"],
            key
        );
    }
    for (index, dimension) in [768, 512, 256].into_iter().enumerate() {
        assert_eq!(
            results[index]
                .as_ref()
                .unwrap()
                .response
                .as_ref()
                .unwrap()
                .embedding
                .values
                .as_ref()
                .unwrap()
                .len(),
            dimension
        );
    }
    assert_eq!(uploaded.len(), 1);
    let uploaded_lines = String::from_utf8(uploaded[0].4.clone()).unwrap();
    assert!(uploaded_lines.contains("\"key\":\"doc-a\""));
    assert!(uploaded_lines.contains("\"model\":\"models/gemini-embedding-2\""));
}

#[tokio::test]
async fn embedding_jsonl_file_results_reject_conflicting_keys_wrong_order_width_and_count() {
    let (reordered, _) = read_keyed_embedding_file_results(vec![
        json!({"key":"doc-b", "embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.5; 256]}}),
    ])
    .await;
    assert!(matches!(
        &reordered[0],
        Err(GeminiBatchError::InvalidResponse(message))
            if message.contains("documented input order")
    ));

    let (conflicting, _) = read_keyed_embedding_file_results(vec![
        json!({
            "key":"doc-a",
            "metadata":{"key":"doc-b"},
            "embedding":{"values":vec![0.25; 768]}
        }),
        json!({"embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.5; 256]}}),
    ])
    .await;
    assert!(matches!(
        &conflicting[0],
        Err(GeminiBatchError::InvalidResponse(message))
            if message.contains("keys conflict")
    ));

    let (wrong_width, _) = read_keyed_embedding_file_results(vec![
        json!({"embedding":{"values":vec![0.25; 768]}}),
        json!({"embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.75; 123]}}),
    ])
    .await;
    assert!(matches!(
        &wrong_width[2],
        Err(GeminiBatchError::InvalidResponse(message))
            if message.contains("outputDimensionality")
    ));

    let (missing_row, _) = read_keyed_embedding_file_results(vec![
        json!({"embedding":{"values":vec![0.25; 768]}}),
        json!({"embedding":{"values":vec![0.5; 512]}}),
    ])
    .await;
    assert!(matches!(
        missing_row.last(),
        Some(Err(GeminiBatchError::InvalidResponse(message)))
            if message.contains("result count differs")
    ));

    let (extra_row, _) = read_keyed_embedding_file_results(vec![
        json!({"embedding":{"values":vec![0.25; 768]}}),
        json!({"embedding":{"values":vec![0.5; 512]}}),
        json!({"embedding":{"values":vec![0.75; 256]}}),
        json!({"embedding":{"values":vec![0.5; 256]}}),
    ])
    .await;
    assert!(matches!(
        extra_row.last(),
        Some(Err(GeminiBatchError::InvalidResponse(message)))
            if message.contains("result count exceeds")
    ));
}

#[tokio::test]
async fn embedding_results_keep_per_item_errors_and_accept_first_party_shape_only() {
    let output = json!({
        "output": {"inlinedResponses": {"inlinedResponses": [
            {
                "metadata": {"key": "doc-0"},
                "response": {
                    "embedding": {"values": vec![0.25; 768], "shape": [1, 768]},
                    "usageMetadata": {"promptTokenCount": 12}
                }
            },
            {
                "metadata": {"key": "doc-1"},
                "error": {"code": 400, "message": "invalid content", "details": []}
            }
        ]}}
    });
    let mock = MockTransport::new([
        reply(
            200,
            embedding_operation("embed-results", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            embedding_operation("embed-results", "BATCH_STATE_SUCCEEDED", Some(output)),
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create_embedding_batch(&embedding_create_request(2), &credential())
        .await
        .unwrap()
        .reference;
    let results = batch_service
        .embedding_batch_results(&reference, &credential())
        .await
        .unwrap();
    let items = results.items.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].metadata.as_ref().unwrap()["key"], "doc-0");
    let response = items[0].response.as_ref().unwrap();
    assert_eq!(response.embedding.values.as_ref().unwrap().len(), 768);
    assert_eq!(
        response.embedding.shape.as_deref(),
        Some([1, 768].as_slice())
    );
    assert_eq!(
        response.usage_metadata.as_ref().unwrap()["promptTokenCount"],
        12
    );
    assert_eq!(items[1].error.as_ref().unwrap().code, Some(400));
    assert_eq!(
        items[1].error.as_ref().unwrap().message.as_deref(),
        Some("invalid content")
    );

    let shape_only_mock = MockTransport::new([
        reply(
            200,
            embedding_operation("embed-shape-only", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            embedding_operation(
                "embed-shape-only",
                "BATCH_STATE_SUCCEEDED",
                Some(json!({
                    "output": {"inlinedResponses": {"inlinedResponses": [{
                        "response": {"embedding": {"shape": [1, 1, 1, 768]}}
                    }]}}
                })),
            ),
        ),
    ]);
    let shape_service = service(&shape_only_mock, "account-a");
    let shape_ref = shape_service
        .create_embedding_batch(&embedding_create_request(1), &credential())
        .await
        .unwrap()
        .reference;
    let shape_results = shape_service
        .embedding_batch_results(&shape_ref, &credential())
        .await
        .unwrap();
    let shape_embedding = shape_results
        .items
        .unwrap()
        .remove(0)
        .response
        .unwrap()
        .embedding;
    assert_eq!(shape_embedding.values, None);
    assert_eq!(
        shape_embedding.shape.as_deref(),
        Some([1, 1, 1, 768].as_slice())
    );
}

#[tokio::test]
async fn embedding_result_dimension_mismatch_is_invalid_and_operation_kind_is_enforced() {
    let wrong_dimension = json!({
        "output": {"inlinedResponses": {"inlinedResponses": [{
            "response": {"embedding": {"values": vec![0.5; 767]}}
        }]}}
    });
    let mock = MockTransport::new([
        reply(
            200,
            embedding_operation("embed-dimension", "BATCH_STATE_PENDING", None),
        ),
        reply(
            200,
            embedding_operation(
                "embed-dimension",
                "BATCH_STATE_SUCCEEDED",
                Some(wrong_dimension),
            ),
        ),
    ]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create_embedding_batch(&embedding_create_request(1), &credential())
        .await
        .unwrap()
        .reference;
    assert!(matches!(
        batch_service
            .embedding_batch_results(&reference, &credential())
            .await,
        Err(GeminiBatchError::InvalidResponse(message))
            if message.contains("outputDimensionality")
    ));

    let mut generation_only = operation("generation-only", "BATCH_STATE_RUNNING", None);
    generation_only["metadata"]["@type"] =
        json!("type.googleapis.com/google.ai.generativelanguage.v1beta.GenerateContentBatch");
    let mixed_kind = MockTransport::new([reply(200, generation_only)]);
    assert!(matches!(
        service(&mixed_kind, "account-a")
            .get_embedding_batch(&reference, &credential())
            .await,
        Err(GeminiBatchError::InvalidResponse(message))
            if message.contains("EmbedContentBatch")
    ));
}

#[tokio::test]
async fn embedding_file_route_and_scope_preflight_are_typed() {
    let mock = MockTransport::new([reply(
        200,
        embedding_operation("embed-file", "BATCH_STATE_PENDING", None),
    )]);
    let batch_service = service(&mock, "account-a");
    let file =
        GeminiEmbeddingBatchFileRef::from_resource_name(batch_service.scope(), "files/input-456")
            .unwrap();
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "file-embeddings",
        GeminiEmbeddingBatchInput::from_file(file),
    )
    .unwrap();
    batch_service
        .create_embedding_batch(&create, &credential())
        .await
        .unwrap();
    assert_eq!(
        body(&mock.requests()[0])["batch"]["input_config"]["file_name"],
        "files/input-456"
    );

    let other = MockTransport::new([]);
    let foreign_file =
        GeminiEmbeddingBatchFileRef::from_resource_name(batch_service.scope(), "files/input-789")
            .unwrap();
    let foreign_create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "foreign-file",
        GeminiEmbeddingBatchInput::from_file(foreign_file),
    )
    .unwrap();
    assert!(matches!(
        service(&other, "account-b")
            .create_embedding_batch(&foreign_create, &credential())
            .await,
        Err(GeminiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(other.requests().is_empty());
}

#[tokio::test]
async fn embedding_list_filters_generation_jobs_and_rejects_stalled_or_duplicate_pages() {
    let mut generation = operation("generate-a", "BATCH_STATE_RUNNING", None);
    generation["metadata"]["@type"] =
        json!("type.googleapis.com/google.ai.generativelanguage.v1beta.GenerateContentBatch");
    let mut model_less_embedding = embedding_operation("embed-a", "BATCH_STATE_RUNNING", None);
    model_less_embedding
        .as_object_mut()
        .unwrap()
        .get_mut("metadata")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("model");
    let mock = MockTransport::new([reply(
        200,
        json!({
            "operations": [
                model_less_embedding,
                generation
            ],
            "nextPageToken": "next"
        }),
    )]);
    let batch_service = service(&mock, "account-a");
    let page = batch_service
        .list_embedding_batches(&GeminiBatchListOptions::new(), &credential())
        .await
        .unwrap();
    assert_eq!(page.batches.len(), 1);
    assert_eq!(page.batches[0].reference.batch_id(), "embed-a");
    assert_eq!(page.batches[0].reference.model(), None);
    assert_eq!(page.batches[0].reference.operation(), "embed_content");
    assert_eq!(page.next_page_token.as_deref(), Some("next"));
    let embedding_ref_json = serde_json::to_value(&page.batches[0].reference).unwrap();
    assert!(serde_json::from_value::<GeminiBatchRef>(embedding_ref_json).is_err());

    let generated_mock = MockTransport::new([reply(
        200,
        operation("generation-ref", "BATCH_STATE_RUNNING", None),
    )]);
    let generated = service(&generated_mock, "account-a");
    let generic_ref = generated
        .create(&create_request(), &credential())
        .await
        .unwrap()
        .reference;
    let generation_ref_json = serde_json::to_value(generic_ref).unwrap();
    assert!(serde_json::from_value::<GeminiEmbeddingBatchRef>(generation_ref_json).is_err());

    let stalled = MockTransport::new([reply(
        200,
        json!({"operations": [], "nextPageToken": "repeat"}),
    )]);
    assert!(matches!(
        service(&stalled, "account-a")
            .list_embedding_batches(
                &GeminiBatchListOptions::new()
                    .with_page_token("repeat")
                    .unwrap(),
                &credential(),
            )
            .await,
        Err(GeminiBatchError::InvalidResponse(message)) if message.contains("did not advance")
    ));

    let duplicate = MockTransport::new([reply(
        200,
        json!({
            "operations": [
                embedding_operation("same", "BATCH_STATE_RUNNING", None),
                embedding_operation("same", "BATCH_STATE_RUNNING", None)
            ]
        }),
    )]);
    assert!(matches!(
        service(&duplicate, "account-a")
            .list_embedding_batches(&GeminiBatchListOptions::new(), &credential())
            .await,
        Err(GeminiBatchError::InvalidResponse(message)) if message.contains("duplicate batch IDs")
    ));
}

#[tokio::test]
async fn embedding_cancel_delete_and_preflight_errors_follow_mutation_semantics() {
    let mock = MockTransport::new([
        reply(
            200,
            embedding_operation("embed-mutate", "BATCH_STATE_RUNNING", None),
        ),
        reply(200, json!({})),
        reply(200, json!({})),
    ]);
    let batch_service = service(&mock, "account-a");
    let reference = batch_service
        .create_embedding_batch(&embedding_create_request(1), &credential())
        .await
        .unwrap()
        .reference;
    batch_service
        .cancel_embedding_batch(&reference, &credential())
        .await
        .unwrap();
    batch_service
        .delete_embedding_batch(&reference, &credential())
        .await
        .unwrap();
    assert!(mock.requests()[1]
        .url
        .ends_with("/batches/embed-mutate:cancel"));
    assert_eq!(mock.requests()[1].method, "POST");
    assert!(mock.requests()[1].body.is_empty());
    assert!(mock.requests()[2].url.ends_with("/batches/embed-mutate"));
    assert_eq!(mock.requests()[2].method, "DELETE");

    let foreign = MockTransport::new([]);
    assert!(matches!(
        service(&foreign, "account-b")
            .delete_embedding_batch(&reference, &credential())
            .await,
        Err(GeminiBatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(foreign.requests().is_empty());

    assert!(GeminiBatchEmbeddingContent::new(Vec::new()).is_err());
    assert!(GeminiBatchEmbeddingConfig::new()
        .with_output_dimensionality(0)
        .is_err());
    let incompatible = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "invalid-config",
        GeminiEmbeddingBatchInput::new(vec![GeminiBatchEmbedContentItem::new(
            GeminiBatchEmbedContentRequest::text("text")
                .unwrap()
                .with_config(
                    GeminiBatchEmbeddingConfig::new()
                        .with_task_type(GeminiBatchEmbeddingTaskType::RetrievalQuery),
                ),
        )])
        .unwrap(),
    )
    .unwrap();
    let invalid_config_transport = MockTransport::new([]);
    assert!(matches!(
        service(&invalid_config_transport, "account-a")
            .create_embedding_batch(&incompatible, &credential())
            .await,
        Err(GeminiBatchError::InvalidInput(message)) if message.contains("does not support taskType")
    ));
    assert!(invalid_config_transport.requests().is_empty());
    let malformed = MockTransport::new([reply(200, operation("missing-kind", "RUNNING", None))]);
    assert!(matches!(
        service(&malformed, "account-a")
            .create_embedding_batch(&embedding_create_request(1), &credential())
            .await,
        Err(GeminiBatchError::EmbeddingOutcomeUnknownResponse {
            operation: "create",
            ..
        })
    ));
}

#[tokio::test]
async fn embedding_result_stream_limits_each_line_without_limiting_chunk_or_file_size() {
    const LINE_LIMIT: usize = 64 * 1024 * 1024;
    let mut too_long_line = vec![b'x'; LINE_LIMIT + 1];
    too_long_line.push(b'\n');
    let oversized_mock = MockTransport::new([raw_reply(200, too_long_line, Vec::new())]);
    let oversized_service = service(&oversized_mock, "account-a");
    let file = GeminiEmbeddingBatchFileRef::from_resource_name(
        oversized_service.scope(),
        "files/oversized-output",
    )
    .unwrap();
    let mut stream = oversized_service
        .download_embedding_results(&file, &credential())
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(GeminiBatchError::InvalidResponse(message)) if message.contains("64 MiB limit")
    ));
    drop(stream);
    drop(oversized_mock);

    let row = b"{\"embedding\":{}}\n";
    let repeats = LINE_LIMIT / row.len() + 1;
    let mut many_small_rows = Vec::with_capacity(row.len() * repeats);
    for _ in 0..repeats {
        many_small_rows.extend_from_slice(row);
    }
    assert!(many_small_rows.len() > LINE_LIMIT);
    let small_rows_mock = MockTransport::new([raw_reply(200, many_small_rows, Vec::new())]);
    let small_rows_service = service(&small_rows_mock, "account-a");
    let file = GeminiEmbeddingBatchFileRef::from_resource_name(
        small_rows_service.scope(),
        "files/many-small-rows",
    )
    .unwrap();
    let mut stream = small_rows_service
        .download_embedding_results(&file, &credential())
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    assert!(stream.next().await.unwrap().is_ok());
}
