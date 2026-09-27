use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::{FilePurpose, FileService, UploadFile},
    protocol::{LlmError, ProviderProfile, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
    xai_batch::*,
    ApiKeyAuthenticator,
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
            body: futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        panic!("this test only covers buffered request and file uploads")
    }
}

fn reply(status: u16, body: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&body).unwrap(),
        headers: vec![("content-type".into(), "application/json".into())],
    })
}

fn raw_reply(status: u16, body: &'static [u8]) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: body.to_vec(),
        headers: vec![("content-type".into(), "application/json".into())],
    })
}

fn batch(id: &str, pending: u64) -> Value {
    json!({
        "batch_id": id,
        "name": "evaluation-set",
        "create_time": "2026-09-25",
        "expire_time": "2026-09-26",
        "cancel_time": null,
        "cancel_by_xai_message": null,
        "state": {
            "num_requests": 3,
            "num_pending": pending,
            "num_success": 1,
            "num_error": 1,
            "num_cancelled": 1
        },
        "future_field": "kept"
    })
}

fn scope(account: &str, endpoint: &str) -> XaiBatchScope {
    XaiBatchScope::new("xai-production", account, endpoint).unwrap()
}

fn service<'a>(mock: &'a MockTransport, account: &str) -> XaiBatchService<'a> {
    XaiBatchService::new(
        mock,
        Secret::new("xai-test-key".into()),
        scope(account, "https://api.x.ai/v1"),
    )
    .unwrap()
}

fn xai_chat_profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "xai",
        "profile_name": "xai-grok-chat",
        "base_url": "https://api.x.ai/v1",
        "protocol": "open_ai_chat",
        "auth": "api_key",
        "models": []
    }))
    .unwrap()
}

fn chat_request(id: &str, model: &str, text: &str) -> XaiBatchRequest {
    let chat = XaiBatchChatCompletion::new(model, vec![json!({"role":"user", "content":text})])
        .unwrap()
        .with_parameter("temperature", json!(0.2))
        .unwrap();
    XaiBatchRequest::new(id, XaiBatchRequestBody::ChatGetCompletion(chat)).unwrap()
}

fn batch_input() -> XaiBatchInput {
    XaiBatchInput::new(vec![chat_request("row-1", "grok-4.3", "hello")]).unwrap()
}

fn request_body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn inline_request_union_is_typed_and_custom_ids_are_unique() {
    let request = XaiBatchRequest::new(
        "responses-1",
        XaiBatchRequestBody::Responses(
            XaiBatchResponses::new("grok-4.3", json!([{"role":"user", "content":"hello"}]))
                .unwrap()
                .with_parameter("tools", json!([{"type":"web_search"}]))
                .unwrap(),
        ),
    )
    .unwrap();
    let input = XaiBatchInput::new(vec![request]).unwrap();
    let encoded = serde_json::to_value(&input.requests()[0]).unwrap();
    assert_eq!(encoded["batch_request_id"], "responses-1");
    assert_eq!(encoded["batch_request"]["responses"]["model"], "grok-4.3");
    assert_eq!(
        encoded["batch_request"]["responses"]["tools"][0]["type"],
        "web_search"
    );

    let duplicate = XaiBatchInput::new(vec![
        chat_request("same-id", "grok-4.3", "first"),
        chat_request("same-id", "grok-4.3", "second"),
    ]);
    assert!(matches!(duplicate, Err(XaiBatchError::InvalidInput(_))));
    assert!(XaiBatchCreateRequest::new(" \n ").is_err());
}

#[test]
fn model_cards_explicitly_without_batch_are_rejected() {
    for model in [
        "grok-4.5",
        "grok-4.5-latest",
        "grok-4.6",
        "grok-4.7",
        "grok-build-0.1",
        "grok-build-latest",
        "grok-code-fast-1",
    ] {
        assert!(
            XaiBatchChatCompletion::new(model, vec![json!({"role":"user","content":"hello"})])
                .is_err()
        );
        assert!(XaiBatchResponses::new(model, json!("hello")).is_err());
    }
    assert!(XaiBatchChatCompletion::new(
        "grok-4.3",
        vec![json!({"role":"user","content":"hello"})]
    )
    .is_ok());
}

#[test]
fn media_request_variants_match_documented_native_envelopes() {
    let image_generation = XaiBatchRequest::new(
        "image-1",
        XaiBatchRequestBody::ImageGeneration(
            XaiBatchMediaPrompt::new("grok-imagine-image-2.0", "A mountain lake").unwrap(),
        ),
    )
    .unwrap();
    let image_edit = XaiBatchRequest::new(
        "image-edit-1",
        XaiBatchRequestBody::ImageEdit(
            XaiBatchImageEdit::new(
                "grok-imagine-image-2.0",
                "Add a rainbow",
                XaiBatchImageInput::image_url("https://images.example/test.png").unwrap(),
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let video_generation = XaiBatchRequest::new(
        "video-1",
        XaiBatchRequestBody::VideoGeneration(
            XaiBatchVideoGeneration::new(
                XaiBatchMediaPrompt::new("grok-imagine-video-1.5", "A rotating product").unwrap(),
                None,
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let video_extension = XaiBatchRequest::new(
        "video-extension-1",
        XaiBatchRequestBody::VideoExtension(
            XaiBatchVideoExtension::new(
                XaiBatchMediaPrompt::new("grok-imagine-video", "Reveal a sunset").unwrap(),
                XaiBatchVideoInput::new("https://video.example/clip.mp4").unwrap(),
                6,
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let input = XaiBatchInput::new(vec![
        image_generation,
        image_edit,
        video_generation,
        video_extension,
    ])
    .unwrap();
    let encoded = serde_json::to_value(&input.requests()[0]).unwrap();
    assert_eq!(
        encoded["batch_request"]["image_generation"]["prompt"],
        "A mountain lake"
    );
    let encoded = serde_json::to_value(&input.requests()[1]).unwrap();
    assert_eq!(
        encoded["batch_request"]["image_edit"]["image"]["type"],
        "image_url"
    );
    let encoded = serde_json::to_value(&input.requests()[2]).unwrap();
    assert_eq!(
        encoded["batch_request"]["video_generation"]["model"],
        "grok-imagine-video-1.5"
    );
    let encoded = serde_json::to_value(&input.requests()[3]).unwrap();
    assert_eq!(encoded["batch_request"]["video_extension"]["duration"], 6);
}

#[tokio::test]
async fn create_submit_get_list_and_cancel_use_documented_rest_routes() {
    let mock = MockTransport::new([
        reply(200, batch("batch_123", 0)),
        raw_reply(200, b""),
        reply(200, batch("batch_123", 1)),
        reply(
            200,
            json!({
                "batches": [batch("batch_123", 1)],
                "pagination_token": "next/page"
            }),
        ),
        reply(200, batch("batch_123", 0)),
    ]);
    let service = service(&mock, "account-a");
    let created = service
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap();
    assert_eq!(created.reference.batch_id(), "batch_123");
    assert_eq!(created.native["future_field"], "kept");
    assert!(created.state.is_terminal());
    assert_eq!(
        service
            .submit(&created.reference, &batch_input())
            .await
            .unwrap(),
        1
    );
    let fetched = service.get(&created.reference).await.unwrap();
    assert!(!fetched.state.is_terminal());
    let page = service
        .list(&XaiBatchPageOptions::new().limit(20))
        .await
        .unwrap();
    assert_eq!(page.batches.len(), 1);
    assert_eq!(page.pagination_token.as_deref(), Some("next/page"));
    let canceled = service.cancel(&created.reference).await.unwrap();
    assert_eq!(canceled.reference.batch_id(), "batch_123");

    let requests = mock.requests();
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://api.x.ai/v1/batches");
    assert_eq!(request_body(&requests[0]), json!({"name":"evaluation-set"}));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| { name == "Authorization" && value == "Bearer xai-test-key" }));
    assert_eq!(
        requests[1].url,
        "https://api.x.ai/v1/batches/batch_123/requests"
    );
    assert_eq!(
        request_body(&requests[1])["batch_requests"][0]["batch_request_id"],
        "row-1"
    );
    assert_eq!(
        request_body(&requests[1])["batch_requests"][0]["batch_request"]["chat_get_completion"]
            ["model"],
        "grok-4.3"
    );
    assert_eq!(requests[2].url, "https://api.x.ai/v1/batches/batch_123");
    assert_eq!(requests[3].url, "https://api.x.ai/v1/batches?limit=20");
    assert_eq!(
        requests[4].url,
        "https://api.x.ai/v1/batches/batch_123:cancel"
    );
}

#[tokio::test]
async fn uploaded_jsonl_from_normal_xai_chat_profile_creates_a_sealed_file_batch() {
    let mock = MockTransport::new([
        reply(
            200,
            json!({
                "id":"file_xai_batch",
                "filename":"requests.jsonl",
                "bytes":19,
                "expires_at":"2099-01-01T00:00:00Z",
                "status":"provider_extension"
            }),
        ),
        // xAI may omit input_file_id from the returned Batch object. The
        // scoped reference must retain the submitted identity in that case.
        reply(200, batch("batch_file_123", 0)),
    ]);
    let profile = xai_chat_profile();
    let auth = ApiKeyAuthenticator;
    let api_key = Secret::new("xai-test-key".to_owned());
    let upload_service = FileService::new(
        &mock,
        &profile,
        Some(&auth),
        Some(&api_key),
        Some("account-a"),
    );
    let uploaded = upload_service
        .upload(
            &UploadFile {
                filename: "requests.jsonl".into(),
                media_type: "application/jsonl".into(),
                bytes: Bytes::from_static(b"{\"custom_id\":\"one\"}\n"),
            },
            FilePurpose::Batch,
        )
        .await
        .unwrap();
    assert_eq!(
        uploaded.protocol,
        lingxi_llm_client::protocol::ProtocolFamily::OpenAiChat
    );
    assert_eq!(uploaded.expires_at.as_deref(), Some("2099-01-01T00:00:00Z"));
    assert_eq!(
        uploaded.processing_status.as_deref(),
        Some("provider_extension")
    );

    let scope = XaiBatchScope::new(
        profile.profile_name.clone(),
        "account-a",
        profile.base_url.as_str(),
    )
    .unwrap();
    let input_file = XaiBatchInputFileRef::from_uploaded_file(&scope, &uploaded).unwrap();
    assert_eq!(input_file.expires_at(), Some("2099-01-01T00:00:00Z"));
    assert_eq!(input_file.processing_status(), Some("provider_extension"));
    let batch_service =
        XaiBatchService::new(&mock, Secret::new("xai-test-key".into()), scope).unwrap();
    let created = batch_service
        .create(
            &XaiBatchCreateRequest::new("jsonl-evaluation")
                .unwrap()
                .with_input_file(input_file),
        )
        .await
        .unwrap();
    assert!(created.reference.is_file_based());
    assert_eq!(created.reference.input_file_id(), Some("file_xai_batch"));
    assert!(matches!(
        batch_service.submit(&created.reference, &batch_input()).await,
        Err(XaiBatchError::InvalidInput(message)) if message.contains("sealed")
    ));

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://api.x.ai/v1/files");
    let multipart = String::from_utf8_lossy(&requests[0].body);
    assert!(multipart.contains("name=\"file\"; filename=\"requests.jsonl\""));
    assert!(!multipart.contains("name=\"purpose\""));
    assert_eq!(requests[1].method, "POST");
    assert_eq!(requests[1].url, "https://api.x.ai/v1/batches");
    assert_eq!(
        request_body(&requests[1]),
        json!({"name":"jsonl-evaluation","input_file_id":"file_xai_batch"})
    );
}

#[tokio::test]
async fn uploaded_batch_file_scope_and_expiry_are_preflighted() {
    let mock = MockTransport::new([reply(
        200,
        json!({
            "id":"file_xai_batch",
            "filename":"requests.jsonl",
            "bytes":19,
            "expires_at":"2099-01-01T00:00:00Z"
        }),
    )]);
    let profile = xai_chat_profile();
    let auth = ApiKeyAuthenticator;
    let api_key = Secret::new("xai-test-key".to_owned());
    let uploaded = FileService::new(
        &mock,
        &profile,
        Some(&auth),
        Some(&api_key),
        Some("account-a"),
    )
    .upload(
        &UploadFile {
            filename: "requests.jsonl".into(),
            media_type: "application/jsonl".into(),
            bytes: Bytes::from_static(b"{\"custom_id\":\"one\"}\n"),
        },
        FilePurpose::Batch,
    )
    .await
    .unwrap();
    let scope = XaiBatchScope::new(
        profile.profile_name.clone(),
        "account-a",
        profile.base_url.as_str(),
    )
    .unwrap();
    assert!(XaiBatchInputFileRef::from_uploaded_file(
        &XaiBatchScope::new("xai-grok-chat", "other-account", "https://api.x.ai/v1").unwrap(),
        &uploaded
    )
    .is_err());

    let mut expired = uploaded;
    expired.expires_at = Some("1".into());
    let input_file = XaiBatchInputFileRef::from_uploaded_file(&scope, &expired).unwrap();
    let batch_service =
        XaiBatchService::new(&mock, Secret::new("xai-test-key".into()), scope).unwrap();
    assert!(matches!(
        batch_service
            .create(&XaiBatchCreateRequest::new("expired").unwrap().with_input_file(input_file))
            .await,
        Err(XaiBatchError::Llm(LlmError::InvalidRequest { message }))
            if message.contains("expired")
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn request_metadata_and_results_are_single_pages_with_typed_item_outcomes() {
    let mock = MockTransport::new([
        reply(200, batch("batch_123", 2)),
        reply(
            200,
            json!({
                "batch_request_metadata": [{
                    "batch_request_id":"row-1",
                    "endpoint":"xai_api.Chat/GetCompletion",
                    "model":"grok-4.3",
                    "state":"succeeded",
                    "create_time":"2026-09-25",
                    "finish_time":"2026-09-25"
                }],
                "pagination_token": null
            }),
        ),
        reply(
            200,
            json!({
                "results": [
                    {
                        "batch_request_id":"ok",
                        "batch_result":{"response":{"chat_get_completion":{"choices":[]}}}
                    },
                    {
                        "batch_request_id":"bad",
                        "error_message":"model unavailable"
                    },
                    {
                        "batch_request_id":"cancelled",
                        "state":"cancelled"
                    },
                    {
                        "batch_request_id":"future",
                        "state":"future_state",
                        "batch_result":{"provider_field":true}
                    }
                ],
                "pagination_token": "results-next"
            }),
        ),
    ]);
    let service = service(&mock, "account-a");
    let reference = service
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap()
        .reference;
    let metadata = service
        .list_requests(&reference, &XaiBatchPageOptions::new().limit(50))
        .await
        .unwrap();
    assert_eq!(metadata.requests[0].state.as_deref(), Some("succeeded"));
    let page = service
        .results(
            &reference,
            &XaiBatchPageOptions::new()
                .limit(100)
                .pagination_token("page/one"),
        )
        .await
        .unwrap();
    assert_eq!(page.results.len(), 4);
    assert!(matches!(
        &page.results[0].outcome,
        XaiBatchResultOutcome::Succeeded { .. }
    ));
    assert!(matches!(
        &page.results[1].outcome,
        XaiBatchResultOutcome::Failed { .. }
    ));
    assert!(matches!(
        &page.results[2].outcome,
        XaiBatchResultOutcome::Canceled
    ));
    assert!(matches!(
        &page.results[3].outcome,
        XaiBatchResultOutcome::Other { .. }
    ));
    assert_eq!(page.pagination_token.as_deref(), Some("results-next"));
    let requests = mock.requests();
    assert_eq!(
        requests[1].url,
        "https://api.x.ai/v1/batches/batch_123/requests?limit=50"
    );
    assert!(requests[2].url.contains("pagination_token=page%2Fone"));
}

#[tokio::test]
async fn mutation_failures_are_explicit_unknown_and_never_retried() {
    let mock = MockTransport::new([Err(LlmError::Transport {
        message: "connection closed after dispatch".into(),
    })]);
    let error = service(&mock, "account-a")
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        XaiBatchError::OutcomeUnknown {
            operation: "create",
            reference: None,
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);

    let mock = MockTransport::new([
        reply(200, batch("batch_123", 1)),
        Err(LlmError::Transport {
            message: "response lost after request dispatch".into(),
        }),
    ]);
    let service = service(&mock, "account-a");
    let reference = service
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap()
        .reference;
    let error = service
        .submit(&reference, &batch_input())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        XaiBatchError::OutcomeUnknown {
            operation: "add requests",
            reference: Some(_),
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 2);
}

#[tokio::test]
async fn references_are_checked_before_requests_and_http_errors_are_known() {
    let create_mock = MockTransport::new([reply(200, batch("batch_123", 1))]);
    let reference = service(&create_mock, "account-a")
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap()
        .reference;
    let mock = MockTransport::new(std::iter::empty::<Result<Reply, LlmError>>());
    let error = service(&mock, "account-b")
        .get(&reference)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        XaiBatchError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests().is_empty());

    let mock = MockTransport::new([
        reply(200, batch("batch_123", 1)),
        Ok(Reply {
            status: 404,
            body: br#"{"error":"missing"}"#.to_vec(),
            headers: vec![("request-id".into(), "req-1".into())],
        }),
    ]);
    let service = service(&mock, "account-a");
    let reference = service
        .create(&XaiBatchCreateRequest::new("evaluation-set").unwrap())
        .await
        .unwrap()
        .reference;
    let error = service.get(&reference).await.unwrap_err();
    assert!(matches!(
        error,
        XaiBatchError::Provider {
            status: 404,
            request_id: Some(ref id),
            ..
        } if id == "req-1"
    ));
}
