use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{ChatRequest, LlmError, ProviderProfile, Secret},
    providers::zhipu::async_tasks::{
        GlmAsyncConfig, GlmAsyncCredentials, GlmAsyncError, GlmAsyncRegion, GlmAsyncRequest,
        GlmAsyncService, GlmAsyncTaskStatus,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct MockTransport {
    replies: Mutex<VecDeque<Result<(u16, Value), LlmError>>>,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        let bytes = serde_json::to_vec(&reply.1).unwrap();
        Ok(StreamResponse {
            status: reply.0,
            headers: vec![("x-request-id".into(), "glm-request-http".into())],
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

fn profile(region: GlmAsyncRegion) -> ProviderProfile {
    let (base_url, regions) = match region {
        GlmAsyncRegion::ChinaMainland => (
            "https://open.bigmodel.cn/api/paas/v4",
            json!(["china_mainland"]),
        ),
        GlmAsyncRegion::International => ("https://api.z.ai/api/paas/v4", json!(["international"])),
    };
    serde_json::from_value(json!({
        "provider_id": "zhipu",
        "profile_name": "glm-account",
        "base_url": base_url,
        "protocol": "open_ai_chat",
        "auth": "bearer",
        "regions": regions
    }))
    .unwrap()
}

fn service<'a>(
    mock: &'a MockTransport,
    region: GlmAsyncRegion,
    account_scope: &str,
    credential_scope: &str,
) -> GlmAsyncService<'a> {
    let config =
        GlmAsyncConfig::new(profile(region), region, account_scope, credential_scope).unwrap();
    GlmAsyncService::new(mock, config).unwrap()
}

fn credentials(scope: &str) -> GlmAsyncCredentials {
    GlmAsyncCredentials::new(scope, Secret::new("secret-glm-key".to_owned()))
}

fn request() -> GlmAsyncRequest {
    serde_json::from_value::<ChatRequest>(json!({
        "model": "glm-4.7",
        "messages": [{"role":"user","content":[{"type":"text","text":"hello"}]}],
        "temperature": 0.2,
        "max_tokens": 50
    }))
    .unwrap()
}

fn mock(replies: impl IntoIterator<Item = Result<(u16, Value), LlmError>>) -> MockTransport {
    MockTransport {
        replies: Mutex::new(replies.into_iter().collect()),
        requests: Mutex::new(Vec::new()),
    }
}

#[tokio::test]
async fn submits_typed_chat_then_gets_one_native_result_under_the_same_scope() {
    let mock = mock([
        Ok((
            200,
            json!({"id":"task-1","request_id":"client-request-1","model":"glm-4.7","task_status":"PROCESSING"}),
        )),
        Ok((
            200,
            json!({"id":"task-1","request_id":"client-request-1","model":"glm-4.7","task_status":"SUCCESS","choices":[{"message":{"content":"done"}}],"usage":{"total_tokens":4}}),
        )),
    ]);
    let client = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "zhipu-user-7",
        "api-key-slot-2",
    );
    let key = credentials("api-key-slot-2");
    let submitted = client.submit(&request(), &key).await.unwrap();
    assert_eq!(submitted.task_status, GlmAsyncTaskStatus::Processing);
    assert_eq!(submitted.reference.task_id(), "task-1");
    assert_eq!(submitted.reference.scope.account_scope, "zhipu-user-7");
    assert_eq!(submitted.reference.scope.credential_scope, "api-key-slot-2");
    assert_eq!(submitted.request_id.as_deref(), Some("client-request-1"));

    let result = client.get(&submitted.reference, &key).await.unwrap();
    assert_eq!(result.task_status, GlmAsyncTaskStatus::Success);
    assert_eq!(result.native["choices"][0]["message"]["content"], "done");
    assert_eq!(result.native["usage"]["total_tokens"], 4);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://open.bigmodel.cn/api/paas/v4/async/chat/completions"
    );
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://open.bigmodel.cn/api/paas/v4/async-result/task-1"
    );
    assert!(requests.iter().all(|request| request
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer secret-glm-key")));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "glm-4.7");
    assert_eq!(body["messages"][0]["content"], "hello");
    assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert_eq!(body["max_tokens"], 50);
    assert_eq!(body["stream"], false);
    assert!(!format!("{key:?}").contains("secret-glm-key"));
}

#[tokio::test]
async fn international_region_is_explicitly_rejected_without_an_async_documented_route() {
    let error = GlmAsyncConfig::new(
        profile(GlmAsyncRegion::International),
        GlmAsyncRegion::International,
        "zai-user-3",
        "zai-key-slot-1",
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GlmAsyncError::UnsupportedRegion {
            region: GlmAsyncRegion::International,
            ..
        }
    ));
}

#[tokio::test]
async fn wrong_credential_scope_is_rejected_before_sending() {
    let mock = mock([]);
    let client = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    );
    let error = client
        .submit(&request(), &credentials("key-slot-2"))
        .await
        .unwrap_err();
    assert!(matches!(error, GlmAsyncError::ScopeMismatch { .. }));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn references_cannot_cross_bound_identity_fields() {
    let mock = mock([Ok((
        200,
        json!({"id":"task-1","task_status":"PROCESSING","model":"glm-4.7"}),
    ))]);
    let mainland = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    );
    let task = mainland
        .submit(&request(), &credentials("key-slot-1"))
        .await
        .unwrap();
    let other_account = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-2",
        "key-slot-1",
    );
    let mut foreign_profile = task.reference.clone();
    foreign_profile.scope.profile_name = "other-profile".into();
    let mut foreign_region = task.reference.clone();
    foreign_region.scope.region = GlmAsyncRegion::International;
    let mut foreign_endpoint = task.reference.clone();
    foreign_endpoint.scope.endpoint_fingerprint = "another-endpoint".into();
    let mut foreign_credential = task.reference.clone();
    foreign_credential.scope.credential_scope = "key-slot-2".into();
    let mut foreign_model = task.reference.clone();
    foreign_model.scope.model = "other-model".into();
    for reference in [
        &task.reference,
        &foreign_profile,
        &foreign_region,
        &foreign_endpoint,
        &foreign_credential,
        &foreign_model,
    ] {
        let result = other_account
            .get(reference, &credentials("key-slot-1"))
            .await;
        assert!(matches!(result, Err(GlmAsyncError::ScopeMismatch { .. })));
    }
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn accepted_response_without_an_id_keeps_the_submission_outcome_unknown() {
    let mock = mock([Ok((200, json!({"task_status":"PROCESSING"})))]);
    let client = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    );
    let error = client
        .submit(&request(), &credentials("key-slot-1"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmAsyncError::SubmitOutcomeUnknown {
            status: Some(200),
            native: Some(_),
            ..
        }
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn transport_and_server_failures_after_submit_are_not_retried() {
    let transport_error = LlmError::Transport {
        message: "connection reset".into(),
    };
    let mock = mock([
        Err(transport_error),
        Ok((
            500,
            json!({"error":{"code":"internal_error","message":"unknown"}}),
        )),
    ]);
    let client = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    );
    for _ in 0..2 {
        let error = client
            .submit(&request(), &credentials("key-slot-1"))
            .await
            .unwrap_err();
        assert!(matches!(error, GlmAsyncError::SubmitOutcomeUnknown { .. }));
    }
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn provider_errors_keep_http_status_native_error_and_request_id() {
    let mock = mock([Ok((
        400,
        json!({"error":{"code":"invalid_parameter","message":"bad messages"}}),
    ))]);
    let client = service(
        &mock,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    );
    let error = client
        .submit(&request(), &credentials("key-slot-1"))
        .await
        .unwrap_err();
    match error {
        GlmAsyncError::Http {
            status,
            request_id,
            native,
            ..
        } => {
            assert_eq!(status, 400);
            assert_eq!(request_id.as_deref(), Some("glm-request-http"));
            assert_eq!(native.unwrap()["error"]["code"], "invalid_parameter");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn profile_route_must_match_the_explicit_region() {
    let mut mismatched = profile(GlmAsyncRegion::ChinaMainland);
    mismatched.base_url = "https://api.z.ai/api/paas/v4".into();
    let error = GlmAsyncConfig::new(
        mismatched,
        GlmAsyncRegion::ChinaMainland,
        "account-1",
        "key-slot-1",
    )
    .unwrap_err();
    assert!(matches!(error, GlmAsyncError::InvalidConfig { .. }));
}
