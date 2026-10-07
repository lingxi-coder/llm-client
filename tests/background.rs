use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, providers::openai::background::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Mock {
    replies: Mutex<VecDeque<Result<(u16, Value), LlmError>>>,
    requests: Mutex<Vec<HttpRequest>>,
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
            .expect("unexpected HTTP call")?;
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "http-id".into())],
            body: futures::stream::once(
                async move { Ok(serde_json::to_vec(&body).unwrap().into()) },
            )
            .boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"openai", "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses", "auth":"none",
        "extra":{"supports_previous_response_id":true},
        "models":[{"display_model":"test","request_model":"test","billing_model":"test"}],
        "background":{"mode":"enabled","value":{"endpoint":"https://api.openai.com/v1/responses","auth":{"type":"bearer"}}}
    })).unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]})).unwrap()
}
fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("key".into())),
        ..Default::default()
    }
}
fn setup(replies: Vec<Result<(u16, Value), LlmError>>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn result(status: &str) -> Value {
    json!({"id":"resp_1", "status":status,"model":"test","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]})
}

#[tokio::test]
async fn submit_poll_complete_and_cancel_use_separate_routes() {
    let (client, mock) = setup(vec![
        Ok((200, result("queued"))),
        Ok((200, result("in_progress"))),
        Ok((200, result("completed"))),
        Ok((200, result("cancelled"))),
    ]);
    let initial = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap();
    assert_eq!(initial.status, BackgroundStatus::Queued);
    assert!(initial.response.is_none());
    let reference = initial.reference;
    let pending = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .get(&reference, &options("account-a"))
        .await
        .unwrap();
    assert!(!pending.status.is_terminal());
    let completed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .get(&reference, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(completed.status, BackgroundStatus::Completed);
    let response = completed.response.unwrap();
    assert_eq!(response.message.content.len(), 1);
    assert_eq!(response.continuation.unwrap().account_scope, "account-a");
    let cancelled = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .cancel(&reference, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(cancelled.status, BackgroundStatus::Cancelled);
    let sent = mock.requests.lock().unwrap();
    assert_eq!(
        sent.iter()
            .map(|r| (r.method.as_str(), r.url.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("POST", "https://api.openai.com/v1/responses"),
            ("GET", "https://api.openai.com/v1/responses/resp_1"),
            ("GET", "https://api.openai.com/v1/responses/resp_1"),
            ("POST", "https://api.openai.com/v1/responses/resp_1/cancel"),
        ]
    );
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["background"], true);
    assert_eq!(body["model"], "test");
    assert_eq!(
        sent[0]
            .headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .unwrap()
            .1,
        "Bearer key"
    );
}

#[tokio::test]
async fn scope_and_endpoint_are_checked_before_network() {
    let (client, mock) = setup(vec![Ok((200, result("queued")))]);
    let job = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap();
    let error = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .get(&job.reference, &options("account-b"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        BackgroundError::Llm(LlmError::PermissionDenied { .. })
    ));
    let mut wrong = job.reference;
    wrong.endpoint_fingerprint = "other".into();
    let error = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .cancel(&wrong, &options("account-a"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        BackgroundError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn continuation_scope_is_checked_before_submission() {
    let (client, mock) = setup(vec![]);
    let mut req = request();
    req.continuation = Some(ContinuationRef {
        protocol: lingxi_llm_client::protocol::ProtocolFamily::OpenAiResponses,
        response_id: ResponseId::new("resp_previous"),
        provider_id: ProviderId::new("openai"),
        profile_name: "openai".into(),
        endpoint_fingerprint: "wrong-endpoint".into(),
        account_scope: "account-a".into(),
        request_model: "test".into(),
        workspace_id: None,
    });
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .submit(&req, &options("account-a"))
            .await,
        Err(BackgroundError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn unknown_submit_and_cancel_outcomes_are_not_retried() {
    let transport_error = || LlmError::Transport {
        message: "lost".into(),
    };
    let (client, mock) = setup(vec![Err(transport_error())]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .submit(&request(), &options("account-a"))
            .await,
        Err(BackgroundError::SubmitOutcomeUnknown { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let (client, mock) = setup(vec![Ok((200, result("queued"))), Err(transport_error())]);
    let job = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .cancel(&job.reference, &options("account-a"))
            .await,
        Err(BackgroundError::CancelOutcomeUnknown { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn terminal_failure_is_preserved_without_decoding_as_success() {
    let body = json!({"id":"resp_1","status":"failed","error":{"message":"failed"}});
    let (client, _) = setup(vec![Ok((200, body.clone()))]);
    let job = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap();
    assert_eq!(job.status, BackgroundStatus::Failed);
    assert!(job.response.is_none());
    assert_eq!(job.native, body);
}

#[test]
fn route_requires_first_party_responses_profile() {
    let mut p = profile();
    p.provider_id = ProviderId::new("other");
    assert!(LlmClientBuilder::with_transport(
        Arc::new(Mock {
            replies: Mutex::new(VecDeque::new()),
            requests: Mutex::new(vec![])
        }),
        &[p]
    )
    .with_region(Region::International)
    .build()
    .is_err());
}

#[tokio::test]
async fn deletion_is_scoped_and_requires_a_matching_receipt() {
    let (client, mock) = setup(vec![
        Ok((200, result("completed"))),
        Ok((
            200,
            json!({"id":"resp_1","object":"response","deleted":true,"extra":"kept"}),
        )),
    ]);
    let reference = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap()
        .reference;
    for mutate in 0..5 {
        let mut wrong = reference.clone();
        match mutate {
            0 => wrong.account_scope = "account-b".into(),
            1 => wrong.endpoint_fingerprint = "elsewhere".into(),
            2 => wrong.provider_id = "other".into(),
            3 => wrong.response_id = "../other".into(),
            _ => wrong.model.clear(),
        }
        assert!(client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .delete(&wrong, &options("account-a"))
            .await
            .is_err());
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }
    let receipt = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .delete(&reference, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(receipt.reference, reference);
    assert_eq!(receipt.native["extra"], "kept");
    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[1].method, "DELETE");
    assert_eq!(sent[1].url, "https://api.openai.com/v1/responses/resp_1");
    assert!(sent[1].body.is_empty());
    assert!(sent[1]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer key"));
}

#[tokio::test]
async fn uncertain_deletions_preserve_reference_and_never_retry() {
    let failures = vec![
        Err(LlmError::Transport {
            message: "lost receipt".into(),
        }),
        Ok((503, json!({"error":"upstream"}))),
        Ok((
            200,
            json!({"id":"resp_other","object":"response","deleted":true}),
        )),
        Ok((
            200,
            json!({"id":"resp_1","object":"response","deleted":false}),
        )),
        Ok((
            200,
            json!({"id":"resp_1","object":"different","deleted":true}),
        )),
    ];
    for failure in failures {
        let (client, mock) = setup(vec![Ok((200, result("completed"))), failure]);
        let reference = client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .submit(&request(), &options("account-a"))
            .await
            .unwrap()
            .reference;
        match client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .delete(&reference, &options("account-a"))
            .await
            .unwrap_err()
        {
            BackgroundError::DeleteOutcomeUnknown {
                reference: saved, ..
            } => assert_eq!(*saved, reference),
            other => panic!("expected unknown deletion, got {other:?}"),
        }
        assert_eq!(mock.requests.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn deletion_provider_rejection_is_not_a_success_receipt() {
    let (client, mock) = setup(vec![
        Ok((200, result("completed"))),
        Ok((404, json!({"error":"not found"}))),
    ]);
    let reference = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit(&request(), &options("account-a"))
        .await
        .unwrap()
        .reference;
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .delete(&reference, &options("account-a"))
            .await,
        Err(BackgroundError::Provider { status: 404, .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}
