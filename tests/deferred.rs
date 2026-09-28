use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, providers::xai::deferred::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Reply {
    method: &'static str,
    url: &'static str,
    result: Result<(u16, Vec<u8>), LlmError>,
}
struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP call");
        assert_eq!(request.method, reply.method);
        assert_eq!(request.url, reply.url);
        self.requests.lock().unwrap().push(request);
        let (status, body) = reply.result?;
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "req-http".into())],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"xai","profile_name":"grok","base_url":"https://api.x.ai/v1",
        "protocol":"open_ai_chat","auth":"none",
        "models":[{"display_model":"grok-test","request_model":"grok-test","billing_model":"grok-test"}],
        "deferred":{"mode":"enabled","value":{"api":"xai_chat",
            "endpoint":"https://api.x.ai/v1/chat/completions",
            "results_endpoint":"https://api.x.ai/v1/chat/deferred-completion",
            "auth":{"type":"bearer"}}}
    })).unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"grok-test",
        "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]
    }))
    .unwrap()
}
fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("key".into())),
        ..Default::default()
    }
}
fn reply(method: &'static str, url: &'static str, status: u16, body: Value) -> Reply {
    Reply {
        method,
        url,
        result: Ok((status, serde_json::to_vec(&body).unwrap())),
    }
}
fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
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
fn completed() -> Value {
    json!({"id":"chat_1","object":"chat.completion","model":"grok-test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"done","reasoning_content":"worked"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}})
}

#[tokio::test]
async fn submit_pending_then_consumed_result_preserves_scope_and_native() {
    let (client, mock) = setup(vec![
        reply(
            "POST",
            "https://api.x.ai/v1/chat/completions",
            200,
            json!({"request_id":"req_1"}),
        ),
        Reply {
            method: "GET",
            url: "https://api.x.ai/v1/chat/deferred-completion/req_1",
            result: Ok((202, vec![])),
        },
        reply(
            "GET",
            "https://api.x.ai/v1/chat/deferred-completion/req_1",
            200,
            completed(),
        ),
    ]);
    let job = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .submit(&request(), &options("acct-a"))
        .await
        .unwrap();
    assert_eq!(job.reference.account_scope, "acct-a");
    assert_eq!(job.native["request_id"], "req_1");
    let pending = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .fetch_once(job.reference, &options("acct-a"))
        .await
        .unwrap();
    let DeferredPoll::Pending(reference) = pending else {
        panic!("expected 202 pending")
    };
    let completed = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .fetch_once(reference, &options("acct-a"))
        .await
        .unwrap();
    let DeferredPoll::Completed(completed) = completed else {
        panic!("expected consumed completion")
    };
    assert_eq!(completed.response.model, "grok-test");
    assert_eq!(completed.response.executed_profile.as_deref(), Some("grok"));
    assert_eq!(
        completed.native["choices"][0]["message"]["reasoning_content"],
        "worked"
    );
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["deferred"], true);
    assert_eq!(body["model"], "grok-test");
    assert!(requests.iter().all(|request| request
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer key")));
}

#[tokio::test]
async fn wrong_account_and_endpoint_fail_before_get() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://api.x.ai/v1/chat/completions",
        200,
        json!({"request_id":"req_1"}),
    )]);
    let job = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .submit(&request(), &options("acct-a"))
        .await
        .unwrap();
    let encoded = serde_json::to_value(&job.reference).unwrap();
    let wrong_scope: DeferredJobRef = serde_json::from_value(encoded.clone()).unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .fetch_once(wrong_scope, &options("acct-b"))
            .await,
        Err(DeferredError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let mut wrong_endpoint = encoded;
    wrong_endpoint["results_endpoint_fingerprint"] = json!("other");
    let wrong_endpoint: DeferredJobRef = serde_json::from_value(wrong_endpoint).unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .fetch_once(wrong_endpoint, &options("acct-a"))
            .await,
        Err(DeferredError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn transport_failure_never_resubmits_or_refetches_single_use_result() {
    let (client, mock) = setup(vec![Reply {
        method: "POST",
        url: "https://api.x.ai/v1/chat/completions",
        result: Err(LlmError::Transport {
            message: "reset".into(),
        }),
    }]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .submit(&request(), &options("acct-a"))
            .await,
        Err(DeferredError::SubmitOutcomeUnknown { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let (client, mock) = setup(vec![
        reply(
            "POST",
            "https://api.x.ai/v1/chat/completions",
            200,
            json!({"request_id":"req_1"}),
        ),
        Reply {
            method: "GET",
            url: "https://api.x.ai/v1/chat/deferred-completion/req_1",
            result: Err(LlmError::Transport {
                message: "reset".into(),
            }),
        },
    ]);
    let job = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .submit(&request(), &options("acct-a"))
        .await
        .unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .fetch_once(job.reference, &options("acct-a"))
            .await,
        Err(DeferredError::FetchOutcomeUnknown { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn malformed_completed_result_is_reported_as_consumed() {
    let (client, mock) = setup(vec![
        reply(
            "POST",
            "https://api.x.ai/v1/chat/completions",
            200,
            json!({"request_id":"req_1"}),
        ),
        reply(
            "GET",
            "https://api.x.ai/v1/chat/deferred-completion/req_1",
            200,
            json!({"id":"chat_1","model":"grok-test","choices":[]}),
        ),
    ]);
    let job = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .submit(&request(), &options("acct-a"))
        .await
        .unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .fetch_once(job.reference, &options("acct-a"))
            .await,
        Err(DeferredError::ConsumedInvalidResult { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn accepted_submission_without_request_id_is_not_safe_to_retry() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://api.x.ai/v1/chat/completions",
        200,
        json!({"unexpected":"accepted"}),
    )]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .submit(&request(), &options("acct-a"))
            .await,
        Err(DeferredError::SubmitAcceptedUnknownId { .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn completed_ticket_remains_readable_after_catalog_model_changes() {
    let (client, _) = setup(vec![reply(
        "POST",
        "https://api.x.ai/v1/chat/completions",
        200,
        json!({"request_id":"req_1"}),
    )]);
    let job = client
        .provider::<lingxi_llm_client::providers::XaiClient>("grok")
        .unwrap()
        .deferred()
        .submit(&request(), &options("acct-a"))
        .await
        .unwrap();
    let mut updated = profile();
    updated.models[0].request_model = "replacement".into();
    updated.models[0].display_model = "replacement".into();
    let mock = Arc::new(Mock {
        replies: Mutex::new(
            vec![reply(
                "GET",
                "https://api.x.ai/v1/chat/deferred-completion/req_1",
                200,
                completed(),
            )]
            .into(),
        ),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[updated])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::XaiClient>("grok")
            .unwrap()
            .deferred()
            .fetch_once(job.reference, &options("acct-a"))
            .await
            .unwrap(),
        DeferredPoll::Completed(_)
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[test]
fn invalid_deferred_route_is_rejected_by_builder() {
    let mut profile = profile();
    if let ServiceSetting::Enabled(route) = &mut profile.deferred {
        route.results_endpoint = "https://other.test/v1/chat/deferred-completion".into();
    }
    let mock = Arc::new(Mock {
        replies: Mutex::new(VecDeque::new()),
        requests: Mutex::new(vec![]),
    });
    assert!(LlmClientBuilder::with_transport(mock, &[profile])
        .with_region(Region::International)
        .build()
        .is_err());
}
