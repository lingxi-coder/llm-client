use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct MockTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let stream = serde_json::from_slice::<Value>(&request.body)
            .ok()
            .and_then(|body| body.get("stream").and_then(Value::as_bool))
            .unwrap_or(false);
        self.requests.lock().unwrap().push(request);
        let body = if stream {
            b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream\",\"model\":\"wire-m\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n".to_vec()
        } else {
            serde_json::to_vec(&json!({
                "id":"resp_complete","model":"wire-m","status":"completed",
                "output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
            }))
            .unwrap()
        };
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(body.into()) }).boxed(),
        })
    }
}

fn profile(name: &str, endpoint: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"acme", "profile_name":name,
        "base_url":endpoint, "protocol":"open_ai_responses", "auth":"none",
        "connection":{"group":"acme","connection_id":name},
        "extra":{"supports_previous_response_id":true},
        "models":[{"request_model":"wire-m","display_model":"m","billing_model":"wire-m"}]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    }))
    .unwrap()
}

fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn completed_response_yields_scoped_reference_and_mismatches_never_send() {
    let http = Arc::new(MockTransport::default());
    let client = LlmClientBuilder::with_transport(
        http.clone(),
        &[
            profile("one", "https://one.test/v1"),
            profile("two", "https://two.test/v1"),
        ],
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    let response = client
        .chat()
        .complete_in("one", &request(), &options("acct-a"))
        .await
        .unwrap();
    let reference = response.continuation.unwrap();
    assert_eq!(reference.response_id.as_str(), "resp_complete");
    assert_eq!(reference.request_model, "wire-m");
    assert_eq!(reference.profile_name, "one");

    let mut next = request();
    next.continuation = Some(reference.clone());
    for (target, scope) in [("two", "acct-a"), ("one", "acct-b"), ("one", "")] {
        let before = http.requests.lock().unwrap().len();
        assert!(matches!(
            client
                .chat()
                .complete_in(target, &next, &options(scope))
                .await,
            Err(LlmError::InvalidRequest { .. })
        ));
        assert_eq!(http.requests.lock().unwrap().len(), before);
    }
    next.continuation.as_mut().unwrap().request_model = "other-model".into();
    assert!(matches!(
        client
            .chat()
            .complete_in("one", &next, &options("acct-a"))
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    next.continuation = Some(reference);
    client
        .chat()
        .complete_in("one", &next, &options("acct-a"))
        .await
        .unwrap();
    let sent = http.requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    let body: Value = serde_json::from_slice(&sent[1].body).unwrap();
    assert_eq!(body["previous_response_id"], "resp_complete");
}

#[tokio::test]
async fn stream_reference_is_available_only_after_terminal_event() {
    let http = Arc::new(MockTransport::default());
    let client = LlmClientBuilder::with_transport(http, &[profile("one", "https://one.test/v1")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut stream = client
        .chat()
        .stream(&request(), &options("acct-a"))
        .await
        .unwrap();
    assert!(stream.continuation().is_none());
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    let reference = stream.continuation().unwrap();
    assert_eq!(reference.response_id.as_str(), "resp_stream");
    assert_eq!(reference.account_scope, "acct-a");
}

#[tokio::test]
async fn a_response_id_is_not_a_reusable_reference_without_profile_opt_in() {
    let http = Arc::new(MockTransport::default());
    let mut p = profile("one", "https://one.test/v1");
    p.extra = Value::Null;
    let client = LlmClientBuilder::with_transport(http, &[p])
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .chat()
        .complete(&request(), &options("acct-a"))
        .await
        .unwrap();
    assert_eq!(response.response_id.unwrap().as_str(), "resp_complete");
    assert!(response.continuation.is_none());
}
