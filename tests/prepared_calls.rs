use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{CompletionRequest, LlmError, ProviderProfile, Region, UsageState},
    HttpRequest, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse, Transport,
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

struct ResponseTransport {
    sends: AtomicUsize,
    requests: Mutex<Vec<HttpRequest>>,
    status: u16,
    body: Bytes,
    stall_after_body: bool,
}
#[async_trait]
impl Transport for ResponseTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(request);
        let initial = stream::iter([Ok(self.body.clone())]);
        Ok(StreamResponse {
            status: self.status,
            headers: vec![("request-id".into(), "test-request".into())],
            body: if self.stall_after_body {
                initial.chain(stream::pending()).boxed()
            } else {
                initial.boxed()
            },
        })
    }
}
fn profiles(protocol: &str) -> Vec<ProviderProfile> {
    ["primary", "secondary"]
        .into_iter()
        .map(|name| {
            serde_json::from_value(json!({
                "profile_name":name, "provider_id":"test", "base_url":"https://example.test",
                "protocol":protocol, "auth":"none",
                "connection":{"group":"test", "connection_id":name},
                "models":[{"request_model":"wire", "display_model":"test", "billing_model":"wire"}]
            }))
            .unwrap()
        })
        .collect()
}
fn request() -> CompletionRequest {
    serde_json::from_value(json!({"model":"test", "messages":[]})).unwrap()
}
fn http(status: u16, body: impl Into<Bytes>, stall: bool) -> Arc<ResponseTransport> {
    Arc::new(ResponseTransport {
        sends: AtomicUsize::new(0),
        requests: Mutex::new(Vec::new()),
        status,
        body: body.into(),
        stall_after_body: stall,
    })
}

#[tokio::test]
async fn exact_counting_uses_count_endpoint_and_never_generates() {
    let http = http(200, r#"{"input_tokens":42}"#, false);
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("anthropic_messages"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = request();
    request.max_tokens = Some(1024);
    assert_eq!(
        client
            .count_tokens_exact_in("primary", &request, &RequestOptions::default())
            .await
            .unwrap(),
        Some(42)
    );
    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].url.ends_with("/v1/messages/count_tokens"));
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("stream").is_none());
}

#[tokio::test]
async fn counting_unsupported_protocol_is_not_an_approximation() {
    let http = http(200, "{}", false);
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    assert_eq!(
        client
            .count_tokens_exact_in("primary", &request(), &RequestOptions::default())
            .await
            .unwrap(),
        None
    );
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn selected_prices_survive_removing_the_profile_during_an_attempt() {
    use lingxi_llm_client::protocol::{InferenceReport, Submission, Usage, UsageReport};
    let http = http(200, "{}", false);
    let mut profiles = profiles("open_ai_chat");
    profiles[0].models[0].pricing = Some(
        serde_json::from_value(json!({"input_per_million":1.0,"output_per_million":2.0})).unwrap(),
    );
    let mut client = LlmClientBuilder::with_transport(http, &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let directory = std::env::temp_dir().join(format!(
        "llm-pricing-snapshot-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    client.set_config_dir(&directory).unwrap();
    let prepared = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let prices = prepared.pricing_snapshot();
    client.remove_provider("primary").unwrap();
    let usage = UsageReport::measured(
        Usage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        },
        UsageState::Complete,
    );
    let cost = prices
        .estimate(&usage, &InferenceReport::default(), Submission::Interactive)
        .unwrap();
    assert!((cost.total_cost - 1.2).abs() < 1e-10);
    std::fs::remove_dir_all(directory).unwrap();
}

struct WsConnection {
    sent: Arc<Mutex<Vec<Bytes>>>,
}
#[async_trait]
impl lingxi_llm_client::transport::WebSocketConnection for WsConnection {
    async fn send(&mut self, body: Bytes) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(body);
        Ok(StreamResponse { status: 200, headers: vec![], body: stream::iter([
            Ok(Bytes::from_static(br#"{"type":"response.created","response":{"id":"r1","model":"wire"}}"#)),
            Ok(Bytes::from_static(br#"{"type":"response.completed","response":{"id":"r1","model":"wire","output":[],"usage":{"input_tokens":10,"output_tokens":2}}}"#)),
        ]).boxed() })
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        Ok(())
    }
}
#[tokio::test]
async fn prepared_websocket_send_uses_one_response_create_and_decodes_usage() {
    let http = http(200, "{}", false);
    let mut profiles = profiles("open_ai_responses");
    profiles[0].supports_websockets = true;
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let mut socket = WsConnection { sent: sent.clone() };
    let received = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Stream,
        )
        .await
        .unwrap()
        .dispatch_websocket_once(&mut socket)
        .await
        .unwrap();
    let mut stream = received
        .into_stream()
        .unwrap_or_else(|_| panic!("success stream"));
    while stream.next_batch().await.is_some() {}
    assert_eq!(stream.usage_report().state, UsageState::Complete);
    assert_eq!(stream.observed_usage().unwrap().output_tokens, 2);
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&sent[0]).unwrap();
    assert_eq!(body["type"], "response.create");
    assert!(body.get("stream").is_none());
}

#[tokio::test]
async fn prepare_sends_nothing_and_failed_dispatch_never_fails_over() {
    let http = http(429, r#"{"error":{"message":"busy"}}"#, false);
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let prepared = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
    assert_eq!(prepared.profile().profile_name, "primary");
    let received = prepared.dispatch_once().await.unwrap();
    assert_eq!(received.status(), 429);
    let response = received.collect().await.unwrap();
    assert!(matches!(
        response.decode(),
        Err(LlmError::RateLimited { .. })
    ));
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn bad_tool_json_cannot_erase_complete_usage() {
    let body = json!({"model":"wire", "choices":[{"message":{"role":"assistant",
        "tool_calls":[{"id":"t1", "type":"function", "function":{"name":"tool", "arguments":"{broken"}}]},
        "finish_reason":"tool_calls"}], "usage":{"prompt_tokens":10,"completion_tokens":7,
        "completion_tokens_details":{"reasoning_tokens":3}}});
    let http = http(200, body.to_string(), false);
    let client = LlmClientBuilder::with_transport(http, &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(response.usage_report().state, UsageState::Complete);
    assert_eq!(response.usage_report().usage.unwrap().reasoning_tokens, 3);
    assert!(response.decode().is_err());
    assert_eq!(response.usage_report().usage.unwrap().output_tokens, 7);
}

#[tokio::test]
async fn error_status_still_exposes_usage_and_actual_tier() {
    let body = json!({"error":{"message":"failed"},"service_tier":"priority",
        "usage":{"input_tokens":10,"output_tokens":2}});
    let http = http(500, body.to_string(), false);
    let client = LlmClientBuilder::with_transport(http, &profiles("open_ai_responses"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(response.usage_report().state, UsageState::Complete);
    assert!(response.decode().is_err());
}

#[tokio::test]
async fn usage_only_frame_returns_before_next_frame_or_cancellation() {
    let http = http(200, "data: {\"model\":\"wire\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n", true);
    let client = LlmClientBuilder::with_transport(http, &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let received = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Stream,
        )
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap();
    let mut stream = received
        .into_stream()
        .unwrap_or_else(|_| panic!("success stream"));
    let batch = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next_batch())
        .await
        .expect("must not wait for a second frame")
        .unwrap();
    assert_eq!(batch.usage.usage.unwrap().input_tokens, 10);
    assert!(!batch.finished);
    drop(stream);
    assert_eq!(batch.usage.usage.unwrap().output_tokens, 2);
}

#[tokio::test]
async fn error_frame_retains_usage_before_terminating() {
    let http = http(200, "data: {\"error\":{\"message\":\"failed\"},\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n", true);
    let client = LlmClientBuilder::with_transport(http, &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let received = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Stream,
        )
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap();
    let mut stream = received
        .into_stream()
        .unwrap_or_else(|_| panic!("success stream"));
    let batch = stream.next_batch().await.unwrap();
    assert!(batch.finished);
    assert!(batch.events.iter().any(Result::is_err));
    assert_eq!(batch.usage.usage.unwrap().output_tokens, 2);
}

#[tokio::test]
async fn group_names_cannot_masquerade_as_selected_connections() {
    let http = http(200, "{}", false);
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(client
        .prepare_on(
            "test",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete
        )
        .await
        .is_err());
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn chat_array_text_is_preserved_and_missing_arguments_are_rejected() {
    for (message, expected_text) in [
        (
            json!({"role":"assistant", "content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}),
            Some("hello world"),
        ),
        (
            json!({"role":"assistant", "tool_calls":[{"id":"call","function":{"name":"tool"}}]}),
            None,
        ),
    ] {
        let transport = http(200, serde_json::to_vec(&json!({"choices":[{"message":message,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}})).unwrap(), false);
        let client = LlmClientBuilder::with_transport(transport, &profiles("open_ai_chat"))
            .with_region(Region::International)
            .build()
            .unwrap();
        let result = client
            .prepare_on(
                "primary",
                &request(),
                &RequestOptions::default(),
                RequestMode::Complete,
            )
            .await
            .unwrap()
            .dispatch_once()
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(result.usage_report().state, UsageState::Complete);
        match expected_text {
            Some(text) => assert_eq!(result.decode().unwrap().message.text(), text),
            None => assert!(result.decode().is_err()),
        }
    }
}

struct BodySigner;
#[async_trait]
impl lingxi_llm_client::Authenticator for BodySigner {
    async fn apply(
        &self,
        req: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&lingxi_llm_client::protocol::Secret<String>>,
    ) -> Result<(), LlmError> {
        req.headers.push((
            "signed-body".into(),
            String::from_utf8(req.body.to_vec()).unwrap(),
        ));
        Ok(())
    }
}
#[tokio::test]
async fn draft_is_signed_only_after_final_exact_bytes_and_dispatch_marker_can_reject() {
    use lingxi_llm_client::protocol::AuthStrategy;
    let http = http(200, "{}", false);
    let mut profiles = profiles("open_ai_chat");
    profiles[0].auth = AuthStrategy::Bearer;
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &profiles);
    builder.register_authenticator(AuthStrategy::Bearer, Arc::new(BodySigner));
    let client = builder.with_region(Region::International).build().unwrap();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    assert!(!draft
        .request()
        .headers
        .iter()
        .any(|(k, _)| k == "signed-body"));
    let bytes = lingxi_llm_client::exact_json::serialize(
        &json!({"text":"display"}),
        &[("/text".into(), vec![0xd800, 65, 0xd83d, 0xde00])]
            .into_iter()
            .collect(),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(bytes.clone()).unwrap(),
        "{\"text\":\"\\ud800A😀\"}"
    );
    draft.request_mut().body = bytes.clone().into();
    let call = draft.seal().await.unwrap();
    assert_eq!(
        call.request()
            .headers
            .iter()
            .find(|(k, _)| k == "signed-body")
            .unwrap()
            .1
            .as_bytes(),
        bytes
    );
    let result = call
        .dispatch_once_with(|| {
            Err(LlmError::InvalidRequest {
                message: "budget rejected".into(),
            })
        })
        .await;
    assert!(result.is_err());
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn finalizer_runs_before_authentication_and_exact_override_rejects_wrong_leaf() {
    use lingxi_llm_client::protocol::AuthStrategy;
    #[derive(Debug)]
    struct Finalizer;
    impl lingxi_llm_client::client::options::RequestFinalizer for Finalizer {
        fn finalize(&self, req: &mut HttpRequest, _: &ProviderProfile) -> Result<(), LlmError> {
            req.body = bytes::Bytes::from_static(b"{\"final\":true}");
            Ok(())
        }
    }
    let http = http(200, "{}", false);
    let mut profiles = profiles("open_ai_chat");
    profiles[0].auth = AuthStrategy::Bearer;
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &profiles);
    builder.register_authenticator(AuthStrategy::Bearer, Arc::new(BodySigner));
    let client = builder.with_region(Region::International).build().unwrap();
    let call = client
        .prepare_on(
            "primary",
            &request(),
            &RequestOptions {
                finalizer: Some(Arc::new(Finalizer)),
                ..Default::default()
            },
            RequestMode::Complete,
        )
        .await
        .unwrap();
    assert_eq!(
        call.request()
            .headers
            .iter()
            .find(|(k, _)| k == "signed-body")
            .unwrap()
            .1,
        "{\"final\":true}"
    );
    call.dispatch_once_with(|| {
        assert_eq!(http.sends.load(Ordering::SeqCst), 0);
        Ok(())
    })
    .await
    .unwrap();
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
    assert!(lingxi_llm_client::exact_json::serialize(
        &json!({"x":1}),
        &[("/x".into(), vec![65])].into_iter().collect()
    )
    .is_err());
}

#[test]
fn host_can_preserve_native_slash_id_precedence_without_duplicating_resolution() {
    let mut profiles = profiles("open_ai_chat");
    profiles[1].models[0].request_model = "primary/wire".into();
    profiles[1].models[0].display_model = "native".into();
    let catalog = lingxi_llm_client::client::RoutingCatalog::new(profiles, Region::International);
    assert!(catalog.resolve_in("primary/wire", None).is_err());
    assert_eq!(
        catalog
            .resolve_in_prefer_native("primary/wire", None)
            .unwrap()
            .profile_name,
        "secondary"
    );
}

#[tokio::test]
async fn final_body_controls_and_exact_utf16_cannot_silently_price_fast_as_standard() {
    let http = http(
        200,
        r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#,
        false,
    );
    let client = LlmClientBuilder::with_transport(http, &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    draft
        .set_json_body(
            json!({"model":"wire","service_tier":"priority","text":"display"}),
            &[("/text".into(), vec![0xd800])].into_iter().collect(),
        )
        .unwrap();
    let received = draft
        .seal()
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        received.inference_report().requested_service_tier,
        Some(lingxi_llm_client::protocol::ServiceTier::Fast)
    );
    assert!(received
        .pricing_snapshot()
        .estimate(
            received.usage_report(),
            received.inference_report(),
            lingxi_llm_client::protocol::Submission::Interactive
        )
        .is_err());
}
