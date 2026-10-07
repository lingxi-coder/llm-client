use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, stream};
use lingxi_llm_client::{
    HttpRequest, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse, Transport,
    protocol::{
        AuthStrategy, ChatRequest, ContentBlock, ConversationMessage, LlmError, MessageRole,
        ProviderProfile, Region, Secret, UsageState,
    },
};
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
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
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"test", "messages":[]})).unwrap()
}
fn request_with_lone_surrogate() -> ChatRequest {
    let mut request = request();
    request.messages.push(ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::TextJsUtf16 {
            text: "�".into(),
            utf16_code_units: vec![0xd800],
            thought_signature: None,
            citations: None,
        }],
    });
    request
}

fn assert_exact_text_request(http: &ResponseTransport, expected_count: usize) {
    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), expected_count);
    for request in requests.iter() {
        let body = std::str::from_utf8(&request.body).unwrap();
        assert!(body.contains("\\ud800"), "{body}");
        assert!(!body.contains("�"), "{body}");
        assert!(!body.contains("utf16_code_units"), "{body}");
        assert!(!body.contains("text_js_utf16"), "{body}");
    }
}

#[tokio::test]
async fn sdk_carried_utf16_replays_on_prepared_and_normal_generation_paths() {
    let request = request_with_lone_surrogate();
    for protocol in [
        "anthropic_messages",
        "open_ai_chat",
        "open_ai_responses",
        "gemini_generate_content",
    ] {
        let http = http(200, "{}", false);
        let client = LlmClientBuilder::with_transport(http, &profiles(protocol))
            .with_region(Region::International)
            .build()
            .unwrap();
        let prepared = client
            .prepare_on(
                "primary",
                &request,
                &RequestOptions::default(),
                RequestMode::Complete,
            )
            .await
            .unwrap();
        let prepared_body = std::str::from_utf8(&prepared.request().body).unwrap();
        assert!(
            prepared_body.contains("\\ud800"),
            "{protocol}: {prepared_body}"
        );
        assert!(
            !prepared_body.contains("utf16_code_units"),
            "{protocol}: {prepared_body}"
        );
        assert!(
            !prepared_body.contains("text_js_utf16"),
            "{protocol}: {prepared_body}"
        );
    }

    let http = http(
        200,
        r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        false,
    );
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    client
        .snapshot()
        .chat()
        .complete_in("primary", &request, &RequestOptions::default())
        .await
        .unwrap();
    assert_exact_text_request(&http, 1);

    let _stream = client
        .snapshot()
        .chat()
        .stream_in("primary", &request, &RequestOptions::default())
        .await
        .unwrap();
    assert_exact_text_request(&http, 2);
}

#[tokio::test]
async fn decoded_anthropic_text_replays_with_exact_units_on_the_next_request() {
    let http = http(
        200,
        br#"{"id":"msg_1","model":"test","stop_reason":"end_turn","content":[{"type":"text","text":"A\ud800B"}],"usage":{"input_tokens":1,"output_tokens":1}}"#.as_slice(),
        false,
    );
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("anthropic_messages"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut first_request = request_with_lone_surrogate();
    first_request.max_tokens = Some(256);
    let response = client
        .snapshot()
        .chat()
        .complete_in("primary", &first_request, &RequestOptions::default())
        .await
        .unwrap();
    assert!(matches!(
        response.message.content.as_slice(),
        [ContentBlock::TextJsUtf16 { text, utf16_code_units, .. }]
            if text == "A�B" && utf16_code_units == &[0x41, 0xd800, 0x42]
    ));

    first_request.messages.push(response.message);
    let prepared = client
        .prepare_on(
            "primary",
            &first_request,
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let body = std::str::from_utf8(&prepared.request().body).unwrap();
    assert_eq!(body.matches("\\ud800").count(), 2, "{body}");
    assert!(!body.contains("utf16_code_units"), "{body}");
    assert!(!body.contains("text_js_utf16"), "{body}");
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
    assert!(
        requests[0]
            .url
            .ends_with("/v1/messages/count_tokens?beta=true")
    );
    assert!(
        requests[0]
            .headers
            .iter()
            .any(|(key, value)| key.eq_ignore_ascii_case("anthropic-beta")
                && value == "token-counting-2024-11-01")
    );
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
async fn exact_count_materializes_virtual_parameters_before_dispatch() {
    for (betas, expected) in [
        (
            json!(["one", null, ["two", "three"]]),
            "one,,two,three,token-counting-2024-11-01",
        ),
        (json!("ab"), "a,b,token-counting-2024-11-01"),
        (serde_json::Value::Null, "token-counting-2024-11-01"),
    ] {
        let http = http(200, r#"{"input_tokens":42}"#, false);
        let mut profiles = profiles("anthropic_messages");
        profiles[0].extra =
            json!({"body": {"betas":betas,"user_profile_id":"profile","workspace_id":"workspace"}});
        let client = LlmClientBuilder::with_transport(http.clone(), &profiles)
            .with_region(Region::International)
            .build()
            .unwrap();
        assert_eq!(
            client
                .count_tokens_exact_in("primary", &request(), &RequestOptions::default())
                .await
                .unwrap(),
            Some(42)
        );
        let requests = http.requests.lock().unwrap();
        let wire = &requests[0];
        let header = |name: &str| {
            wire.headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(header("anthropic-beta"), Some(expected));
        assert_eq!(header("anthropic-user-profile-id"), Some("profile"));
        assert_eq!(header("anthropic-workspace-id"), Some("workspace"));
        let body: serde_json::Value = serde_json::from_slice(&wire.body).unwrap();
        for field in ["betas", "user_profile_id", "workspace_id"] {
            assert!(body.get(field).is_none());
        }
    }
}

#[tokio::test]
async fn sdk_dispatch_preserves_hook_prompt_kind_override_on_non_anthropic_codec() {
    let http = http(200, "{}", false);
    let client = LlmClientBuilder::with_transport(http.clone(), &profiles("open_ai_chat"))
        .with_region(Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"test",
        "messages":[{"role":"user","content":[{"type":"text","text":"\u{fffd}"}]}]
    }))
    .unwrap();
    let options = RequestOptions {
        anthropic_request_kind:
            lingxi_llm_client::providers::anthropic::request_policy::AnthropicRequestKind::HookPrompt,
        message_text_utf16_overrides: [("/messages/0/content/0/text".into(), vec![0xd800])]
            .into_iter()
            .collect(),
        ..Default::default()
    };

    let _response = client
        .prepare_on("primary", &request, &options, RequestMode::Complete)
        .await
        .unwrap()
        .dispatch_once()
        .await
        .unwrap();

    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body = std::str::from_utf8(&requests[0].body).unwrap();
    assert!(body.contains("\\ud800"), "{body}");
    assert!(!body.contains('\u{fffd}'), "{body}");
}

#[tokio::test]
async fn chatgpt_oauth_body_policy_precedes_exact_responses_serialization() {
    let transport = http(200, "{}", false);
    let mut profiles = profiles("open_ai_responses");
    profiles[0].auth = AuthStrategy::ChatGptOAuth;
    let mut builder = LlmClientBuilder::with_transport(transport, &profiles);
    builder.register_authenticator(
        AuthStrategy::ChatGptOAuth,
        Arc::new(lingxi_llm_client::BearerAuthenticator),
    );
    let client = builder.with_region(Region::International).build().unwrap();

    let mut input = ChatRequest::new("test");
    input
        .messages
        .push(lingxi_llm_client::protocol::ConversationMessage::user_text(
            "�",
        ));
    input.max_tokens = Some(96);
    input.temperature = Some(0.4);
    let options = RequestOptions {
        credential: Some(Secret::new("synthetic-chatgpt-token".into())),
        message_text_utf16_overrides: [("/messages/0/content/0/text".into(), vec![0xd800])]
            .into_iter()
            .collect(),
        ..Default::default()
    };

    let mut draft = client
        .prepare_draft_on("primary", &input, &options, RequestMode::Complete)
        .await
        .unwrap();
    let mut overrides = draft.message_json_string_overrides().clone();
    let mut body = draft.semantic_body_json().unwrap();
    // Simulate the final host body policy after request preparation.
    body["instructions"] = json!("�");
    body["metadata"] = json!({"annotation":"�"});
    body["max_output_tokens"] = json!("discarded exact field");
    body["temperature"] = json!(0.4);
    body["top_p"] = json!({"annotation":"discarded exact descendant"});
    overrides.insert("/instructions".into(), vec![0xd801]);
    overrides.insert("/metadata/annotation".into(), vec![0xd802]);
    overrides.insert("/max_output_tokens".into(), vec![0xd803]);
    overrides.insert("/top_p/annotation".into(), vec![0xd804]);
    assert!(
        !draft
            .message_json_string_overrides()
            .contains_key("/instructions")
    );
    draft.set_json_body(body, &overrides).unwrap();
    assert!(
        !draft
            .message_json_string_overrides()
            .contains_key("/metadata/annotation")
    );
    assert_eq!(
        draft
            .request_json_string_overrides()
            .get("/metadata/annotation"),
        Some(&vec![0xd802])
    );
    let exact_body = draft.request().body.clone();
    let semantic_body = draft.semantic_body_json().unwrap();
    draft.set_body_bytes(exact_body, semantic_body);
    let call = draft.seal().await.unwrap();

    let bytes = &call.request().body;
    let wire = std::str::from_utf8(bytes).unwrap();
    assert!(wire.contains("\\ud800"), "{wire}");
    assert!(wire.contains("\"store\":false"), "{wire}");
    assert!(wire.contains("\"instructions\":\"\\ud801\""), "{wire}");
    assert!(wire.contains("\\ud802"), "{wire}");
    assert!(!wire.contains("\\ud803"), "{wire}");
    assert!(!wire.contains("\\ud804"), "{wire}");
    for removed in ["max_output_tokens", "temperature", "top_p"] {
        assert!(!wire.contains(removed), "{wire}");
    }
    assert!(
        call.request()
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                && value == "Bearer synthetic-chatgpt-token")
    );
}

#[tokio::test]
async fn selected_prices_survive_removing_the_profile_during_an_attempt() {
    use lingxi_llm_client::protocol::{InferenceReport, Submission, Usage, UsageReport};
    let http = http(200, "{}", false);
    let mut profiles = profiles("open_ai_chat");
    profiles[0].models[0].pricing = Some(
        serde_json::from_value(json!({"input_per_million":1.0,"output_per_million":2.0})).unwrap(),
    );
    let (client, manager) = LlmClientBuilder::with_transport(http, &profiles)
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let directory = std::env::temp_dir().join(format!(
        "llm-pricing-snapshot-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    manager.set_config_dir(&directory).await.unwrap();
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
    manager.remove_provider("primary").await.unwrap();
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
    let http = http(
        200,
        "data: {\"model\":\"wire\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
        true,
    );
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
    let http = http(
        200,
        "data: {\"error\":{\"message\":\"failed\"},\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
        true,
    );
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
    assert!(
        client
            .prepare_on(
                "test",
                &request(),
                &RequestOptions::default(),
                RequestMode::Complete
            )
            .await
            .is_err()
    );
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
    assert!(
        !draft
            .request()
            .headers
            .iter()
            .any(|(k, _)| k == "signed-body")
    );
    let bytes = lingxi_llm_client::exact_json::serialize(
        &json!({"text":"display"}),
        &[("/text".into(), vec![0xd800, 65, 0xd83d, 0xde00])]
            .into_iter()
            .collect(),
        lingxi_llm_client::exact_json::JsonEncoding::JavaScript,
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
    assert!(
        lingxi_llm_client::exact_json::serialize(
            &json!({"x":1}),
            &[("/x".into(), vec![65])].into_iter().collect(),
            lingxi_llm_client::exact_json::JsonEncoding::JavaScript,
        )
        .is_err()
    );
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
    assert!(
        received
            .pricing_snapshot()
            .estimate(
                received.usage_report(),
                received.inference_report(),
                lingxi_llm_client::protocol::Submission::Interactive
            )
            .is_err()
    );
}

#[tokio::test]
async fn native_client_request_id_uses_seal_and_fetch_fallback_stages() {
    use lingxi_llm_client::protocol::ProtocolFamily;
    use lingxi_llm_client::providers::response_headers::{
        AnthropicAwsBaseUrlFact, FirstPartyBaseUrlFact, NativeAnthropicProvider,
        NativeAnthropicRequestHeaderFacts,
    };

    let direct_http = http(200, "{}", false);
    let mut direct_profiles = profiles("anthropic_messages");
    for profile in &mut direct_profiles {
        profile.provider_id = "anthropic".into();
        profile.base_url = "https://api.anthropic.com".into();
    }
    let direct_client = LlmClientBuilder::with_transport(direct_http.clone(), &direct_profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut direct_draft = direct_client
        .prepare_draft_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    direct_draft
        .set_native_anthropic_request_header_facts(NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::AnthropicApiHost,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        })
        .unwrap();
    let direct = direct_draft.seal().await.unwrap();
    let direct_id = direct.client_request_id().unwrap().to_owned();
    assert_uuid_v4(&direct_id);
    let received = direct
        .dispatch_once_using(direct_http.as_ref(), || Ok(()))
        .await
        .unwrap();
    assert_eq!(received.client_request_id(), Some(direct_id.as_str()));
    assert_eq!(
        direct_http.requests.lock().unwrap()[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
            .map(|(_, value)| value.as_str()),
        Some(direct_id.as_str())
    );

    let aws_http = http(200, "{}", false);
    let mut aws_profiles = profiles("anthropic_messages");
    for profile in &mut aws_profiles {
        profile.provider_id = "anthropicAws".into();
        profile.base_url = "https://bedrock-gateway.example".into();
    }
    let aws_client = LlmClientBuilder::with_transport(aws_http.clone(), &aws_profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut aws_draft = aws_client
        .prepare_draft_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    aws_draft
        .set_native_anthropic_request_header_facts(NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::AnthropicAws,
            first_party_base_url: FirstPartyBaseUrlFact::Custom,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::DefinedEmpty,
        })
        .unwrap();
    let aws = aws_draft.seal().await.unwrap();
    assert_eq!(aws.client_request_id(), None);
    let received = aws
        .dispatch_once_using(aws_http.as_ref(), || Ok(()))
        .await
        .unwrap();
    let aws_id = received.client_request_id().unwrap().to_owned();
    assert_uuid_v4(&aws_id);
    assert_eq!(
        aws_http.requests.lock().unwrap()[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
            .map(|(_, value)| value.as_str()),
        Some(aws_id.as_str())
    );
}

#[tokio::test]
async fn native_client_request_id_preserves_empty_caller_header_and_custom_route() {
    use lingxi_llm_client::protocol::ProtocolFamily;
    use lingxi_llm_client::providers::response_headers::{
        AnthropicAwsBaseUrlFact, FirstPartyBaseUrlFact, NativeAnthropicProvider,
        NativeAnthropicRequestHeaderFacts,
    };

    let http = http(200, "{}", false);
    let mut provider_profiles = profiles("anthropic_messages");
    for profile in &mut provider_profiles {
        profile.provider_id = "anthropic".into();
        profile.base_url = "https://gateway.example".into();
    }
    let client = LlmClientBuilder::with_transport(http.clone(), &provider_profiles)
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
        .request_mut()
        .headers
        .push(("X-Client-Request-Id".into(), String::new()));
    draft
        .set_native_anthropic_request_header_facts(NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        })
        .unwrap();
    let call = draft.seal().await.unwrap();
    assert_eq!(call.client_request_id(), Some(""));
    call.dispatch_once_using(http.as_ref(), || Ok(()))
        .await
        .unwrap();
    let requests = http.requests.lock().unwrap();
    let id_headers: Vec<_> = requests[0]
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
        .collect();
    assert_eq!(id_headers.len(), 1);
    assert_eq!(id_headers[0].1, "");
}

#[tokio::test]
async fn native_count_tokens_uses_fetch_fallback_without_per_attempt_id() {
    use lingxi_llm_client::protocol::ProtocolFamily;
    use lingxi_llm_client::providers::response_headers::{
        AnthropicAwsBaseUrlFact, FirstPartyBaseUrlFact, NativeAnthropicProvider,
        NativeAnthropicRequestHeaderFacts,
    };

    let transport = http(200, "{}", false);
    let client =
        LlmClientBuilder::with_transport(transport.clone(), &profiles("anthropic_messages"))
            .with_region(Region::International)
            .build()
            .unwrap();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &RequestOptions::default(),
            RequestMode::CountTokens,
        )
        .await
        .unwrap();
    draft
        .set_native_anthropic_request_header_facts(NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::AnthropicApiHost,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        })
        .unwrap();
    let call = draft.seal().await.unwrap();
    assert_eq!(call.client_request_id(), None);
    let received = call
        .dispatch_once_using(transport.as_ref(), || Ok(()))
        .await
        .unwrap();
    let client_request_id = received.client_request_id().unwrap().to_owned();
    assert_uuid_v4(&client_request_id);
    let sent = transport.requests.lock().unwrap();
    assert!(sent[0].url.contains("/v1/messages/count_tokens"));
    assert_eq!(
        sent[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
            .map(|(_, value)| value.as_str()),
        Some(client_request_id.as_str())
    );
}

#[tokio::test]
async fn native_traceparent_is_inserted_after_the_per_attempt_gate() {
    use lingxi_llm_client::protocol::ProtocolFamily;
    use lingxi_llm_client::providers::response_headers::{
        AnthropicAwsBaseUrlFact, FirstPartyBaseUrlFact, NativeAnthropicProvider,
        NativeAnthropicRequestHeaderFacts,
    };

    const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let transport = http(200, "{}", false);
    let mut provider_profiles = profiles("anthropic_messages");
    for profile in &mut provider_profiles {
        profile.provider_id = "anthropic".into();
        profile.base_url = "https://api.anthropic.com".into();
    }
    let client = LlmClientBuilder::with_transport(transport, &provider_profiles)
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
        .set_native_anthropic_request_header_facts(NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::AnthropicApiHost,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        })
        .unwrap();
    draft.set_native_traceparent(TRACEPARENT.into()).unwrap();
    let call = draft.seal().await.unwrap();
    assert_eq!(
        call.request()
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("traceparent"))
            .map(|(_, value)| value.as_str()),
        Some(TRACEPARENT)
    );
}

fn assert_uuid_v4(value: &str) {
    assert_eq!(value.len(), 36);
    assert_eq!(&value[8..9], "-");
    assert_eq!(&value[13..14], "-");
    assert_eq!(&value[18..19], "-");
    assert_eq!(&value[23..24], "-");
    assert_eq!(&value[14..15], "4");
    assert!(matches!(&value[19..20], "8" | "9" | "a" | "b"));
}

struct MergeClock(std::sync::atomic::AtomicU64);
impl lingxi_llm_client::transport::Clock for MergeClock {
    fn now(&self) -> std::time::SystemTime {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(self.0.load(Ordering::SeqCst))
    }
}

#[tokio::test]
async fn prepared_file_expiry_is_rechecked_before_the_dispatch_marker() {
    use lingxi_llm_client::{files::provider_file_endpoint_fingerprint, protocol::*};
    let transport = http(200, "{}", false);
    let mut profile = profiles("open_ai_responses").remove(0);
    profile.provider_id = "openai".into();
    profile.base_url = "https://api.openai.com/v1".into();
    profile.models[0].capability_support =
        Some(serde_json::from_value(json!({"documents":"supported"})).unwrap());
    let clock = Arc::new(MergeClock(std::sync::atomic::AtomicU64::new(2_000_000_000)));
    let mut builder = LlmClientBuilder::with_transport(transport.clone(), &[profile.clone()]);
    builder.with_clock(clock.clone());
    let client = builder.with_region(Region::International).build().unwrap();
    let mut request = request();
    request.messages = vec![serde_json::from_value(json!({"role":"user","content":[{
        "type":"document","source":{"type":"provider_file","file":{
            "provider_id":"openai","profile_name":"primary","protocol":"open_ai_responses",
            "endpoint_fingerprint":provider_file_endpoint_fingerprint(&profile.base_url),
            "account_scope":"acct","file_id":"file_123","media_type":"application/pdf","expires_at":"2000000001"
        }}
    }]})).unwrap()];
    let options = RequestOptions {
        file_account_scope: Some("acct".into()),
        ..Default::default()
    };
    let call = client
        .prepare_on("primary", &request, &options, RequestMode::Complete)
        .await
        .unwrap();
    clock.0.store(2_000_000_001, Ordering::SeqCst);
    let marked = AtomicUsize::new(0);
    assert!(matches!(
        call.dispatch_once_with(|| {
            marked.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert_eq!(marked.load(Ordering::SeqCst), 0);
    assert_eq!(transport.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn prepared_batches_preserve_scoped_continuation_and_frozen_prices() {
    let wire = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_merge\",\"model\":\"wire\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_merge\",\"model\":\"wire\",\"status\":\"completed\",\"output\":[]}}\n\n"
    );
    let transport = http(200, wire, false);
    let mut profiles = profiles("open_ai_responses");
    profiles[0].extra["supports_previous_response_id"] = json!(true);
    let client = LlmClientBuilder::with_transport(transport, &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        account_scope: Some("acct".into()),
        ..Default::default()
    };
    let call = client
        .prepare_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    let mut stream = call
        .dispatch_once()
        .await
        .unwrap()
        .into_stream()
        .ok()
        .unwrap();
    while let Some(batch) = stream.next_batch().await {
        for event in batch.events {
            event.unwrap();
        }
    }
    assert!(stream.pricing_snapshot().is_some());
    assert_eq!(
        stream.continuation().unwrap().response_id.as_str(),
        "resp_merge"
    );
    assert_eq!(stream.continuation().unwrap().account_scope, "acct");
}

#[tokio::test]
async fn host_controls_and_native_tools_survive_borrowed_body_encoding() {
    let transport = http(200, "{}", false);
    let client =
        LlmClientBuilder::with_transport(transport.clone(), &profiles("anthropic_messages"))
            .with_region(Region::International)
            .build()
            .unwrap();
    let mut request = request();
    request.controls.top_p = Some(0.7);
    request.tools = vec![
        serde_json::from_value(json!({
            "name":"computer","tool_type":"computer_20250124","description":"desktop",
            "input_schema":{},"extra":{"display_width_px":1024,"display_height_px":768}
        }))
        .unwrap(),
    ];
    let call = client
        .prepare_on(
            "primary",
            &request,
            &RequestOptions::default(),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(body["top_p"], 0.7);
    assert_eq!(body["tools"][0]["type"], "computer_20250124");
    assert_eq!(body["tools"][0]["display_width_px"], 1024);
    assert!(body["tools"][0].get("input_schema").is_none());
    assert!(
        call.request().headers.iter().any(
            |(key, value)| key == "anthropic-beta" && value.contains("computer-use-2025-01-24")
        )
    );
    assert_eq!(transport.sends.load(Ordering::SeqCst), 0);
}
