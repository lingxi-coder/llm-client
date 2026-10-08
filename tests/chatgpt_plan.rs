use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::client::options::{RequestAuthenticator, RequestFinalizer};
use lingxi_llm_client::{
    codecs::openai::responses::classify_error,
    protocol::{
        AttachmentRef, AuthStrategy, ChatRequest, ContentBlock, ImageSource, LlmError,
        LlmErrorKind, ProviderProfile, Region, Secret, StreamEvent,
    },
    providers::openai::{
        chatgpt_plan::ChatGptPlanModelError, containers::OpenAiContainerScope, OpenAiClient,
    },
    transport::WebSocketConnection,
    Authenticator, ChatGptPlanAuthenticator, HttpRequest, LlmClientBuilder, RequestMode,
    RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"chatgpt-plan",
        "base_url":"https://api.openai.com/v1", "protocol":"open_ai_responses",
        "auth":"chat_gpt_plan", "model_list":"none",
        "pricing":{"billingMode":"subscription"},
        "models":[{"display_model":"gpt-6.1-sol","request_model":"gpt-6.1-sol","billing_model":"gpt-6.1-sol"}]
    }))
    .unwrap()
}

fn request(body: Value) -> HttpRequest {
    HttpRequest {
        http1_header_layout: None,
        method: "POST".into(),
        url: "https://api.openai.com/v1/responses".into(),
        headers: vec![],
        body: serde_json::to_vec(&body).unwrap().into(),
        timeout: None,
    }
}

fn valid_body() -> Value {
    json!({"model":"gpt-6.1-sol", "input":[{"role":"user","content":"Hi"}], "store":false, "stream":true})
}

fn chat(store: bool) -> ChatRequest {
    let mut chat = ChatRequest::new("gpt-6.1-sol");
    chat.messages
        .push(lingxi_llm_client::protocol::ConversationMessage::user_text(
            "Hi",
        ));
    chat.controls.responses.store = Some(store);
    chat
}

fn change_model(request: &mut HttpRequest) {
    let mut body: Value = serde_json::from_slice(&request.body).unwrap();
    body["model"] = json!("gpt-6-astra");
    request.body = serde_json::to_vec(&body).unwrap().into();
}

#[derive(Debug)]
struct ModelFinalizer;

impl RequestFinalizer for ModelFinalizer {
    fn finalize(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
    ) -> Result<(), LlmError> {
        change_model(request);
        Ok(())
    }
}

#[tokio::test]
async fn plan_rejects_model_changes_after_route_selection() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        credential: Some(Secret::new("plan-token".into())),
        ..Default::default()
    };
    let mut draft = client
        .prepare_draft_on("chatgpt-plan", &chat(false), &options, RequestMode::Stream)
        .await
        .unwrap();
    change_model(draft.request_mut());
    assert!(draft.seal().await.is_err());

    let finalized = RequestOptions {
        finalizer: Some(Arc::new(ModelFinalizer)),
        ..options
    };
    assert!(client
        .chat()
        .stream_in("chatgpt-plan", &chat(false), &finalized)
        .await
        .is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
}

struct PermissiveAuthenticator;

#[async_trait]
impl Authenticator for PermissiveAuthenticator {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        request
            .headers
            .push(("authorization".into(), "Bearer custom-token".into()));
        Ok(())
    }
}

struct MutatingAuthenticator;

#[async_trait]
impl Authenticator for MutatingAuthenticator {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _profile: &ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        let mut body: Value = serde_json::from_slice(&request.body).unwrap();
        body["store"] = json!(true);
        request.body = serde_json::to_vec(&body).unwrap().into();
        request
            .headers
            .push(("authorization".into(), "Bearer custom-token".into()));
        Ok(())
    }
}

#[tokio::test]
async fn custom_authenticator_cannot_skip_final_plan_validation() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let options = RequestOptions {
        authenticator: Some(RequestAuthenticator(Arc::new(PermissiveAuthenticator))),
        ..Default::default()
    };
    assert!(client
        .chat()
        .stream_in("chatgpt-plan", &chat(true), &options)
        .await
        .is_err());
    let draft = client
        .prepare_draft_on("chatgpt-plan", &chat(true), &options, RequestMode::Stream)
        .await
        .unwrap();
    assert!(draft.seal().await.is_err());
    let mutating = RequestOptions {
        authenticator: Some(RequestAuthenticator(Arc::new(MutatingAuthenticator))),
        ..Default::default()
    };
    assert!(client
        .chat()
        .stream_in("chatgpt-plan", &chat(false), &mutating)
        .await
        .is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn plan_auth_only_accepts_the_documented_public_streaming_request() {
    let profile = profile();
    let token = Secret::new("account-token".to_owned());
    let mut allowed = request(valid_body());
    ChatGptPlanAuthenticator
        .apply(&mut allowed, &profile, Some(&token))
        .await
        .unwrap();
    assert!(allowed.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer account-token"
    }));

    for (name, value) in [
        ("store", json!(true)),
        ("stream", json!(false)),
        ("max_output_tokens", json!(200)),
        ("previous_response_id", json!("resp_1")),
        ("tools", json!([{"type":"file_search"}])),
        ("tools", json!([{"type":"function","name":"run"}])),
    ] {
        let mut body = valid_body();
        body[name] = value;
        let mut rejected = request(body);
        assert!(
            ChatGptPlanAuthenticator
                .apply(&mut rejected, &profile, Some(&token))
                .await
                .is_err(),
            "{name}"
        );
        assert!(!rejected
            .headers
            .iter()
            .any(|(header, _)| header.eq_ignore_ascii_case("authorization")));
    }

    let mut system = valid_body();
    system["input"] = json!([{"role":"system","content":"hidden"}]);
    assert!(ChatGptPlanAuthenticator
        .apply(&mut request(system), &profile, Some(&token))
        .await
        .is_err());

    let mut private_url = request(valid_body());
    private_url.url = "https://chatgpt.com/backend-api/codex/responses".into();
    assert!(ChatGptPlanAuthenticator
        .apply(&mut private_url, &profile, Some(&token))
        .await
        .is_err());

    for header in [
        "ChatGPT-Account-ID",
        "X-OpenAI-Fedramp",
        "x-api-key",
        "OpenAI-Project",
    ] {
        let mut mixed = request(valid_body());
        mixed
            .headers
            .push((header.into(), "stale-account-or-key".into()));
        assert!(
            ChatGptPlanAuthenticator
                .apply(&mut mixed, &profile, Some(&token))
                .await
                .is_err(),
            "{header}"
        );
    }
}

#[tokio::test]
async fn plan_rejects_unsupported_nested_inputs_and_tools() {
    let profile = profile();
    let token = Secret::new("plan-token".into());
    let forbidden_inputs = [
        json!([{"type":"message","role":"user","content":[{"type":"input_audio","input_audio":{"data":"abc","format":"wav"}}]}]),
        json!([{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file_123"}]}]),
        json!([{"type":"additional_tools","role":"developer","tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]}]),
        json!([{"type":"mcp_approval_request","id":"mcpr_1","server_label":"remote"}]),
        json!([{"type":"mcp_approval_response","approval_request_id":"mcpr_1","approve":true}]),
        json!([{"type":"future_hosted_call","id":"hosted_1"}]),
    ];
    for input in forbidden_inputs {
        let mut body = valid_body();
        body["input"] = input;
        assert!(ChatGptPlanAuthenticator
            .apply(&mut request(body), &profile, Some(&token))
            .await
            .is_err());
    }
    let mut body = valid_body();
    body["tools"] = json!([{"type":"namespace","name":"local","description":"local tools","tools":[{"type":"file_search"}]}]);
    assert!(ChatGptPlanAuthenticator
        .apply(&mut request(body), &profile, Some(&token))
        .await
        .is_err());

    let mut supported = valid_body();
    supported["input"] = json!([{"type":"additional_tools","role":"developer","tools":[{"type":"function","name":"local","parameters":{"type":"object"}}]}]);
    supported["tools"] = json!([{"type":"namespace","name":"local","description":"local tools","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}]);
    ChatGptPlanAuthenticator
        .apply(&mut request(supported), &profile, Some(&token))
        .await
        .unwrap();
}

#[tokio::test]
async fn draft_rejects_hosted_mcp_approval_before_dispatch() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut draft = client
        .prepare_draft_on(
            "chatgpt-plan",
            &chat(false),
            &RequestOptions {
                credential: Some(Secret::new("plan-token".into())),
                ..Default::default()
            },
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let mut body: Value = serde_json::from_slice(&draft.request().body).unwrap();
    body["input"].as_array_mut().unwrap().push(json!({
        "type":"mcp_approval_response", "approval_request_id":"mcpr_1", "approve":true
    }));
    draft.request_mut().body = serde_json::to_vec(&body).unwrap().into();
    assert!(matches!(
        draft.seal().await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[derive(Default)]
struct RejectableSocket {
    sends: usize,
}

#[async_trait]
impl WebSocketConnection for RejectableSocket {
    async fn send(&mut self, _payload: Bytes) -> Result<StreamResponse, LlmError> {
        self.sends += 1;
        Err(LlmError::Transport {
            message: "unexpected WebSocket send".into(),
        })
    }

    async fn close(&mut self) -> Result<(), LlmError> {
        Ok(())
    }
}

#[tokio::test]
async fn sealed_plan_call_rejects_externally_supplied_websocket() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let call = client
        .prepare_draft_on(
            "chatgpt-plan",
            &chat(false),
            &RequestOptions {
                credential: Some(Secret::new("plan-token".into())),
                ..Default::default()
            },
            RequestMode::Stream,
        )
        .await
        .unwrap()
        .seal()
        .await
        .unwrap();
    let mut socket = RejectableSocket::default();
    let dispatched = AtomicUsize::new(0);
    let result = call
        .dispatch_websocket_once_with(&mut socket, || {
            dispatched.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .await;
    assert!(matches!(
        result,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(socket.sends, 0);
    assert_eq!(dispatched.load(Ordering::SeqCst), 0);
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn raw_provider_audio_cannot_bypass_plan_input_validation() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut request = chat(false);
    request.messages[0].content.push(ContentBlock::ProviderContent {
        protocol: lingxi_llm_client::protocol::ProtocolFamily::OpenAiResponses,
        value: json!({"type":"message","role":"user","content":[{"type":"input_audio","input_audio":{"data":"abc","format":"wav"}}]}),
    });
    assert!(client
        .chat()
        .stream_in(
            "chatgpt-plan",
            &request,
            &RequestOptions {
                credential: Some(Secret::new("plan-token".into())),
                ..Default::default()
            },
        )
        .await
        .is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[derive(Default)]
struct ModelTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for ModelTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let is_inference = request.method == "POST";
        self.requests.lock().unwrap().push(request);
        let models = json!({"models":[
            {"slug":"gpt-6.1-sol","display_name":"GPT-6.1 Sol","visibility":"list"},
            {"slug":"internal-model","display_name":"Internal","visibility":"hidden"}
        ]});
        let body = if is_inference {
            Bytes::from_static(b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-6.1-sol\",\"status\":\"completed\",\"output\":[]}}\n\n")
        } else {
            Bytes::from(serde_json::to_vec(&models).unwrap())
        };
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: stream::once(async move { Ok(body) }).boxed(),
        })
    }
}

#[tokio::test]
async fn plan_models_use_the_public_endpoint_and_selected_account_token() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let openai = client.provider::<OpenAiClient>("chatgpt-plan").unwrap();
    let models = openai
        .chatgpt_plan_models(&Secret::new("selected-account-token".into()))
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].slug, "gpt-6.1-sol");
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].url, "https://api.openai.com/v1/models");
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer selected-account-token"));
}

#[tokio::test]
async fn prepared_plan_call_requires_stream_and_no_storage() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut chat = ChatRequest::new("gpt-6.1-sol");
    chat.messages
        .push(lingxi_llm_client::protocol::ConversationMessage::user_text(
            "Hi",
        ));
    let options = RequestOptions {
        credential: Some(Secret::new("token".into())),
        ..Default::default()
    };
    let draft = client
        .prepare_draft_on("chatgpt-plan", &chat, &options, RequestMode::Stream)
        .await
        .unwrap();
    assert!(draft.seal().await.is_err());
    chat.controls.responses.store = Some(false);
    let draft = client
        .prepare_draft_on("chatgpt-plan", &chat, &options, RequestMode::Stream)
        .await
        .unwrap();
    let call = draft.seal().await.unwrap();
    assert_eq!(call.request().url, "https://api.openai.com/v1/responses");
    let _stream = client
        .chat()
        .stream_in("chatgpt-plan", &chat, &options)
        .await
        .unwrap();
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[0].url, "https://api.openai.com/v1/responses");
    assert!(
        requests[0]
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                && value == "Bearer token")
    );
    let sent: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(sent["store"], false);
    assert_eq!(sent["stream"], true);
}

#[tokio::test]
async fn plan_rejects_attachment_before_any_upload_or_inference() {
    let transport = Arc::new(ModelTransport::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut chat = ChatRequest::new("gpt-6.1-sol");
    let mut message = lingxi_llm_client::protocol::ConversationMessage::user_text("inspect");
    message.content.push(ContentBlock::Image {
        source: ImageSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "image-1".into(),
                revision: "r1".into(),
                filename: "image.png".into(),
                media_type: "image/png".into(),
                size_bytes: 10,
            },
        },
    });
    chat.messages.push(message);
    chat.controls.responses.store = Some(false);
    let result = client
        .prepare_draft_on(
            "chatgpt-plan",
            &chat,
            &RequestOptions {
                credential: Some(Secret::new("token".into())),
                ..Default::default()
            },
            RequestMode::Stream,
        )
        .await;
    assert!(result.is_err());
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[test]
fn plan_profile_rejects_private_endpoint_and_websocket() {
    let mut invalid = profile();
    invalid.base_url = "https://chatgpt.com/backend-api/codex".into();
    assert!(
        LlmClientBuilder::with_transport(Arc::new(ModelTransport::default()), &[invalid])
            .with_region(Region::International)
            .build()
            .is_err()
    );
    let mut invalid = profile();
    invalid.supports_websockets = true;
    assert!(
        LlmClientBuilder::with_transport(Arc::new(ModelTransport::default()), &[invalid])
            .with_region(Region::International)
            .build()
            .is_err()
    );
    assert_eq!(profile().auth, AuthStrategy::ChatGptPlan);

    let mut invalid = profile();
    invalid.pricing.billing_mode = lingxi_llm_client::protocol::BillingMode::Unknown;
    assert!(
        LlmClientBuilder::with_transport(Arc::new(ModelTransport::default()), &[invalid])
            .with_region(Region::International)
            .build()
            .is_err()
    );

    let mut invalid = profile();
    invalid.background = serde_json::from_value(json!({
        "mode":"enabled",
        "value":{"endpoint":"https://api.openai.com/v1/responses","auth":{"type":"bearer"}}
    }))
    .unwrap();
    assert!(
        LlmClientBuilder::with_transport(Arc::new(ModelTransport::default()), &[invalid])
            .with_region(Region::International)
            .build()
            .is_err()
    );
}

#[test]
fn plan_error_codes_preserve_recovery_categories() {
    for (code, expected) in [
        ("subscription_sharing_usage_limit_exceeded", "quota"),
        ("subscription_sharing_user_not_eligible", "permission"),
        ("subscription_sharing_usage_unavailable", "provider"),
        ("subscription_sharing_unsupported_capability", "invalid"),
        ("subscription_sharing_invalid_user", "auth"),
    ] {
        let error = classify_error(500, &json!({"error":{"code":code,"message":"test"}}), None);
        let category = match error {
            LlmError::QuotaExceeded { .. } => "quota",
            LlmError::PermissionDenied { .. } => "permission",
            LlmError::ProviderInternal { .. } => "provider",
            LlmError::InvalidRequest { .. } => "invalid",
            LlmError::Authentication { .. } => "auth",
            other => panic!("unexpected error for {code}: {other}"),
        };
        assert_eq!(category, expected);
    }
}

#[derive(Default)]
struct FailoverTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FailoverTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(StreamResponse {
            status: 429,
            headers: vec![("x-request-id".into(), "req-plan-limit".into())],
            body: stream::once(async {
                Ok(Bytes::from_static(b"{\"error\":{\"code\":\"subscription_sharing_usage_limit_exceeded\",\"param\":\"model\",\"message\":\"app limit\"}}"))
            })
            .boxed(),
        })
    }
}

#[tokio::test]
async fn plan_usage_failure_never_switches_to_api_key_billing() {
    let transport = Arc::new(FailoverTransport::default());
    let mut plan = profile();
    plan.connection.group = Some("billing-group".into());
    plan.connection.connection_id = Some("plan".into());
    plan.connection.failover.rate_limit = true;
    let mut key = profile();
    key.profile_name = "api-key".into();
    key.auth = AuthStrategy::ApiKey;
    key.connection.group = Some("billing-group".into());
    key.connection.connection_id = Some("key".into());
    key.connection.order = 1;
    key.connection.failover.rate_limit = true;
    let client = LlmClientBuilder::with_transport(transport.clone(), &[plan, key])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(client
        .resolve_in("gpt-6.1-sol", Some("chatgpt-plan"))
        .unwrap()
        .connection_chain
        .is_empty());
    let mut chat = ChatRequest::new("gpt-6.1-sol");
    chat.messages
        .push(lingxi_llm_client::protocol::ConversationMessage::user_text(
            "Hi",
        ));
    chat.controls.responses.store = Some(false);
    let options = RequestOptions {
        credential: Some(Secret::new("plan-token".into())),
        fallback_credentials: [("api-key".into(), Secret::new("key-token".into()))].into(),
        ..Default::default()
    };
    let result = client
        .chat()
        .stream_in("chatgpt-plan", &chat, &options)
        .await;
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("expected a plan usage failure"),
    };
    match error {
        LlmError::ProviderResponse {
            status,
            request_id,
            body,
            classification,
            ..
        } => {
            assert_eq!(status, 429);
            assert_eq!(request_id.as_deref(), Some("req-plan-limit"));
            assert_eq!(classification, LlmErrorKind::QuotaExceeded);
            assert_eq!(
                body["error"]["code"],
                "subscription_sharing_usage_limit_exceeded"
            );
            assert_eq!(body["error"]["param"], "model");
        }
        other => panic!("expected structured plan failure, got {other}"),
    }
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn api_key_failure_never_switches_to_chatgpt_plan_billing() {
    let transport = Arc::new(FailoverTransport::default());
    let mut key = profile();
    key.profile_name = "api-key".into();
    key.auth = AuthStrategy::ApiKey;
    key.connection.group = Some("billing-group".into());
    key.connection.connection_id = Some("key".into());
    key.connection.failover.rate_limit = true;
    let mut plan = profile();
    plan.connection.group = Some("billing-group".into());
    plan.connection.connection_id = Some("plan".into());
    plan.connection.order = 1;
    plan.connection.failover.rate_limit = true;
    let client = LlmClientBuilder::with_transport(transport.clone(), &[key, plan])
        .with_region(Region::International)
        .build()
        .unwrap();
    assert!(client
        .resolve_in("gpt-6.1-sol", Some("api-key"))
        .unwrap()
        .connection_chain
        .is_empty());
    let mut chat = ChatRequest::new("gpt-6.1-sol");
    chat.messages
        .push(lingxi_llm_client::protocol::ConversationMessage::user_text(
            "Hi",
        ));
    chat.controls.responses.store = Some(false);
    let options = RequestOptions {
        credential: Some(Secret::new("key-token".into())),
        fallback_credentials: [("chatgpt-plan".into(), Secret::new("plan-token".into()))].into(),
        ..Default::default()
    };
    let result = client.chat().stream_in("api-key", &chat, &options).await;
    assert!(matches!(result, Err(LlmError::QuotaExceeded { .. })));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn model_discovery_keeps_status_request_id_and_error_body() {
    let transport = Arc::new(FailoverTransport::default());
    let client = LlmClientBuilder::with_transport(transport, &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let openai = client.provider::<OpenAiClient>("chatgpt-plan").unwrap();
    let error = openai
        .chatgpt_plan_models(&Secret::new("plan-token".into()))
        .await
        .unwrap_err();
    match error {
        ChatGptPlanModelError::Provider {
            status,
            request_id,
            body,
        } => {
            assert_eq!(status, 429);
            assert_eq!(request_id.as_deref(), Some("req-plan-limit"));
            assert_eq!(
                body["error"]["code"],
                "subscription_sharing_usage_limit_exceeded"
            );
        }
        other => panic!("expected structured provider error, got {other}"),
    }
}

#[test]
fn plan_profile_cannot_open_container_service() {
    let client =
        LlmClientBuilder::with_transport(Arc::new(ModelTransport::default()), &[profile()])
            .with_region(Region::International)
            .build()
            .unwrap();
    let openai = client.provider::<OpenAiClient>("chatgpt-plan").unwrap();
    let scope = OpenAiContainerScope::new("chatgpt-plan", "account-1").unwrap();
    assert!(openai.containers(scope).is_err());
}

struct StreamPayloadTransport {
    payload: &'static [u8],
    status: u16,
}

#[async_trait]
impl Transport for StreamPayloadTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let payload = self.payload;
        Ok(StreamResponse {
            status: self.status,
            headers: vec![("x-request-id".into(), "req-stream".into())],
            body: stream::once(async move { Ok(Bytes::from_static(payload)) }).boxed(),
        })
    }
}

async fn plan_stream_events(payload: &'static [u8]) -> Vec<Result<StreamEvent, LlmError>> {
    plan_stream_events_with_status(payload, 200).await
}

async fn plan_stream_events_with_status(
    payload: &'static [u8],
    status: u16,
) -> Vec<Result<StreamEvent, LlmError>> {
    let client = LlmClientBuilder::with_transport(
        Arc::new(StreamPayloadTransport { payload, status }),
        &[profile()],
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    let mut stream = client
        .chat()
        .stream_in(
            "chatgpt-plan",
            &chat(false),
            &RequestOptions {
                credential: Some(Secret::new("plan-token".into())),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn plan_stream_needs_response_completed_not_done_sentinel() {
    let events = plan_stream_events(
        b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"partial\"}\n\ndata: [DONE]\n\n",
    )
    .await;
    assert!(!events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::End { .. }))));
    assert!(events
        .iter()
        .any(|event| matches!(event, Err(LlmError::StreamInterrupted { .. }))));
}

#[tokio::test]
async fn plan_stream_preserves_failed_response_details() {
    let events = plan_stream_events(
        b"data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"subscription_sharing_usage_limit_exceeded\",\"param\":\"model\",\"message\":\"app limit\"}}}\n\n",
    )
    .await;
    let error = events.into_iter().find_map(Result::err).unwrap();
    match error {
        LlmError::ProviderResponse {
            status,
            request_id,
            body,
            classification,
            ..
        } => {
            assert_eq!(status, 200);
            assert_eq!(request_id.as_deref(), Some("req-stream"));
            assert_eq!(classification, LlmErrorKind::QuotaExceeded);
            assert_eq!(
                body["error"]["code"],
                "subscription_sharing_usage_limit_exceeded"
            );
            assert_eq!(body["error"]["param"], "model");
        }
        other => panic!("expected structured stream failure, got {other}"),
    }
}

#[tokio::test]
async fn plan_stream_error_event_keeps_http_status() {
    for http_status in [200, 201] {
        let events = plan_stream_events_with_status(
            b"data: {\"type\":\"error\",\"error\":{\"code\":\"subscription_sharing_usage_unavailable\",\"message\":\"try later\"}}\n\n",
            http_status,
        )
        .await;
        let error = events.into_iter().find_map(Result::err).unwrap();
        match error {
            LlmError::ProviderResponse {
                status,
                request_id,
                body,
                classification,
                ..
            } => {
                assert_eq!(status, http_status);
                assert_eq!(request_id.as_deref(), Some("req-stream"));
                assert_eq!(classification, LlmErrorKind::ProviderInternal);
                assert_eq!(
                    body["error"]["code"],
                    "subscription_sharing_usage_unavailable"
                );
            }
            other => panic!("expected structured stream error, got {other}"),
        }
    }
}

#[tokio::test]
async fn plan_stream_reports_incomplete_response_as_failure() {
    let events = plan_stream_events(
        b"data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
    )
    .await;
    assert!(!events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::End { .. }))));
    assert!(events.iter().any(|event| matches!(
        event,
        Err(LlmError::ProviderResponse {
            classification: LlmErrorKind::StreamInterrupted,
            ..
        })
    )));
}
