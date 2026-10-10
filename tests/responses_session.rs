use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{
        AuthStrategy, ChatRequest, ContinuationRef, LlmError, ProviderProfile, Region, ResponseId,
        Secret,
    },
    transport::WebSocketConnection,
    Authenticator, HttpRequest, LlmClient, LlmClientBuilder, RequestDraft, RequestMode,
    RequestOptions, ResponsesSession, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
struct RecordingTransport {
    handshakes: Mutex<Vec<HttpRequest>>,
    sent: Arc<Mutex<Vec<Value>>>,
    closes: Arc<AtomicUsize>,
    reject_next_upgrade: AtomicUsize,
    output: Vec<Value>,
}
#[async_trait]
impl Transport for RecordingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("these tests only dispatch WebSocket requests")
    }
    async fn connect_websocket(
        &self,
        request: HttpRequest,
    ) -> Result<Box<dyn WebSocketConnection>, LlmError> {
        self.handshakes.lock().unwrap().push(request);
        if self.reject_next_upgrade.swap(0, Ordering::SeqCst) != 0 {
            return Err(LlmError::InvalidRequest {
                message: "426 Upgrade Required".into(),
            });
        }
        Ok(Box::new(RecordingConnection {
            sent: self.sent.clone(),
            closes: self.closes.clone(),
            output: self.output.clone(),
        }))
    }
}
struct RecordingConnection {
    sent: Arc<Mutex<Vec<Value>>>,
    closes: Arc<AtomicUsize>,
    output: Vec<Value>,
}
#[async_trait]
impl WebSocketConnection for RecordingConnection {
    async fn send(&mut self, body: Bytes) -> Result<StreamResponse, LlmError> {
        let mut sent = self.sent.lock().unwrap();
        sent.push(serde_json::from_slice(&body).unwrap());
        let frame = json!({
            "type":"response.completed",
            "response":{"id":format!("resp_{}", sent.len()), "model":"wire", "status":"completed", "output":self.output}
        });
        let created = json!({
            "type":"response.created",
            "response":{"id":format!("resp_{}", sent.len()), "model":"wire"}
        });
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: stream::iter([
                Ok(Bytes::from(serde_json::to_vec(&created).unwrap())),
                Ok(Bytes::from(serde_json::to_vec(&frame).unwrap())),
            ])
            .boxed(),
        })
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name":"primary", "provider_id":"test", "base_url":"https://example.test",
        "protocol":"open_ai_responses", "auth":"bearer", "supports_websockets":true,
        "extra":{"supports_previous_response_id":true},
        "models":[{"request_model":"wire", "display_model":"test", "billing_model":"wire"}]
    }))
    .unwrap()
}
fn client(http: Arc<RecordingTransport>, profile: ProviderProfile) -> LlmClient {
    let mut builder =
        LlmClientBuilder::with_transport(http, &[profile]).with_region(Region::International);
    builder.register_authenticator(
        AuthStrategy::OAuthBearer,
        Arc::new(lingxi_llm_client::BearerAuthenticator),
    );
    builder.build().unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"test", "messages":[]})).unwrap()
}
fn options(scope: &str, credential: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new(credential.into())),
        ..Default::default()
    }
}
async fn run(
    client: &LlmClient,
    profile: &str,
    request: &ChatRequest,
    options: &RequestOptions,
    session: &mut ResponsesSession,
) -> ContinuationRef {
    let mut draft = client
        .prepare_draft_on(profile, request, options, RequestMode::Stream)
        .await
        .unwrap();
    session.prepare(&mut draft, false, false).await.unwrap();
    let mut stream = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
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
    stream.continuation().unwrap().clone()
}

#[tokio::test]
async fn same_identity_reuses_connection_and_implicit_continuation() {
    let http = Arc::new(RecordingTransport::default());
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let options = options("account-a", "key-a");
    run(&client, "primary", &request(), &options, &mut session).await;
    run(&client, "primary", &request(), &options, &mut session).await;
    assert_eq!(http.handshakes.lock().unwrap().len(), 1);
    assert_eq!(
        http.sent.lock().unwrap()[1]["previous_response_id"],
        "resp_1"
    );
}

#[tokio::test]
async fn account_and_connection_identity_changes_reconnect_without_continuation() {
    for changed in [
        "account",
        "credential",
        "endpoint",
        "profile",
        "provider",
        "connection",
        "auth",
    ] {
        let http = Arc::new(RecordingTransport::default());
        let mut next_profile = profile();
        let first = client(http.clone(), profile());
        let mut next_options = options("account-a", "key-a");
        match changed {
            "account" => next_options.account_scope = Some("account-b".into()),
            "credential" => next_options.credential = Some(Secret::new("key-b".into())),
            "endpoint" => next_profile.base_url = "https://other.test".into(),
            "profile" => next_profile.profile_name = "secondary".into(),
            "provider" => next_profile.provider_id = "other".into(),
            "connection" => next_profile.connection.connection_id = Some("secondary".into()),
            "auth" => next_profile.auth = AuthStrategy::OAuthBearer,
            _ => unreachable!(),
        }
        let next_name = next_profile.profile_name.clone();
        let second = client(http.clone(), next_profile);
        let mut session = ResponsesSession::new();
        run(
            &first,
            "primary",
            &request(),
            &options("account-a", "key-a"),
            &mut session,
        )
        .await;
        let continuation = run(&second, &next_name, &request(), &next_options, &mut session).await;
        assert_eq!(http.handshakes.lock().unwrap().len(), 2, "{changed}");
        assert_eq!(http.closes.load(Ordering::SeqCst), 1, "{changed}");
        assert!(
            http.sent.lock().unwrap()[1]
                .get("previous_response_id")
                .is_none(),
            "{changed}"
        );
        assert_eq!(
            continuation.account_scope,
            next_options.account_scope.unwrap()
        );
    }
}

struct MutableAuthenticator {
    value: Mutex<Option<String>>,
    calls: AtomicUsize,
}
#[async_trait]
impl Authenticator for MutableAuthenticator {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let value = self
            .value
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "credential unavailable".into(),
            })?;
        request.headers.push(("authorization".into(), value));
        Ok(())
    }
}

#[tokio::test]
async fn custom_authentication_changes_and_failures_invalidate_connection() {
    let http = Arc::new(RecordingTransport::default());
    let auth = Arc::new(MutableAuthenticator {
        value: Mutex::new(Some("key-a".into())),
        calls: AtomicUsize::new(0),
    });
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile()])
        .with_region(Region::International);
    builder.register_authenticator(AuthStrategy::Bearer, auth.clone());
    let client = builder.build().unwrap();
    let options = RequestOptions {
        account_scope: Some("same-account".into()),
        ..Default::default()
    };
    let mut session = ResponsesSession::new();
    run(&client, "primary", &request(), &options, &mut session).await;
    *auth.value.lock().unwrap() = Some("key-b".into());
    run(&client, "primary", &request(), &options, &mut session).await;
    assert_eq!(http.handshakes.lock().unwrap().len(), 2);
    // One handshake authentication and one generation authentication per turn.
    assert_eq!(auth.calls.load(Ordering::SeqCst), 4);
    assert!(http.sent.lock().unwrap()[1]
        .get("previous_response_id")
        .is_none());
    *auth.value.lock().unwrap() = None;
    let mut draft = client
        .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    assert!(session.prepare(&mut draft, false, false).await.is_err());
    assert_eq!(http.closes.load(Ordering::SeqCst), 2);
    assert!(session.last_response_id().is_none());
}

#[tokio::test]
async fn explicit_branch_wins_over_cached_response_and_does_not_seed_implicit_history() {
    let http = Arc::new(RecordingTransport::default());
    let client = client(http.clone(), profile());
    let options = options("account-a", "key-a");
    let mut session = ResponsesSession::new();
    let mut branch = run(&client, "primary", &request(), &options, &mut session).await;
    branch.response_id = ResponseId::new("resp_external_branch");
    let mut explicit = request();
    explicit.continuation = Some(branch);
    run(&client, "primary", &explicit, &options, &mut session).await;
    run(&client, "primary", &request(), &options, &mut session).await;
    let sent = http.sent.lock().unwrap();
    assert_eq!(sent[1]["previous_response_id"], "resp_external_branch");
    assert!(sent[2].get("previous_response_id").is_none());
    assert_eq!(http.handshakes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn changing_accounts_resets_http_fallback() {
    let http = Arc::new(RecordingTransport::default());
    http.reject_next_upgrade.store(1, Ordering::SeqCst);
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account-a", "key-a"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut draft, false, true).await.unwrap();
    assert!(session.fallback_to_http());
    run(
        &client,
        "primary",
        &request(),
        &options("account-b", "key-a"),
        &mut session,
    )
    .await;
    assert!(!session.fallback_to_http());
    assert_eq!(http.handshakes.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn old_response_observer_cannot_repopulate_a_new_accounts_state() {
    let http = Arc::new(RecordingTransport::default());
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account-a", "key-a"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut draft, false, false).await.unwrap();
    let old_received = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
        .await
        .unwrap();
    run(
        &client,
        "primary",
        &request(),
        &options("account-b", "key-a"),
        &mut session,
    )
    .await;
    let mut old_stream = old_received.into_stream().ok().unwrap();
    while old_stream.next_batch().await.is_some() {}
    assert_eq!(session.last_response_id().as_deref(), Some("resp_2"));
}

#[tokio::test]
async fn dispatch_rejects_an_old_preparation_without_consuming_the_current_one() {
    let http = Arc::new(RecordingTransport::default());
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let mut first = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account-a", "key-a"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut first, false, false).await.unwrap();
    let first = first.seal().await.unwrap();
    let mut second = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account-b", "key-b"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut second, false, false).await.unwrap();
    let second = second.seal().await.unwrap();
    assert!(matches!(
        session
            .dispatch(first, || panic!("must not admit stale call"))
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert!(http.sent.lock().unwrap().is_empty());
    assert!(session.dispatch(second, || Ok(())).await.is_ok());
    assert_eq!(http.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn dispatch_rejects_drafts_mutated_after_session_preparation() {
    let http = Arc::new(RecordingTransport::default());
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account-a", "key-a"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut draft, false, false).await.unwrap();
    draft.request_mut().url = "https://other.test/v1/responses".into();
    assert!(matches!(
        session
            .dispatch(draft.seal().await.unwrap(), || panic!(
                "must not admit changed call"
            ))
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert!(http.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cached_http_fallback_still_rejects_websocket_only_prewarm() {
    let http = Arc::new(RecordingTransport::default());
    http.reject_next_upgrade.store(1, Ordering::SeqCst);
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    for _ in 0..2 {
        let mut draft = client
            .prepare_draft_on(
                "primary",
                &request(),
                &options("account", "key"),
                RequestMode::Stream,
            )
            .await
            .unwrap();
        assert!(session.prepare(&mut draft, true, false).await.is_err());
        assert!(session.fallback_to_http());
        // A rejected preparation must not leave a dispatchable binding.
        assert!(session
            .dispatch(draft.seal().await.unwrap(), || panic!(
                "must not admit rejected prewarm"
            ))
            .await
            .is_err());
    }
    assert_eq!(http.handshakes.lock().unwrap().len(), 1);
    assert!(http.sent.lock().unwrap().is_empty());
    let mut allowed = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account", "key"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    assert!(session.prepare(&mut allowed, false, true).await.is_ok());
}

#[tokio::test]
async fn trace_header_changes_preserve_connection_and_continuation() {
    for prewarm in [false, true] {
        let http = Arc::new(RecordingTransport::default());
        let client = client(http.clone(), profile());
        let mut session = ResponsesSession::new();
        let options = options("account-a", "key-a");
        for (index, trace) in ["request-1", "request-2"].into_iter().enumerate() {
            let mut draft = client
                .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
                .await
                .unwrap();
            draft
                .request_mut()
                .headers
                .push(("x-request-id".into(), trace.into()));
            session
                .prepare(&mut draft, prewarm && index == 0, false)
                .await
                .unwrap();
            let mut stream = session
                .dispatch(draft.seal().await.unwrap(), || Ok(()))
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
        }
        assert_eq!(
            http.sent.lock().unwrap()[1]["previous_response_id"],
            "resp_1"
        );
        assert_eq!(http.closes.load(Ordering::SeqCst), 0);
        assert_eq!(
            http.handshakes.lock().unwrap().len(),
            1,
            "trace header must not force reconnect"
        );
        assert_eq!(
            session
                .last_request_snapshot()
                .wire_used_prewarm_response_id,
            prewarm
        );
    }
}

#[tokio::test]
async fn full_history_continuation_omits_completed_assistant_output() {
    let http = Arc::new(RecordingTransport {
        output: vec![
            json!({"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello","annotations":[],"logprobs":[]} ]}),
        ],
        ..Default::default()
    });
    let client = client(http.clone(), profile());
    let mut session = ResponsesSession::new();
    let options = options("account", "key");
    let first = serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]})).unwrap();
    let next = serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]},{"role":"assistant","content":[{"type":"text","text":"Hello"}]},{"role":"user","content":[{"type":"text","text":"Next"}]}]})).unwrap();
    run(&client, "primary", &first, &options, &mut session).await;
    run(&client, "primary", &next, &options, &mut session).await;
    let sent = http.sent.lock().unwrap();
    assert_eq!(sent[1]["previous_response_id"], "resp_1");
    assert_eq!(sent[1]["input"].as_array().unwrap().len(), 1);
    assert_eq!(sent[1]["input"][0]["content"][0]["text"], "Next");
}

#[tokio::test]
async fn replacing_client_transport_reconnects_but_shared_transport_reuses() {
    let a = Arc::new(RecordingTransport::default());
    let b = Arc::new(RecordingTransport::default());
    let ca = client(a.clone(), profile());
    let cb = client(b.clone(), profile());
    let cb2 = client(b.clone(), profile());
    let mut session = ResponsesSession::new();
    let options = options("account", "key");
    run(&ca, "primary", &request(), &options, &mut session).await;
    run(&cb, "primary", &request(), &options, &mut session).await;
    assert_eq!(a.handshakes.lock().unwrap().len(), 1);
    assert_eq!(a.closes.load(Ordering::SeqCst), 1);
    assert_eq!(a.sent.lock().unwrap().len(), 1);
    assert_eq!(b.handshakes.lock().unwrap().len(), 1);
    assert!(b.sent.lock().unwrap()[0]
        .get("previous_response_id")
        .is_none());
    run(&cb2, "primary", &request(), &options, &mut session).await;
    assert_eq!(b.handshakes.lock().unwrap().len(), 1);
    assert_eq!(b.sent.lock().unwrap()[1]["previous_response_id"], "resp_1");
}

#[tokio::test]
async fn explicit_transport_replacement_rebinds_before_dispatch() {
    let a = Arc::new(RecordingTransport::default());
    let b = Arc::new(RecordingTransport::default());
    let client = client(a.clone(), profile());
    let options = options("account", "key");
    let mut session = ResponsesSession::new();
    run(&client, "primary", &request(), &options, &mut session).await;
    let mut draft = client
        .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    session
        .prepare_using(&mut draft, false, false, Some(b.clone()))
        .await
        .unwrap();
    let mut stream = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
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
    assert_eq!(a.closes.load(Ordering::SeqCst), 1);
    assert_eq!(a.sent.lock().unwrap().len(), 1);
    assert_eq!(b.handshakes.lock().unwrap().len(), 1);
    assert!(b.sent.lock().unwrap()[0]
        .get("previous_response_id")
        .is_none());
    // Changing the dispatch transport after preparation must fail before admission.
    let mut draft = client
        .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    session
        .prepare_using(&mut draft, false, false, Some(b.clone()))
        .await
        .unwrap();
    assert!(session
        .dispatch_using(
            draft.seal().await.unwrap(),
            || panic!("wrong transport must not admit"),
            Some(a.as_ref())
        )
        .await
        .is_err());
    assert_eq!(b.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn changing_transport_clears_cached_http_fallback() {
    let a = Arc::new(RecordingTransport::default());
    a.reject_next_upgrade.store(1, Ordering::SeqCst);
    let b = Arc::new(RecordingTransport::default());
    let client = client(a.clone(), profile());
    let options = options("account", "key");
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    session.prepare(&mut draft, false, true).await.unwrap();
    assert!(session.fallback_to_http());
    let mut next = client
        .prepare_draft_on("primary", &request(), &options, RequestMode::Stream)
        .await
        .unwrap();
    session
        .prepare_using(&mut next, false, false, Some(b.clone()))
        .await
        .unwrap();
    assert!(!session.fallback_to_http());
    assert_eq!(b.handshakes.lock().unwrap().len(), 1);
    assert!(session
        .dispatch(draft.seal().await.unwrap(), || panic!(
            "old transport draft must not admit"
        ))
        .await
        .is_err());
    assert!(session
        .dispatch(next.seal().await.unwrap(), || Ok(()))
        .await
        .is_ok());
}

#[tokio::test]
async fn http_fallback_uses_the_transport_selected_during_preparation() {
    #[derive(Default)]
    struct HttpOnly(AtomicUsize);
    #[async_trait]
    impl Transport for HttpOnly {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(lingxi_llm_client::HttpResponse {
                status: 200,
                headers: vec![],
                body: br#"{"id":"r","output":[]}"#.as_slice().into(),
            }
            .into())
        }
        async fn connect_websocket(
            &self,
            _: HttpRequest,
        ) -> Result<Box<dyn WebSocketConnection>, LlmError> {
            Err(LlmError::Transport {
                message: "426 Upgrade Required".into(),
            })
        }
    }
    let default_transport = Arc::new(HttpOnly::default());
    let selected = Arc::new(HttpOnly::default());
    let client = LlmClientBuilder::with_transport(default_transport.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &request(),
            &options("account", "key"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session
        .prepare_using(&mut draft, false, true, Some(selected.clone()))
        .await
        .unwrap();
    assert!(session.fallback_to_http());
    let received = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
        .await
        .unwrap();
    received.collect().await.unwrap().finish().await;
    assert_eq!(selected.0.load(Ordering::SeqCst), 1);
    assert_eq!(default_transport.0.load(Ordering::SeqCst), 0);
}

struct AcceptAnyCredential;
#[async_trait]
impl Authenticator for AcceptAnyCredential {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}
fn chatgpt_client(transport: Arc<dyn Transport>) -> LlmClient {
    let mut profile = profile();
    profile.auth = AuthStrategy::ChatGptOAuth;
    let mut builder =
        LlmClientBuilder::with_transport(transport, &[profile]).with_region(Region::International);
    builder.register_authenticator(AuthStrategy::ChatGptOAuth, Arc::new(AcceptAnyCredential));
    builder.build().unwrap()
}
fn generation_request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"test",
        "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],
        "max_tokens":128,
        "temperature":0.2
    }))
    .unwrap()
}
fn assert_chatgpt_generation_wire(frame: &Value) {
    assert_eq!(frame["store"], false);
    assert!(frame.get("temperature").is_none());
    assert!(frame.get("max_output_tokens").is_none());
    assert!(frame.get("instructions").is_some());
}
async fn dispatch_and_drain(session: &mut ResponsesSession, draft: RequestDraft) {
    let mut stream = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
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
}

#[tokio::test]
async fn chatgpt_oauth_generation_policy_is_applied_before_the_session_prepares() {
    let http = Arc::new(RecordingTransport::default());
    let client = chatgpt_client(http.clone());
    let options = options("account-a", "key-a");
    let mut session = ResponsesSession::new();
    run(&client, "primary", &generation_request(), &options, &mut session).await;
    run(&client, "primary", &generation_request(), &options, &mut session).await;
    assert_eq!(http.handshakes.lock().unwrap().len(), 1);
    let sent = http.sent.lock().unwrap();
    assert_eq!(sent.len(), 2);
    sent.iter().for_each(assert_chatgpt_generation_wire);
    assert_eq!(sent[1]["previous_response_id"], "resp_1");
}

#[tokio::test]
async fn host_rewriting_the_body_before_preparation_keeps_the_session_binding() {
    let http = Arc::new(RecordingTransport::default());
    let client = chatgpt_client(http.clone());
    let options = options("account-a", "key-a");
    let mut session = ResponsesSession::new();
    for _ in 0..2 {
        let mut draft = client
            .prepare_draft_on(
                "primary",
                &generation_request(),
                &options,
                RequestMode::Stream,
            )
            .await
            .unwrap();
        // The host applies the route's body policy, reads the JSON, then writes
        // it back before the session prepares the draft.
        draft
            .apply_request_body_auth_policy(AuthStrategy::ChatGptOAuth)
            .unwrap();
        let body = draft.semantic_body_json().unwrap();
        draft.set_json_body(body, &Default::default()).unwrap();
        session.prepare(&mut draft, false, false).await.unwrap();
        dispatch_and_drain(&mut session, draft).await;
    }
    assert_eq!(http.handshakes.lock().unwrap().len(), 1);
    let sent = http.sent.lock().unwrap();
    assert_eq!(sent.len(), 2);
    sent.iter().for_each(assert_chatgpt_generation_wire);
    assert_eq!(sent[1]["previous_response_id"], "resp_1");
}

#[tokio::test]
async fn chatgpt_oauth_prewarm_applies_the_policy_and_keeps_the_binding() {
    let http = Arc::new(RecordingTransport::default());
    let client = chatgpt_client(http.clone());
    let options = options("account-a", "key-a");
    let mut session = ResponsesSession::new();
    let mut warm = client
        .prepare_draft_on(
            "primary",
            &generation_request(),
            &options,
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut warm, true, false).await.unwrap();
    dispatch_and_drain(&mut session, warm).await;
    run(&client, "primary", &generation_request(), &options, &mut session).await;
    let sent = http.sent.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0]["generate"], false);
    sent.iter().for_each(assert_chatgpt_generation_wire);
}

#[tokio::test]
async fn chatgpt_oauth_http_fallback_keeps_the_session_binding() {
    #[derive(Default)]
    struct HttpFallback(Mutex<Vec<Value>>);
    #[async_trait]
    impl Transport for HttpFallback {
        async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.0
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&request.body).unwrap());
            Ok(lingxi_llm_client::HttpResponse {
                status: 200,
                headers: vec![],
                body: br#"{"id":"r","output":[]}"#.as_slice().into(),
            }
            .into())
        }
        async fn connect_websocket(
            &self,
            _: HttpRequest,
        ) -> Result<Box<dyn WebSocketConnection>, LlmError> {
            Err(LlmError::Transport {
                message: "426 Upgrade Required".into(),
            })
        }
    }
    let transport = Arc::new(HttpFallback::default());
    let client = chatgpt_client(transport.clone());
    let mut session = ResponsesSession::new();
    let mut draft = client
        .prepare_draft_on(
            "primary",
            &generation_request(),
            &options("account", "key"),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    session.prepare(&mut draft, false, true).await.unwrap();
    assert!(session.fallback_to_http());
    let received = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
        .await
        .unwrap();
    received.collect().await.unwrap().finish().await;
    let sent = transport.0.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_chatgpt_generation_wire(&sent[0]);
}
