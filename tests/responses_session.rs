use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{
        AuthStrategy, ChatRequest, ContinuationRef, LlmError, ProviderProfile, Region, ResponseId,
        Secret,
    },
    transport::WebSocketConnection,
    Authenticator, HttpRequest, LlmClient, LlmClientBuilder, RequestMode, RequestOptions,
    ResponsesSession, StreamResponse, Transport,
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
        }))
    }
}
struct RecordingConnection {
    sent: Arc<Mutex<Vec<Value>>>,
    closes: Arc<AtomicUsize>,
}
#[async_trait]
impl WebSocketConnection for RecordingConnection {
    async fn send(&mut self, body: Bytes) -> Result<StreamResponse, LlmError> {
        let mut sent = self.sent.lock().unwrap();
        sent.push(serde_json::from_slice(&body).unwrap());
        let frame = json!({
            "type":"response.completed",
            "response":{"id":format!("resp_{}", sent.len()), "model":"wire", "status":"completed", "output":[]}
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
